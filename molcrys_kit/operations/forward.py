"""Deterministic symmetry-constrained molecular-crystal construction.

This module implements the forward half of the crystallographic data model
described in Issue #172.  It deliberately does not search, relax, or infer a
packing arrangement.  Callers provide a complete rigid molecule, a cell, a
validated operation set, and one or more poses.  The same prepared context is
then used for site analysis, ordered expansion, and the explicit statistical
disorder path.

Coordinates follow the package convention: lattice vectors are rows, a
fractional row ``f`` maps to Cartesian coordinates as ``f @ L``, and a
``FractionalAffineOperation`` acts as ``f @ W.T + w``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import networkx as nx
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from ..constants.config import (
    KEY_ASSEMBLY,
    KEY_ASYM_ID,
    KEY_ATOM_ID,
    KEY_DISORDER_GROUP,
    KEY_LABEL,
    KEY_IMAGE_SHIFT,
    KEY_OCCUPANCY,
    KEY_SITE_SYMMETRY_ORDER,
    KEY_SYM_OP_INDEX,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
    identity_operation,
)


DEFAULT_GEOMETRY_TOLERANCE_A = 1.0e-5
DEFAULT_FRACTIONAL_TOLERANCE = 1.0e-7


class ConstructionValidationError(ValueError):
    """Base error for invalid forward-construction input."""


class InvalidOrientationError(ConstructionValidationError):
    """Raised when a pose orientation is not a proper rigid rotation."""


class InvalidSymmetryError(ConstructionValidationError):
    """Raised when a supplied finite operation set is not a group."""


class IncompatibleSiteError(ConstructionValidationError):
    """Raised when strict ordered construction meets an incompatible site."""

    def __init__(self, analysis: "SiteSymmetryAnalysis") -> None:
        self.analysis = analysis
        names = ", ".join(str(op.index if op.index is not None else i)
                           for i, op in enumerate(analysis.conflicting_operations))
        super().__init__(
            "molecular pose is incompatible with a special position: "
            f"|S|={analysis.site_symmetry_order}, |K|={analysis.stabilizer_order}, "
            f"m={analysis.orbit_multiplicity}, k={analysis.orientation_multiplicity}; "
            f"conflicting operations [{names}]"
        )


def _readonly_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ConstructionValidationError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ConstructionValidationError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _canonical_fractional(value: np.ndarray) -> np.ndarray:
    """Return a coordinate in the half-open unit cube."""
    result = np.asarray(value, dtype=float) - np.floor(np.asarray(value, dtype=float))
    result[np.isclose(result, 1.0, atol=DEFAULT_FRACTIONAL_TOLERANCE)] = 0.0
    return result


def _fractional_equal(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return bool(np.allclose(delta, np.rint(delta), atol=tolerance, rtol=0.0))


def _equivalent(operation: FractionalAffineOperation,
                operations: Iterable[FractionalAffineOperation],
                tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE) -> FractionalAffineOperation | None:
    return next((item for item in operations if operation.equivalent_to(item, tolerance=tolerance)), None)


def _deduplicate_operations(operations: Iterable[FractionalAffineOperation], tolerance: float) -> tuple[FractionalAffineOperation, ...]:
    result: list[FractionalAffineOperation] = []
    for operation in operations:
        if _equivalent(operation, result, tolerance) is None:
            result.append(operation)
    return tuple(result)


def _validate_group(operations: Sequence[FractionalAffineOperation], lattice: np.ndarray,
                    tolerance: float) -> tuple[FractionalAffineOperation, ...]:
    raw_group = tuple(operations)
    for index, operation in enumerate(raw_group):
        if _equivalent(operation, raw_group[:index], tolerance) is not None:
            raise InvalidSymmetryError("symmetry operation set contains duplicate operations")
    group = raw_group
    if not group:
        raise InvalidSymmetryError("at least one symmetry operation is required")
    if _equivalent(identity_operation(), group, tolerance) is None:
        raise InvalidSymmetryError("symmetry operation set does not contain identity")
    for operation in group:
        operation.validate_metric(lattice)
        if _equivalent(operation.inverse(), group, tolerance) is None:
            raise InvalidSymmetryError("symmetry operation set is missing an inverse")
    for left in group:
        for right in group:
            if _equivalent(left.compose(right), group, tolerance) is None:
                raise InvalidSymmetryError("symmetry operation set is not closed under composition")
    # Source CIF order is not a crystallographic invariant.  Canonicalizing
    # the finite set makes all generated molecule/atom order deterministic
    # even when an equivalent operation loop is permuted by an input reader.
    return tuple(
        sorted(
            group,
            key=lambda operation: (
                tuple(np.rint(operation.rotation).astype(int).ravel()),
                tuple(np.round(operation.canonical_translation(), 12)),
                operation.index if operation.index is not None else -1,
            ),
        )
    )


def _validate_lattice(lattice: Any) -> np.ndarray:
    result = _readonly_array(lattice, (3, 3), "lattice")
    determinant = float(np.linalg.det(result))
    if determinant <= 1.0e-8:
        raise ConstructionValidationError(
            "lattice must have finite, non-degenerate, positively handed volume"
        )
    return result


def _validate_orientation(value: Any, tolerance: float) -> np.ndarray:
    result = _readonly_array(value, (3, 3), "orientation")
    if not np.allclose(result.T @ result, np.eye(3), atol=tolerance, rtol=0.0):
        raise InvalidOrientationError("orientation must be orthonormal")
    if not np.isclose(float(np.linalg.det(result)), 1.0, atol=tolerance, rtol=0.0):
        raise InvalidOrientationError("orientation must be a proper rotation with determinant +1")
    return result


def _atom_identity_key(atoms: Atoms, index: int) -> tuple[Any, ...]:
    symbol = atoms.get_chemical_symbols()[index]
    isotope = atoms.arrays.get("isotope")
    charge = atoms.arrays.get("formal_charge")
    # Stereochemical labels are part of molecular identity.  They are
    # optional because ordinary ASE templates do not carry them, but when a
    # template does provide the annotation a symmetry operation may only map
    # atoms with the same annotation.
    stereo = None
    for key in (
        "stereochemistry",
        "stereo",
        "cip_code",
        "chemical_identity",
        "atom_identity",
    ):
        values = atoms.arrays.get(key)
        if values is not None and len(values) > index:
            stereo = str(values[index])
            break
    return (
        symbol,
        int(isotope[index]) if isotope is not None else 0,
        float(charge[index]) if charge is not None else 0.0,
        stereo,
    )


@dataclass(frozen=True)
class MolecularTemplate:
    """Immutable rigid molecular template used by the forward builder."""

    atoms: Atoms
    identifier: str = "template"
    anchor_index: int | None = None
    anchor_cartesian: np.ndarray | None = None
    connectivity: tuple[tuple[int, int], ...] = ()
    stereochemistry: tuple[Any, ...] = ()
    chemical_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        copy = self.atoms.copy()
        copy.set_cell(np.zeros((3, 3)))
        copy.set_pbc(False)
        copy.positions = np.asarray(self.atoms.get_positions(), dtype=float).copy()
        if self.anchor_index is not None and not 0 <= int(self.anchor_index) < len(copy):
            raise ConstructionValidationError("anchor_index is outside the template")
        if self.anchor_cartesian is not None:
            anchor = _readonly_array(self.anchor_cartesian, (3,), "anchor_cartesian")
        else:
            anchor = None
        object.__setattr__(self, "atoms", copy)
        object.__setattr__(self, "anchor_cartesian", anchor)
        connectivity_values: set[tuple[int, int]] = set()
        for edge in self.connectivity:
            if len(edge) != 2:
                raise ConstructionValidationError(
                    "connectivity edges must contain exactly two atom indices"
                )
            left, right = (int(edge[0]), int(edge[1]))
            if not 0 <= left < len(copy) or not 0 <= right < len(copy):
                raise ConstructionValidationError(
                    "connectivity edge index is outside the template"
                )
            if left == right:
                raise ConstructionValidationError("connectivity cannot contain self-edges")
            connectivity_values.add(tuple(sorted((left, right))))
        connectivity = tuple(sorted(connectivity_values))
        if not connectivity and getattr(self.atoms, "_graph", None) is not None:
            connectivity = tuple(
                sorted((min(int(left), int(right)), max(int(left), int(right)))
                       for left, right in self.atoms._graph.edges())
            )
        object.__setattr__(self, "connectivity", connectivity)
        stereo = tuple(self.stereochemistry)
        if stereo and len(stereo) != len(copy):
            raise ConstructionValidationError(
                "stereochemistry must contain one value per template atom"
            )
        if not stereo:
            for key in ("stereochemistry", "stereo", "cip_code"):
                values = copy.arrays.get(key)
                if values is not None and len(values) == len(copy):
                    stereo = tuple(str(value) for value in values)
                    break
        if stereo:
            copy.set_array("stereochemistry", np.asarray(stereo, dtype=str))
        object.__setattr__(self, "stereochemistry", stereo)
        if self.chemical_identity is not None:
            object.__setattr__(self, "chemical_identity", str(self.chemical_identity))

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    def anchor(self, mode: str | int | np.ndarray | None = None) -> np.ndarray:
        selected = self.anchor_index if mode is None else mode
        if isinstance(selected, (list, tuple, np.ndarray)):
            return _readonly_array(selected, (3,), "anchor")
        if isinstance(selected, (int, np.integer)):
            return np.asarray(self.atoms.positions[int(selected)], dtype=float).copy()
        if selected in (None, "explicit") and self.anchor_cartesian is not None:
            return np.asarray(self.anchor_cartesian, dtype=float).copy()
        if selected in (None, "geometric_center", "center"):
            return np.asarray(self.atoms.positions, dtype=float).mean(axis=0)
        if selected in ("center_of_mass", "com"):
            return np.asarray(self.atoms.get_center_of_mass(), dtype=float)
        raise ConstructionValidationError(f"unknown anchor definition: {selected!r}")


@dataclass(frozen=True)
class MolecularPose:
    """A complete template pose in a prepared unit cell."""

    template: MolecularTemplate | Atoms
    anchor_fractional: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    anchor: str | int | np.ndarray | None = None
    species: str | None = None
    occupation: float = 1.0

    def __post_init__(self) -> None:
        template = self.template
        if isinstance(template, Atoms):
            anchor_index = int(self.anchor) if isinstance(self.anchor, (int, np.integer)) else None
            template = MolecularTemplate(template, anchor_index=anchor_index)
            object.__setattr__(self, "template", template)
        if not isinstance(template, MolecularTemplate):
            raise TypeError("template must be a MolecularTemplate or ASE Atoms object")
        anchor = _readonly_array(self.anchor_fractional, (3,), "anchor_fractional")
        orientation = _validate_orientation(self.orientation, DEFAULT_FRACTIONAL_TOLERANCE)
        occupation = float(self.occupation)
        if not np.isfinite(occupation) or occupation < 0.0 or occupation > 1.0:
            raise ConstructionValidationError("occupation must be finite and lie in [0, 1]")
        object.__setattr__(self, "anchor_fractional", anchor)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "occupation", occupation)


@dataclass(frozen=True)
class ForwardConstructionContext:
    """Validated immutable cell, group, and provenance context."""

    lattice: np.ndarray
    symmetry: CrystalSymmetry
    setting: str | None = None
    origin_choice: str | None = None
    basis: str | None = None
    tolerance_A: float = DEFAULT_GEOMETRY_TOLERANCE_A
    fractional_tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE
    chemistry_policy: str = "unspecified"

    def __post_init__(self) -> None:
        lattice = _validate_lattice(self.lattice)
        if self.tolerance_A <= 0 or not np.isfinite(self.tolerance_A):
            raise ConstructionValidationError("tolerance_A must be positive and finite")
        if self.fractional_tolerance <= 0 or not np.isfinite(self.fractional_tolerance):
            raise ConstructionValidationError("fractional_tolerance must be positive and finite")
        operations = _validate_group(tuple(self.symmetry.operations), lattice, self.fractional_tolerance)
        symmetry = CrystalSymmetry(
            operations=operations,
            space_group_number=self.symmetry.space_group_number,
            space_group_symbol=self.symmetry.space_group_symbol,
            hall_symbol=self.symmetry.hall_symbol,
            source=self.symmetry.source,
            expanded_from_declaration=self.symmetry.expanded_from_declaration,
        )
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "symmetry", symmetry)

    @property
    def operations(self) -> tuple[FractionalAffineOperation, ...]:
        return self.symmetry.operations

    @property
    def space_group_number(self) -> int | None:
        return self.symmetry.space_group_number

    def to_metadata(self) -> dict[str, Any]:
        return {
            "space_group_number": self.space_group_number,
            "space_group_symbol": self.symmetry.space_group_symbol,
            "hall_symbol": self.symmetry.hall_symbol,
            "setting": self.setting,
            "origin_choice": self.origin_choice,
            "basis": self.basis,
            "tolerance_A": float(self.tolerance_A),
            "fractional_tolerance": float(self.fractional_tolerance),
            "chemistry_policy": self.chemistry_policy,
            "operations": tuple(self.operations),
        }


# Public spelling used by callers that prefer the shorter name.
CrystalConstructionContext = ForwardConstructionContext


def prepare_construction_context(
    lattice: Any,
    symmetry: CrystalSymmetry | Iterable[FractionalAffineOperation],
    *,
    space_group_number: int | None = None,
    space_group_symbol: str | None = None,
    hall_symbol: str | None = None,
    setting: str | None = None,
    origin_choice: str | None = None,
    basis: str | None = None,
    tolerance_A: float = DEFAULT_GEOMETRY_TOLERANCE_A,
    fractional_tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE,
    chemistry_policy: str = "unspecified",
) -> ForwardConstructionContext:
    """Validate and freeze a construction context."""
    if isinstance(symmetry, CrystalSymmetry):
        canonical = symmetry
    else:
        canonical = CrystalSymmetry(
            tuple(symmetry),
            space_group_number=space_group_number,
            space_group_symbol=space_group_symbol,
            hall_symbol=hall_symbol,
        )
    return ForwardConstructionContext(
        lattice=lattice,
        symmetry=canonical,
        setting=setting,
        origin_choice=origin_choice,
        basis=basis,
        tolerance_A=tolerance_A,
        fractional_tolerance=fractional_tolerance,
        chemistry_policy=chemistry_policy,
    )


def prepare_context_from_cif(
    filepath: str,
    *,
    setting: str | None = None,
    origin_choice: str | None = None,
    basis: str | None = None,
    tolerance_A: float = DEFAULT_GEOMETRY_TOLERANCE_A,
    fractional_tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE,
    chemistry_policy: str = "unspecified",
) -> ForwardConstructionContext:
    """Prepare a strict context from a CIF's cell and explicit operation set.

    This helper intentionally requests ``strict=True`` and
    ``expand_symmetry=False`` from the existing CIF symmetry reader.  It never
    fabricates missing operations or silently falls back to identity; callers
    that need a declared space group with incomplete loops must resolve that
    declaration before entering the forward-construction contract.
    """
    from ase.io import read as ase_read
    from ..io.cif import read_cif_symmetry

    atoms = ase_read(filepath, format="cif", index=0)
    symmetry = read_cif_symmetry(filepath, expand_symmetry=False, strict=True)
    return prepare_construction_context(
        np.asarray(atoms.cell.array, dtype=float),
        symmetry,
        setting=setting,
        origin_choice=origin_choice,
        basis=basis,
        tolerance_A=tolerance_A,
        fractional_tolerance=fractional_tolerance,
        chemistry_policy=chemistry_policy,
    )


def _template_fractional_pose(context: ForwardConstructionContext, pose: MolecularPose) -> np.ndarray:
    anchor = pose.template.anchor(pose.anchor)
    cart_anchor = pose.anchor_fractional @ context.lattice
    positioned = (np.asarray(pose.template.atoms.positions, dtype=float) - anchor) @ pose.orientation.T
    positioned += cart_anchor
    return positioned @ np.linalg.inv(context.lattice)


def _mapped_template(
    context: ForwardConstructionContext,
    pose: MolecularPose,
    operation: FractionalAffineOperation,
) -> tuple[np.ndarray, np.ndarray]:
    fractional = _template_fractional_pose(context, pose)
    mapped = operation.apply(fractional, wrap=False)
    cartesian = mapped @ context.lattice
    return mapped, cartesian


def _match_operation_to_template(
    context: ForwardConstructionContext,
    pose: MolecularPose,
    operation: FractionalAffineOperation,
) -> tuple[bool, tuple[int, ...] | None, float]:
    """Compare a whole molecule under an operation, allowing one image shift."""
    original = _template_fractional_pose(context, pose)
    mapped = operation.apply(original, wrap=False)
    anchor = pose.anchor_fractional
    common_shift = np.rint(operation.apply(anchor, wrap=False) - anchor)
    mapped -= common_shift
    symbols = pose.template.atoms.get_chemical_symbols()
    permutation = [-1] * len(symbols)
    max_distance = 0.0
    for key in sorted({_atom_identity_key(pose.template.atoms, i) for i in range(len(symbols))}):
        left = [i for i in range(len(symbols)) if _atom_identity_key(pose.template.atoms, i) == key]
        if not left:
            continue
        # A molecule may straddle a unit-cell boundary.  Stabilizer matching
        # compares the nearest periodic image for each candidate atom rather
        # than requiring a single Cartesian image shift for the whole pose.
        delta = mapped[left, None, :] - original[None, left, :]
        delta -= np.rint(delta)
        cost = np.linalg.norm(delta @ context.lattice, axis=2)
        rows, columns = linear_sum_assignment(cost)
        if len(rows) != len(left):
            return False, None, float(np.max(cost)) if cost.size else 0.0
        if np.any(cost[rows, columns] > context.tolerance_A):
            return False, None, float(np.max(cost[rows, columns]))
        for row, column in zip(rows, columns):
            permutation[left[int(row)]] = left[int(column)]
            max_distance = max(max_distance, float(cost[row, column]))
    if any(index < 0 for index in permutation):
        return False, None, max_distance
    # A geometric permutation is only a molecular stabilizer if it preserves
    # the supplied molecular graph.  Coordinate-only matching would certify a
    # square/path counterexample whose atoms occupy symmetric positions but
    # whose bonds are changed by the operation.
    connectivity = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in pose.template.connectivity
        if len(edge) >= 2
    }
    if connectivity:
        mapped_connectivity = {
            tuple(sorted((permutation[left], permutation[right])))
            for left, right in connectivity
        }
        if mapped_connectivity != connectivity:
            return False, tuple(permutation), max_distance
    # A declared chemical identity is an explicit stereochemical/tautomeric
    # contract.  An improper operation changes handedness and therefore may
    # not stabilize an identity-bearing template unless the caller selected a
    # chemistry policy that explicitly permits that image.
    if (
        pose.template.chemical_identity
        and operation.determinant < 0
        and context.chemistry_policy.lower() not in {"racemic", "allow_improper"}
    ):
        return False, tuple(permutation), max_distance
    if operation.determinant < 0 and context.chemistry_policy.lower() in {
        "enantiopure", "enantiopure_only", "forbid_improper"
    }:
        return False, tuple(permutation), max_distance
    return True, tuple(permutation), max_distance


def _site_operations(context: ForwardConstructionContext, anchor: np.ndarray) -> tuple[FractionalAffineOperation, ...]:
    return tuple(
        operation
        for operation in context.operations
        if _fractional_equal(operation.apply(anchor, wrap=False), anchor, context.fractional_tolerance)
    )


def _orbit_representatives(context: ForwardConstructionContext, anchor: np.ndarray) -> tuple[FractionalAffineOperation, ...]:
    representatives: list[FractionalAffineOperation] = []
    positions: list[np.ndarray] = []
    for operation in context.operations:
        target = operation.apply(anchor, wrap=False)
        if any(_fractional_equal(target, existing, context.fractional_tolerance) for existing in positions):
            continue
        positions.append(target)
        representatives.append(operation)
    return tuple(representatives)


@dataclass(frozen=True)
class SiteSymmetryAnalysis:
    """Machine-readable site and molecular stabilizer analysis."""

    anchor_fractional: np.ndarray
    anchor_orbit: tuple[np.ndarray, ...]
    site_operations: tuple[FractionalAffineOperation, ...]
    stabilizer_operations: tuple[FractionalAffineOperation, ...]
    atom_mappings: tuple[tuple[int, ...], ...]
    orbit_multiplicity: int
    site_symmetry_order: int
    stabilizer_order: int
    orientation_multiplicity: int
    ordered_compatible: bool
    near_special_distance: float
    conflicting_operations: tuple[FractionalAffineOperation, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_fractional", np.asarray(self.anchor_fractional, dtype=float).copy())
        self.anchor_fractional.setflags(write=False)
        object.__setattr__(self, "anchor_orbit", tuple(np.asarray(item, dtype=float).copy() for item in self.anchor_orbit))

    @property
    def S(self) -> tuple[FractionalAffineOperation, ...]:
        return self.site_operations

    @property
    def K(self) -> tuple[FractionalAffineOperation, ...]:
        return self.stabilizer_operations

    @property
    def site_stabilizer(self) -> tuple[FractionalAffineOperation, ...]:
        return self.site_operations

    @property
    def molecular_stabilizer(self) -> tuple[FractionalAffineOperation, ...]:
        return self.stabilizer_operations

    @property
    def m(self) -> int:
        return self.orbit_multiplicity

    @property
    def k(self) -> int:
        return self.orientation_multiplicity

    @property
    def multiplicity(self) -> int:
        return self.orbit_multiplicity

    @property
    def compatible(self) -> bool:
        return self.ordered_compatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_fractional": self.anchor_fractional.tolist(),
            "anchor_orbit": [item.tolist() for item in self.anchor_orbit],
            "orbit_multiplicity": self.orbit_multiplicity,
            "site_symmetry_order": self.site_symmetry_order,
            "stabilizer_order": self.stabilizer_order,
            "orientation_multiplicity": self.orientation_multiplicity,
            "ordered_compatible": self.ordered_compatible,
            "near_special_distance": self.near_special_distance,
            "reason_codes": list(self.reason_codes),
            "conflicting_operations": [op.xyz for op in self.conflicting_operations],
        }


def analyze_molecular_pose(context: ForwardConstructionContext, pose: MolecularPose) -> SiteSymmetryAnalysis:
    """Analyze S, K, multiplicities, atom maps, and ordered compatibility."""
    anchor = pose.anchor_fractional
    site = _site_operations(context, anchor)
    stabilizers: list[FractionalAffineOperation] = []
    mappings: list[tuple[int, ...]] = []
    conflicts: list[FractionalAffineOperation] = []
    max_distance = 0.0
    for operation in site:
        compatible, mapping, distance = _match_operation_to_template(context, pose, operation)
        max_distance = max(max_distance, distance)
        if compatible:
            stabilizers.append(operation)
            if mapping is not None:
                mappings.append(mapping)
        else:
            conflicts.append(operation)
    orbit = _orbit_representatives(context, anchor)
    non_identity_residuals = []
    for operation in context.operations:
        if _equivalent(operation, (identity_operation(),), context.fractional_tolerance) is not None:
            continue
        delta = operation.apply(anchor, wrap=False) - anchor
        non_identity_residuals.append(float(np.linalg.norm(delta - np.rint(delta))))
    distance_to_special = min(non_identity_residuals) if non_identity_residuals else float("inf")
    site_order = len(site)
    stabilizer_order = len(stabilizers)
    orientation_count = site_order // stabilizer_order if stabilizer_order else 0
    reasons: list[str] = []
    if site_order > 1:
        reasons.append("special_position")
    if conflicts:
        reasons.append("molecular_stabilizer_incomplete")
    if context.chemistry_policy.lower() in {"enantiopure", "enantiopure_only", "forbid_improper"} and any(
        operation.determinant < 0 for operation in conflicts
    ):
        reasons.append("improper_operation_conflicts_with_chemistry_policy")
    return SiteSymmetryAnalysis(
        anchor_fractional=anchor,
        anchor_orbit=tuple(operation.apply(anchor, wrap=False) for operation in orbit),
        site_operations=site,
        stabilizer_operations=tuple(stabilizers),
        atom_mappings=tuple(mappings),
        orbit_multiplicity=len(orbit),
        site_symmetry_order=site_order,
        stabilizer_order=stabilizer_order,
        orientation_multiplicity=orientation_count,
        ordered_compatible=not conflicts,
        near_special_distance=distance_to_special,
        conflicting_operations=tuple(conflicts),
        reason_codes=tuple(reasons),
    )


# Alias requested by the issue wording.
analyze_site_symmetry = analyze_molecular_pose


def _coset_representatives(
    group: Sequence[FractionalAffineOperation],
    subgroup: Sequence[FractionalAffineOperation],
    tolerance: float,
) -> tuple[FractionalAffineOperation, ...]:
    remaining = list(group)
    result: list[FractionalAffineOperation] = []
    while remaining:
        representative = remaining.pop(0)
        result.append(representative)
        coset = [representative.compose(member) for member in subgroup]
        remaining = [item for item in remaining if _equivalent(item, coset, tolerance) is None]
    return tuple(result)


def _copy_generated_atoms(
    context: ForwardConstructionContext,
    pose: MolecularPose,
    operation: FractionalAffineOperation,
    *,
    pose_index: int,
    operation_index: int,
    occupancy: float = 1.0,
    disorder_group: int = 0,
    assembly: str = "",
    asym_id: int = -1,
    site_symmetry_order: int = 1,
    atom_identities: Sequence[str] | None = None,
) -> CrystalMolecule:
    _, cartesian = _mapped_template(context, pose, operation)
    source = pose.template.atoms
    atoms = Atoms(symbols=source.get_chemical_symbols(), positions=cartesian)
    for key, values in source.arrays.items():
        if key in {"numbers", "positions"}:
            continue
        if len(values) == len(atoms):
            atoms.set_array(key, np.asarray(values).copy())
    n = len(atoms)
    atoms.set_array(KEY_OCCUPANCY, np.full(n, float(occupancy), dtype=float))
    atoms.set_array(KEY_DISORDER_GROUP, np.full(n, int(disorder_group), dtype=int))
    assembly_dtype = f"U{max(1, len(str(assembly)))}"
    atoms.set_array(KEY_ASSEMBLY, np.full(n, assembly, dtype=assembly_dtype))
    atoms.set_array(KEY_SYM_OP_INDEX, np.full(n, int(operation_index), dtype=int))
    atoms.set_array(KEY_ASYM_ID, np.full(n, int(asym_id), dtype=int))
    atoms.set_array(
        KEY_SITE_SYMMETRY_ORDER,
        np.full(n, int(site_symmetry_order), dtype=int),
    )
    image_shift = np.rint(
        operation.apply(pose.anchor_fractional, wrap=False)
        - _canonical_fractional(operation.apply(pose.anchor_fractional, wrap=False))
    ).astype(int)
    atoms.set_array(KEY_IMAGE_SHIFT, np.tile(image_shift, (n, 1)))
    source_labels = source.arrays.get(KEY_LABEL)
    if source_labels is None or len(source_labels) != n:
        source_labels = np.asarray(
            [f"{symbol}{index + 1}" for index, symbol in enumerate(source.get_chemical_symbols())],
            dtype=str,
        )
    atoms.set_array(KEY_LABEL, np.asarray(source_labels, dtype=str).copy())
    atoms.set_array(
        KEY_ATOM_ID,
        np.asarray([f"pose{pose_index}:op{operation_index}:a{index}" for index in range(n)], dtype=str),
    )
    molecule = CrystalMolecule(atoms, check_pbc=False)
    # Keep a stable entity/atom identity across operation mates.  The atom ID
    # retains the generated operation for provenance, while the pose/template
    # metadata lets ASU reduction recognise the same atom in another mate
    # without collapsing independent same-element entities.
    molecule.info["forward_template_identifier"] = pose.template.identifier
    molecule.info["forward_pose_index"] = int(pose_index)
    molecule.info["forward_operation_index"] = int(operation_index)
    if atom_identities is None:
        atom_identities = [f"a{index}" for index in range(n)]
    if len(atom_identities) != n:
        raise ConstructionValidationError("atom_identities must match template atom count")
    molecule.info["forward_atom_identities"] = tuple(str(item) for item in atom_identities)
    if pose.template.connectivity:
        connectivity = [
            (int(edge[0]), int(edge[1]))
            for edge in pose.template.connectivity
            if len(edge) >= 2 and 0 <= int(edge[0]) < n and 0 <= int(edge[1]) < n
        ]
        molecule.info["forward_connectivity"] = [list(edge) for edge in connectivity]
        molecule.info["bond_pairs"] = [
            tuple(sorted(edge)) for edge in connectivity if edge[0] != edge[1]
        ]
        molecule._graph = nx.Graph()
        molecule._graph.add_nodes_from(range(n))
        positions = np.asarray(molecule.get_positions(), dtype=float)
        for left, right in connectivity:
            if left == right:
                continue
            vector = positions[right] - positions[left]
            molecule._graph.add_edge(
                left,
                right,
                vector=vector.copy(),
                distance=float(np.linalg.norm(vector)),
                image_shift=np.zeros(3, dtype=int),
            )
    return molecule


def construct_molecular_crystal(
    context: ForwardConstructionContext,
    poses: MolecularPose | Sequence[MolecularPose],
    *,
    strict: bool = True,
    ordered: bool = True,
) -> MolecularCrystal:
    """Construct complete ordered molecules from explicit poses."""
    if not ordered:
        return construct_statistical_disorder(context, poses)
    if isinstance(poses, MolecularPose):
        poses = (poses,)
    poses = tuple(poses)
    if not poses:
        raise ConstructionValidationError("at least one molecular pose is required")
    molecules: list[CrystalMolecule] = []
    analyses: list[SiteSymmetryAnalysis] = []
    seen_centers: set[tuple[tuple[float, ...], ...]] = set()
    for pose_index, pose in enumerate(poses):
        if pose.occupation < 1.0 - 1e-8:
            raise ConstructionValidationError(
                "fractional molecular occupation requires the explicit statistical-disorder path"
            )
        analysis = analyze_molecular_pose(context, pose)
        analyses.append(analysis)
        if strict and not analysis.ordered_compatible:
            raise IncompatibleSiteError(analysis)
        center_orbit = _center_orbit_key(context, pose.anchor_fractional)
        subgroup = analysis.stabilizer_operations if analysis.ordered_compatible else (identity_operation(),)
        # A center is a crystallographic occupation slot.  Template names and
        # element species do not make two full-occupancy poses independent
        # slots, so reject collisions before expanding either pose.
        if center_orbit in seen_centers:
            raise ConstructionValidationError(
                "duplicate or conflicting molecular pose center orbit"
            )
        seen_centers.add(center_orbit)
        atom_identities = _stabilizer_atom_identities(analysis, pose.template.atom_count)
        representatives = _coset_representatives(context.operations, subgroup, context.fractional_tolerance)
        for operation in representatives:
            molecules.append(
                _copy_generated_atoms(
                    context,
                    pose,
                    operation,
                    pose_index=pose_index,
                    operation_index=_context_operation_index(context, operation),
                    occupancy=1.0,
                    asym_id=pose_index,
                    site_symmetry_order=analysis.site_symmetry_order,
                    atom_identities=atom_identities,
                )
            )
    metadata = {
        "representation": "ordered",
        "source_symmetry": context.to_metadata(),
        "forward_context": context,
        "pose_analyses": tuple(analysis.to_dict() for analysis in analyses),
    }
    return MolecularCrystal(context.lattice, molecules, metadata=metadata)


# Friendly aliases used in examples and downstream integrations.
build_molecular_crystal = construct_molecular_crystal
build_symmetry_complete_crystal = construct_molecular_crystal


@dataclass(frozen=True)
class MolecularAlternative:
    """One complete, mutually exclusive molecular state."""

    orbit_id: str
    alternative_id: str
    molecule: CrystalMolecule
    weight: float
    operation_index: int
    center_fractional: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_fractional, dtype=float).copy()
        center.setflags(write=False)
        object.__setattr__(self, "center_fractional", center)
        if self.weight < 0 or not np.isfinite(self.weight):
            raise ConstructionValidationError("alternative weight must be finite and non-negative")


@dataclass(frozen=True)
class AverageAtomSite:
    """Derived average atom site and its CIF occupancy."""

    symbol: str
    fractional: np.ndarray
    occupancy: float
    orbit_id: str
    template_atom_index: int
    alternative_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fractional = _readonly_array(self.fractional, (3,), "fractional")
        object.__setattr__(self, "fractional", _canonical_fractional(fractional))
        if self.occupancy < 0 or self.occupancy > 1.0 + 1e-8:
            raise ConstructionValidationError("average atom occupancy must lie in [0, 1]")


@dataclass(frozen=True)
class StatisticalDisorderModel:
    """Explicit whole-molecule alternatives plus a derived average view."""

    context: ForwardConstructionContext
    alternatives: tuple[MolecularAlternative, ...]
    average_sites: tuple[AverageAtomSite, ...]
    orbit_probabilities: Mapping[str, float]
    correlation_status: str = "local-only"
    correlations: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expected_formula(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for alternative in self.alternatives:
            for symbol in alternative.molecule.get_chemical_symbols():
                result[symbol] = result.get(symbol, 0.0) + alternative.weight
        return result

    @property
    def alternatives_by_orbit(self) -> dict[str, tuple[MolecularAlternative, ...]]:
        result: dict[str, list[MolecularAlternative]] = {}
        for alternative in self.alternatives:
            result.setdefault(alternative.orbit_id, []).append(alternative)
        return {key: tuple(value) for key, value in result.items()}

    def select_ordered_realization(
        self, selections: Mapping[str, str | int] | None = None
    ) -> MolecularCrystal:
        """Select one mutually exclusive state per orbit."""
        return select_ordered_realization(self, selections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": "statistical_average",
            "correlation_status": self.correlation_status,
            "correlations": dict(self.correlations),
            "orbit_probabilities": dict(self.orbit_probabilities),
            "alternatives": [
                {
                    "orbit_id": item.orbit_id,
                    "alternative_id": item.alternative_id,
                    "weight": item.weight,
                    "operation_index": item.operation_index,
                    "center_fractional": item.center_fractional.tolist(),
                }
                for item in self.alternatives
            ],
            "average_sites": [
                {
                    "symbol": item.symbol,
                    "fractional": item.fractional.tolist(),
                    "occupancy": item.occupancy,
                    "orbit_id": item.orbit_id,
                    "template_atom_index": item.template_atom_index,
                    "alternative_ids": list(item.alternative_ids),
                }
                for item in self.average_sites
            ],
        }


def _merge_average_site(
    sites: list[AverageAtomSite],
    candidate: AverageAtomSite,
    tolerance: float,
) -> None:
    for index, existing in enumerate(sites):
        if existing.symbol != candidate.symbol or existing.orbit_id != candidate.orbit_id:
            continue
        if _fractional_equal(existing.fractional, candidate.fractional, tolerance):
            alternatives = tuple(dict.fromkeys(existing.alternative_ids + candidate.alternative_ids))
            sites[index] = AverageAtomSite(
                existing.symbol,
                existing.fractional,
                existing.occupancy + candidate.occupancy,
                existing.orbit_id,
                existing.template_atom_index,
                alternatives,
            )
            return
    sites.append(candidate)


def _center_orbit_key(
    context: ForwardConstructionContext,
    anchor: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    """Canonical key for one crystallographic molecular center orbit."""
    points = _orbit_representatives(context, anchor)
    return tuple(
        sorted(
            tuple(np.round(_canonical_fractional(operation.apply(anchor, wrap=False)), 9))
            for operation in points
        )
    )


def _stabilizer_atom_identities(
    analysis: SiteSymmetryAnalysis,
    atom_count: int,
) -> tuple[str, ...]:
    """Return canonical atom-orbit IDs induced by molecular stabilizers."""
    parent = list(range(atom_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for mapping in analysis.atom_mappings:
        for source, target in enumerate(mapping):
            union(source, int(target))
    roots = {root: index for index, root in enumerate(sorted({find(i) for i in range(atom_count)}))}
    return tuple(f"atom_orbit{roots[find(index)]}" for index in range(atom_count))


def _context_operation_index(
    context: ForwardConstructionContext,
    operation: FractionalAffineOperation,
) -> int:
    """Return the canonical zero-based index for a group operation."""
    for index, candidate in enumerate(context.operations):
        if operation.equivalent_to(candidate, tolerance=context.fractional_tolerance):
            return index
    raise ConstructionValidationError("generated operation is outside the validated symmetry group")


def construct_statistical_disorder(
    context: ForwardConstructionContext,
    poses: MolecularPose | Sequence[MolecularPose],
    *,
    center_occupation: float | None = None,
    center_occupancies: Mapping[int | str, float] | None = None,
    correlation_status: str = "local-only",
    correlations: Mapping[str, Any] | None = None,
) -> StatisticalDisorderModel:
    """Build an explicit symmetry-derived whole-molecule disorder model.

    Each incompatible special-site pose yields ``k`` mutually exclusive local
    states.  With full center occupation their weights are ``1/k``; a supplied
    ``center_occupation=q`` scales those weights to ``q/k``.  Atomic sites are
    merged only when both chemical identity and periodic position agree.
    """
    if isinstance(poses, MolecularPose):
        poses = (poses,)
    poses = tuple(poses)
    if not poses:
        raise ConstructionValidationError("at least one molecular pose is required")
    requested_center_occupation = center_occupation
    if center_occupation is None:
        center_occupation = 1.0
    center_occupation = float(center_occupation)
    if not 0.0 <= center_occupation <= 1.0:
        raise ConstructionValidationError("center_occupation must lie in [0, 1]")
    center_occupancies = dict(center_occupancies or {})
    if correlations is None:
        correlations = {}
    elif not isinstance(correlations, Mapping):
        raise ConstructionValidationError("correlations must be a mapping or None")
    else:
        correlations = dict(correlations)

    alternatives: list[MolecularAlternative] = []
    average_sites: list[AverageAtomSite] = []
    probabilities: dict[str, float] = {}
    center_groups: dict[tuple[tuple[float, ...], ...], int] = {}
    center_totals: dict[tuple[tuple[float, ...], ...], float] = {}
    for pose_index, pose in enumerate(poses):
        center_key = _center_orbit_key(context, pose.anchor_fractional)
        default_pose_occupation = (
            float(pose.occupation)
            if requested_center_occupation is None
            else center_occupation
        )
        pose_occupation = float(
            center_occupancies.get(
                pose_index,
                center_occupancies.get(str(pose_index), default_pose_occupation),
            )
        )
        center_totals[center_key] = center_totals.get(center_key, 0.0) + pose_occupation
        if center_key not in center_groups:
            center_groups[center_key] = len(center_groups)
    if any(total > 1.0 + 1e-8 for total in center_totals.values()):
        raise ConstructionValidationError(
            "molecular poses overoccupy a crystallographic center orbit"
        )
    for pose_index, pose in enumerate(poses):
        analysis = analyze_molecular_pose(context, pose)
        default_pose_occupation = (
            float(pose.occupation)
            if requested_center_occupation is None
            else center_occupation
        )
        pose_occupation = float(
            center_occupancies.get(
                pose_index,
                center_occupancies.get(str(pose_index), default_pose_occupation),
            )
        )
        if not 0.0 <= pose_occupation <= 1.0:
            raise ConstructionValidationError("each center occupation must lie in [0, 1]")
        if analysis.ordered_compatible:
            # A compatible pose has one local molecular state.  Stabilizer
            # operations describe its invariance; they must not be emitted as
            # multiple alternatives.
            local_representatives = (identity_operation(),)
        else:
            local_representatives = _coset_representatives(
                analysis.site_operations,
                analysis.stabilizer_operations or (identity_operation(),),
                context.fractional_tolerance,
            )
        orbit_representatives = _orbit_representatives(context, pose.anchor_fractional)
        center_key = _center_orbit_key(context, pose.anchor_fractional)
        center_group = center_groups[center_key]
        atom_identities = _stabilizer_atom_identities(analysis, pose.template.atom_count)
        local_count = len(local_representatives)
        if local_count != analysis.orientation_multiplicity and analysis.ordered_compatible:
            local_count = 1
        weight = pose_occupation / float(local_count)
        for orbit_index, orbit_operation in enumerate(orbit_representatives):
            orbit_id = f"center{center_group}:orbit{orbit_index}"
            probabilities[orbit_id] = probabilities.get(orbit_id, 0.0) + pose_occupation
            for local_index, local_operation in enumerate(local_representatives):
                operation = orbit_operation.compose(local_operation)
                operation_index = _context_operation_index(context, operation)
                # Pose index is part of the identity: different templates
                # may occupy the same center orbit with partial weights, and
                # their local state numbers would otherwise collide.
                alternative_id = f"{orbit_id}:pose{pose_index}:state{local_index}"
                molecule = _copy_generated_atoms(
                    context,
                    pose,
                    operation,
                    pose_index=pose_index,
                    operation_index=operation_index,
                    occupancy=1.0,
                    disorder_group=local_index + 1 if local_count > 1 else 0,
                    assembly=orbit_id,
                    asym_id=orbit_index,
                    site_symmetry_order=analysis.site_symmetry_order,
                    atom_identities=atom_identities,
                )
                alternative = MolecularAlternative(
                    orbit_id,
                    alternative_id,
                    molecule,
                    weight,
                    operation_index,
                    _canonical_fractional(operation.apply(pose.anchor_fractional, wrap=False)),
                )
                alternatives.append(alternative)
                mapped_fractional, _ = _mapped_template(context, pose, operation)
                for atom_index, (symbol, fractional) in enumerate(
                    zip(pose.template.atoms.get_chemical_symbols(), mapped_fractional)
                ):
                    _merge_average_site(
                        average_sites,
                        AverageAtomSite(
                            symbol,
                            fractional,
                            weight,
                            orbit_id,
                            atom_index,
                            (alternative_id,),
                        ),
                        context.fractional_tolerance,
                    )
    # Whole-state alternatives must not overoccupy an orbit.
    for orbit_id, probability in probabilities.items():
        total = sum(item.weight for item in alternatives if item.orbit_id == orbit_id)
        if total > 1.0 + 1e-8:
            raise ConstructionValidationError(f"alternatives overoccupy center orbit {orbit_id}")
    metadata = {
        "representation": "statistical_average",
        "source_symmetry": context.to_metadata(),
        "correlation_status": correlation_status,
        "correlations": dict(correlations),
    }
    return StatisticalDisorderModel(
        context=context,
        alternatives=tuple(alternatives),
        average_sites=tuple(average_sites),
        orbit_probabilities=probabilities,
        correlation_status=correlation_status,
        correlations=correlations,
        metadata=metadata,
    )


build_orientational_disorder = construct_statistical_disorder


def select_ordered_realization(
    model: StatisticalDisorderModel,
    selections: Mapping[str, str | int] | None = None,
) -> MolecularCrystal:
    """Select one whole-molecule state per orbit without materialising others."""
    selections = dict(selections or {})
    chosen: list[CrystalMolecule] = []
    for orbit_id, alternatives in model.alternatives_by_orbit.items():
        choice = selections.get(orbit_id, 0)
        if isinstance(choice, str):
            selected = next((item for item in alternatives if item.alternative_id == choice), None)
        else:
            selected = alternatives[int(choice)] if 0 <= int(choice) < len(alternatives) else None
        if selected is None:
            raise ConstructionValidationError(f"no alternative {choice!r} for orbit {orbit_id}")
        chosen.append(selected.molecule.copy())
    metadata = dict(model.metadata)
    metadata["representation"] = "ordered_realization"
    metadata["source_disorder"] = model.to_dict()
    return MolecularCrystal(model.context.lattice, chosen, metadata=metadata)


__all__ = [
    "AverageAtomSite",
    "ConstructionValidationError",
    "CrystalConstructionContext",
    "DEFAULT_FRACTIONAL_TOLERANCE",
    "DEFAULT_GEOMETRY_TOLERANCE_A",
    "ForwardConstructionContext",
    "IncompatibleSiteError",
    "InvalidOrientationError",
    "InvalidSymmetryError",
    "MolecularAlternative",
    "MolecularPose",
    "MolecularTemplate",
    "SiteSymmetryAnalysis",
    "StatisticalDisorderModel",
    "analyze_molecular_pose",
    "analyze_site_symmetry",
    "build_molecular_crystal",
    "build_orientational_disorder",
    "build_symmetry_complete_crystal",
    "construct_molecular_crystal",
    "construct_statistical_disorder",
    "prepare_construction_context",
    "prepare_context_from_cif",
    "select_ordered_realization",
]
