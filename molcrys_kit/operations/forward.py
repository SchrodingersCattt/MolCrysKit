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

A declared space-group number, symbol, Hall symbol, setting, origin, or basis
must match the complete operation set.  An unlabelled closed set remains a
custom group.  ``chemistry_policy="enantiopure"`` applies to the whole group,
not only the site stabilizer.  Stabilizer comparisons use one coherent
molecular image.  Typed bonds, isotopes, and explicit charges are preserved.
Statistical models use companion schema ``molcrys.statistical_disorder.v2``;
older id-only payloads are rejected instead of being completed by guesswork.
"""

from __future__ import annotations

import re
import copy
import hashlib
from dataclasses import dataclass, field, replace
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
    PARTIAL_OCCUPANCY_TOLERANCE,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
    LatticeBasisChange,
    identity_operation,
)


DISORDER_COMPANION_SCHEMA = "molcrys.statistical_disorder.v2"
DISORDER_CONSTRAINT_SCHEMA = "molcrys.disorder_constraints.v1"
_ENANTIOPURE_POLICIES = {"enantiopure", "enantiopure_only", "forbid_improper"}
_RACEMIC_POLICIES = {"racemic", "allow_improper"}
_UNKNOWN_STEREO = {"", ".", "?", "none", "nan", "unknown", "null"}
_KNOWN_BOND_TYPES = {
    1: "single",
    2: "double",
    3: "triple",
}


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
        names = ", ".join(
            str(op.index if op.index is not None else i)
            for i, op in enumerate(analysis.conflicting_operations)
        )
        super().__init__(
            "molecular pose is incompatible with a special position: "
            f"|S|={analysis.site_symmetry_order}, |K|={analysis.stabilizer_order}, "
            f"m={analysis.orbit_multiplicity}, k={analysis.orientation_multiplicity}; "
            f"conflicting operations [{names}]"
        )


class IncompatibleChemistryError(ConstructionValidationError):
    """Raised when an explicit chemistry policy disagrees with the full group."""

    def __init__(self, analysis: "SiteSymmetryAnalysis", message: str) -> None:
        self.analysis = analysis
        super().__init__(message)


class DisorderCompanionLossError(ConstructionValidationError):
    """Raised when a disorder companion cannot be reconstructed losslessly."""


@dataclass(frozen=True)
class PeriodicBondRecord:
    """One typed periodic bond.

    Legacy two-index edges are accepted and keep unknown order and bond type.
    A known order is never inferred from bond length.
    """

    left: int
    right: int
    order: float | None = None
    bond_type: str | None = None
    kind: str | None = None
    image_shift: tuple[int, int, int] = (0, 0, 0)
    order_known: bool | None = None
    bond_type_known: bool | None = None

    def pair(self) -> tuple[int, int]:
        left, right = int(self.left), int(self.right)
        if left <= right:
            return left, right
        return right, left


def _unknown_token(value: Any) -> bool:
    return str(value).strip().lower() in _UNKNOWN_STEREO


def _normalize_bond_record(edge: Any, atom_count: int) -> PeriodicBondRecord:
    """Accept a pair, a typed triple, a mapping, or a :class:`PeriodicBondRecord`."""
    order = None
    bond_type = None
    kind = None
    image_shift = (0, 0, 0)
    order_known = False
    bond_type_known = False
    infer_bond_type = True
    if isinstance(edge, PeriodicBondRecord):
        left, right = _integer_atom_index(edge.left), _integer_atom_index(edge.right)
        infer_bond_type = edge.bond_type_known is None
        order = edge.order
        bond_type = edge.bond_type
        kind = edge.kind
        image_shift = _integer_image_shift(edge.image_shift)
        order_known = bool(
            edge.order is not None if edge.order_known is None else edge.order_known
        )
        bond_type_known = bool(
            (edge.bond_type not in (None, "") and not _unknown_token(edge.bond_type))
            if edge.bond_type_known is None
            else edge.bond_type_known
        )
    elif isinstance(edge, Mapping):
        try:
            left, right = (
                _integer_atom_index(edge["left"]),
                _integer_atom_index(edge["right"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConstructionValidationError(
                "bond mappings require integer 'left' and 'right' endpoints"
            ) from error
        order = edge.get("order")
        infer_bond_type = "bond_type_known" not in edge
        bond_type = edge.get("bond_type")
        kind = edge.get("kind")
        raw_shift = edge.get("image_shift", (0, 0, 0))
        image_shift = _integer_image_shift(raw_shift)
        order_known = bool(edge.get("order_known", order is not None))
        bond_type_known = bool(
            edge.get(
                "bond_type_known",
                bond_type not in (None, "") and not _unknown_token(bond_type),
            )
        )
    else:
        try:
            if len(edge) not in {2, 3}:
                raise ConstructionValidationError(
                    "connectivity edge must have two endpoints and at most one annotation"
                )
            left, right = _integer_atom_index(edge[0]), _integer_atom_index(edge[1])
        except (TypeError, ValueError, IndexError) as error:
            raise ConstructionValidationError(
                "connectivity edges must contain exactly two atom indices"
            ) from error
        extra = edge[2] if len(edge) >= 3 else None
        if isinstance(extra, Mapping):
            return _normalize_bond_record(
                {"left": left, "right": right, **dict(extra)},
                atom_count,
            )
        if isinstance(extra, str):
            bond_type = extra
            bond_type_known = not _unknown_token(extra)
        elif isinstance(extra, (int, float)) and not isinstance(extra, bool):
            order = float(extra)
            order_known = True
        elif extra is not None:
            raise ConstructionValidationError(
                "connectivity annotation must be a bond type, order, or mapping"
            )
    if isinstance(edge, (Mapping, PeriodicBondRecord)):
        if order_known and order is None:
            raise ConstructionValidationError("known bond order requires a value")
        if not order_known:
            order = None
        if not bond_type_known:
            bond_type = None
    if not 0 <= left < atom_count or not 0 <= right < atom_count:
        raise ConstructionValidationError(
            "connectivity edge index is outside the template"
        )
    if left == right and image_shift == (0, 0, 0):
        raise ConstructionValidationError("connectivity cannot contain self-edges")
    if order is not None:
        order = float(order)
        if not np.isfinite(order) or order <= 0:
            raise ConstructionValidationError("bond order must be positive and finite")
        order_known = True
        if (
            not bond_type_known
            and infer_bond_type
            and int(order) in _KNOWN_BOND_TYPES
            and float(order) == int(order)
        ):
            bond_type = _KNOWN_BOND_TYPES[int(order)]
            bond_type_known = True
    if bond_type is not None:
        bond_type = str(bond_type).strip().lower()
        if _unknown_token(bond_type):
            bond_type = None
            bond_type_known = False
        else:
            bond_type_known = True
    if kind is not None:
        kind = str(kind).strip().lower()
    if left > right:
        left, right = right, left
        image_shift = tuple(-int(value) for value in image_shift)
    elif left == right:
        image_shift = tuple(int(value) for value in image_shift)
    return PeriodicBondRecord(
        left,
        right,
        order if order_known else None,
        bond_type if bond_type_known else None,
        kind,
        image_shift,
        order_known,
        bond_type_known,
    )


def _integer_image_shift(value):
    try:
        array = np.asarray(value, dtype=float)
        if (
            array.shape == (3,)
            and np.all(np.isfinite(array))
            and np.all(array == np.rint(array))
        ):
            return tuple(int(item) for item in array)
    except (TypeError, ValueError):
        pass
    raise ConstructionValidationError("bond image_shift must contain three integers")


def _integer_atom_index(value):
    try:
        result = int(value)
        if not isinstance(value, (bool, np.bool_)) and float(value) == result:
            return result
    except (TypeError, ValueError, OverflowError):
        pass
    raise ConstructionValidationError("bond endpoints must be integer atom indices")


def _readonly_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ConstructionValidationError(
            f"{name} must have shape {shape}; got {array.shape}"
        )
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


def _equivalent(
    operation: FractionalAffineOperation,
    operations: Iterable[FractionalAffineOperation],
    tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE,
) -> FractionalAffineOperation | None:
    return next(
        (
            item
            for item in operations
            if operation.equivalent_to(item, tolerance=tolerance)
        ),
        None,
    )


def _deduplicate_operations(
    operations: Iterable[FractionalAffineOperation], tolerance: float
) -> tuple[FractionalAffineOperation, ...]:
    result: list[FractionalAffineOperation] = []
    for operation in operations:
        if _equivalent(operation, result, tolerance) is None:
            result.append(operation)
    return tuple(result)


def _validate_group(
    operations: Sequence[FractionalAffineOperation],
    lattice: np.ndarray,
    tolerance: float,
) -> tuple[FractionalAffineOperation, ...]:
    raw_group = tuple(operations)
    for index, operation in enumerate(raw_group):
        if _equivalent(operation, raw_group[:index], tolerance) is not None:
            raise InvalidSymmetryError(
                "symmetry operation set contains duplicate operations"
            )
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
                raise InvalidSymmetryError(
                    "symmetry operation set is not closed under composition"
                )
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
        raise InvalidOrientationError(
            "orientation must be a proper rotation with determinant +1"
        )
    return result


def _atom_identity_key(atoms: Atoms, index: int) -> tuple[Any, ...]:
    """Return every supplied chemical annotation for one template atom.

    Stabilizer matching must preserve the complete atom identity.  Keeping
    only the first optional annotation (the previous behavior) allowed a
    mapping to ignore a second tag, for example a custom chemical identity
    attached alongside a CIP label.
    """
    symbol = atoms.get_chemical_symbols()[index]
    isotope = atoms.arrays.get("isotope")
    charge = atoms.arrays.get("formal_charge")
    charge_known = atoms.arrays.get("formal_charge_known")
    annotations: list[tuple[str, str]] = []
    for key in (
        "stereochemistry",
        "stereo",
        "cip_code",
        "chemical_identity",
        "atom_identity",
    ):
        values = atoms.arrays.get(key)
        if values is not None and len(values) > index:
            annotations.append((key, str(values[index])))
    return (
        symbol,
        int(isotope[index]) if isotope is not None else 0,
        ("known", float(charge[index]))
        if charge is not None and (charge_known is None or charge_known[index])
        else ("unknown",),
        tuple(annotations),
    )


def _validate_template_annotations(atoms):
    for key in ("isotope", "formal_charge"):
        if key not in atoms.arrays:
            continue
        values = atoms.arrays[key]
        if (
            values.shape != (len(atoms),)
            or not np.all(np.isfinite(values))
            or not np.all(values == np.rint(values))
        ):
            raise ConstructionValidationError(
                f"{key} annotations must be finite integers per atom"
            )
    if "isotope" in atoms.arrays and np.any(atoms.arrays["isotope"] < 0):
        raise ConstructionValidationError("isotope mass numbers must be non-negative")
    if "formal_charge_known" in atoms.arrays and "formal_charge" not in atoms.arrays:
        raise ConstructionValidationError(
            "formal_charge_known requires a formal_charge array"
        )
    if "formal_charge" in atoms.arrays and "formal_charge_known" not in atoms.arrays:
        atoms.set_array("formal_charge_known", np.ones(len(atoms), dtype=bool))
    if "masses" in atoms.arrays and (
        not np.all(np.isfinite(atoms.arrays["masses"]))
        or np.any(atoms.arrays["masses"] <= 0)
    ):
        raise ConstructionValidationError(
            "explicit atomic masses must be positive and finite"
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
    bonds: tuple[PeriodicBondRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        copy_atoms = self.atoms.copy()
        copy_atoms.info = copy.deepcopy(self.atoms.info)
        copy_atoms.set_cell(np.zeros((3, 3)))
        copy_atoms.set_pbc(False)
        copy_atoms.positions = np.asarray(
            self.atoms.get_positions(), dtype=float
        ).copy()
        if not len(copy_atoms) or not np.all(np.isfinite(copy_atoms.positions)):
            raise ConstructionValidationError(
                "template requires nonempty finite atom coordinates"
            )
        _validate_template_annotations(copy_atoms)
        if self.anchor_index is not None and not 0 <= int(self.anchor_index) < len(
            copy_atoms
        ):
            raise ConstructionValidationError("anchor_index is outside the template")
        if self.anchor_cartesian is not None:
            anchor = _readonly_array(self.anchor_cartesian, (3,), "anchor_cartesian")
        else:
            anchor = None
        object.__setattr__(self, "atoms", copy_atoms)
        object.__setattr__(self, "anchor_cartesian", anchor)
        bond_records: list[PeriodicBondRecord] = []
        if self.connectivity:
            bond_records.extend(
                _normalize_bond_record(edge, len(copy_atoms))
                for edge in self.connectivity
            )
        elif getattr(self.atoms, "_graph", None) is not None:
            for left, right, edge_data in self.atoms._graph.edges(data=True):
                payload = {
                    "left": int(left),
                    "right": int(right),
                    "order": edge_data.get("order"),
                    "bond_type": edge_data.get("bond_type"),
                    "kind": edge_data.get("kind"),
                    "image_shift": edge_data.get("image_shift", (0, 0, 0)),
                    "order_known": bool(
                        edge_data.get("order_known", edge_data.get("order") is not None)
                    ),
                    "bond_type_known": bool(
                        edge_data.get(
                            "bond_type_known",
                            edge_data.get("bond_type") not in (None, ""),
                        )
                    ),
                }
                bond_records.append(_normalize_bond_record(payload, len(copy_atoms)))
        if self.bonds:
            bond_records.extend(
                _normalize_bond_record(edge, len(copy_atoms)) for edge in self.bonds
            )
        deduped: dict[tuple[int, int, tuple[int, int, int]], PeriodicBondRecord] = {}
        for record in bond_records:
            key = (record.left, record.right, record.image_shift)
            if key in deduped and deduped[key] != record:
                raise ConstructionValidationError(
                    "conflicting typed bonds name the same endpoint image"
                )
            deduped.setdefault(key, record)
        bond_records = list(deduped.values())
        connectivity = tuple(sorted(record.pair() for record in bond_records))
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "bonds", tuple(bond_records))
        stereo = tuple(self.stereochemistry)
        if stereo and len(stereo) != len(copy_atoms):
            raise ConstructionValidationError(
                "stereochemistry must contain one value per template atom"
            )
        if not stereo:
            for key in ("stereochemistry", "stereo", "cip_code"):
                values = copy_atoms.arrays.get(key)
                if values is not None and len(values) == len(copy_atoms):
                    stereo = tuple(str(value) for value in values)
                    break
        if stereo:
            copy_atoms.set_array("stereochemistry", np.asarray(stereo, dtype=str))
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
            anchor_index = (
                int(self.anchor) if isinstance(self.anchor, (int, np.integer)) else None
            )
            template = MolecularTemplate(template, anchor_index=anchor_index)
            object.__setattr__(self, "template", template)
        if not isinstance(template, MolecularTemplate):
            raise TypeError("template must be a MolecularTemplate or ASE Atoms object")
        anchor = _readonly_array(self.anchor_fractional, (3,), "anchor_fractional")
        orientation = _validate_orientation(
            self.orientation, DEFAULT_FRACTIONAL_TOLERANCE
        )
        occupation = float(self.occupation)
        if not np.isfinite(occupation) or occupation < 0.0 or occupation > 1.0:
            raise ConstructionValidationError(
                "occupation must be finite and lie in [0, 1]"
            )
        object.__setattr__(self, "anchor_fractional", anchor)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "occupation", occupation)


def _operation_key(operation: FractionalAffineOperation) -> tuple[Any, ...]:
    translation = np.asarray(operation.canonical_translation(), dtype=float)
    translation = np.where(np.isclose(translation, 1.0, atol=1.0e-8), 0.0, translation)
    return (
        tuple(int(value) for value in np.rint(operation.rotation).astype(int).ravel()),
        tuple(float(np.round(value, 8)) for value in translation),
    )


def _same_operation_set(
    left: Sequence[FractionalAffineOperation],
    right: Sequence[FractionalAffineOperation],
) -> bool:
    return {_operation_key(operation) for operation in left} == {
        _operation_key(operation) for operation in right
    }


def _pymatgen_space_group(label: str):
    from pymatgen.symmetry.groups import SpaceGroup

    compact = re.sub(r"\s+", "", label.strip())
    variants = [
        label.strip(),
        compact,
        re.sub(r"([2346])([123456])", r"\1_\2", compact),
    ]
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        try:
            return SpaceGroup(variant)
        except Exception:
            continue
    return None


def _fractional_operation_from_symmop(operation) -> FractionalAffineOperation:
    return FractionalAffineOperation(
        operation.rotation_matrix, operation.translation_vector
    )


def _operations_from_xyz(
    tokens: Sequence[str],
) -> tuple[FractionalAffineOperation, ...]:
    from pymatgen.core.operations import SymmOp

    return tuple(
        _fractional_operation_from_symmop(SymmOp.from_xyz_str(str(token)))
        for token in tokens
    )


def _hall_symmetry_entry(hall: str):
    from pymatgen.symmetry.groups import SpaceGroup

    target = re.sub(r"\s+", " ", hall.strip())
    for entry in SpaceGroup.SYMM_OPS:
        candidate = re.sub(r"\s+", " ", str(entry.get("hall", "")).strip())
        if candidate == target:
            return entry
    return None


def _origin_choice_operations(number: int | None, choice: str):
    from pymatgen.symmetry.groups import SpaceGroup

    matches = []
    for entry in SpaceGroup.SYMM_OPS:
        if number is not None and int(entry["number"]) != int(number):
            continue
        universal = str(entry.get("universal_h_m", ""))
        if universal.endswith(f":{choice}") or f":{choice}" in universal:
            matches.append(entry)
    if len(matches) != 1:
        return None
    return _operations_from_xyz(matches[0]["symops"])


def _parse_fraction_token(token: str) -> float:
    token = token.strip()
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return float(numerator) / float(denominator)
    return float(token)


def _parse_basis_matrix(basis: str | None) -> np.ndarray | None:
    if basis is None:
        return None
    text = str(basis).strip()
    if text.lower() in {"", "i", "identity", "a,b,c", "a, b, c"}:
        return None
    try:
        if text.startswith("["):
            import json

            matrix = np.asarray(json.loads(text), dtype=float)
        else:
            values = [float(part) for part in re.split(r"[\s,]+", text) if part]
            matrix = np.asarray(values, dtype=float).reshape(3, 3)
    except (TypeError, ValueError) as error:
        raise InvalidSymmetryError(f"unrecognized basis transform {basis!r}") from error
    if matrix.shape != (3, 3):
        raise InvalidSymmetryError("basis transform must be a 3x3 unimodular matrix")
    return matrix


def _parse_origin_choice(origin_choice: str | None):
    if origin_choice is None:
        return None
    token = str(origin_choice).strip()
    if token.lower() in {"", "0", "1", "default", "standard", "origin1"}:
        return None
    if re.fullmatch(r"\d+", token):
        return ("choice", token)
    parts = [part.strip() for part in token.split(",")]
    if len(parts) != 3:
        raise InvalidSymmetryError(f"unrecognized origin choice {origin_choice!r}")
    try:
        return (
            "shift",
            np.asarray([_parse_fraction_token(part) for part in parts], dtype=float),
        )
    except (TypeError, ValueError) as error:
        raise InvalidSymmetryError(
            f"unrecognized origin choice {origin_choice!r}"
        ) from error


def _validate_declared_space_group(
    operations: Sequence[FractionalAffineOperation],
    *,
    space_group_number: int | None,
    space_group_symbol: str | None,
    hall_symbol: str | None,
    setting: str | None,
    origin_choice: str | None,
    basis: str | None,
) -> None:
    """Require a declared group to match its complete reference operation set.

    Absent crystallographic labels leave a closed custom group valid.  A
    declared number, Hermann–Mauguin symbol, Hall symbol, setting, origin, or
    basis is checked against the same pymatgen space-group tables used by the
    strict CIF reader.  A self-consistent subgroup with a conflicting label is
    rejected.
    """
    labels_absent = all(
        value in (None, "")
        for value in (
            space_group_number,
            space_group_symbol,
            hall_symbol,
            setting,
            origin_choice,
        )
    )
    if labels_absent and basis in (None, ""):
        return
    reference: list[FractionalAffineOperation] | None = None
    number = None if space_group_number is None else int(space_group_number)
    symbol_group = None
    if space_group_symbol:
        symbol_group = _pymatgen_space_group(str(space_group_symbol))
        if symbol_group is None:
            raise InvalidSymmetryError(
                f"unrecognized space-group symbol {space_group_symbol!r}"
            )
        if number is not None and int(symbol_group.int_number) != number:
            raise InvalidSymmetryError(
                "space-group number "
                f"{number} disagrees with symbol {space_group_symbol!r} "
                f"(#{symbol_group.int_number})"
            )
        number = int(symbol_group.int_number)
        reference = [
            _fractional_operation_from_symmop(operation)
            for operation in symbol_group.symmetry_ops
        ]
    elif number is not None:
        from pymatgen.symmetry.groups import SpaceGroup

        try:
            symbol_group = SpaceGroup.from_int_number(number)
        except Exception as error:
            raise InvalidSymmetryError(
                f"unrecognized space-group number {number}"
            ) from error
        reference = [
            _fractional_operation_from_symmop(operation)
            for operation in symbol_group.symmetry_ops
        ]
    if hall_symbol:
        entry = _hall_symmetry_entry(str(hall_symbol))
        if entry is None:
            raise InvalidSymmetryError(f"unrecognized Hall symbol {hall_symbol!r}")
        if number is not None and int(entry["number"]) != number:
            raise InvalidSymmetryError(
                f"Hall symbol {hall_symbol!r} belongs to space group #{entry['number']}, "
                f"not #{number}"
            )
        number = int(entry["number"])
        hall_ops = list(_operations_from_xyz(entry["symops"]))
        if reference is not None and not _same_operation_set(reference, hall_ops):
            raise InvalidSymmetryError(
                "Hall symbol selects a different setting or origin than the declared symbol"
            )
        reference = hall_ops
    if setting not in (None, ""):
        setting_text = str(setting).strip()
        if setting_text.lower() not in {"standard", "default"}:
            setting_group = _pymatgen_space_group(setting_text)
            if setting_group is None:
                raise InvalidSymmetryError(
                    f"unrecognized space-group setting {setting!r}"
                )
            if number is not None and int(setting_group.int_number) != number:
                raise InvalidSymmetryError(
                    f"setting {setting!r} belongs to space group #{setting_group.int_number}, "
                    f"not #{number}"
                )
            setting_ops = [
                _fractional_operation_from_symmop(operation)
                for operation in setting_group.symmetry_ops
            ]
            if reference is not None and not _same_operation_set(
                reference, setting_ops
            ):
                raise InvalidSymmetryError(
                    "declared setting does not match the declared space-group operations"
                )
            reference = setting_ops
            number = int(setting_group.int_number)
    origin = _parse_origin_choice(origin_choice)
    if origin is not None and origin[0] == "choice":
        if number is None:
            raise InvalidSymmetryError(
                "an origin choice requires a declared space group"
            )
        choice_ops = _origin_choice_operations(number, origin[1])
        if choice_ops is None:
            raise InvalidSymmetryError(
                f"origin choice {origin[1]!r} is not a unique setting of space group #{number}"
            )
        reference = list(choice_ops)
    basis_matrix = _parse_basis_matrix(basis)
    origin_shift = origin[1] if origin is not None and origin[0] == "shift" else None
    if reference is not None and (basis_matrix is not None or origin_shift is not None):
        change = LatticeBasisChange(
            np.eye(3) if basis_matrix is None else basis_matrix,
            np.zeros(3) if origin_shift is None else origin_shift,
        )
        try:
            reference = [
                change.transform_operation(operation) for operation in reference
            ]
        except ValueError as error:
            raise InvalidSymmetryError(str(error)) from error
    if reference is None:
        if basis_matrix is not None or origin is not None:
            return
        raise InvalidSymmetryError(
            "crystallographic labels do not identify a reference group"
        )
    if not _same_operation_set(operations, reference):
        label = space_group_symbol or (
            f"#{number}" if number is not None else "the declared group"
        )
        raise InvalidSymmetryError(
            f"supplied symmetry operations do not match {label}: "
            f"got {len(operations)} operation(s), reference has {len(reference)}. "
            "A closed subgroup, a different setting, or a different origin is not the declared group."
        )


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
            raise ConstructionValidationError(
                "fractional_tolerance must be positive and finite"
            )
        operations = _validate_group(
            tuple(self.symmetry.operations), lattice, self.fractional_tolerance
        )
        _validate_declared_space_group(
            operations,
            space_group_number=self.symmetry.space_group_number,
            space_group_symbol=self.symmetry.space_group_symbol,
            hall_symbol=self.symmetry.hall_symbol,
            setting=self.setting,
            origin_choice=self.origin_choice,
            basis=self.basis,
        )
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


def _template_fractional_pose(
    context: ForwardConstructionContext, pose: MolecularPose
) -> np.ndarray:
    anchor = pose.template.anchor(pose.anchor)
    cart_anchor = pose.anchor_fractional @ context.lattice
    positioned = (
        np.asarray(pose.template.atoms.positions, dtype=float) - anchor
    ) @ pose.orientation.T
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
    for key in sorted(
        {_atom_identity_key(pose.template.atoms, i) for i in range(len(symbols))}
    ):
        left = [
            i
            for i in range(len(symbols))
            if _atom_identity_key(pose.template.atoms, i) == key
        ]
        if not left:
            continue
        # One common lattice shift has already been removed.  Matching the
        # unwrapped molecule atom-by-atom would certify a false stabilizer
        # whenever two atoms are a lattice vector apart.  Boundary-crossing
        # molecules still match when a single coherent image brings the whole
        # pose into coincidence.
        delta = mapped[left, None, :] - original[None, left, :]
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
    if pose.template.bonds:
        original_bonds = {_bond_identity(record) for record in pose.template.bonds}
        mapped_bonds = {
            _bond_identity(
                replace(
                    record,
                    left=permutation[record.left],
                    right=permutation[record.right],
                    image_shift=_transform_bond_image_shift(
                        operation, record.image_shift
                    ),
                )
            )
            for record in pose.template.bonds
        }
        if mapped_bonds != original_bonds:
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
        "enantiopure",
        "enantiopure_only",
        "forbid_improper",
    }:
        return False, tuple(permutation), max_distance
    return True, tuple(permutation), max_distance


def _site_operations(
    context: ForwardConstructionContext, anchor: np.ndarray
) -> tuple[FractionalAffineOperation, ...]:
    return tuple(
        operation
        for operation in context.operations
        if _fractional_equal(
            operation.apply(anchor, wrap=False), anchor, context.fractional_tolerance
        )
    )


def _orbit_representatives(
    context: ForwardConstructionContext, anchor: np.ndarray
) -> tuple[FractionalAffineOperation, ...]:
    representatives: list[FractionalAffineOperation] = []
    positions: list[np.ndarray] = []
    for operation in context.operations:
        target = operation.apply(anchor, wrap=False)
        if any(
            _fractional_equal(target, existing, context.fractional_tolerance)
            for existing in positions
        ):
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
    near_special_distance_A: float = float("inf")
    group_chemistry_compatible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "anchor_fractional",
            np.asarray(self.anchor_fractional, dtype=float).copy(),
        )
        self.anchor_fractional.setflags(write=False)
        object.__setattr__(
            self,
            "anchor_orbit",
            tuple(np.asarray(item, dtype=float).copy() for item in self.anchor_orbit),
        )

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
            "near_special_distance_A": self.near_special_distance_A,
            "near_special_distance_units": {
                "near_special_distance": "fractional residual, dimensionless",
                "near_special_distance_A": "angstrom",
            },
            "group_chemistry_compatible": self.group_chemistry_compatible,
            "reason_codes": list(self.reason_codes),
            "conflicting_operations": [op.xyz for op in self.conflicting_operations],
        }


def _enantiopure_group_conflict(
    context: ForwardConstructionContext,
    stabilizer_operations: Sequence[FractionalAffineOperation],
) -> bool:
    """Return whether an explicit enantiopure policy meets an improper mate.

    Site-subgroup checks do not see a general-position inversion.  The policy
    applies to every operation that would generate a distinct enantiomeric
    copy, while an improper operation already inside the molecular stabilizer
    remains an ordinary site incompatibility.
    """
    if context.chemistry_policy.lower() not in _ENANTIOPURE_POLICIES:
        return False
    for operation in context.operations:
        if (
            operation.determinant < 0
            and _equivalent(
                operation, stabilizer_operations, context.fractional_tolerance
            )
            is None
        ):
            return True
    return False


def analyze_molecular_pose(
    context: ForwardConstructionContext, pose: MolecularPose
) -> SiteSymmetryAnalysis:
    """Analyze S, K, multiplicities, atom maps, and ordered compatibility."""
    anchor = pose.anchor_fractional
    site = _site_operations(context, anchor)
    stabilizers: list[FractionalAffineOperation] = []
    mappings: list[tuple[int, ...]] = []
    conflicts: list[FractionalAffineOperation] = []
    max_distance = 0.0
    for operation in site:
        compatible, mapping, distance = _match_operation_to_template(
            context, pose, operation
        )
        max_distance = max(max_distance, distance)
        if compatible:
            stabilizers.append(operation)
            if mapping is not None:
                mappings.append(mapping)
        else:
            conflicts.append(operation)
    orbit = _orbit_representatives(context, anchor)
    non_identity_residuals = []
    non_identity_residuals_A = []
    for operation in context.operations:
        if (
            _equivalent(
                operation, (identity_operation(),), context.fractional_tolerance
            )
            is not None
        ):
            continue
        delta = operation.apply(anchor, wrap=False) - anchor
        fractional_residual = delta - np.rint(delta)
        non_identity_residuals.append(float(np.linalg.norm(fractional_residual)))
        non_identity_residuals_A.append(
            float(np.linalg.norm(fractional_residual @ context.lattice))
        )
    # ``near_special_distance`` is the dimensionless fractional residual.
    # ``near_special_distance_A`` is that same residual measured in Angstrom.
    # Neither value snaps the pose or changes the molecule count.
    distance_to_special = (
        min(non_identity_residuals) if non_identity_residuals else float("inf")
    )
    distance_to_special_A = (
        min(non_identity_residuals_A) if non_identity_residuals_A else float("inf")
    )
    group_chemistry_compatible = not _enantiopure_group_conflict(
        context, tuple(stabilizers)
    )
    site_order = len(site)
    stabilizer_order = len(stabilizers)
    orientation_count = site_order // stabilizer_order if stabilizer_order else 0
    reasons: list[str] = []
    if site_order > 1:
        reasons.append("special_position")
    if conflicts:
        reasons.append("molecular_stabilizer_incomplete")
    if context.chemistry_policy.lower() in _ENANTIOPURE_POLICIES and any(
        operation.determinant < 0 for operation in conflicts
    ):
        reasons.append("improper_operation_conflicts_with_chemistry_policy")
    if not group_chemistry_compatible:
        reasons.append("enantiopure_policy_conflicts_with_improper_operations")
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
        near_special_distance_A=distance_to_special_A,
        group_chemistry_compatible=group_chemistry_compatible,
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
        remaining = [
            item for item in remaining if _equivalent(item, coset, tolerance) is None
        ]
    return tuple(result)


def _transform_bond_image_shift(
    operation: FractionalAffineOperation,
    image_shift: Sequence[int],
) -> tuple[int, int, int]:
    """Rotate a bond's lattice image by the generating operation.

    Fractional row vectors transform as ``f @ W.T``.  The integer image that
    takes the right endpoint into the bonded cell therefore transforms by the
    same linear map, not by being copied onto the mate.
    """
    shift = (
        np.asarray(image_shift, dtype=float)
        @ np.asarray(operation.rotation, dtype=float).T
    )
    rounded = np.rint(shift)
    if not np.allclose(shift, rounded, atol=1.0e-8, rtol=0.0):
        raise ConstructionValidationError(
            "symmetry operation does not map the bond image onto an integer lattice shift"
        )
    return tuple(int(value) for value in rounded)


def _bond_record_payload(record: PeriodicBondRecord) -> dict[str, Any]:
    return {
        "left": int(record.left),
        "right": int(record.right),
        "order": record.order,
        "order_known": bool(record.order_known),
        "bond_type": record.bond_type,
        "bond_type_known": bool(record.bond_type_known),
        "kind": record.kind,
        "image_shift": [int(value) for value in record.image_shift],
    }


def _bond_identity(record: PeriodicBondRecord) -> tuple[Any, ...]:
    """Undirected typed edge key, retaining its directed lattice image."""
    forward = (record.left, record.right, tuple(record.image_shift))
    reverse = (record.right, record.left, tuple(-x for x in record.image_shift))
    return (
        min(forward, reverse),
        record.order_known,
        record.order if record.order_known else None,
        record.bond_type_known,
        record.bond_type if record.bond_type_known else None,
        record.kind,
    )


def _has_known_stereo(atoms: Atoms) -> bool:
    values = atoms.arrays.get("stereochemistry")
    if values is None:
        return False
    return any(not _unknown_token(value) for value in values)


def _reassign_improper_stereochemistry(
    molecule: CrystalMolecule,
    bonds: Sequence[PeriodicBondRecord],
) -> None:
    """Replace copied stereo labels using the transformed typed graph.

    An improper isometry is not a blanket R/S or E/Z swap.  Known labels are
    reassigned from the CIP engine on the transformed coordinates.  If that
    assignment is indeterminate, the improper mate keeps an unknown label
    instead of the untransformed descriptor.  Empty source labels stay empty.
    """
    from ..chemistry.models import (
        BondKind,
        ChemicalAtom,
        ChemicalBond,
        Embedding,
        FiniteChemicalEntity,
    )
    from ..chemistry.stereo import StereoKind, assign_stereochemistry

    count = len(molecule)
    symbols = molecule.get_chemical_symbols()
    positions = np.asarray(molecule.get_positions(), dtype=float)
    source = molecule.arrays.get("stereochemistry")
    source_values = [
        "" if source is None else str(source[index]) for index in range(count)
    ]
    isotopes = molecule.arrays.get("isotope")
    atom_ids = [f"a{index}" for index in range(count)]
    chemical_atoms = tuple(
        ChemicalAtom(
            atom_ids[index],
            symbols[index],
            isotope=int(isotopes[index])
            if isotopes is not None and int(isotopes[index]) > 0
            else None,
        )
        for index in range(count)
    )
    chemical_bonds = []
    for record in bonds:
        if not (0 <= record.left < count and 0 <= record.right < count):
            continue
        kind_name = (record.kind or "unknown").upper()
        try:
            kind = BondKind[kind_name]
        except KeyError:
            kind = BondKind.UNKNOWN
        chemical_bonds.append(
            ChemicalBond(
                atom_ids[record.left],
                atom_ids[record.right],
                order=record.order if record.order_known else None,
                kind=kind,
                aromatic=bool(record.bond_type == "aromatic"),
                atom2_image_shift=tuple(int(value) for value in record.image_shift),
            )
        )
    embedding = Embedding(
        tuple(
            (atom_ids[index], tuple(float(value) for value in positions[index]))
            for index in range(count)
        )
    )
    entity = FiniteChemicalEntity(
        "forward-improper-mate",
        chemical_atoms,
        tuple(chemical_bonds),
        embedding=embedding,
    )
    report = assign_stereochemistry(entity, embedding)
    assigned: dict[int, str] = {}
    for descriptor in report.descriptors:
        if descriptor.descriptor not in {"R", "S", "E", "Z"}:
            continue
        try:
            index = atom_ids.index(descriptor.center_atom_id)
        except ValueError:
            continue
        if descriptor.kind is StereoKind.DOUBLE_BOND:
            # E/Z is geometric, not a chiral label.  Keep the geometry result
            # on the bond's reference atom rather than exchanging E with Z.
            assigned[index] = descriptor.descriptor
        else:
            assigned[index] = descriptor.descriptor
    updated = []
    for index, value in enumerate(source_values):
        if _unknown_token(value):
            updated.append("")
            continue
        updated.append(assigned.get(index, ""))
    molecule.set_array("stereochemistry", np.asarray(updated, dtype=str))
    molecule.info["forward_stereo_reassigned"] = True


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
    atoms.info = copy.deepcopy(source.info)
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
            [
                f"{symbol}{index + 1}"
                for index, symbol in enumerate(source.get_chemical_symbols())
            ],
            dtype=str,
        )
    atoms.set_array(KEY_LABEL, np.asarray(source_labels, dtype=str).copy())
    atoms.set_array(
        KEY_ATOM_ID,
        np.asarray(
            [f"pose{pose_index}:op{operation_index}:a{index}" for index in range(n)],
            dtype=str,
        ),
    )
    molecule = CrystalMolecule(atoms, check_pbc=False)
    # Keep a stable entity/atom identity across operation mates.  The atom ID
    # retains the generated operation for provenance, while the pose/template
    # metadata lets ASU reduction recognise the same atom in another mate
    # without collapsing independent same-element entities.
    molecule.info["forward_template_identifier"] = pose.template.identifier
    molecule.info["forward_chemical_identity"] = pose.template.chemical_identity
    molecule.info["forward_pose_index"] = int(pose_index)
    molecule.info["forward_operation_index"] = int(operation_index)
    if atom_identities is None:
        atom_identities = [f"a{index}" for index in range(n)]
    if len(atom_identities) != n:
        raise ConstructionValidationError(
            "atom_identities must match template atom count"
        )
    molecule.info["forward_atom_identities"] = tuple(
        str(item) for item in atom_identities
    )
    if pose.template.bonds:
        transformed_bonds = tuple(
            replace(
                record,
                image_shift=_transform_bond_image_shift(operation, record.image_shift),
            )
            for record in pose.template.bonds
        )
        molecule.info["forward_connectivity"] = [
            [int(record.left), int(record.right)] for record in transformed_bonds
        ]
        molecule.info["bond_pairs"] = [
            record.pair() for record in transformed_bonds if record.left != record.right
        ]
        molecule.info["forward_bonds"] = [
            _bond_record_payload(record) for record in transformed_bonds
        ]
        molecule._graph = nx.Graph()
        molecule._graph.add_nodes_from(range(n))
        positions = np.asarray(molecule.get_positions(), dtype=float)
        for record in transformed_bonds:
            left, right = int(record.left), int(record.right)
            if left == right or not (0 <= left < n and 0 <= right < n):
                continue
            vector = positions[right] - positions[left]
            molecule._graph.add_edge(
                left,
                right,
                vector=vector.copy(),
                distance=float(np.linalg.norm(vector)),
                image_shift=np.asarray(record.image_shift, dtype=int),
                order=record.order,
                order_known=bool(record.order_known),
                bond_type=record.bond_type,
                bond_type_known=bool(record.bond_type_known),
                kind=record.kind,
            )
    else:
        transformed_bonds = ()
    if operation.determinant < 0 and _has_known_stereo(molecule):
        _reassign_improper_stereochemistry(molecule, transformed_bonds)
    return molecule


def construct_molecular_crystal(
    context: ForwardConstructionContext,
    poses: MolecularPose | Sequence[MolecularPose],
    *,
    strict: bool = True,
    ordered: bool = True,
    reject_contacts: bool = False,
    expected_formula: Mapping[str, float] | None = None,
    formula_unit: Mapping[str, int] | None = None,
    Z: int | None = None,
    expected_charge: float | None = None,
) -> MolecularCrystal:
    """Construct complete ordered molecules from explicit poses.

    ``reject_contacts`` reports and rejects intermolecular approaches closer
    than the summed covalent radii.  It never moves the supplied pose.
    """
    if not ordered:
        return construct_statistical_disorder(
            context,
            poses,
            expected_formula=expected_formula,
            formula_unit=formula_unit,
            Z=Z,
            expected_charge=expected_charge,
        )
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
        if not analysis.group_chemistry_compatible:
            raise IncompatibleChemistryError(
                analysis,
                "enantiopure chemistry policy is incompatible with an improper "
                "operation outside the molecular stabilizer; the full group would "
                "generate an enantiomeric mate",
            )
        if strict and not analysis.ordered_compatible:
            raise IncompatibleSiteError(analysis)
        center_orbit = _center_orbit_key(context, pose.anchor_fractional)
        subgroup = (
            analysis.stabilizer_operations
            if analysis.ordered_compatible
            else (identity_operation(),)
        )
        # A center is a crystallographic occupation slot.  Template names and
        # element species do not make two full-occupancy poses independent
        # slots, so reject collisions before expanding either pose.
        if center_orbit in seen_centers:
            raise ConstructionValidationError(
                "duplicate or conflicting molecular pose center orbit"
            )
        seen_centers.add(center_orbit)
        atom_identities = _stabilizer_atom_identities(
            analysis, pose.template.atom_count
        )
        representatives = _coset_representatives(
            context.operations, subgroup, context.fractional_tolerance
        )
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
        "hydrogen_policy": "explicit_template",
    }
    crystal = MolecularCrystal(context.lattice, molecules, metadata=metadata)
    crystal.metadata["composition_accounting"] = _composition_accounting(
        [(molecule, 1.0) for molecule in molecules],
        expected_formula=expected_formula,
        formula_unit=formula_unit,
        Z=Z,
        expected_charge=expected_charge,
    )
    contacts = intermolecular_contacts(crystal)
    crystal.metadata["intermolecular_contacts"] = contacts
    crystal.metadata["realized_symmetry"] = _realized_symmetry(context, molecules)
    if reject_contacts and contacts:
        raise ConstructionValidationError(
            f"ordered construction has {len(contacts)} intermolecular contact(s) "
            "inside the summed covalent radii"
        )
    return crystal


def _composition_accounting(
    states, *, expected_formula=None, formula_unit=None, Z=None, expected_charge=None
):
    """Check explicit composition/charge contracts without guessing charges."""
    totals, charge, unknown = {}, 0.0, 0.0
    for molecule, weight in states:
        if molecule is None or weight == 0:
            continue
        for index, symbol in enumerate(molecule.get_chemical_symbols()):
            totals[symbol] = totals.get(symbol, 0.0) + weight
            charges = molecule.arrays.get("formal_charge")
            known = molecule.arrays.get("formal_charge_known")
            if charges is not None and known is not None and known[index]:
                charge += weight * float(charges[index])
            else:
                unknown += weight
    if (formula_unit is None) != (Z is None):
        raise ConstructionValidationError(
            "formula_unit and Z must be supplied together"
        )
    if Z is not None:
        if isinstance(Z, bool) or not isinstance(Z, (int, np.integer)) or Z <= 0:
            raise ConstructionValidationError("Z must be a positive integer")
        formula = {key: float(value) * Z for key, value in formula_unit.items()}
        if expected_formula is not None and formula != dict(expected_formula):
            raise ConstructionValidationError(
                "formula_unit/Z contradicts expected_formula"
            )
        expected_formula = formula
    if expected_formula is not None:
        expected = {str(key): float(value) for key, value in expected_formula.items()}
        if any(not np.isfinite(value) or value < 0 for value in expected.values()):
            raise ConstructionValidationError(
                "expected formula counts must be finite and non-negative"
            )
        if any(
            abs(totals.get(key, 0.0) - expected.get(key, 0.0))
            > PARTIAL_OCCUPANCY_TOLERANCE
            for key in totals.keys() | expected.keys()
        ):
            raise ConstructionValidationError(
                "constructed composition disagrees with expected formula"
            )
    if expected_charge is not None:
        if not np.isfinite(expected_charge):
            raise ConstructionValidationError("expected charge must be finite")
        if unknown:
            raise ConstructionValidationError(
                "charge balance is indeterminate: formal charges are unknown"
            )
        if abs(charge - expected_charge) > PARTIAL_OCCUPANCY_TOLERANCE:
            raise ConstructionValidationError(
                "constructed charge disagrees with expected charge"
            )
    return {
        "formula": totals,
        "known_charge": charge,
        "unknown_charge_atoms": unknown,
        "total_charge": charge if not unknown else None,
    }


def intermolecular_contacts(crystal: MolecularCrystal) -> tuple[dict[str, Any], ...]:
    """Return intermolecular clashes without changing coordinates.

    Distances are Cartesian angstroms. All images inside the summed covalent
    radii are included, including nonzero images of the same molecule. The
    crystal is not snapped, expanded, or otherwise repaired.
    """
    from ..constants import get_atomic_radius, has_atomic_radius
    from ase.neighborlist import neighbor_list

    molecules = list(crystal.molecules)
    if not molecules:
        return ()
    radii = []
    symbols = []
    positions = []
    owners = []
    for molecule_index, molecule in enumerate(molecules):
        molecule_symbols = molecule.get_chemical_symbols()
        molecule_positions = np.asarray(molecule.get_positions(), dtype=float)
        for atom_index, symbol in enumerate(molecule_symbols):
            symbols.append(symbol)
            positions.append(molecule_positions[atom_index])
            owners.append(molecule_index)
            radii.append(
                float(get_atomic_radius(symbol)) if has_atomic_radius(symbol) else 0.7
            )
    atoms = Atoms(symbols=symbols, positions=positions, cell=crystal.lattice, pbc=True)
    lefts, rights, shifts, distances = neighbor_list("ijSd", atoms, radii)
    contacts = []
    seen = set()
    for left, right, shift, distance in zip(lefts, rights, shifts, distances):
        if owners[left] == owners[right] and not np.any(shift):
            continue
        edge = (int(left), int(right), tuple(int(x) for x in shift))
        reverse = (int(right), int(left), tuple(-int(x) for x in shift))
        if min(edge, reverse) in seen:
            continue
        seen.add(min(edge, reverse))
        limit = radii[left] + radii[right]
        if distance < limit:
            contacts.append(
                {
                    "left_molecule": int(owners[left]),
                    "right_molecule": int(owners[right]),
                    "left_symbol": symbols[left],
                    "right_symbol": symbols[right],
                    "distance_A": distance,
                    "limit_A": limit,
                    "image_shift": tuple(int(x) for x in shift),
                }
            )
    return tuple(
        sorted(
            contacts,
            key=lambda item: (
                item["left_molecule"],
                item["right_molecule"],
                item["image_shift"],
                item["distance_A"],
            ),
        )
    )


# Friendly aliases used in examples and downstream integrations.
build_molecular_crystal = construct_molecular_crystal
build_symmetry_complete_crystal = construct_molecular_crystal


@dataclass(frozen=True)
class MolecularAlternative:
    """One complete, mutually exclusive molecular state."""

    orbit_id: str
    alternative_id: str
    molecule: CrystalMolecule | None
    weight: float
    operation_index: int
    center_fractional: np.ndarray
    vacancy: bool = False

    def __post_init__(self) -> None:
        center = np.asarray(self.center_fractional, dtype=float).copy()
        center.setflags(write=False)
        object.__setattr__(self, "center_fractional", center)
        if self.weight < 0 or not np.isfinite(self.weight):
            raise ConstructionValidationError(
                "alternative weight must be finite and non-negative"
            )
        if self.vacancy and self.molecule is not None:
            raise ConstructionValidationError(
                "a vacancy alternative cannot contain a molecule"
            )
        if not self.vacancy and self.molecule is None:
            raise ConstructionValidationError(
                "an occupied alternative requires a molecule"
            )


@dataclass(frozen=True)
class AverageAtomSite:
    """Derived average atom site and its CIF occupancy."""

    symbol: str
    fractional: np.ndarray
    occupancy: float
    orbit_id: str
    template_atom_index: int
    alternative_ids: tuple[str, ...] = ()
    isotope: int = 0
    formal_charge: float | None = None
    formal_charge_known: bool = False
    site_identity: str = ""
    mass: float | None = None
    chemical_annotations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        fractional = _readonly_array(self.fractional, (3,), "fractional")
        object.__setattr__(self, "fractional", _canonical_fractional(fractional))
        if (
            not np.isfinite(self.occupancy)
            or self.occupancy < 0
            or self.occupancy > 1.0 + 1e-8
        ):
            raise ConstructionValidationError(
                "average atom occupancy must lie in [0, 1]"
            )


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
    vacancies: tuple[MolecularAlternative, ...] = ()
    constraint_report: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expected_formula(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for alternative in self.alternatives:
            if alternative.vacancy or alternative.molecule is None:
                continue
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
        self,
        selections: Mapping[str, str | int] | None = None,
        *,
        reject_contacts: bool = False,
    ) -> MolecularCrystal:
        """Select one mutually exclusive state per orbit."""
        return select_ordered_realization(
            self, selections, reject_contacts=reject_contacts
        )

    def to_dict(self) -> dict[str, Any]:
        return _statistical_model_payload(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StatisticalDisorderModel":
        """Rebuild a model from a versioned lossless companion payload."""
        return read_statistical_disorder_companion(payload)


def _chemical_site_key(site: AverageAtomSite) -> tuple[Any, ...]:
    charge = (
        ("known", float(site.formal_charge))
        if site.formal_charge_known
        else ("unknown",)
    )
    return (
        site.symbol,
        int(site.isotope),
        charge,
        str(site.site_identity),
        site.mass,
        tuple(site.chemical_annotations),
    )


def _merge_average_site(
    sites: list[AverageAtomSite],
    candidate: AverageAtomSite,
    tolerance: float,
) -> None:
    for index, existing in enumerate(sites):
        if existing.orbit_id != candidate.orbit_id:
            continue
        if _chemical_site_key(existing) != _chemical_site_key(candidate):
            continue
        if _fractional_equal(existing.fractional, candidate.fractional, tolerance):
            alternatives = tuple(
                dict.fromkeys(existing.alternative_ids + candidate.alternative_ids)
            )
            sites[index] = AverageAtomSite(
                existing.symbol,
                existing.fractional,
                existing.occupancy + candidate.occupancy,
                existing.orbit_id,
                existing.template_atom_index,
                alternatives,
                isotope=existing.isotope,
                formal_charge=existing.formal_charge,
                formal_charge_known=existing.formal_charge_known,
                site_identity=existing.site_identity,
                mass=existing.mass,
                chemical_annotations=existing.chemical_annotations,
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
            tuple(
                np.round(_canonical_fractional(operation.apply(anchor, wrap=False)), 9)
            )
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
    roots = {
        root: index
        for index, root in enumerate(sorted({find(i) for i in range(atom_count)}))
    }
    return tuple(f"atom_orbit{roots[find(index)]}" for index in range(atom_count))


def _context_operation_index(
    context: ForwardConstructionContext,
    operation: FractionalAffineOperation,
) -> int:
    """Return the canonical zero-based index for a group operation."""
    for index, candidate in enumerate(context.operations):
        if operation.equivalent_to(candidate, tolerance=context.fractional_tolerance):
            return index
    raise ConstructionValidationError(
        "generated operation is outside the validated symmetry group"
    )


def construct_statistical_disorder(
    context: ForwardConstructionContext,
    poses: MolecularPose | Sequence[MolecularPose],
    *,
    center_occupation: float | None = None,
    center_occupancies: Mapping[int | str, float] | None = None,
    correlation_status: str = "local-only",
    correlations: Mapping[str, Any] | None = None,
    expected_formula: Mapping[str, float] | None = None,
    formula_unit: Mapping[str, int] | None = None,
    Z: int | None = None,
    expected_charge: float | None = None,
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
        correlations = copy.deepcopy(dict(correlations))

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
        if not analysis.group_chemistry_compatible:
            raise IncompatibleChemistryError(
                analysis,
                "enantiopure chemistry policy is incompatible with the full source group",
            )
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
            raise ConstructionValidationError(
                "each center occupation must lie in [0, 1]"
            )
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
        atom_identities = _stabilizer_atom_identities(
            analysis, pose.template.atom_count
        )
        local_count = len(local_representatives)
        if (
            local_count != analysis.orientation_multiplicity
            and analysis.ordered_compatible
        ):
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
                    _canonical_fractional(
                        operation.apply(pose.anchor_fractional, wrap=False)
                    ),
                )
                alternatives.append(alternative)
                mapped_fractional, _ = _mapped_template(context, pose, operation)
                template_atoms = pose.template.atoms
                isotope_values = template_atoms.arrays.get("isotope")
                charge_values = template_atoms.arrays.get("formal_charge")
                charge_known = template_atoms.arrays.get("formal_charge_known")
                for atom_index, (symbol, fractional) in enumerate(
                    zip(template_atoms.get_chemical_symbols(), mapped_fractional)
                ):
                    isotope = (
                        int(isotope_values[atom_index])
                        if isotope_values is not None
                        else 0
                    )
                    known_charge = (
                        bool(charge_known[atom_index])
                        if charge_known is not None
                        else False
                    )
                    charge = (
                        float(charge_values[atom_index])
                        if known_charge and charge_values is not None
                        else None
                    )
                    identity = (
                        atom_identities[atom_index]
                        if atom_index < len(atom_identities)
                        else f"a{atom_index}"
                    )
                    _merge_average_site(
                        average_sites,
                        AverageAtomSite(
                            symbol,
                            fractional,
                            weight,
                            orbit_id,
                            atom_index,
                            (alternative_id,),
                            isotope=isotope,
                            formal_charge=charge,
                            formal_charge_known=known_charge,
                            site_identity=str(identity),
                            mass=float(template_atoms.arrays["masses"][atom_index])
                            if "masses" in template_atoms.arrays
                            else None,
                            chemical_annotations=(
                                (
                                    "molecular_identity",
                                    pose.template.chemical_identity or "",
                                ),
                                *_atom_identity_key(molecule, atom_index)[3],
                            ),
                        ),
                        context.fractional_tolerance,
                    )
    # Whole-state alternatives must not overoccupy an orbit.
    for orbit_id, probability in probabilities.items():
        total = sum(item.weight for item in alternatives if item.orbit_id == orbit_id)
        if total > 1.0 + 1e-8:
            raise ConstructionValidationError(
                f"alternatives overoccupy center orbit {orbit_id}"
            )
    vacancies: list[MolecularAlternative] = []
    occupied_weight: dict[str, float] = {}
    centers: dict[str, np.ndarray] = {}
    for alternative in alternatives:
        occupied_weight[alternative.orbit_id] = occupied_weight.get(
            alternative.orbit_id, 0.0
        ) + float(alternative.weight)
        centers.setdefault(alternative.orbit_id, alternative.center_fractional)
    for orbit_id, occupied in occupied_weight.items():
        residual = 1.0 - occupied
        if residual > 1.0e-8:
            vacancies.append(
                MolecularAlternative(
                    orbit_id,
                    f"{orbit_id}:vacancy",
                    None,
                    residual,
                    -1,
                    centers[orbit_id],
                    vacancy=True,
                )
            )
    constraint_report = _validate_disorder_constraints(
        context,
        tuple(alternatives),
        tuple(vacancies),
        correlations,
        correlation_status,
    )
    metadata = {
        "representation": "statistical_average",
        # Keep the validated object available to ordered-realization and CIF
        # export paths.  ``source_symmetry`` is intentionally a lightweight
        # metadata snapshot, but it is not sufficient to reconstruct the
        # lattice/operation provenance needed for a true ASU export.
        "forward_context": context,
        "source_symmetry": context.to_metadata(),
        "correlation_status": correlation_status,
        "correlations": dict(correlations),
        "constraint_report": constraint_report,
        "composition_accounting": _composition_accounting(
            [(item.molecule, item.weight) for item in alternatives],
            expected_formula=expected_formula,
            formula_unit=formula_unit,
            Z=Z,
            expected_charge=expected_charge,
        ),
    }
    return StatisticalDisorderModel(
        context=context,
        alternatives=tuple(alternatives),
        average_sites=tuple(average_sites),
        orbit_probabilities=probabilities,
        correlation_status=correlation_status,
        correlations=correlations,
        metadata=metadata,
        vacancies=tuple(vacancies),
        constraint_report=constraint_report,
    )


build_orientational_disorder = construct_statistical_disorder


def _alternatives_for_selection(
    model: StatisticalDisorderModel,
) -> dict[str, tuple[MolecularAlternative, ...]]:
    grouped: dict[str, list[MolecularAlternative]] = {}
    for alternative in (*model.alternatives, *model.vacancies):
        grouped.setdefault(alternative.orbit_id, []).append(alternative)
    return {key: tuple(value) for key, value in grouped.items()}


def select_ordered_realization(
    model: StatisticalDisorderModel,
    selections: Mapping[str, str | int] | None = None,
    *,
    reject_contacts: bool = False,
) -> MolecularCrystal:
    """Select one admissible whole-molecule state per orbit.

    A zero-weight occupied state is rejected.  When an orbit has both a
    positive occupied weight and a positive vacancy weight, that orbit needs
    its own occupied or vacancy choice.  ``None`` and ``{}`` do not supply
    one, and naming a different orbit does not either.  The returned metadata
    records the symmetry of the selected coordinates separately from the
    source average symmetry.
    """
    available = _alternatives_for_selection(model)
    selections = dict(selections or {})
    if set(selections) - set(available):
        raise ConstructionValidationError("selection contains unknown center orbit(s)")
    chosen: list[CrystalMolecule] = []
    selected_alternatives: dict[str, str] = {}
    for orbit_id, alternatives in available.items():
        occupied = [
            item for item in alternatives if not item.vacancy and item.weight > 1.0e-8
        ]
        vacant = [
            item for item in alternatives if item.vacancy and item.weight > 1.0e-8
        ]
        if orbit_id not in selections:
            if occupied and vacant:
                raise ConstructionValidationError(
                    "partial center occupation requires an explicit occupied or "
                    f"vacancy selection for orbit {orbit_id}"
                )
            if not occupied and vacant:
                choice: str | int = vacant[0].alternative_id
            elif len(occupied) == 1 and not vacant:
                choice = occupied[0].alternative_id
            else:
                choice = occupied[0].alternative_id if occupied else 0
        else:
            choice = selections[orbit_id]
        if isinstance(choice, str):
            selected = next(
                (item for item in alternatives if item.alternative_id == choice), None
            )
        else:
            if isinstance(choice, bool) or not isinstance(choice, (int, np.integer)):
                raise ConstructionValidationError(
                    "alternative index must be an integer"
                )
            occupied_only = [item for item in alternatives if not item.vacancy]
            selected = (
                occupied_only[int(choice)]
                if 0 <= int(choice) < len(occupied_only)
                else None
            )
        if selected is None:
            raise ConstructionValidationError(
                f"no alternative {choice!r} for orbit {orbit_id}"
            )
        if selected.weight <= 1.0e-8:
            raise ConstructionValidationError(
                f"zero-weight occupied state {selected.alternative_id} cannot be selected"
            )
        if selected.vacancy or selected.molecule is None:
            selected_alternatives[orbit_id] = selected.alternative_id
            continue
        molecule = selected.molecule.copy()
        molecule.info = copy.deepcopy(selected.molecule.info)
        chosen.append(molecule)
        selected_alternatives[orbit_id] = selected.alternative_id
    _reject_infeasible_joint(model, selected_alternatives)
    realized = _realized_symmetry(model.context, chosen)
    metadata = copy.deepcopy(dict(model.metadata))
    metadata["representation"] = "ordered_realization"
    metadata["source_disorder"] = model.to_dict()
    metadata["forward_context"] = model.context
    metadata["source_symmetry"] = model.context.to_metadata()
    metadata["selected_alternatives"] = selected_alternatives
    metadata["realized_symmetry"] = realized
    crystal = MolecularCrystal(model.context.lattice, chosen, metadata=metadata)
    crystal.metadata["composition_accounting"] = _composition_accounting(
        [(molecule, 1.0) for molecule in chosen]
    )
    contacts = intermolecular_contacts(crystal)
    crystal.metadata["intermolecular_contacts"] = contacts
    if reject_contacts and contacts:
        raise ConstructionValidationError(
            f"selected realization has {len(contacts)} intermolecular contact(s)"
        )
    return crystal


def _validate_disorder_constraints(
    context: ForwardConstructionContext,
    alternatives: Sequence[MolecularAlternative],
    vacancies: Sequence[MolecularAlternative],
    correlations: Mapping[str, Any],
    correlation_status: str,
) -> dict[str, Any]:
    """Check an explicit joint distribution against marginals and symmetry.

    Missing joints stay ``unknown``.  A declared feasible distribution must be
    normalized, reproduce the one-site weights, and be closed under the source
    group.  Contradictions raise instead of being stored as a physical model.
    """
    status = str(correlation_status or "local-only")
    if status not in {"local-only", "unknown", "truncated", "feasible", "infeasible"}:
        raise ConstructionValidationError(
            f"unrecognized correlation_status {correlation_status!r}"
        )
    if not correlations:
        if status in {"feasible", "infeasible"}:
            raise ConstructionValidationError(
                "local marginals alone do not establish a feasible or infeasible "
                "joint distribution"
            )
        return {
            "schema": None,
            "status": "unknown",
            "reason": "no joint distribution was supplied",
        }
    schema = correlations.get("schema") if isinstance(correlations, Mapping) else None
    if schema != DISORDER_CONSTRAINT_SCHEMA:
        if status == "local-only":
            return {
                "schema": None,
                "status": "unknown",
                "reason": "opaque correlation metadata is not a validated joint distribution",
            }
        raise ConstructionValidationError(
            f"correlations must use schema {DISORDER_CONSTRAINT_SCHEMA} "
            "before they can be validated"
        )
    by_id = {item.alternative_id: item for item in (*alternatives, *vacancies)}
    orbits = {item.orbit_id for item in (*alternatives, *vacancies)}
    joints = []
    weight_sum = 0.0
    for joint in correlations.get("joints") or []:
        if not isinstance(joint, Mapping):
            raise ConstructionValidationError("each joint must be a mapping")
        weight = float(joint.get("weight", 0.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ConstructionValidationError(
                "joint weight must be finite and non-negative"
            )
        assignment = {
            str(key): str(value)
            for key, value in dict(joint.get("assignment") or {}).items()
        }
        if set(assignment) != orbits:
            raise ConstructionValidationError(
                "each joint must assign every center orbit exactly once"
            )
        for orbit_id, alternative_id in assignment.items():
            alternative = by_id.get(alternative_id)
            if alternative is None or alternative.orbit_id != orbit_id:
                raise ConstructionValidationError(
                    f"joint state {alternative_id!r} is not in orbit {orbit_id}"
                )
            if alternative.weight <= 1.0e-8 and not alternative.vacancy:
                raise ConstructionValidationError(
                    "a joint cannot select a zero-weight occupied alternative"
                )
        joints.append({"weight": weight, "assignment": assignment})
        weight_sum += weight
    marginals: dict[str, float] = {}
    for joint in joints:
        for alternative_id in joint["assignment"].values():
            marginals[alternative_id] = (
                marginals.get(alternative_id, 0.0) + joint["weight"]
            )
    if status == "feasible":
        if abs(weight_sum - 1.0) > PARTIAL_OCCUPANCY_TOLERANCE:
            raise ConstructionValidationError(
                "a feasible joint distribution must sum to 1"
            )
        for alternative in (*alternatives, *vacancies):
            predicted = marginals.get(alternative.alternative_id, 0.0)
            if abs(predicted - float(alternative.weight)) > PARTIAL_OCCUPANCY_TOLERANCE:
                raise ConstructionValidationError(
                    "joint marginals disagree with the declared alternative weight of "
                    f"{alternative.alternative_id}"
                )
        _require_symmetry_covariant_joints(context, by_id, joints)
    elif status == "truncated" and weight_sum > 1.0 + PARTIAL_OCCUPANCY_TOLERANCE:
        raise ConstructionValidationError("truncated joint weights exceed 1")
    elif status == "infeasible":
        raise ConstructionValidationError(
            "infeasible disorder constraints are rejected and are not serialized as a model"
        )
    for item in correlations.get("forbidden") or []:
        assignment = _forbidden_assignment(item, by_id, orbits)
        if any(
            joint["weight"] > 1.0e-8
            and all(
                joint["assignment"].get(key) == value
                for key, value in assignment.items()
            )
            for joint in joints
        ):
            raise ConstructionValidationError("a positive joint is also forbidden")
    return {
        "schema": DISORDER_CONSTRAINT_SCHEMA,
        "status": status if status in {"feasible", "truncated"} else "unknown",
        "joint_count": len(joints),
        "weight_sum": weight_sum,
    }


def _forbidden_assignment(item, by_id, orbits):
    if not isinstance(item, Mapping) or not isinstance(item.get("assignment"), Mapping):
        raise ConstructionValidationError(
            "forbidden constraint requires an assignment mapping"
        )
    assignment = {str(key): str(value) for key, value in item["assignment"].items()}
    if not assignment or not set(assignment) <= orbits:
        raise ConstructionValidationError(
            "forbidden assignment names unknown or empty center orbits"
        )
    if any(
        value not in by_id or by_id[value].orbit_id != key
        for key, value in assignment.items()
    ):
        raise ConstructionValidationError(
            "forbidden assignment names an invalid alternative"
        )
    return assignment


def _image_alternative(
    context: ForwardConstructionContext,
    alternative: MolecularAlternative,
    operation: FractionalAffineOperation,
    alternatives: Sequence[MolecularAlternative],
) -> MolecularAlternative | None:
    center = operation.apply(alternative.center_fractional, wrap=False)
    if alternative.vacancy or alternative.operation_index < 0:
        for candidate in alternatives:
            if candidate.vacancy and _fractional_equal(
                candidate.center_fractional, center, context.fractional_tolerance
            ):
                return candidate
        if _fractional_equal(
            alternative.center_fractional, center, context.fractional_tolerance
        ):
            return alternative
        return None
    for candidate in alternatives:
        if candidate.vacancy or candidate.operation_index < 0:
            continue
        if alternative.molecule.info.get(
            "forward_pose_index"
        ) != candidate.molecule.info.get("forward_pose_index"):
            continue
        if _fractional_equal(
            candidate.center_fractional, center, context.fractional_tolerance
        ) and _molecule_image_matches(
            context,
            alternative.molecule,
            candidate.molecule,
            operation,
        ):
            return candidate
    return None


def _molecule_image_matches(context, source, target, operation) -> bool:
    """Compare complete unwrapped molecular images, including typed topology.

    An operation label is only a coset representative, not a unique molecular
    orientation. Geometry and the graph determine the action on alternatives.
    """
    if len(source) != len(target):
        return False
    inverse = np.linalg.inv(context.lattice)
    mapped = operation.apply(source.positions @ inverse, wrap=False)
    original = target.positions @ inverse
    center_delta = mapped.mean(axis=0) - original.mean(axis=0)
    mapped -= np.rint(center_delta)
    delta = mapped[:, None, :] - original[None, :, :]
    cost = np.linalg.norm(delta @ context.lattice, axis=2)
    for left in range(len(source)):
        for right in range(len(target)):
            # Stereo is transformed by an improper operation; the typed graph
            # plus geometry establishes orientation, never a copied CIP label.
            left_key = _atom_identity_key(source, left)
            right_key = _atom_identity_key(target, right)
            if operation.determinant < 0:
                left_key = (
                    *left_key[:3],
                    tuple(
                        x
                        for x in left_key[3]
                        if x[0] not in {"stereo", "stereochemistry", "cip_code"}
                    ),
                )
                right_key = (
                    *right_key[:3],
                    tuple(
                        x
                        for x in right_key[3]
                        if x[0] not in {"stereo", "stereochemistry", "cip_code"}
                    ),
                )
            if left_key != right_key:
                cost[left, right] = np.inf
    try:
        rows, columns = linear_sum_assignment(cost)
    except ValueError:
        return False
    if np.any(cost[rows, columns] > context.tolerance_A):
        return False
    permutation = dict(zip(rows, columns))
    source_bonds = [
        _normalize_bond_record(item, len(source))
        for item in _molecule_bond_payload(source)
    ]
    target_bonds = [
        _normalize_bond_record(item, len(target))
        for item in _molecule_bond_payload(target)
    ]
    return {
        _bond_identity(
            replace(
                record,
                left=int(permutation[record.left]),
                right=int(permutation[record.right]),
                image_shift=_transform_bond_image_shift(operation, record.image_shift),
            )
        )
        for record in source_bonds
    } == {_bond_identity(record) for record in target_bonds}


def _require_symmetry_covariant_joints(
    context: ForwardConstructionContext,
    by_id: Mapping[str, MolecularAlternative],
    joints: Sequence[Mapping[str, Any]],
) -> None:
    alternatives = tuple(by_id.values())
    weights = {}
    for joint in joints:
        key = tuple(sorted(joint["assignment"].items()))
        weights[key] = weights.get(key, 0.0) + float(joint["weight"])
    for operation in context.operations:
        for assignment, weight in weights.items():
            if weight == 0:
                continue
            image: dict[str, str] = {}
            for _, alternative_id in assignment:
                mapped = _image_alternative(
                    context, by_id[alternative_id], operation, alternatives
                )
                if mapped is None:
                    raise ConstructionValidationError(
                        "joint distribution is not closed under the source symmetry group"
                    )
                image[mapped.orbit_id] = mapped.alternative_id
            image_weight = weights.get(tuple(sorted(image.items())))
            if (
                image_weight is None
                or abs(image_weight - weight) > PARTIAL_OCCUPANCY_TOLERANCE
            ):
                raise ConstructionValidationError(
                    "joint distribution is not symmetry-covariant"
                )


def _reject_infeasible_joint(
    model: StatisticalDisorderModel,
    selected: Mapping[str, str],
) -> None:
    correlations = dict(model.correlations or {})
    if correlations.get("schema") != DISORDER_CONSTRAINT_SCHEMA:
        return
    for item in correlations.get("forbidden") or []:
        assignment = {
            str(key): str(value)
            for key, value in dict(item.get("assignment") or {}).items()
        }
        if all(selected.get(key) == value for key, value in assignment.items()):
            raise ConstructionValidationError(
                "selected realization is forbidden by the disorder constraints"
            )
    if dict(model.constraint_report or {}).get("status") == "feasible":
        matched = False
        for joint in correlations.get("joints") or []:
            assignment = {
                str(key): str(value)
                for key, value in dict(joint.get("assignment") or {}).items()
            }
            if (
                assignment == dict(selected)
                and float(joint.get("weight", 0.0)) > 1.0e-8
            ):
                matched = True
        if not matched:
            raise ConstructionValidationError(
                "selected realization is not a positive-weight state of the feasible joint distribution"
            )


def _operation_preserves_atom_set(
    context: ForwardConstructionContext,
    operation: FractionalAffineOperation,
    molecules: Sequence[CrystalMolecule],
) -> bool:
    matches = np.asarray(
        [
            [
                0.0
                if _molecule_image_matches(context, source, target, operation)
                else 1.0
                for target in molecules
            ]
            for source in molecules
        ]
    )
    rows, columns = linear_sum_assignment(matches)
    return bool(np.all(matches[rows, columns] == 0))


def _realized_symmetry(
    context: ForwardConstructionContext,
    molecules: Sequence[CrystalMolecule],
) -> dict[str, Any]:
    if not molecules:
        return {
            "status": "vacant",
            "operation_indices": [],
            "operation_count": 0,
            "source_operation_count": len(context.operations),
            "matches_source_group": False,
        }
    indices = [
        index
        for index, operation in enumerate(context.operations)
        if _operation_preserves_atom_set(context, operation, molecules)
    ]
    report = {
        "status": "occupied",
        "operation_indices": indices,
        "operation_count": len(indices),
        "source_operation_count": len(context.operations),
        "matches_source_group": len(indices) == len(context.operations),
    }
    # Independent coordinate-based detection, kept separate from verification
    # of the requested whole-molecule operation set. No cell standardization
    # or coordinate modification is performed.
    import spglib

    points, types = [], []
    identities = {}
    for molecule in molecules:
        for index, fractional in enumerate(
            molecule.positions @ np.linalg.inv(context.lattice)
        ):
            identity = _atom_identity_key(molecule, index)[:3]
            identities.setdefault(identity, len(identities) + 1)
            types.append(identities[identity])
            points.append(fractional)
    dataset = spglib.get_symmetry_dataset(
        (context.lattice, points, types), symprec=context.tolerance_A
    )
    report["detection_status"] = "detected" if dataset is not None else "indeterminate"
    if dataset is not None:
        report.update(
            detected_space_group_number=int(dataset.number),
            detected_operation_count=len(dataset.rotations),
            accidental_supergroup=bool(
                len(dataset.rotations) > len(context.operations)
                and report["matches_source_group"]
            ),
        )
    return report


def _molecule_atom_payload(molecule: CrystalMolecule) -> list[dict[str, Any]]:
    symbols = molecule.get_chemical_symbols()
    positions = np.asarray(molecule.get_positions(), dtype=float)
    isotopes = molecule.arrays.get("isotope")
    charges = molecule.arrays.get("formal_charge")
    known = molecule.arrays.get("formal_charge_known")
    stereo = molecule.arrays.get("stereochemistry")
    labels = molecule.arrays.get(KEY_LABEL)
    identities = (molecule.info or {}).get("forward_atom_identities")
    rows = []
    for index, symbol in enumerate(symbols):
        charge_known = bool(known[index]) if known is not None else False
        rows.append(
            {
                "symbol": str(symbol),
                "cartesian": [float(value) for value in positions[index]],
                "isotope": int(isotopes[index]) if isotopes is not None else 0,
                "formal_charge": int(charges[index])
                if charge_known and charges is not None
                else None,
                "formal_charge_known": charge_known,
                "stereochemistry": "" if stereo is None else str(stereo[index]),
                "label": "" if labels is None else str(labels[index]),
                "atom_identity": None if identities is None else str(identities[index]),
            }
        )
    return rows


def _molecule_bond_payload(molecule: CrystalMolecule) -> list[dict[str, Any]]:
    if "forward_bonds" in molecule.info:
        return copy.deepcopy(molecule.info["forward_bonds"])
    graph = getattr(molecule, "_graph", None)
    if graph is None:
        return []
    payload = []
    for left, right, data in graph.edges(data=True):
        shift = np.asarray(data.get("image_shift", (0, 0, 0)), dtype=int).reshape(-1)
        payload.append(
            {
                "left": int(left),
                "right": int(right),
                "order": data.get("order") if data.get("order_known") else None,
                "order_known": bool(data.get("order_known")),
                "bond_type": data.get("bond_type")
                if data.get("bond_type_known")
                else None,
                "bond_type_known": bool(data.get("bond_type_known")),
                "kind": data.get("kind"),
                "image_shift": [int(value) for value in shift[:3]],
            }
        )
    return payload


def _alternative_payload(item: MolecularAlternative) -> dict[str, Any]:
    payload = {
        "orbit_id": item.orbit_id,
        "alternative_id": item.alternative_id,
        "weight": float(item.weight),
        "operation_index": int(item.operation_index),
        "center_fractional": [float(value) for value in item.center_fractional],
        "vacancy": bool(item.vacancy),
        "atoms": [] if item.molecule is None else _molecule_atom_payload(item.molecule),
        "bonds": [] if item.molecule is None else _molecule_bond_payload(item.molecule),
        "info": {},
        "arrays": {},
    }
    if item.molecule is not None:
        payload["info"] = _json_value(item.molecule.info)
        for key, array in item.molecule.arrays.items():
            if key in {"numbers", "positions"}:
                continue
            if array.dtype.kind not in "biufUS":
                raise DisorderCompanionLossError(
                    f"array {key!r} has an unsupported companion dtype"
                )
            payload["arrays"][key] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "data": _json_value(array.tolist()),
            }
    return payload


def _json_value(value):
    """Encode supported metadata without silently stringifying lost objects."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise DisorderCompanionLossError(
        "companion metadata must contain finite JSON values"
    )


def _statistical_model_payload(model: StatisticalDisorderModel) -> dict[str, Any]:
    context = model.context
    payload = {
        "schema": DISORDER_COMPANION_SCHEMA,
        "representation": "statistical_average",
        "correlation_status": model.correlation_status,
        "correlations": _json_value(model.correlations),
        "constraint_report": dict(model.constraint_report),
        "orbit_probabilities": {
            key: float(value) for key, value in model.orbit_probabilities.items()
        },
        "cell": {
            "lattice": np.asarray(context.lattice, dtype=float).tolist(),
            "space_group_number": context.space_group_number,
            "space_group_symbol": context.symmetry.space_group_symbol,
            "hall_symbol": context.symmetry.hall_symbol,
            "setting": context.setting,
            "origin_choice": context.origin_choice,
            "basis": context.basis,
            "tolerance_A": float(context.tolerance_A),
            "fractional_tolerance": float(context.fractional_tolerance),
            "chemistry_policy": context.chemistry_policy,
            "operations": [
                {
                    "rotation": np.asarray(operation.rotation, dtype=float).tolist(),
                    "translation": np.asarray(
                        operation.translation, dtype=float
                    ).tolist(),
                    "xyz": operation.xyz,
                    "index": operation.index,
                }
                for operation in context.operations
            ],
        },
        "alternatives": [_alternative_payload(item) for item in model.alternatives],
        "vacancies": [_alternative_payload(item) for item in model.vacancies],
        "average_sites": [
            {
                "symbol": item.symbol,
                "fractional": [float(value) for value in item.fractional],
                "occupancy": float(item.occupancy),
                "orbit_id": item.orbit_id,
                "template_atom_index": int(item.template_atom_index),
                "alternative_ids": list(item.alternative_ids),
                "isotope": int(item.isotope),
                "formal_charge": item.formal_charge,
                "formal_charge_known": bool(item.formal_charge_known),
                "site_identity": item.site_identity,
                "mass": item.mass,
                "chemical_annotations": [
                    list(pair) for pair in item.chemical_annotations
                ],
            }
            for item in model.average_sites
        ],
    }
    if model.metadata.get("cif_binding") is not None:
        payload["cif_binding"] = copy.deepcopy(model.metadata["cif_binding"])
    return payload


def _molecule_from_payload(payload: Mapping[str, Any]) -> CrystalMolecule | None:
    if payload.get("vacancy"):
        return None
    rows = list(payload.get("atoms") or [])
    if not rows:
        raise DisorderCompanionLossError(
            "occupied alternative is missing its complete atom records"
        )
    atoms = Atoms(
        symbols=[str(row["symbol"]) for row in rows],
        positions=np.asarray([row["cartesian"] for row in rows], dtype=float),
    )
    if not np.all(np.isfinite(atoms.positions)):
        raise DisorderCompanionLossError("companion atom coordinates must be finite")
    atoms.set_array(
        "isotope", np.asarray([int(row.get("isotope") or 0) for row in rows], dtype=int)
    )
    charges = [row.get("formal_charge") for row in rows]
    known = [bool(row.get("formal_charge_known")) for row in rows]
    atoms.set_array(
        "formal_charge",
        np.asarray(
            [0 if value is None else int(value) for value in charges], dtype=int
        ),
    )
    atoms.set_array("formal_charge_known", np.asarray(known, dtype=bool))
    atoms.set_array(
        "stereochemistry",
        np.asarray([str(row.get("stereochemistry") or "") for row in rows], dtype=str),
    )
    atoms.set_array(
        KEY_LABEL, np.asarray([str(row.get("label") or "") for row in rows], dtype=str)
    )
    # v2 preserves complete ASE arrays, including masses, provenance, identity
    # masks, and vector-valued arrays. The atom records remain human-auditable.
    if "arrays" not in payload:
        raise DisorderCompanionLossError(
            "companion alternative lacks lossless atom arrays"
        )
    for key in list(atoms.arrays):
        if key not in {"numbers", "positions"}:
            del atoms.arrays[key]
    for key, encoded in payload["arrays"].items():
        dtype = np.dtype(encoded["dtype"])
        if key in {"numbers", "positions"} or dtype.kind not in "biufUS":
            raise DisorderCompanionLossError("invalid companion array name or dtype")
        array = np.asarray(encoded["data"], dtype=dtype)
        if (
            not array.ndim
            or list(array.shape) != encoded["shape"]
            or array.shape[0] != len(atoms)
        ):
            raise DisorderCompanionLossError(
                "companion atom array has inconsistent dimensions"
            )
        atoms.set_array(str(key), array)
    _validate_template_annotations(atoms)
    molecule = CrystalMolecule(atoms, check_pbc=False)
    molecule.info = copy.deepcopy(dict(payload.get("info") or {}))
    if _molecule_atom_payload(molecule) != rows:
        raise DisorderCompanionLossError(
            "companion atom annotations disagree with its arrays and mappings"
        )
    if (
        "forward_bonds" in molecule.info
        and molecule.info["forward_bonds"] != payload["bonds"]
    ):
        raise DisorderCompanionLossError(
            "companion typed bond records disagree with molecular provenance"
        )
    molecule._graph = nx.Graph()
    molecule._graph.add_nodes_from(range(len(molecule)))
    positions = np.asarray(molecule.get_positions(), dtype=float)
    for raw_bond in payload.get("bonds") or []:
        bond = _bond_record_payload(_normalize_bond_record(raw_bond, len(atoms)))
        left, right = int(bond["left"]), int(bond["right"])
        vector = positions[right] - positions[left]
        molecule._graph.add_edge(
            left,
            right,
            vector=vector.copy(),
            distance=float(np.linalg.norm(vector)),
            image_shift=np.asarray(bond.get("image_shift") or (0, 0, 0), dtype=int),
            order=bond.get("order"),
            order_known=bool(bond.get("order_known")),
            bond_type=bond.get("bond_type"),
            bond_type_known=bool(bond.get("bond_type_known")),
            kind=bond.get("kind"),
        )
    return molecule


def _alternative_from_payload(payload: Mapping[str, Any]) -> MolecularAlternative:
    if "atoms" not in payload or "bonds" not in payload:
        raise DisorderCompanionLossError(
            "legacy disorder companion has no complete molecular records; "
            "refusing to invent coordinates, bonds, or correlations"
        )
    vacancy = bool(payload.get("vacancy"))
    return MolecularAlternative(
        str(payload["orbit_id"]),
        str(payload["alternative_id"]),
        None if vacancy else _molecule_from_payload(payload),
        float(payload["weight"]),
        int(payload["operation_index"]),
        np.asarray(payload["center_fractional"], dtype=float),
        vacancy=vacancy,
    )


def _validate_restored_marginals(
    context, alternatives, vacancies, sites, probabilities
):
    """Recompute the average from full states; never trust cached marginals."""
    by_id = {item.alternative_id: item for item in (*alternatives, *vacancies)}
    if not alternatives or len(by_id) != len(alternatives) + len(vacancies):
        raise DisorderCompanionLossError(
            "companion requires nonempty, uniquely identified alternatives"
        )
    totals, occupied = {}, {}
    expected_sites = []
    for item in (*alternatives, *vacancies):
        if item.center_fractional.shape != (3,) or not np.all(
            np.isfinite(item.center_fractional)
        ):
            raise DisorderCompanionLossError(
                "alternative center must be a finite three-vector"
            )
        totals[item.orbit_id] = totals.get(item.orbit_id, 0.0) + item.weight
        occupied.setdefault(item.orbit_id, 0.0)
        if item.vacancy:
            if item.alternative_id not in {state.alternative_id for state in vacancies}:
                raise DisorderCompanionLossError(
                    "vacancy is in the occupied alternative list"
                )
            continue
        if item.operation_index < 0 or item.operation_index >= len(context.operations):
            raise DisorderCompanionLossError(
                "alternative operation index is outside the source group"
            )
        occupied[item.orbit_id] += item.weight
        molecule = item.molecule
        identities = molecule.info.get("forward_atom_identities", ())
        if len(identities) != len(molecule):
            raise DisorderCompanionLossError(
                "alternative lacks complete atom/site mappings"
            )
        for index, fractional in enumerate(
            molecule.positions @ np.linalg.inv(context.lattice)
        ):
            key = _atom_identity_key(molecule, index)
            charge = key[2]
            _merge_average_site(
                expected_sites,
                AverageAtomSite(
                    key[0],
                    fractional,
                    item.weight,
                    item.orbit_id,
                    index,
                    (item.alternative_id,),
                    isotope=key[1],
                    formal_charge=charge[1] if charge[0] == "known" else None,
                    formal_charge_known=charge[0] == "known",
                    site_identity=str(identities[index]),
                    mass=float(molecule.arrays["masses"][index])
                    if "masses" in molecule.arrays
                    else None,
                    chemical_annotations=(
                        (
                            "molecular_identity",
                            molecule.info.get("forward_chemical_identity") or "",
                        ),
                        *key[3],
                    ),
                ),
                context.fractional_tolerance,
            )
    if any(abs(value - 1.0) > PARTIAL_OCCUPANCY_TOLERANCE for value in totals.values()):
        raise DisorderCompanionLossError(
            "companion orbit weights including vacancies must sum to 1"
        )
    if (
        not isinstance(probabilities, Mapping)
        or set(probabilities) != set(occupied)
        or any(
            abs(float(probabilities[key]) - value) > PARTIAL_OCCUPANCY_TOLERANCE
            for key, value in occupied.items()
        )
    ):
        raise DisorderCompanionLossError(
            "companion occupation probabilities disagree with its marginal weights"
        )
    if len(expected_sites) != len(sites):
        raise DisorderCompanionLossError(
            "companion marginal site inventory disagrees with complete alternatives"
        )
    remaining = list(sites)
    for expected in expected_sites:
        match = next(
            (
                index
                for index, site in enumerate(remaining)
                if site.orbit_id == expected.orbit_id
                and _chemical_site_key(site) == _chemical_site_key(expected)
                and _fractional_equal(
                    site.fractional, expected.fractional, context.fractional_tolerance
                )
            ),
            None,
        )
        if (
            match is None
            or abs(remaining[match].occupancy - expected.occupancy)
            > PARTIAL_OCCUPANCY_TOLERANCE
            or set(remaining[match].alternative_ids) != set(expected.alternative_ids)
        ):
            raise DisorderCompanionLossError(
                "companion marginal site disagrees with complete alternatives"
            )
        remaining.pop(match)


def read_statistical_disorder_companion(
    payload: Mapping[str, Any],
    *,
    cif_text: str | None = None,
) -> StatisticalDisorderModel:
    """Restore a v2 companion and validate complete states and constraints.

    If ``cif_text`` is supplied its UTF-8 SHA-256 must match the writer's
    binding. Legacy id-only and v1 companions fail explicitly because masses,
    arbitrary atom arrays, and identity mappings cannot be recovered from them.
    """
    try:
        return _read_statistical_disorder_companion(payload, cif_text=cif_text)
    except ConstructionValidationError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        raise DisorderCompanionLossError(
            f"invalid disorder companion: {error}"
        ) from error


def _read_statistical_disorder_companion(
    payload: Mapping[str, Any],
    *,
    cif_text: str | None = None,
) -> StatisticalDisorderModel:
    """Read a versioned statistical-disorder companion.

    The public wrapper turns malformed fields into typed validation errors.
    """
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != DISORDER_COMPANION_SCHEMA
    ):
        raise DisorderCompanionLossError(
            "disorder companion is not schema "
            f"{DISORDER_COMPANION_SCHEMA}; refusing to invent molecular records"
        )
    if cif_text is not None:
        binding = payload.get("cif_binding") or {}
        if (
            binding.get("sha256")
            != hashlib.sha256(cif_text.encode("utf-8")).hexdigest()
        ):
            raise DisorderCompanionLossError(
                "companion CIF hash binding is absent or does not match"
            )
    cell = dict(payload.get("cell") or {})
    if "lattice" not in cell or "operations" not in cell:
        raise DisorderCompanionLossError(
            "disorder companion is missing cell and operation context"
        )
    operations = tuple(
        FractionalAffineOperation(
            np.asarray(operation["rotation"], dtype=float),
            np.asarray(operation["translation"], dtype=float),
            xyz=operation.get("xyz"),
            index=operation.get("index"),
        )
        for operation in cell["operations"]
    )
    context = prepare_construction_context(
        np.asarray(cell["lattice"], dtype=float),
        CrystalSymmetry(
            operations,
            space_group_number=cell.get("space_group_number"),
            space_group_symbol=cell.get("space_group_symbol"),
            hall_symbol=cell.get("hall_symbol"),
        ),
        setting=cell.get("setting"),
        origin_choice=cell.get("origin_choice"),
        basis=cell.get("basis"),
        tolerance_A=float(cell.get("tolerance_A", DEFAULT_GEOMETRY_TOLERANCE_A)),
        fractional_tolerance=float(
            cell.get("fractional_tolerance", DEFAULT_FRACTIONAL_TOLERANCE)
        ),
        chemistry_policy=str(cell.get("chemistry_policy") or "unspecified"),
    )
    alternatives = tuple(
        _alternative_from_payload(item) for item in payload.get("alternatives") or []
    )
    vacancies = tuple(
        _alternative_from_payload(item) for item in payload.get("vacancies") or []
    )
    average_sites = tuple(
        AverageAtomSite(
            str(item["symbol"]),
            np.asarray(item["fractional"], dtype=float),
            float(item["occupancy"]),
            str(item["orbit_id"]),
            int(item["template_atom_index"]),
            tuple(item.get("alternative_ids") or []),
            isotope=int(item.get("isotope") or 0),
            formal_charge=item.get("formal_charge"),
            formal_charge_known=bool(item.get("formal_charge_known")),
            site_identity=str(item.get("site_identity") or ""),
            mass=item.get("mass"),
            chemical_annotations=tuple(
                tuple(pair) for pair in item.get("chemical_annotations", [])
            ),
        )
        for item in payload.get("average_sites") or []
    )
    correlation_status = str(payload.get("correlation_status") or "local-only")
    correlations = copy.deepcopy(dict(payload.get("correlations") or {}))
    report = _validate_disorder_constraints(
        context, alternatives, vacancies, correlations, correlation_status
    )
    _validate_restored_marginals(
        context,
        alternatives,
        vacancies,
        average_sites,
        payload.get("orbit_probabilities"),
    )
    return StatisticalDisorderModel(
        context=context,
        alternatives=alternatives,
        average_sites=average_sites,
        orbit_probabilities={
            str(key): float(value)
            for key, value in dict(payload.get("orbit_probabilities") or {}).items()
        },
        correlation_status=correlation_status,
        correlations=correlations,
        metadata={
            "representation": "statistical_average",
            "forward_context": context,
            "source_symmetry": context.to_metadata(),
            "restored_from_companion": True,
            **(
                {"cif_binding": copy.deepcopy(payload["cif_binding"])}
                if "cif_binding" in payload
                else {}
            ),
        },
        vacancies=vacancies,
        constraint_report=report,
    )


__all__ = [
    "AverageAtomSite",
    "ConstructionValidationError",
    "CrystalConstructionContext",
    "DEFAULT_FRACTIONAL_TOLERANCE",
    "DEFAULT_GEOMETRY_TOLERANCE_A",
    "DISORDER_COMPANION_SCHEMA",
    "DISORDER_CONSTRAINT_SCHEMA",
    "DisorderCompanionLossError",
    "ForwardConstructionContext",
    "IncompatibleChemistryError",
    "IncompatibleSiteError",
    "InvalidOrientationError",
    "InvalidSymmetryError",
    "MolecularAlternative",
    "MolecularPose",
    "MolecularTemplate",
    "PeriodicBondRecord",
    "SiteSymmetryAnalysis",
    "StatisticalDisorderModel",
    "analyze_molecular_pose",
    "analyze_site_symmetry",
    "build_molecular_crystal",
    "build_orientational_disorder",
    "build_symmetry_complete_crystal",
    "construct_molecular_crystal",
    "construct_statistical_disorder",
    "intermolecular_contacts",
    "prepare_construction_context",
    "prepare_context_from_cif",
    "read_statistical_disorder_companion",
    "select_ordered_realization",
]
