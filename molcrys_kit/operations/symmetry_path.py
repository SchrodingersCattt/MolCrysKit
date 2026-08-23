"""Strict, auditable rigid paths generated from crystallographic operations.

A crystallographic affine operation defines an endpoint relation, not a kinetic
trajectory.  This module accepts only molecular endpoint pairs that can be
related by proper rigid motions within explicit tolerances.  It never repairs a
failed fit with Cartesian shape interpolation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..constants import get_atomic_mass
from ..constants.symmetry_path import (
    ASSIGNMENT_ATOM_RMSD_WEIGHT,
    ASSIGNMENT_COM_DISTANCE_WEIGHT,
    ASSIGNMENT_INFEASIBLE_COST,
    CORRESPONDENCE_DISTANCE_TOLERANCE_ANGSTROM,
    MINIMUM_NONTRIVIAL_PATH_DISPLACEMENT_ANGSTROM,
    RIGID_MASS_WEIGHTED_RMSD_TOLERANCE_ANGSTROM,
    RIGID_MAX_BOND_RELATIVE_ERROR,
)
from ..structures.crystal import MolecularCrystal
from ..structures.symmetry import FractionalAffineOperation
from ..utils.geometry import (
    cart_to_frac,
    frac_to_cart,
    kabsch_align,
)
from ._path_core import (
    InterpolationMethod,
    coerce_interpolation_method,
    copy_crystal_with_molecule_positions,
    interpolate_rigid_positions,
    minimum_image_displacement,
    path_lambda_values,
)
from .interpolation import best_atom_mapping


@dataclass(frozen=True)
class RigidReachabilityTolerance:
    """Configurable strict molecular-rigidity thresholds."""

    mass_weighted_rmsd_angstrom: float = RIGID_MASS_WEIGHTED_RMSD_TOLERANCE_ANGSTROM
    max_bond_relative_error: float = RIGID_MAX_BOND_RELATIVE_ERROR

    def __post_init__(self) -> None:
        if self.mass_weighted_rmsd_angstrom < 0:
            raise ValueError("mass-weighted RMSD tolerance must be non-negative")
        if self.max_bond_relative_error < 0:
            raise ValueError("bond relative-error tolerance must be non-negative")


@dataclass(frozen=True)
class SymmetryPathConfig:
    """Configuration for strict rigid symmetry-path planning."""

    method: InterpolationMethod | str = InterpolationMethod.SLERP
    n_images: int = 11
    include_endpoints: bool = True
    tolerance: RigidReachabilityTolerance = field(
        default_factory=RigidReachabilityTolerance
    )
    max_isomorphisms: int = 4096
    correspondence_tolerance_angstrom: float = (
        CORRESPONDENCE_DISTANCE_TOLERANCE_ANGSTROM
    )
    allow_partial_occupancy: bool = False
    minimum_nontrivial_displacement_angstrom: float = (
        MINIMUM_NONTRIVIAL_PATH_DISPLACEMENT_ANGSTROM
    )

    def __post_init__(self) -> None:
        path_lambda_values(self.n_images, self.include_endpoints)
        if (
            isinstance(self.max_isomorphisms, bool)
            or int(self.max_isomorphisms) != self.max_isomorphisms
            or self.max_isomorphisms < 1
        ):
            raise ValueError("max_isomorphisms must be an integer >= 1")
        if self.correspondence_tolerance_angstrom < 0:
            raise ValueError("correspondence tolerance must be non-negative")
        if self.minimum_nontrivial_displacement_angstrom < 0:
            raise ValueError("minimum path displacement must be non-negative")
        object.__setattr__(self, "method", coerce_interpolation_method(self.method))


@dataclass(frozen=True)
class AtomCorrespondence:
    source_to_target: np.ndarray
    method: str
    proper_fit_rmsd_angstrom: float

    def __post_init__(self) -> None:
        mapping = np.asarray(self.source_to_target, dtype=int).copy()
        mapping.setflags(write=False)
        object.__setattr__(self, "source_to_target", mapping)


@dataclass(frozen=True)
class SymmetryMoleculeMatch:
    source_molecule_index: int
    target_molecule_index: int
    atom_correspondence: AtomCorrespondence
    target_image_shift_fractional: np.ndarray
    proper_rotation: np.ndarray
    com_translation_cartesian: np.ndarray
    mass_weighted_rmsd_angstrom: float
    max_bond_relative_error: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("target_image_shift_fractional", (3,)),
            ("proper_rotation", (3, 3)),
            ("com_translation_cartesian", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CrystalCorrespondence:
    molecule_matches: tuple[SymmetryMoleculeMatch, ...]


@dataclass(frozen=True)
class SymmetryPathProvenance:
    operation: FractionalAffineOperation
    config: SymmetryPathConfig
    correspondence: CrystalCorrespondence
    target_generated_from_operation: bool

    def to_dict(self) -> dict:
        return {
            "operation": {
                "rotation": self.operation.rotation.tolist(),
                "translation": self.operation.translation.tolist(),
                "xyz": self.operation.xyz,
                "index": self.operation.index,
                "source": self.operation.source,
                "determinant": self.operation.determinant,
            },
            "config": {
                "method": self.config.method.value,
                "n_images": self.config.n_images,
                "include_endpoints": self.config.include_endpoints,
                "tolerance": asdict(self.config.tolerance),
                "max_isomorphisms": self.config.max_isomorphisms,
                "correspondence_tolerance_angstrom": (
                    self.config.correspondence_tolerance_angstrom
                ),
                "allow_partial_occupancy": self.config.allow_partial_occupancy,
                "minimum_nontrivial_displacement_angstrom": (
                    self.config.minimum_nontrivial_displacement_angstrom
                ),
            },
            "target_generated_from_operation": self.target_generated_from_operation,
            "molecule_matches": [
                {
                    "source_molecule_index": match.source_molecule_index,
                    "target_molecule_index": match.target_molecule_index,
                    "source_to_target": (
                        match.atom_correspondence.source_to_target.tolist()
                    ),
                    "mapping_method": match.atom_correspondence.method,
                    "proper_fit_rmsd_angstrom": (
                        match.atom_correspondence.proper_fit_rmsd_angstrom
                    ),
                    "target_image_shift_fractional": (
                        match.target_image_shift_fractional.tolist()
                    ),
                    "proper_rotation": match.proper_rotation.tolist(),
                    "com_translation_cartesian": (
                        match.com_translation_cartesian.tolist()
                    ),
                    "mass_weighted_rmsd_angstrom": (match.mass_weighted_rmsd_angstrom),
                    "max_bond_relative_error": match.max_bond_relative_error,
                }
                for match in self.correspondence.molecule_matches
            ],
        }


@dataclass(frozen=True)
class SymmetryPathPlan:
    start: MolecularCrystal
    target: MolecularCrystal
    provenance: SymmetryPathProvenance


class RigidReachabilityError(ValueError):
    """Raised when a requested endpoint needs non-rigid molecular deformation."""


def _require_resolved_occupancy(
    crystal: MolecularCrystal, config: SymmetryPathConfig, label: str
) -> None:
    if config.allow_partial_occupancy:
        return
    for molecule_index, molecule in enumerate(crystal.molecules):
        occupancy = molecule.arrays.get("occupancy")
        if occupancy is not None and not np.allclose(occupancy, 1.0):
            raise ValueError(
                f"{label} molecule {molecule_index} has partial occupancy; "
                "resolve disorder before building a symmetry path"
            )


def _copy_crystal(
    crystal: MolecularCrystal,
    *,
    positions_by_molecule: dict[int, np.ndarray] | None = None,
    metadata_update: dict | None = None,
) -> MolecularCrystal:
    frame = copy_crystal_with_molecule_positions(crystal, positions_by_molecule or {})
    frame.metadata.update(copy.deepcopy(metadata_update or {}))
    return frame


def transform_crystal_fractional(
    crystal: MolecularCrystal,
    operation: FractionalAffineOperation,
    *,
    molecule_indices: Sequence[int] | None = None,
) -> MolecularCrystal:
    """Apply one fractional affine operation without atom-wise wrapping."""
    operation.validate_metric(crystal.lattice)
    selected = (
        set(range(len(crystal.molecules)))
        if molecule_indices is None
        else {int(index) for index in molecule_indices}
    )
    if selected - set(range(len(crystal.molecules))):
        raise IndexError("molecule index out of range")
    positions = {}
    for index in selected:
        molecule = crystal.molecules[index]
        fractional = cart_to_frac(molecule.get_positions(), crystal.lattice)
        transformed = operation.apply(fractional, wrap=False)
        source_center = cart_to_frac(molecule.get_center_of_mass(), crystal.lattice)
        transformed_center = operation.apply(source_center, wrap=False)
        transformed -= np.round(transformed_center - source_center)
        positions[index] = frac_to_cart(transformed, crystal.lattice)
    return _copy_crystal(
        crystal,
        positions_by_molecule=positions,
        metadata_update={
            "fractional_affine_operation": {
                "rotation": operation.rotation.tolist(),
                "translation": operation.translation.tolist(),
                "xyz": operation.xyz,
            }
        },
    )


def _mass_weighted_rmsd(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    symbols: Sequence[str],
    rotation: np.ndarray,
) -> float:
    weights = np.asarray([get_atomic_mass(symbol) for symbol in symbols], dtype=float)
    source_center = np.average(source_positions, axis=0, weights=weights)
    target_center = np.average(target_positions, axis=0, weights=weights)
    fitted = (source_positions - source_center) @ rotation.T + target_center
    squared = np.sum((fitted - target_positions) ** 2, axis=1)
    return float(np.sqrt(np.average(squared, weights=weights)))


def _max_bond_relative_error(
    source_molecule,
    target_positions: np.ndarray,
) -> float:
    source_positions = np.asarray(source_molecule.get_positions(), dtype=float)
    errors = []
    for left, right in source_molecule.get_graph().edges:
        source_length = np.linalg.norm(source_positions[left] - source_positions[right])
        target_length = np.linalg.norm(target_positions[left] - target_positions[right])
        if source_length > 0:
            errors.append(abs(target_length / source_length - 1.0))
    return float(max(errors, default=0.0))


def _candidate_match(source, target, *, max_isomorphisms: int):
    if (
        len(source) != len(target)
        or source.get_chemical_formula() != target.get_chemical_formula()
    ):
        return None
    mapping = best_atom_mapping(source, target, max_isomorphisms=max_isomorphisms)
    source_positions = np.asarray(source.get_positions(), dtype=float)
    target_positions = np.asarray(target.get_positions(), dtype=float)[mapping]
    source_centered = source_positions - source.get_center_of_mass()
    target_centered = target_positions - np.average(
        target_positions,
        axis=0,
        weights=[get_atomic_mass(symbol) for symbol in source.get_chemical_symbols()],
    )
    rotation, unweighted_rmsd = kabsch_align(source_centered, target_centered)
    weighted_rmsd = _mass_weighted_rmsd(
        source_positions,
        target_positions,
        source.get_chemical_symbols(),
        rotation,
    )
    bond_error = _max_bond_relative_error(source, target_positions)
    return mapping, rotation, float(unweighted_rmsd), weighted_rmsd, bond_error


def _build_correspondence(
    start: MolecularCrystal,
    target: MolecularCrystal,
    config: SymmetryPathConfig,
) -> CrystalCorrespondence:
    if len(start.molecules) != len(target.molecules):
        raise ValueError("start and target must contain the same number of molecules")
    if not np.allclose(start.lattice, target.lattice):
        raise ValueError("strict rigid symmetry paths require identical lattices")
    if not np.array_equal(
        np.asarray(start.pbc, dtype=bool), np.asarray(target.pbc, dtype=bool)
    ):
        raise ValueError("strict rigid symmetry paths require identical PBC flags")
    count = len(start.molecules)
    costs = np.full((count, count), ASSIGNMENT_INFEASIBLE_COST, dtype=float)
    candidates = {}
    for source_index, source in enumerate(start.molecules):
        source_com = np.asarray(source.get_center_of_mass(), dtype=float)
        for target_index, target_molecule in enumerate(target.molecules):
            candidate = _candidate_match(
                source, target_molecule, max_isomorphisms=config.max_isomorphisms
            )
            if candidate is None:
                continue
            target_com = np.asarray(target_molecule.get_center_of_mass(), dtype=float)
            com_translation, _ = minimum_image_displacement(
                target_com - source_com, start.lattice, start.pbc
            )
            _mapping, _rotation, unweighted_rmsd, _weighted, _bond = candidate
            costs[source_index, target_index] = (
                ASSIGNMENT_ATOM_RMSD_WEIGHT * unweighted_rmsd
                + ASSIGNMENT_COM_DISTANCE_WEIGHT * np.linalg.norm(com_translation)
            )
            candidates[source_index, target_index] = candidate
    source_indices, target_indices = linear_sum_assignment(costs)
    if np.any(costs[source_indices, target_indices] >= ASSIGNMENT_INFEASIBLE_COST):
        raise ValueError("could not find a global graph-isomorphic molecule assignment")

    matches = []
    for source_index, target_index in zip(source_indices, target_indices):
        source = start.molecules[source_index]
        target_molecule = target.molecules[target_index]
        mapping, rotation, unweighted_rmsd, weighted_rmsd, bond_error = candidates[
            source_index, target_index
        ]
        source_com = np.asarray(source.get_center_of_mass(), dtype=float)
        target_com = np.asarray(target_molecule.get_center_of_mass(), dtype=float)
        com_translation, image_shift = minimum_image_displacement(
            target_com - source_com, start.lattice, start.pbc
        )
        if (
            weighted_rmsd > config.tolerance.mass_weighted_rmsd_angstrom
            or bond_error > config.tolerance.max_bond_relative_error
        ):
            raise RigidReachabilityError(
                "molecule "
                f"{source_index}->{target_index} is not rigid-reachable: "
                f"mass-weighted RMSD={weighted_rmsd:.6f} Å "
                f"(limit {config.tolerance.mass_weighted_rmsd_angstrom:.6f} Å), "
                f"max bond relative error={bond_error:.6f} "
                f"(limit {config.tolerance.max_bond_relative_error:.6f})"
            )
        matches.append(
            SymmetryMoleculeMatch(
                source_molecule_index=int(source_index),
                target_molecule_index=int(target_index),
                atom_correspondence=AtomCorrespondence(
                    mapping, "element_graph_isomorphism", unweighted_rmsd
                ),
                target_image_shift_fractional=np.asarray(image_shift, dtype=float),
                proper_rotation=rotation,
                com_translation_cartesian=com_translation,
                mass_weighted_rmsd_angstrom=weighted_rmsd,
                max_bond_relative_error=bond_error,
            )
        )
    matches.sort(key=lambda match: match.source_molecule_index)
    return CrystalCorrespondence(tuple(matches))


def _maximum_correspondence_error(
    source: MolecularCrystal,
    target: MolecularCrystal,
    correspondence: CrystalCorrespondence,
) -> float:
    """Return maximum mapped atom distance between two equivalent endpoints."""
    maximum = 0.0
    lattice = np.asarray(target.lattice, dtype=float)
    for match in correspondence.molecule_matches:
        source_positions = np.asarray(
            source.molecules[match.source_molecule_index].get_positions(), dtype=float
        )
        target_positions = np.asarray(
            target.molecules[match.target_molecule_index].get_positions(), dtype=float
        )[match.atom_correspondence.source_to_target]
        target_positions += match.target_image_shift_fractional @ lattice
        maximum = max(
            maximum,
            float(np.linalg.norm(target_positions - source_positions, axis=1).max()),
        )
    return maximum


def build_symmetry_path_plan(
    crystal: MolecularCrystal,
    operation: FractionalAffineOperation,
    *,
    target: MolecularCrystal | None = None,
    config: SymmetryPathConfig | None = None,
) -> SymmetryPathPlan:
    """Build and validate a strict rigid path before generating any images."""
    config = config or SymmetryPathConfig()
    _require_resolved_occupancy(crystal, config, "start")
    if target is not None:
        _require_resolved_occupancy(target, config, "target")
    generated_target = transform_crystal_fractional(crystal, operation)
    generated = target is None
    resolved_target = generated_target if target is None else target
    if target is not None:
        endpoint_correspondence = _build_correspondence(
            generated_target, target, config
        )
        endpoint_error = _maximum_correspondence_error(
            generated_target, target, endpoint_correspondence
        )
        if endpoint_error > config.correspondence_tolerance_angstrom:
            raise ValueError(
                "explicit target does not match the endpoint generated by the "
                f"symmetry operation: max error {endpoint_error:.6g} Å exceeds "
                f"{config.correspondence_tolerance_angstrom:.6g} Å"
            )
    correspondence = _build_correspondence(crystal, resolved_target, config)
    maximum_displacement = 0.0
    lattice = np.asarray(resolved_target.lattice, dtype=float)
    for match in correspondence.molecule_matches:
        source_positions = np.asarray(
            crystal.molecules[match.source_molecule_index].get_positions(),
            dtype=float,
        )
        target_positions = np.asarray(
            resolved_target.molecules[match.target_molecule_index].get_positions(),
            dtype=float,
        )[match.atom_correspondence.source_to_target]
        target_positions += match.target_image_shift_fractional @ lattice
        maximum_displacement = max(
            maximum_displacement,
            float(np.linalg.norm(target_positions - source_positions, axis=1).max()),
        )
    if (
        config.minimum_nontrivial_displacement_angstrom > 0.0
        and maximum_displacement <= config.minimum_nontrivial_displacement_angstrom
    ):
        raise ValueError(
            "symmetry operation produces only a permutation of already present "
            "molecules and no nontrivial physical path"
        )
    provenance = SymmetryPathProvenance(
        operation=operation,
        config=config,
        correspondence=correspondence,
        target_generated_from_operation=generated,
    )
    return SymmetryPathPlan(crystal, resolved_target, provenance)


def _target_positions_in_source_order(
    plan: SymmetryPathPlan,
) -> dict[int, np.ndarray]:
    """Return exact target coordinates in the source atom/molecule order."""
    positions = {}
    lattice = np.asarray(plan.target.lattice, dtype=float)
    for match in plan.provenance.correspondence.molecule_matches:
        target_molecule = plan.target.molecules[match.target_molecule_index]
        mapped = np.asarray(target_molecule.get_positions(), dtype=float)[
            match.atom_correspondence.source_to_target
        ]
        positions[match.source_molecule_index] = (
            mapped + match.target_image_shift_fractional @ lattice
        )
    return positions


def interpolate_symmetry_path(
    plan: SymmetryPathPlan,
    *,
    n_images: int | None = None,
    include_endpoints: bool | None = None,
) -> list[MolecularCrystal]:
    """Generate proper rigid molecular motions with exact endpoint copies."""
    config = plan.provenance.config
    count = config.n_images if n_images is None else int(n_images)
    endpoints = (
        config.include_endpoints if include_endpoints is None else include_endpoints
    )
    if count < 1:
        raise ValueError("n_images must be at least 1")
    lambdas = path_lambda_values(count, endpoints)
    matches = plan.provenance.correspondence.molecule_matches
    frames = []
    for frame_index, lambda_value in enumerate(lambdas):
        if endpoints and frame_index == 0:
            frame = _copy_crystal(plan.start)
        elif endpoints and frame_index == len(lambdas) - 1:
            frame = _copy_crystal(
                plan.start,
                positions_by_molecule=_target_positions_in_source_order(plan),
                metadata_update={"target_metadata": dict(plan.target.metadata)},
            )
        else:
            positions = {}
            for match in matches:
                molecule = plan.start.molecules[match.source_molecule_index]
                source_positions = np.asarray(molecule.get_positions(), dtype=float)
                source_com = np.asarray(molecule.get_center_of_mass(), dtype=float)
                positions[match.source_molecule_index] = interpolate_rigid_positions(
                    source_positions,
                    center=source_com,
                    rotation=match.proper_rotation,
                    translation=match.com_translation_cartesian,
                    lam=float(lambda_value),
                    method=config.method,
                )
            frame = _copy_crystal(plan.start, positions_by_molecule=positions)
        frame.metadata["symmetry_path"] = plan.provenance.to_dict()
        frame.metadata["path_lambda"] = float(lambda_value)
        frame.metadata["path_frame_index"] = frame_index
        frames.append(frame)
    return frames


def generate_collective_symmetry_path(
    crystal: MolecularCrystal,
    operation: FractionalAffineOperation,
    *,
    target: MolecularCrystal | None = None,
    config: SymmetryPathConfig | None = None,
) -> list[MolecularCrystal]:
    """Plan, validate, and generate a strict collective rigid path."""
    return interpolate_symmetry_path(
        build_symmetry_path_plan(crystal, operation, target=target, config=config)
    )
