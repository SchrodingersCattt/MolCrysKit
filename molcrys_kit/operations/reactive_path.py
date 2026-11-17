"""Atom-mapped reactive initial paths with independently rigid groups.

The routines in this module generate geometric initial guesses.  They do not
perform energy minimization or a nudged elastic band calculation.
``build_proton_transfer_path`` and ``build_bond_cleavage_path`` construct
multi-species endpoints, then call :func:`interpolate_reactive_path`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from ..analysis.interactions.bonding import get_bonding_threshold
from ..constants import get_atomic_radius, has_atomic_radius, is_metal_element
from ..constants.config import (
    BOND_LENGTHS,
    BOND_ROTATION_AXIS_TOLERANCE,
    BONDING_CONFIG,
)
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import frac_to_cart, kabsch_align, rotation_to_axis_angle
from ._path_core import (
    InterpolationMethod,
    coerce_interpolation_method,
    interpolate_rigid_positions,
    materialize_atoms_frame,
    minimum_image_displacement,
    path_lambda_values,
)
from .bond_rotation import RingBondRotationError, partition_at_bond

REACTIVE_PATH_SCHEMA_V3 = "molcrys_kit.reactive_path.v3"
_ELECTRON_PAIRINGS = frozenset({"closed_shell", "diradical"})


ImageShift = tuple[int, int, int]


@dataclass(frozen=True)
class RigidGroup:
    """A set of reactant-global atom identities that moves as one rigid body.

    Rigid fits use equal-weight geometric centroids, not mass-weighted centers.
    """

    atom_indices: tuple[int, ...]
    name: str | None = None


@dataclass(frozen=True)
class BondChange:
    """An expected endpoint connectivity change in reactant-global identity."""

    atom_i: int
    atom_j: int
    reactant_bonded: bool
    product_bonded: bool


@dataclass(frozen=True)
class ReactivePathConfig:
    """Configuration for a fixed-cell reactive initial path."""

    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW
    n_images: int = 11
    include_endpoints: bool = True
    rigid_fit_tolerance_A: float = 1.0e-5
    endpoint_tolerance_A: float = 1.0e-8
    bond_scale: float = 1.0
    validate_bond_changes: bool = True


@dataclass
class ReactivePathResult:
    """Flat atom-ordered images and correspondence used to build them.

    Reactive images are ASE ``Atoms`` rather than ``MolecularCrystal`` because
    bond changes can merge or split molecular components. The input molecule
    partition is retained only as the ``reactant_molecule_index`` provenance
    array on each image.
    """

    images: list[Atoms]
    product_index_by_reactant: tuple[int, ...]
    product_image_shifts: tuple[ImageShift, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _RigidGroupPose:
    group: RigidGroup
    center_a: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray
    angle_rad: float
    fit_rmsd_A: float
    pose_rmsd_A: float
    image_shift: ImageShift


def _validate_mapping(
    symbols_a: Sequence[str],
    symbols_b: Sequence[str],
    mapping: Sequence[int] | None,
) -> tuple[int, ...]:
    n_atoms = len(symbols_a)
    if len(symbols_b) != n_atoms:
        raise ValueError(
            "Reactive interpolation requires equal atom counts; "
            f"got {n_atoms} and {len(symbols_b)}"
        )
    if mapping is None:
        resolved = tuple(range(n_atoms))
    else:
        resolved = tuple(int(index) for index in mapping)
    if len(resolved) != n_atoms or set(resolved) != set(range(n_atoms)):
        raise ValueError(
            "product_index_by_reactant must be a permutation of all atom indices"
        )
    mismatches = [
        index
        for index, product_index in enumerate(resolved)
        if symbols_a[index] != symbols_b[product_index]
    ]
    if mismatches:
        raise ValueError(
            "Atom mapping must preserve element identity; mismatched reactant "
            f"indices: {mismatches[:8]}"
        )
    return resolved


def _validate_groups(
    groups: Sequence[RigidGroup], n_atoms: int
) -> tuple[tuple[RigidGroup, ...], set[int]]:
    resolved = tuple(groups)
    if len(resolved) < 2:
        raise ValueError("Reactive interpolation requires at least two rigid groups")
    used: set[int] = set()
    for group_index, group in enumerate(resolved):
        indices = tuple(int(index) for index in group.atom_indices)
        if not indices:
            raise ValueError(f"Rigid group {group_index} is empty")
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"Rigid group {group_index} contains duplicate atom indices"
            )
        invalid = [index for index in indices if index < 0 or index >= n_atoms]
        if invalid:
            raise ValueError(
                f"Rigid group {group_index} has out-of-range indices: {invalid}"
            )
        overlap = used.intersection(indices)
        if overlap:
            raise ValueError(f"Rigid groups overlap at atom indices: {sorted(overlap)}")
        used.update(indices)
    return resolved, used


def _fit_rigid_group(
    group: RigidGroup,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    tolerance_A: float,
    image_shift: ImageShift | None = None,
) -> tuple[_RigidGroupPose, np.ndarray]:
    indices = np.asarray(group.atom_indices, dtype=int)
    group_a = positions_a[indices]
    group_b_raw = positions_b[indices]
    center_a = np.mean(group_a, axis=0)
    center_b_raw = np.mean(group_b_raw, axis=0)
    if image_shift is None:
        translation, image_shift = minimum_image_displacement(
            center_b_raw - center_a, lattice, pbc
        )
    else:
        image_shift = tuple(int(value) for value in image_shift)
        translation = (
            center_b_raw
            - center_a
            + frac_to_cart(np.asarray(image_shift, dtype=float), lattice)
        )
    shift_cart = frac_to_cart(np.asarray(image_shift, dtype=float), lattice)
    group_b = group_b_raw + shift_cart
    center_b = center_b_raw + shift_cart

    if len(indices) == 1:
        rotation = np.eye(3)
        fit_rmsd = 0.0
    else:
        rotation, fit_rmsd = kabsch_align(group_a - center_a, group_b - center_b)
    if fit_rmsd > tolerance_A:
        label = group.name or str(tuple(group.atom_indices))
        raise ValueError(
            f"Rigid group {label!r} is not rigid-reachable: fit RMSD "
            f"{fit_rmsd:.6g} Å exceeds {tolerance_A:.6g} Å"
        )
    axis, angle = rotation_to_axis_angle(rotation)
    fitted = (group_a - center_a) @ rotation.T + center_a + translation
    pose_rmsd = float(np.sqrt(np.mean(np.sum((fitted - group_b) ** 2, axis=1))))
    pose = _RigidGroupPose(
        group=group,
        center_a=center_a,
        translation=translation,
        rotation=rotation,
        axis=axis,
        angle_rad=float(angle),
        fit_rmsd_A=float(fit_rmsd),
        pose_rmsd_A=pose_rmsd,
        image_shift=image_shift,
    )
    return pose, group_b


def _interpolate_group(
    positions_a: np.ndarray,
    pose: _RigidGroupPose,
    lam: float,
    method: InterpolationMethod,
) -> np.ndarray:
    return interpolate_rigid_positions(
        positions_a,
        center=pose.center_a,
        rotation=pose.rotation,
        translation=pose.translation,
        lam=lam,
        method=method,
    )


def _pair_is_bonded(
    symbols: Sequence[str],
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    bond_scale: float,
) -> tuple[bool, float, float]:
    symbol_i = symbols[atom_i]
    symbol_j = symbols[atom_j]
    if not has_atomic_radius(symbol_i) or not has_atomic_radius(symbol_j):
        raise ValueError(
            f"Missing atomic radius for bond validation: {symbol_i}-{symbol_j}"
        )
    displacement, _ = minimum_image_displacement(
        positions[atom_j] - positions[atom_i], lattice, pbc
    )
    distance = float(np.linalg.norm(displacement))
    threshold = float(bond_scale) * get_bonding_threshold(
        get_atomic_radius(symbol_i),
        get_atomic_radius(symbol_j),
        is_metal_element(symbol_i),
        is_metal_element(symbol_j),
    )
    return distance <= threshold, distance, threshold


def _validate_bond_changes(
    changes: Sequence[BondChange],
    symbols: Sequence[str],
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    config: ReactivePathConfig,
) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    n_atoms = len(symbols)
    for change in changes:
        atom_i, atom_j = int(change.atom_i), int(change.atom_j)
        if atom_i == atom_j:
            raise ValueError("BondChange atoms must be distinct")
        if min(atom_i, atom_j) < 0 or max(atom_i, atom_j) >= n_atoms:
            raise ValueError(
                f"BondChange has out-of-range atom indices: {(atom_i, atom_j)}"
            )
        pair = tuple(sorted((atom_i, atom_j)))
        if pair in seen:
            raise ValueError(f"Duplicate BondChange for atom pair {pair}")
        seen.add(pair)
        if bool(change.reactant_bonded) == bool(change.product_bonded):
            raise ValueError(f"BondChange {pair} must change bonded state")

        record: dict[str, Any] = {
            "atoms": list(pair),
            "reactant_bonded": bool(change.reactant_bonded),
            "product_bonded": bool(change.product_bonded),
        }
        if config.validate_bond_changes:
            actual_a, distance_a, threshold_a = _pair_is_bonded(
                symbols, positions_a, atom_i, atom_j, lattice, pbc, config.bond_scale
            )
            actual_b, distance_b, threshold_b = _pair_is_bonded(
                symbols, positions_b, atom_i, atom_j, lattice, pbc, config.bond_scale
            )
            if actual_a != bool(change.reactant_bonded) or actual_b != bool(
                change.product_bonded
            ):
                raise ValueError(
                    f"BondChange {pair} does not match endpoint geometry: "
                    f"observed {actual_a}->{actual_b}, declared "
                    f"{change.reactant_bonded}->{change.product_bonded}"
                )
            record.update(
                {
                    "reactant_distance_A": distance_a,
                    "product_distance_A": distance_b,
                    "reactant_threshold_A": threshold_a,
                    "product_threshold_A": threshold_b,
                    "validated": True,
                }
            )
        else:
            record["validated"] = False
        records.append(record)
    return records


def _coerce_group_image_shifts(
    shifts: Sequence[ImageShift | None] | None,
    n_groups: int,
) -> tuple[ImageShift | None, ...]:
    if shifts is None:
        return (None,) * n_groups
    if len(shifts) != n_groups:
        raise ValueError(
            "rigid_group_image_shifts must contain one shift per rigid group"
        )
    resolved: list[ImageShift | None] = []
    for shift in shifts:
        if shift is None:
            resolved.append(None)
            continue
        if len(shift) != 3:
            raise ValueError("rigid group image shift must contain three integers")
        resolved.append(
            (
                _require_int(shift[0], "image shift"),
                _require_int(shift[1], "image shift"),
                _require_int(shift[2], "image shift"),
            )
        )
    return tuple(resolved)


def interpolate_reactive_path(
    reactant: MolecularCrystal,
    product_crystal: MolecularCrystal,
    *,
    rigid_groups: Sequence[RigidGroup],
    product_index_by_reactant: Sequence[int] | None = None,
    bond_changes: Sequence[BondChange] = (),
    config: ReactivePathConfig | None = None,
    rigid_group_image_shifts: Sequence[ImageShift | None] | None = None,
) -> ReactivePathResult:
    """Generate a fixed-cell atom-mapped reactive initial path.

    Rigid groups move independently but share the same interpolation parameter.
    Atoms outside all rigid groups interpolate linearly between endpoint images.
    ``rigid_group_image_shifts`` selects the lattice image of each product
    group. The default, ``None``, uses the minimum image of the group centre.
    An explicit shift keeps that image of the supplied coordinates, so a
    requested displacement longer than half a cell edge is not replaced by
    the shorter periodic return.
    """
    config = config or ReactivePathConfig()
    method = coerce_interpolation_method(config.method)
    lambdas = path_lambda_values(config.n_images, config.include_endpoints)
    if config.rigid_fit_tolerance_A < 0 or config.endpoint_tolerance_A < 0:
        raise ValueError("Path tolerances must be non-negative")
    if config.bond_scale <= 0:
        raise ValueError("bond_scale must be positive")

    atoms_a = reactant.to_ase()
    atoms_b = product_crystal.to_ase()
    lattice_a = np.asarray(reactant.lattice, dtype=float)
    lattice_b = np.asarray(product_crystal.lattice, dtype=float)
    if not np.allclose(lattice_a, lattice_b, atol=1.0e-6, rtol=0.0):
        raise ValueError("Reactive interpolation currently requires identical lattices")
    if not np.array_equal(
        np.asarray(reactant.pbc, dtype=bool),
        np.asarray(product_crystal.pbc, dtype=bool),
    ):
        raise ValueError("Reactive interpolation requires identical PBC flags")

    symbols_a = atoms_a.get_chemical_symbols()
    symbols_b = atoms_b.get_chemical_symbols()
    mapping = _validate_mapping(symbols_a, symbols_b, product_index_by_reactant)
    groups, rigid_indices = _validate_groups(rigid_groups, len(atoms_a))
    group_shifts = _coerce_group_image_shifts(rigid_group_image_shifts, len(groups))
    positions_a = np.asarray(atoms_a.positions, dtype=float)
    positions_b = np.asarray(atoms_b.positions, dtype=float)[
        np.asarray(mapping, dtype=int)
    ]

    target_positions = positions_b.copy()
    shifts = np.zeros((len(atoms_a), 3), dtype=int)
    poses: list[_RigidGroupPose] = []
    for group, group_shift in zip(groups, group_shifts):
        pose, group_target = _fit_rigid_group(
            group,
            positions_a,
            positions_b,
            lattice_a,
            reactant.pbc,
            config.rigid_fit_tolerance_A,
            image_shift=group_shift,
        )
        indices = np.asarray(group.atom_indices, dtype=int)
        target_positions[indices] = group_target
        shifts[indices] = np.asarray(pose.image_shift, dtype=int)
        poses.append(pose)

    free_indices = sorted(set(range(len(atoms_a))) - rigid_indices)
    for atom_index in free_indices:
        _, image_shift = minimum_image_displacement(
            positions_b[atom_index] - positions_a[atom_index],
            lattice_a,
            reactant.pbc,
        )
        shifts[atom_index] = np.asarray(image_shift, dtype=int)
        target_positions[atom_index] = positions_b[atom_index] + frac_to_cart(
            shifts[atom_index], lattice_a
        )

    bond_records = _validate_bond_changes(
        bond_changes,
        symbols_a,
        positions_a,
        target_positions,
        lattice_a,
        reactant.pbc,
        config,
    )

    reference_atoms = atoms_a.copy()
    reactant_partition = reference_atoms.arrays.get("molecule_index")
    if reactant_partition is not None:
        reference_atoms.set_array(
            "reactant_molecule_index", np.asarray(reactant_partition, dtype=int).copy()
        )
        del reference_atoms.arrays["molecule_index"]

    images: list[Atoms] = []
    for image_index, lam_value in enumerate(lambdas):
        lam = float(lam_value)
        if lam == 0.0:
            positions = positions_a.copy()
        elif lam == 1.0:
            positions = target_positions.copy()
        else:
            positions = positions_a.copy()
            for pose in poses:
                indices = np.asarray(pose.group.atom_indices, dtype=int)
                positions[indices] = _interpolate_group(
                    positions_a[indices], pose, lam, method
                )
            if free_indices:
                indices = np.asarray(free_indices, dtype=int)
                positions[indices] = (1.0 - lam) * positions_a[
                    indices
                ] + lam * target_positions[indices]

        images.append(
            materialize_atoms_frame(
                reference_atoms,
                positions,
                info_updates={
                    "path_kind": "reactive",
                    "path_image_index": int(image_index),
                    "path_lambda": lam,
                    "path_method": method.value,
                },
            )
        )

    endpoint_error = 0.0
    if len(lambdas) and float(lambdas[-1]) == 1.0:
        endpoint_error = float(
            np.max(np.linalg.norm(images[-1].positions - target_positions, axis=1))
        )
        if endpoint_error > config.endpoint_tolerance_A:
            raise RuntimeError(
                f"Reactive path endpoint error {endpoint_error:.6g} Å exceeds "
                f"{config.endpoint_tolerance_A:.6g} Å"
            )

    for frame in images:
        if frame.get_chemical_symbols() != symbols_a:
            raise RuntimeError("Reactive path changed global atom order")
        if "molecule_index" in frame.arrays:
            raise RuntimeError("Reactive path exposed a stale molecular partition")
        if reactant_partition is not None and not np.array_equal(
            frame.arrays.get("reactant_molecule_index"), reactant_partition
        ):
            raise RuntimeError("Reactive path changed reactant partition provenance")
        if not np.all(np.isfinite(frame.positions)):
            raise RuntimeError("Reactive path contains non-finite coordinates")

    group_metadata = []
    for pose in poses:
        group_metadata.append(
            {
                "name": pose.group.name,
                "atom_indices": list(pose.group.atom_indices),
                "product_image_shift": list(pose.image_shift),
                "fit_rmsd_A": pose.fit_rmsd_A,
                "pose_rmsd_A": pose.pose_rmsd_A,
                "angle_deg": float(np.degrees(pose.angle_rad)),
                "displacement_A": float(np.linalg.norm(pose.translation)),
            }
        )
    metadata: dict[str, Any] = {
        "schema": "molcrys_kit.reactive_path.v2",
        "method": method.value,
        "n_images": int(config.n_images),
        "include_endpoints": bool(config.include_endpoints),
        "product_index_by_reactant": list(mapping),
        "product_image_shifts": shifts.tolist(),
        "rigid_groups": group_metadata,
        "free_atom_indices": free_indices,
        "bond_changes": bond_records,
        "bond_validation_enabled": bool(config.validate_bond_changes),
        "endpoint_max_error_A": endpoint_error,
    }
    return ReactivePathResult(
        images=images,
        product_index_by_reactant=mapping,
        product_image_shifts=tuple(
            tuple(int(value) for value in row) for row in shifts.tolist()
        ),
        metadata=metadata,
    )


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _molecule_indices_of(values: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _charge_map(
    charges: Sequence[tuple[int, int]],
    expected: set[int],
    cell_charge: int,
    label: str,
) -> dict[int, int]:
    if isinstance(charges, (str, bytes)) or not isinstance(charges, Sequence):
        raise ValueError(
            f"{label} species charges must be (molecule_index, formal_charge) pairs"
        )
    mapping: dict[int, int] = {}
    for item in charges:
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise ValueError(
                f"{label} species charges must be (molecule_index, formal_charge) pairs"
            )
        index = _require_int(item[0], "molecule_index")
        formal_charge = _require_int(item[1], "formal_charge")
        if index in mapping:
            raise ValueError(f"duplicate charge for molecule {index}")
        mapping[index] = formal_charge
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(
            f"{label} species charges do not match molecules: "
            f"missing {missing}, extra {extra}"
        )
    total = sum(mapping.values())
    if total != cell_charge:
        raise ValueError(
            f"{label} species charges sum to {total}, cell charge is {cell_charge}"
        )
    return mapping


def _require_spectator_charges(
    reactant_charges: dict[int, int],
    product_charges: dict[int, int],
    spectators: Sequence[int],
) -> None:
    for index in spectators:
        if reactant_charges[index] != product_charges[index]:
            raise ValueError(
                f"spectator molecule {index} charge changed from "
                f"{reactant_charges[index]} to {product_charges[index]}"
            )


def _require_index(index: int, size: int, name: str) -> int:
    resolved = _require_int(index, name)
    if resolved < 0 or resolved >= size:
        raise ValueError(f"{name} {resolved} is out of range")
    return resolved


def _flat_molecules(crystal: MolecularCrystal) -> tuple[Atoms, np.ndarray]:
    atoms = crystal.to_ase()
    molecule_index = atoms.arrays.get("molecule_index")
    if molecule_index is None:
        raise ValueError("crystal is missing molecule_index")
    return atoms, np.asarray(molecule_index, dtype=int)


def _require_molecule(molecule_index: np.ndarray, index: int, name: str) -> int:
    resolved = _require_int(index, name)
    if resolved not in set(molecule_index.tolist()):
        raise ValueError(f"{name} molecule {resolved} is not in the crystal")
    return resolved


def _bond_search_cutoff(symbols: Sequence[str], bond_scale: float) -> float:
    """Return a neighbor-list cutoff that covers every present bonding threshold.

    ASE's neighbor list keeps pairs strictly inside the cutoff, while bond
    validation uses ``distance <= threshold``. The cutoff is the next
    floating-point value above the largest scaled threshold so an exact
    threshold pair is still visited.
    """
    unique = tuple(sorted(set(symbols)))
    max_threshold = 0.0
    for left_index, left in enumerate(unique):
        if not has_atomic_radius(left):
            raise ValueError(f"Missing atomic radius for bond validation: {left}")
        for right in unique[left_index:]:
            if not has_atomic_radius(right):
                raise ValueError(f"Missing atomic radius for bond validation: {right}")
            threshold = get_bonding_threshold(
                get_atomic_radius(left),
                get_atomic_radius(right),
                is_metal_element(left),
                is_metal_element(right),
            )
            max_threshold = max(max_threshold, float(threshold))
    limit = float(bond_scale) * max_threshold
    return float(np.nextafter(limit, np.inf))


def _moved_atom_indices(positions_a: np.ndarray, positions_b: np.ndarray) -> list[int]:
    delta = np.asarray(positions_b, dtype=float) - np.asarray(positions_a, dtype=float)
    return [
        index
        for index, displacement in enumerate(delta)
        if float(np.linalg.norm(displacement)) > BOND_ROTATION_AXIS_TOLERANCE
    ]


def _candidate_bond_pairs(
    symbols: Sequence[str],
    positions: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    cutoff: float,
    moved: set[int],
) -> set[tuple[int, int]]:
    atoms = Atoms(
        symbols=list(symbols),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=pbc,
    )
    first, second = neighbor_list("ij", atoms, cutoff=cutoff)
    pairs: set[tuple[int, int]] = set()
    for atom_i, atom_j in zip(first, second):
        left, right = int(atom_i), int(atom_j)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        if left not in moved and right not in moved:
            continue
        pairs.add((left, right))
    return pairs


def _connectivity_changes(
    symbols: Sequence[str],
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    bond_scale: float,
) -> set[tuple[int, int]]:
    """Return pairs whose bonded state changes, searching only around moving atoms.

    Atoms that share a displacement cannot change their mutual distance, so
    spectator pairs and pairs inside one rigid fragment are not visited.
    """
    moved = _moved_atom_indices(positions_a, positions_b)
    if not moved:
        return set()
    moved_set = set(moved)
    cutoff = _bond_search_cutoff(symbols, bond_scale)
    candidates: set[tuple[int, int]] = set()
    for positions in (positions_a, positions_b):
        candidates.update(
            _candidate_bond_pairs(symbols, positions, lattice, pbc, cutoff, moved_set)
        )
    delta = np.asarray(positions_b, dtype=float) - np.asarray(positions_a, dtype=float)
    changed: set[tuple[int, int]] = set()
    for atom_i, atom_j in candidates:
        relative = delta[atom_j] - delta[atom_i]
        if float(np.linalg.norm(relative)) <= BOND_ROTATION_AXIS_TOLERANCE:
            continue
        bonded_a, _, _ = _pair_is_bonded(
            symbols, positions_a, atom_i, atom_j, lattice, pbc, bond_scale
        )
        bonded_b, _, _ = _pair_is_bonded(
            symbols, positions_b, atom_i, atom_j, lattice, pbc, bond_scale
        )
        if bonded_a != bonded_b:
            changed.add((atom_i, atom_j))
    return changed


def _require_exact_changes(
    actual: set[tuple[int, int]],
    expected: set[tuple[int, int]],
) -> None:
    if actual != expected:
        raise ValueError(
            "endpoint connectivity changes "
            f"{sorted(actual)} differ from the declared reaction {sorted(expected)}"
        )


def _species_records(
    symbols: Sequence[str],
    molecule_index: np.ndarray,
    charges: dict[int, int],
    roles: dict[int, str],
) -> list[dict[str, Any]]:
    records = []
    for mol_index in sorted(charges):
        atom_indices = [
            int(index)
            for index, owner in enumerate(molecule_index)
            if int(owner) == mol_index
        ]
        counts: dict[str, int] = {}
        for index in atom_indices:
            symbol = symbols[index]
            counts[symbol] = counts.get(symbol, 0) + 1
        records.append(
            {
                "molecule_index": int(mol_index),
                "atom_indices": atom_indices,
                "counts": counts,
                "formal_charge": int(charges[mol_index]),
                "role": roles.get(mol_index, "spectator"),
            }
        )
    return records


def _rebuild_crystal(
    crystal: MolecularCrystal,
    positions: np.ndarray,
    molecule_index: np.ndarray,
) -> MolecularCrystal:
    atoms = crystal.to_ase()
    rebuilt = atoms.copy()
    rebuilt.calc = None
    rebuilt.positions[:] = np.asarray(positions, dtype=float)
    rebuilt.set_array("molecule_index", np.asarray(molecule_index, dtype=int))
    product = MolecularCrystal.from_ase_atoms(rebuilt)
    restored = product.to_ase()
    if restored.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise RuntimeError("rebuilt crystal changed atom order")
    if not np.allclose(restored.positions, positions, atol=1.0e-8, rtol=0.0):
        raise RuntimeError("rebuilt crystal changed Cartesian coordinates")
    return product


def _annotate_result(
    result: ReactivePathResult,
    *,
    cell_charge: int,
    multiplicity: int,
    electron_pairing: str | None,
    reactant_species: list[dict[str, Any]],
    product_species: list[dict[str, Any]],
    product_molecule_index: np.ndarray,
) -> ReactivePathResult:
    result.metadata["schema"] = REACTIVE_PATH_SCHEMA_V3
    result.metadata["cell_charge"] = int(cell_charge)
    result.metadata["multiplicity"] = int(multiplicity)
    result.metadata["electron_pairing"] = electron_pairing
    result.metadata["reactant_species"] = reactant_species
    result.metadata["product_species"] = product_species
    result.metadata["product_molecule_index"] = [
        int(value) for value in product_molecule_index
    ]
    for frame in result.images:
        frame.info["cell_charge"] = int(cell_charge)
        frame.info["multiplicity"] = int(multiplicity)
        frame.info["electron_pairing"] = electron_pairing
        frame.info["path_schema"] = REACTIVE_PATH_SCHEMA_V3
    return result


def _positive_distance(value: float, name: str) -> float:
    distance = float(value)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"{name} must be positive")
    return distance


def _product_bond_length(acceptor_symbol: str, requested: float | None) -> float:
    if requested is not None:
        return _positive_distance(requested, "product_distance_A")
    for key in (f"{acceptor_symbol}-H", f"H-{acceptor_symbol}"):
        if key in BOND_LENGTHS:
            return float(BOND_LENGTHS[key])
    raise ValueError(f"no bond length for {acceptor_symbol}-H; pass product_distance_A")


def _bonded_neighbors(
    symbols: Sequence[str],
    positions: np.ndarray,
    atom_index: int,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    bond_scale: float,
) -> list[int]:
    neighbors = []
    for other in range(len(symbols)):
        if other == atom_index:
            continue
        bonded, _, _ = _pair_is_bonded(
            symbols, positions, atom_index, other, lattice, pbc, bond_scale
        )
        if bonded:
            neighbors.append(other)
    return neighbors


def _rigid_groups_by_molecule(
    molecule_index: np.ndarray,
    roles: dict[int, str],
    skip: set[int],
) -> list[RigidGroup]:
    groups: list[RigidGroup] = []
    for mol_index in _molecule_indices_of(molecule_index):
        indices = tuple(
            int(index)
            for index, owner in enumerate(molecule_index)
            if int(owner) == mol_index and int(index) not in skip
        )
        if not indices:
            continue
        name = roles.get(mol_index, f"molecule-{mol_index}")
        if name == "spectator":
            name = f"spectator-{mol_index}"
        groups.append(RigidGroup(indices, name=name))
    if len(groups) < 2:
        raise ValueError("Reactive interpolation requires at least two rigid groups")
    return groups


def _run_declared_path(
    reactant: MolecularCrystal,
    product: MolecularCrystal,
    groups: Sequence[RigidGroup],
    bond_changes: Sequence[BondChange],
    config: ReactivePathConfig,
    *,
    rigid_group_image_shifts: Sequence[ImageShift | None] | None = None,
    cell_charge: int,
    multiplicity: int,
    electron_pairing: str | None,
    reactant_species: list[dict[str, Any]],
    product_species: list[dict[str, Any]],
    product_molecule_index: np.ndarray,
) -> ReactivePathResult:
    result = interpolate_reactive_path(
        reactant,
        product,
        rigid_groups=groups,
        bond_changes=bond_changes,
        config=config,
        rigid_group_image_shifts=rigid_group_image_shifts,
    )
    return _annotate_result(
        result,
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        electron_pairing=electron_pairing,
        reactant_species=reactant_species,
        product_species=product_species,
        product_molecule_index=product_molecule_index,
    )


def build_proton_transfer_path(
    crystal: MolecularCrystal,
    *,
    donor_molecule: int,
    hydrogen_index: int,
    acceptor_molecule: int,
    acceptor_index: int,
    cell_charge: int,
    multiplicity: int,
    reactant_species_charges: Sequence[tuple[int, int]],
    product_species_charges: Sequence[tuple[int, int]],
    product_distance_A: float | None = None,
    config: ReactivePathConfig | None = None,
) -> ReactivePathResult:
    """Build a fixed-cell path that moves one hydrogen onto another species.

    ``hydrogen_index`` and ``acceptor_index`` are global indices in
    ``crystal.to_ase()``. Spectator molecules stay fixed. The transferring
    hydrogen is the only free atom. Charges are declared bookkeeping: both
    species-charge maps must sum to ``cell_charge``, and spectator charges
    must be unchanged. The geometric frames are an initial guess for an
    external NEB calculation.
    """
    config = config or ReactivePathConfig()
    cell_charge = _require_int(cell_charge, "cell_charge")
    multiplicity = _require_int(multiplicity, "multiplicity")
    if multiplicity < 1:
        raise ValueError("multiplicity must be a positive integer")
    donor_molecule = _require_int(donor_molecule, "donor_molecule")
    acceptor_molecule = _require_int(acceptor_molecule, "acceptor_molecule")
    if donor_molecule == acceptor_molecule:
        raise ValueError("donor and acceptor must be different molecules")

    atoms, molecule_index = _flat_molecules(crystal)
    symbols = atoms.get_chemical_symbols()
    hydrogen_index = _require_index(hydrogen_index, len(atoms), "hydrogen_index")
    acceptor_index = _require_index(acceptor_index, len(atoms), "acceptor_index")
    _require_molecule(molecule_index, donor_molecule, "donor")
    _require_molecule(molecule_index, acceptor_molecule, "acceptor")
    if symbols[hydrogen_index] != "H":
        raise ValueError(f"atom {hydrogen_index} is not hydrogen")
    if int(molecule_index[hydrogen_index]) != donor_molecule:
        raise ValueError(
            f"hydrogen atom {hydrogen_index} is not on donor molecule {donor_molecule}"
        )
    if int(molecule_index[acceptor_index]) != acceptor_molecule:
        raise ValueError(
            f"atom {acceptor_index} is not on acceptor molecule {acceptor_molecule}"
        )

    present = set(_molecule_indices_of(molecule_index))
    reactant_charges = _charge_map(
        reactant_species_charges, present, cell_charge, "reactant"
    )
    product_charges = _charge_map(
        product_species_charges, present, cell_charge, "product"
    )
    spectators = sorted(present - {donor_molecule, acceptor_molecule})
    _require_spectator_charges(reactant_charges, product_charges, spectators)

    positions = np.asarray(atoms.positions, dtype=float).copy()
    lattice = np.asarray(crystal.lattice, dtype=float)
    neighbors = _bonded_neighbors(
        symbols, positions, hydrogen_index, lattice, crystal.pbc, config.bond_scale
    )
    if len(neighbors) != 1 or int(molecule_index[neighbors[0]]) != donor_molecule:
        raise ValueError(
            f"hydrogen atom {hydrogen_index} must have exactly one bonded "
            f"neighbor on donor molecule {donor_molecule}"
        )
    partner = neighbors[0]
    distance = _product_bond_length(symbols[acceptor_index], product_distance_A)
    displacement, _ = minimum_image_displacement(
        positions[acceptor_index] - positions[hydrogen_index], lattice, crystal.pbc
    )
    current = float(np.linalg.norm(displacement))
    limit = float(BONDING_CONFIG["MAX_HYDROGEN_BOND_DISTANCE"])
    if current > limit:
        raise ValueError(
            f"H...acceptor distance {current:.6g} Å exceeds the "
            f"hydrogen-bond limit {limit:.6g} Å"
        )
    if current <= BOND_ROTATION_AXIS_TOLERANCE:
        raise ValueError("hydrogen and acceptor atom coincide")
    product_positions = positions.copy()
    product_positions[hydrogen_index] = positions[hydrogen_index] + displacement * (
        1.0 - distance / current
    )
    product_molecule_index = molecule_index.copy()
    product_molecule_index[hydrogen_index] = acceptor_molecule
    expected = {
        tuple(sorted((partner, hydrogen_index))),
        tuple(sorted((hydrogen_index, acceptor_index))),
    }
    _require_exact_changes(
        _connectivity_changes(
            symbols,
            positions,
            product_positions,
            lattice,
            crystal.pbc,
            config.bond_scale,
        ),
        expected,
    )
    roles = {index: "spectator" for index in present}
    roles[donor_molecule] = "donor"
    roles[acceptor_molecule] = "acceptor"
    product = _rebuild_crystal(crystal, product_positions, product_molecule_index)
    return _run_declared_path(
        crystal,
        product,
        _rigid_groups_by_molecule(molecule_index, roles, {hydrogen_index}),
        [
            BondChange(partner, hydrogen_index, True, False),
            BondChange(hydrogen_index, acceptor_index, False, True),
        ],
        config,
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        electron_pairing=None,
        reactant_species=_species_records(
            symbols, molecule_index, reactant_charges, roles
        ),
        product_species=_species_records(
            symbols, product_molecule_index, product_charges, roles
        ),
        product_molecule_index=product_molecule_index,
    )


def build_bond_cleavage_path(
    crystal: MolecularCrystal,
    *,
    molecule_index: int,
    atom_i: int,
    atom_j: int,
    separation_A: float,
    cell_charge: int,
    multiplicity: int,
    reactant_species_charges: Sequence[tuple[int, int]],
    product_species_charges: Sequence[tuple[int, int]],
    electron_pairing: str | None = None,
    config: ReactivePathConfig | None = None,
) -> ReactivePathResult:
    """Build a fixed-cell path that separates one bridge-bond fragment.

    ``atom_i`` and ``atom_j`` are molecule-local indices. The fragment
    containing ``atom_j`` translates by ``separation_A`` along the bond.
    A ring bond raises :class:`RingBondRotationError`. ``electron_pairing``
    is the caller-declared label ``"closed_shell"`` or ``"diradical"`` and
    is not inferred. Spectator charges stay unchanged.
    """
    if electron_pairing not in _ELECTRON_PAIRINGS:
        raise ValueError("electron_pairing must be 'closed_shell' or 'diradical'")
    config = config or ReactivePathConfig()
    cell_charge = _require_int(cell_charge, "cell_charge")
    multiplicity = _require_int(multiplicity, "multiplicity")
    if multiplicity < 1:
        raise ValueError("multiplicity must be a positive integer")
    separation_A = _positive_distance(separation_A, "separation_A")

    atoms, owners = _flat_molecules(crystal)
    symbols = atoms.get_chemical_symbols()
    molecule_index = _require_molecule(owners, molecule_index, "cleaved")
    if molecule_index < 0 or molecule_index >= len(crystal.molecules):
        raise ValueError(f"cleaved molecule {molecule_index} is not in the crystal")
    molecule = crystal.molecules[molecule_index]
    atom_i = _require_index(atom_i, len(molecule), "atom_i")
    atom_j = _require_index(atom_j, len(molecule), "atom_j")
    partition = partition_at_bond(molecule, atom_i, atom_j)
    if partition.is_ring_bond:
        cycle_text = (
            f"; cycle atoms={list(partition.cycle_atoms)}"
            if partition.cycle_atoms
            else ""
        )
        raise RingBondRotationError(
            f"Bond {atom_i}-{atom_j} is a ring bond{cycle_text}. "
            "Cleavage of a ring bond is not a rigid fragment separation."
        )

    stored = molecule.info.get("atom_indices")
    if stored is not None and len(stored) == len(molecule):
        local_to_global = tuple(int(index) for index in stored)
    else:
        offset = sum(len(item) for item in crystal.molecules[:molecule_index])
        local_to_global = tuple(range(offset, offset + len(molecule)))
    global_i = local_to_global[atom_i]
    global_j = local_to_global[atom_j]
    fixed = tuple(local_to_global[index] for index in partition.fixed_atoms)
    moving = tuple(local_to_global[index] for index in partition.moving_atoms)

    positions = np.asarray(atoms.positions, dtype=float).copy()
    lattice = np.asarray(crystal.lattice, dtype=float)
    axis, _ = minimum_image_displacement(
        positions[global_j] - positions[global_i], lattice, crystal.pbc
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= BOND_ROTATION_AXIS_TOLERANCE:
        raise ValueError(f"bond axis {atom_i}-{atom_j} has zero length")
    product_positions = positions.copy()
    product_positions[list(moving)] += axis / axis_norm * separation_A
    product_owners = owners.copy()
    new_index = int(owners.max()) + 1
    product_owners[list(moving)] = new_index
    expected = {tuple(sorted((global_i, global_j)))}
    _require_exact_changes(
        _connectivity_changes(
            symbols,
            positions,
            product_positions,
            lattice,
            crystal.pbc,
            config.bond_scale,
        ),
        expected,
    )

    present = set(_molecule_indices_of(owners))
    product_present = present | {new_index}
    reactant_charges = _charge_map(
        reactant_species_charges, present, cell_charge, "reactant"
    )
    product_charges = _charge_map(
        product_species_charges, product_present, cell_charge, "product"
    )
    spectators = sorted(present - {molecule_index})
    _require_spectator_charges(reactant_charges, product_charges, spectators)
    roles = {index: "spectator" for index in present}
    roles[molecule_index] = "fragment_fixed"
    roles[new_index] = "fragment_moving"
    groups = [
        RigidGroup(fixed, name="fragment_fixed"),
        RigidGroup(moving, name="fragment_moving"),
    ]
    for spectator in spectators:
        indices = tuple(int(index) for index in np.where(owners == spectator)[0])
        if indices:
            groups.append(RigidGroup(indices, name=f"spectator-{spectator}"))
    if len(groups) < 2:
        raise ValueError("Reactive interpolation requires at least two rigid groups")

    product = _rebuild_crystal(crystal, product_positions, product_owners)
    # Product coordinates already contain the requested unwrapped displacement.
    # A zero image shift keeps that direction when it is longer than half a cell.
    return _run_declared_path(
        crystal,
        product,
        groups,
        [BondChange(global_i, global_j, True, False)],
        config,
        rigid_group_image_shifts=tuple((0, 0, 0) for _ in groups),
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        electron_pairing=electron_pairing,
        reactant_species=_species_records(symbols, owners, reactant_charges, roles),
        product_species=_species_records(
            symbols, product_owners, product_charges, roles
        ),
        product_molecule_index=product_owners,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"reactive path metadata is not JSON serializable: {type(value)!r}")


def write_reactive_path(result: ReactivePathResult, path: str | Path) -> None:
    """Write reactive images to extxyz and the result metadata to JSON.

    The JSON file uses the same stem and a ``.json`` suffix. Frames stay
    flat ASE atoms. This does not run a calculator or an NEB solver.
    """
    import ase.io

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in result.images:
        image = frame.copy()
        image.info = {
            key: value for key, value in frame.info.items() if value is not None
        }
        images.append(image)
    ase.io.write(target, images, format="extxyz")
    sidecar = target.with_suffix(".json")
    sidecar.write_text(
        json.dumps(_json_ready(result.metadata), indent=2) + "\n",
        encoding="utf-8",
    )
