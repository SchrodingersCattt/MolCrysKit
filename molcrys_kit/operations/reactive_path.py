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
from ..constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..constants.config import (
    BOND_LENGTHS,
    BOND_ROTATION_AXIS_TOLERANCE,
    BONDING_CONFIG,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import (
    cart_to_frac,
    frac_to_cart,
    get_rotation_matrix,
    kabsch_align,
    rotation_to_axis_angle,
)
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
class FreeAtomArc:
    """Polar arc of one free atom about a pivot, on the shared path parameter.

    The direction from the pivot is slerped from the reactant vector to the
    product vector, and the distance changes linearly. The pivot may belong
    to a rigid group; its motion and the arc are applied in one pass.
    ``atom_index`` and ``pivot_index`` are reactant-global indices.
    """

    atom_index: int
    pivot_index: int


@dataclass(frozen=True)
class ReactivePathConfig:
    """Configuration for a fixed-cell reactive initial path."""

    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW
    n_images: int = 11
    include_endpoints: bool = True
    # Construction accuracy. Independently relaxed endpoints usually need
    # about 0.1 Å; a residual of 0.01-0.1 Å is normal and is not a geometry bug.
    rigid_fit_tolerance_A: float = 1.0e-5
    endpoint_tolerance_A: float = 1.0e-8
    # Groups closer than this are treated as unmoved, so a singular Kabsch
    # rotation cannot drift them. This is independent of endpoint_tolerance_A:
    # widening the endpoint check must not erase a real rigid motion.
    stationary_tolerance_A: float = 1.0e-8
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
    reactant_positions: np.ndarray
    reactant_atom_image_shifts: tuple[ImageShift, ...]
    atom_image_shifts: tuple[ImageShift, ...]


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


def _unwrap_along_bonds(
    positions: np.ndarray,
    symbols: Sequence[str],
    lattice: np.ndarray,
    pbc: Sequence[bool],
    bond_scale: float,
) -> tuple[np.ndarray, tuple[ImageShift, ...]]:
    """Place a group by walking its periodic bonds from the first atom.

    A minimum image taken independently against the first atom folds a chain
    that spans more than half a cell. The bond walk keeps each step on the
    bond's own image. A group that is not one connected component is refused.
    """
    count = len(positions)
    unwrapped = np.array(positions, dtype=float, copy=True)
    shifts: list[ImageShift] = [(0, 0, 0)] * count
    if count <= 1:
        return unwrapped, tuple(shifts)
    neighbours: list[list[int]] = [[] for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            bonded, _, _ = _pair_is_bonded(
                symbols, positions, left, right, lattice, pbc, bond_scale
            )
            if bonded:
                neighbours[left].append(right)
                neighbours[right].append(left)
    seen = {0}
    queue = [0]
    while queue:
        origin = queue.pop(0)
        for target in neighbours[origin]:
            if target in seen:
                continue
            mic, shift = minimum_image_displacement(
                positions[target] - unwrapped[origin], lattice, pbc
            )
            unwrapped[target] = unwrapped[origin] + mic
            shifts[target] = shift
            seen.add(target)
            queue.append(target)
    if len(seen) != count:
        raise ValueError(
            "rigid group is not one periodic bond component; "
            f"placed {len(seen)} of {count} atoms"
        )
    return unwrapped, tuple(shifts)


def _fit_rigid_group(
    group: RigidGroup,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    tolerance_A: float,
    image_shift: ImageShift | None = None,
    stationary_tolerance_A: float = 1.0e-8,
    symbols: Sequence[str] | None = None,
    bond_scale: float = 1.0,
) -> tuple[_RigidGroupPose, np.ndarray]:
    indices = np.asarray(group.atom_indices, dtype=int)
    if symbols is None:
        raise ValueError("rigid-group unwrap requires chemical symbols")
    group_symbols = [symbols[int(index)] for index in indices]
    group_a, reactant_atom_image_shifts = _unwrap_along_bonds(
        positions_a[indices], group_symbols, lattice, pbc, bond_scale
    )
    group_b, atom_image_shifts = _unwrap_along_bonds(
        positions_b[indices], group_symbols, lattice, pbc, bond_scale
    )
    center_a = np.mean(group_a, axis=0)
    center_b_raw = np.mean(group_b, axis=0)
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
    group_b = group_b + shift_cart
    center_b = center_b_raw + shift_cart
    # Identical groups skip Kabsch. A near-zero singular rotation otherwise
    # wanders by about 1e-8 Å on some BLAS builds. The threshold is the
    # endpoint tolerance, so a real displacement above that tolerance is kept.
    atom_shift = np.linalg.norm(group_b - group_a, axis=1)
    if float(np.max(atom_shift)) <= stationary_tolerance_A:
        pose = _RigidGroupPose(
            group=group,
            center_a=center_a,
            translation=np.zeros(3),
            rotation=np.eye(3),
            axis=np.array([1.0, 0.0, 0.0]),
            angle_rad=0.0,
            fit_rmsd_A=0.0,
            pose_rmsd_A=0.0,
            image_shift=image_shift,
            reactant_positions=group_a,
            reactant_atom_image_shifts=reactant_atom_image_shifts,
            atom_image_shifts=atom_image_shifts,
        )
        return pose, group_a.copy()

    if len(indices) == 1:
        rotation = np.eye(3)
        fit_rmsd = 0.0
    else:
        rotation, fit_rmsd = kabsch_align(group_a - center_a, group_b - center_b)
    if fit_rmsd > tolerance_A:
        label = group.name or str(tuple(group.atom_indices))
        raise ValueError(
            f"Rigid group {label!r} is not rigid-reachable: fit RMSD "
            f"{fit_rmsd:.6g} Å exceeds rigid_fit_tolerance_A="
            f"{tolerance_A:.6g} Å. Constructed endpoints can keep the "
            "default 1e-5 Å. Endpoints relaxed independently under a "
            "potential usually need ReactivePathConfig("
            "rigid_fit_tolerance_A=0.1); a residual of 0.01-0.1 Å is "
            "normal and is not a geometry bug."
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
        reactant_positions=group_a,
        reactant_atom_image_shifts=reactant_atom_image_shifts,
        atom_image_shifts=atom_image_shifts,
    )
    return pose, group_b


def _interpolate_group(
    positions_a: np.ndarray,
    pose: _RigidGroupPose,
    lam: float,
    method: InterpolationMethod,
) -> np.ndarray:
    if pose.angle_rad == 0.0 and not np.any(pose.translation):
        return np.array(positions_a, dtype=float, copy=True)
    return interpolate_rigid_positions(
        positions_a,
        center=pose.center_a,
        rotation=pose.rotation,
        translation=pose.translation,
        lam=lam,
        method=method,
    )


@dataclass(frozen=True)
class _PreparedArc:
    atom_index: int
    pivot_index: int
    reactant_vector: np.ndarray
    product_vector: np.ndarray


def _polar_offset(
    reactant_vector: np.ndarray, product_vector: np.ndarray, lam: float
) -> np.ndarray:
    """Slerp the direction and linearly interpolate the length."""
    start = np.asarray(reactant_vector, dtype=float)
    end = np.asarray(product_vector, dtype=float)
    start_length = float(np.linalg.norm(start))
    end_length = float(np.linalg.norm(end))
    if (
        start_length <= BOND_ROTATION_AXIS_TOLERANCE
        or end_length <= BOND_ROTATION_AXIS_TOLERANCE
    ):
        raise ValueError(
            "polar arc requires a nonzero distance from the pivot at both endpoints"
        )
    direction_start = start / start_length
    direction_end = end / end_length
    cross = np.cross(direction_start, direction_end)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(direction_start, direction_end), -1.0, 1.0))
    if sine <= BOND_ROTATION_AXIS_TOLERANCE:
        if cosine > 0.0:
            direction = direction_start
        else:
            helper = (
                np.array([1.0, 0.0, 0.0])
                if abs(float(direction_start[0])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            axis = np.cross(direction_start, helper)
            axis = axis / np.linalg.norm(axis)
            direction = (
                direction_start @ get_rotation_matrix(axis, float(lam) * np.pi).T
            )
    else:
        axis = cross / sine
        angle = float(np.arctan2(sine, cosine))
        direction = direction_start @ get_rotation_matrix(axis, float(lam) * angle).T
    length = (1.0 - float(lam)) * start_length + float(lam) * end_length
    return direction * length


def _minimum_image_endpoint_error(
    frame_positions: np.ndarray,
    target_positions: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
) -> float:
    """Largest minimum-image gap between a frame and the recorded target.

    A lattice-vector offset is the same structure. The raw Cartesian gap can
    be one cell edge when a free atom joins another molecule and is re-imaged
    with that molecule.
    """
    worst = 0.0
    for index in range(len(frame_positions)):
        mic, _ = minimum_image_displacement(
            frame_positions[index] - target_positions[index], lattice, pbc
        )
        worst = max(worst, float(np.linalg.norm(mic)))
    return worst


def _prepare_free_atom_arcs(
    arcs: Sequence[FreeAtomArc],
    free_indices: Sequence[int],
    poses: Sequence[_RigidGroupPose],
    positions_a: np.ndarray,
    target_positions: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
) -> dict[int, _PreparedArc]:
    free = set(free_indices)
    reactant_frame = np.array(positions_a, dtype=float, copy=True)
    for pose in poses:
        reactant_frame[list(pose.group.atom_indices)] = pose.reactant_positions
    prepared: dict[int, _PreparedArc] = {}
    for arc in arcs:
        atom_index = _require_int(arc.atom_index, "FreeAtomArc.atom_index")
        pivot_index = _require_int(arc.pivot_index, "FreeAtomArc.pivot_index")
        if atom_index == pivot_index:
            raise ValueError("a polar arc pivot must be a different atom")
        if atom_index not in free:
            raise ValueError(
                f"polar arc atom {atom_index} must be outside the rigid groups"
            )
        if atom_index in prepared:
            raise ValueError(f"duplicate polar arc for atom {atom_index}")
        n_atoms = len(positions_a)
        if (
            atom_index < 0
            or atom_index >= n_atoms
            or pivot_index < 0
            or pivot_index >= n_atoms
        ):
            raise ValueError("polar arc atom index is out of range")
        reactant_vector, _ = minimum_image_displacement(
            reactant_frame[atom_index] - reactant_frame[pivot_index], lattice, pbc
        )
        product_vector, _ = minimum_image_displacement(
            target_positions[atom_index] - target_positions[pivot_index], lattice, pbc
        )
        prepared[atom_index] = _PreparedArc(
            atom_index=atom_index,
            pivot_index=pivot_index,
            reactant_vector=reactant_vector,
            product_vector=product_vector,
        )
    for arc in prepared.values():
        if arc.pivot_index in prepared:
            raise ValueError("a polar arc pivot cannot itself follow an arc")
    return prepared


def _reactant_bonded_pairs(
    symbols: Sequence[str],
    positions: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    bond_scale: float,
) -> set[tuple[int, int]]:
    atoms = Atoms(
        symbols=list(symbols),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=pbc,
    )
    first, second = neighbor_list(
        "ij", atoms, cutoff=_bond_search_cutoff(symbols, bond_scale)
    )
    bonded_pairs: set[tuple[int, int]] = set()
    for atom_i, atom_j in zip(first, second):
        left, right = int(atom_i), int(atom_j)
        if left >= right:
            continue
        bonded, _, _ = _pair_is_bonded(
            symbols, positions, left, right, lattice, pbc, bond_scale
        )
        if bonded:
            bonded_pairs.add((left, right))
    return bonded_pairs


def _image_diagnostics(
    symbols: Sequence[str],
    positions_by_image: Sequence[np.ndarray],
    lambdas: np.ndarray,
    scheduled_angles: Sequence[Sequence[float]],
    lattice: np.ndarray,
    pbc: Sequence[bool],
    poses: Sequence[_RigidGroupPose],
    free_indices: Sequence[int],
    bond_changes: Sequence[BondChange],
    arcs: dict[int, _PreparedArc],
    bond_scale: float,
    reactant_positions: np.ndarray,
) -> list[dict[str, Any]]:
    """Closest approach that is not a reactant bond or a declared bond change.

    Whether a pair is bonded is fixed from the reactant geometry plus
    ``bond_changes``. An intermediate image that merely looks bonded does not
    drop the contact from the distance, and is listed in ``new_contacts``.
    """
    moving = set(free_indices)
    group_of: dict[int, int] = {}
    for group_index, pose in enumerate(poses):
        active = pose.angle_rad != 0.0 or bool(np.any(pose.translation))
        for atom_index in pose.group.atom_indices:
            group_of[int(atom_index)] = group_index
            if active:
                moving.add(int(atom_index))
    ignored = {
        tuple(sorted((int(change.atom_i), int(change.atom_j))))
        for change in bond_changes
    }
    for arc in arcs.values():
        ignored.add(tuple(sorted((arc.atom_index, arc.pivot_index))))
    reactant_bonds = _reactant_bonded_pairs(
        symbols, reactant_positions, lattice, pbc, bond_scale
    )
    diagnostics = []
    for image_index, (positions, lam, angles) in enumerate(
        zip(positions_by_image, lambdas, scheduled_angles)
    ):
        distance, new_contacts = _min_nonbonded_distance(
            symbols,
            positions,
            lattice,
            pbc,
            moving,
            group_of,
            ignored,
            reactant_bonds,
            bond_scale,
        )
        diagnostics.append(
            {
                "image_index": int(image_index),
                "lambda": float(lam),
                "min_nonbonded_distance_A": distance,
                "new_contacts": [list(pair) for pair in new_contacts],
                "rigid_group_angles_deg": [float(angle) for angle in angles],
            }
        )
    return diagnostics


def _min_nonbonded_distance(
    symbols: Sequence[str],
    positions: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    moving: set[int],
    group_of: dict[int, int],
    ignored_pairs: set[tuple[int, int]],
    reactant_bonds: set[tuple[int, int]],
    bond_scale: float,
) -> tuple[float | None, list[tuple[int, int]]]:
    if not moving:
        return None, []
    atoms = Atoms(
        symbols=list(symbols),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=pbc,
    )
    first, second = neighbor_list(
        "ij", atoms, cutoff=_bond_search_cutoff(symbols, bond_scale)
    )
    best: float | None = None
    new_contacts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for atom_i, atom_j in zip(first, second):
        left, right = int(atom_i), int(atom_j)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        pair = (left, right)
        if pair in seen or pair in ignored_pairs or pair in reactant_bonds:
            continue
        seen.add(pair)
        if left not in moving and right not in moving:
            continue
        if group_of.get(left) is not None and group_of.get(left) == group_of.get(right):
            continue
        # Neighbor lists can yield a longer periodic image before the closest
        # one. The reported distance is the minimum image, not that row.
        bonded_now, mic_distance, _ = _pair_is_bonded(
            symbols, positions, left, right, lattice, pbc, bond_scale
        )
        if best is None or mic_distance < best:
            best = mic_distance
        if bonded_now:
            new_contacts.append(pair)
    return best, new_contacts


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
    free_atom_arcs: Sequence[FreeAtomArc] = (),
) -> ReactivePathResult:
    """Generate a fixed-cell atom-mapped reactive initial path.

    Rigid groups move independently but share the same interpolation parameter.
    Each group is unwrapped by a walk along its periodic bonds, so a chain
    that spans more than half a cell is not folded back onto the first atom.
    A group that is not one bond component is refused. The default
    ``rigid_fit_tolerance_A`` of 1e-5 Å accepts only a construction-accurate
    rigid motion. Endpoints taken from independent relaxations usually
    differ by 0.01-0.1 Å inside a group; pass
    ``ReactivePathConfig(rigid_fit_tolerance_A=0.1)`` for that input.
    Atoms outside the groups, and not listed in ``free_atom_arcs``, move on
    the straight minimum-image line between their own two endpoint
    coordinates, including when the surrounding groups were relaxed
    separately. ``free_atom_arcs`` replaces that line for one atom with a
    polar arc about a pivot. Arcs use the same path parameter as the
    rigid motion.

    ``rigid_group_image_shifts`` is an optional override for the lattice image
    of each product-group centroid after that internal unwrap. It is relative
    to the reactant centroid. ``None`` selects the minimum image. Pass an
    explicit shift only to keep a displacement that is longer than half a
    cell edge.
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
            stationary_tolerance_A=config.stationary_tolerance_A,
            symbols=symbols_a,
            bond_scale=config.bond_scale,
        )
        indices = np.asarray(group.atom_indices, dtype=int)
        target_positions[indices] = group_target
        atom_shifts = np.asarray(pose.atom_image_shifts, dtype=int)
        group_shift = np.asarray(pose.image_shift, dtype=int)
        shifts[indices] = atom_shifts + group_shift
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

    arcs = _prepare_free_atom_arcs(
        free_atom_arcs,
        free_indices,
        poses,
        positions_a,
        target_positions,
        lattice_a,
        reactant.pbc,
    )
    for arc in arcs.values():
        # Land on the bond image about the pivot. The reactant-minimum-image
        # continuation can sit a lattice vector away once the atom has joined
        # the pivot's molecule.
        landing = target_positions[arc.pivot_index] + arc.product_vector
        fractional = cart_to_frac(landing - positions_b[arc.atom_index], lattice_a)
        image_shift = np.rint(fractional)
        image_shift[~np.asarray(reactant.pbc, dtype=bool)] = 0
        target_positions[arc.atom_index] = landing
        shifts[arc.atom_index] = image_shift.astype(int)
    linear_free = [index for index in free_indices if index not in arcs]
    images: list[Atoms] = []
    scheduled_angles: list[list[float]] = []
    for image_index, lam_value in enumerate(lambdas):
        lam = float(lam_value)
        positions = positions_a.copy()
        if lam == 1.0:
            positions = target_positions.copy()
        else:
            for pose in poses:
                indices = np.asarray(pose.group.atom_indices, dtype=int)
                if lam == 0.0:
                    positions[indices] = pose.reactant_positions
                else:
                    positions[indices] = _interpolate_group(
                        pose.reactant_positions, pose, lam, method
                    )
            if linear_free and lam != 0.0:
                indices = np.asarray(linear_free, dtype=int)
                positions[indices] = (1.0 - lam) * positions_a[
                    indices
                ] + lam * target_positions[indices]
        for arc in arcs.values():
            positions[arc.atom_index] = positions[arc.pivot_index] + _polar_offset(
                arc.reactant_vector, arc.product_vector, lam
            )
        scheduled_angles.append(
            [float(np.degrees(pose.angle_rad) * lam) for pose in poses]
        )

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
        endpoint_error = _minimum_image_endpoint_error(
            images[-1].positions, target_positions, lattice_a, reactant.pbc
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
                "reactant_atom_image_shifts": [
                    list(shift) for shift in pose.reactant_atom_image_shifts
                ],
                "atom_image_shifts": [list(shift) for shift in pose.atom_image_shifts],
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
        "free_atom_arcs": [
            {"atom_index": arc.atom_index, "pivot_index": arc.pivot_index}
            for arc in arcs.values()
        ],
        "images": _image_diagnostics(
            symbols_a,
            [frame.positions for frame in images],
            lambdas,
            scheduled_angles,
            lattice_a,
            reactant.pbc,
            poses,
            free_indices,
            bond_changes,
            arcs,
            config.bond_scale,
            positions_a,
        ),
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


def _indices_by_molecule(molecule_index: np.ndarray) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for atom_index, owner in enumerate(np.asarray(molecule_index, dtype=int)):
        grouped.setdefault(int(owner), []).append(int(atom_index))
    return {owner: tuple(indices) for owner, indices in grouped.items()}


def _species_records(
    symbols: Sequence[str],
    molecule_index: np.ndarray,
    charges: dict[int, int],
    roles: dict[int, str],
) -> list[dict[str, Any]]:
    grouped = _indices_by_molecule(molecule_index)
    records = []
    for mol_index in sorted(charges):
        atom_indices = list(grouped.get(mol_index, ()))
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


def _electron_count(
    atomic_numbers: np.ndarray, cell_charge: int, multiplicity: int
) -> int:
    """Return the cell electron count when it can form the declared multiplicity.

    ``N_e = sum(Z) - cell_charge`` must leave a non-negative even number of
    paired electrons after ``multiplicity - 1`` unpaired electrons are placed.
    """
    n_electrons = int(np.sum(atomic_numbers)) - int(cell_charge)
    unpaired = int(multiplicity) - 1
    if n_electrons < unpaired or (n_electrons - unpaired) % 2 != 0:
        raise ValueError(
            f"multiplicity {multiplicity} is incompatible with {n_electrons} electrons"
        )
    return n_electrons


def _annotate_result(
    result: ReactivePathResult,
    *,
    cell_charge: int,
    multiplicity: int,
    n_electrons: int,
    electron_pairing: str | None,
    reactant_species: list[dict[str, Any]],
    product_species: list[dict[str, Any]],
    product_molecule_index: np.ndarray,
) -> ReactivePathResult:
    result.metadata["schema"] = REACTIVE_PATH_SCHEMA_V3
    result.metadata["cell_charge"] = int(cell_charge)
    result.metadata["multiplicity"] = int(multiplicity)
    result.metadata["n_electrons"] = int(n_electrons)
    result.metadata["electron_pairing"] = electron_pairing
    result.metadata["reactant_species"] = reactant_species
    result.metadata["product_species"] = product_species
    result.metadata["product_molecule_index"] = [
        int(value) for value in product_molecule_index
    ]
    for frame in result.images:
        frame.info["cell_charge"] = int(cell_charge)
        frame.info["multiplicity"] = int(multiplicity)
        frame.info["n_electrons"] = int(n_electrons)
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
    cutoff = _bond_search_cutoff(symbols, bond_scale)
    pairs = _candidate_bond_pairs(
        symbols, positions, lattice, pbc, cutoff, {int(atom_index)}
    )
    neighbors = []
    for left, right in pairs:
        other = right if left == atom_index else left
        bonded, _, _ = _pair_is_bonded(
            symbols, positions, atom_index, other, lattice, pbc, bond_scale
        )
        if bonded:
            neighbors.append(other)
    return neighbors


def _unwrapped_for_partition(
    crystal: MolecularCrystal, molecule_index: int
) -> CrystalMolecule:
    """Return a copy whose bonds are contiguous across periodic boundaries.

    ``partition_at_bond`` reads a nonperiodic molecular graph. A bridge that
    crosses a cell face is invisible until the molecule is unwrapped with the
    periodic bond vectors.
    """
    molecule = crystal.molecules[molecule_index].copy()
    molecule.set_cell(np.asarray(crystal.lattice, dtype=float))
    molecule.set_pbc(tuple(bool(flag) for flag in crystal.pbc))
    molecule._graph = None
    molecule._adjust_positions_for_pbc()
    molecule._graph = None
    return molecule


def _rigid_groups_by_molecule(
    molecule_index: np.ndarray,
    roles: dict[int, str],
    skip: set[int],
) -> list[RigidGroup]:
    grouped = _indices_by_molecule(molecule_index)
    groups: list[RigidGroup] = []
    for mol_index in sorted(grouped):
        indices = tuple(index for index in grouped[mol_index] if index not in skip)
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
    free_atom_arcs: Sequence[FreeAtomArc] = (),
    cell_charge: int,
    multiplicity: int,
    n_electrons: int,
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
        free_atom_arcs=free_atom_arcs,
    )
    return _annotate_result(
        result,
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        n_electrons=n_electrons,
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
    free_atom_arcs: Sequence[FreeAtomArc] = (),
) -> ReactivePathResult:
    """Build a fixed-cell path that moves one hydrogen onto another species.

    ``hydrogen_index`` and ``acceptor_index`` are global indices in
    ``crystal.to_ase()``. Spectator molecules stay fixed. The transferring
    hydrogen is the only free atom. This is an H+ transfer: the donor formal
    charge decreases by 1 and the acceptor formal charge increases by 1.
    Both species-charge maps must sum to ``cell_charge``, and spectator
    charges must be unchanged. ``multiplicity`` must be compatible with the
    cell electron count. The geometric frames are an initial guess for an
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
    n_electrons = _electron_count(atoms.get_atomic_numbers(), cell_charge, multiplicity)
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
    if product_charges[donor_molecule] != reactant_charges[donor_molecule] - 1:
        raise ValueError("proton transfer must decrease the donor formal charge by 1")
    if product_charges[acceptor_molecule] != reactant_charges[acceptor_molecule] + 1:
        raise ValueError(
            "proton transfer must increase the acceptor formal charge by 1"
        )

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
        free_atom_arcs=free_atom_arcs,
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        n_electrons=n_electrons,
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
    containing ``atom_j`` translates by ``separation_A`` along the periodic
    image of the bond. A separation longer than half a cell edge stays on
    that Cartesian image.
    A ring bond raises :class:`RingBondRotationError`. A bridge that crosses
    a cell face is unwrapped before the partition is read. ``electron_pairing``
    is the caller-declared label ``"closed_shell"`` or ``"diradical"`` and
    is not inferred. ``multiplicity`` must be compatible with the cell
    electron count. Spectator charges stay unchanged.
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
    n_electrons = _electron_count(atoms.get_atomic_numbers(), cell_charge, multiplicity)
    molecule_index = _require_molecule(owners, molecule_index, "cleaved")
    if molecule_index < 0 or molecule_index >= len(crystal.molecules):
        raise ValueError(f"cleaved molecule {molecule_index} is not in the crystal")
    molecule = crystal.molecules[molecule_index]
    atom_i = _require_index(atom_i, len(molecule), "atom_i")
    atom_j = _require_index(atom_j, len(molecule), "atom_j")
    partition = partition_at_bond(
        _unwrapped_for_partition(crystal, molecule_index), atom_i, atom_j
    )
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
    axis, image_shift = minimum_image_displacement(
        positions[global_j] - positions[global_i], lattice, crystal.pbc
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= BOND_ROTATION_AXIS_TOLERANCE:
        raise ValueError(f"bond axis {atom_i}-{atom_j} has zero length")
    # ``axis`` points at one periodic image of the moving atom. Put the
    # fragment on that image before interpolating, then keep the requested
    # Cartesian separation even when it is longer than half a cell edge.
    unwrap = frac_to_cart(np.asarray(image_shift, dtype=float), lattice)
    reactant_positions = positions.copy()
    reactant_positions[list(moving)] += unwrap
    product_positions = reactant_positions.copy()
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
    grouped = _indices_by_molecule(owners)
    groups = [
        RigidGroup(fixed, name="fragment_fixed"),
        RigidGroup(moving, name="fragment_moving"),
    ]
    for spectator in spectators:
        indices = grouped.get(spectator, ())
        if indices:
            groups.append(RigidGroup(indices, name=f"spectator-{spectator}"))
    if len(groups) < 2:
        raise ValueError("Reactive interpolation requires at least two rigid groups")

    reactant = _rebuild_crystal(crystal, reactant_positions, owners)
    product = _rebuild_crystal(crystal, product_positions, product_owners)
    # Both endpoints already sit on the bond image. A zero image shift keeps
    # a separation that is longer than half a cell.
    return _run_declared_path(
        reactant,
        product,
        groups,
        [BondChange(global_i, global_j, True, False)],
        config,
        rigid_group_image_shifts=tuple((0, 0, 0) for _ in groups),
        cell_charge=cell_charge,
        multiplicity=multiplicity,
        n_electrons=n_electrons,
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
