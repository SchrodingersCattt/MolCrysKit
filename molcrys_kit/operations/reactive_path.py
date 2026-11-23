"""Atom-mapped reactive initial paths with independently rigid groups.

The routines in this module generate geometric initial guesses.  They do not
perform energy minimization or a nudged elastic band calculation.
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
    BOND_ROTATION_AXIS_TOLERANCE,
)
from ..constants.path_constraints import REACTIVE_PATH_STATIONARY_TOLERANCE_A
from ..structures.crystal import MolecularCrystal
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

ImageShift = tuple[int, int, int]


@dataclass(frozen=True)
class RigidGroup:
    """A set of reactant-global atom identities that moves as one rigid body.

    Rigid fits use equal-weight geometric centroids, not mass-weighted centers.
    ``unwrap_mode='auto'`` uses a bond walk when both endpoints are connected;
    otherwise it retains the supplied Cartesian embedding and reports that
    fallback. ``'bond_walk'`` requires connected endpoints; ``'as_provided'``
    explicitly retains the supplied embedding without inferring bonds.
    """

    atom_indices: tuple[int, ...]
    name: str | None = None
    unwrap_mode: str = "auto"

    def __post_init__(self):
        if self.unwrap_mode not in {"auto", "bond_walk", "as_provided"}:
            raise ValueError("unwrap_mode must be 'auto', 'bond_walk', or 'as_provided'")


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
    rigid_fit_tolerance_A: float = 1.0e-5
    endpoint_tolerance_A: float = 1.0e-8
    bond_scale: float = 1.0
    validate_bond_changes: bool = True
    # Added last to retain the existing positional argument order.
    # Widening the endpoint tolerance must not erase real rigid motion.
    stationary_tolerance_A: float = REACTIVE_PATH_STATIONARY_TOLERANCE_A


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
    unwrap_mode_used: str


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
        raw = tuple(mapping)
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in raw):
            raise ValueError("product_index_by_reactant must contain integer indices")
        resolved = tuple(int(index) for index in raw)
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
    if not resolved:
        raise ValueError("Reactive interpolation requires at least one rigid group")
    used: set[int] = set()
    for group_index, group in enumerate(resolved):
        if not isinstance(group, RigidGroup):
            raise TypeError("rigid_groups must contain RigidGroup objects")
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in group.atom_indices):
            raise ValueError("rigid group atom indices must be integers")
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


class _DisconnectedRigidGroup(ValueError):
    """Bond-walk unwrapping could not connect all selected atoms."""


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
        raise _DisconnectedRigidGroup(
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
    stationary_tolerance_A: float = REACTIVE_PATH_STATIONARY_TOLERANCE_A,
    symbols: Sequence[str] | None = None,
    bond_scale: float = 1.0,
) -> tuple[_RigidGroupPose, np.ndarray]:
    indices = np.asarray(group.atom_indices, dtype=int)
    if symbols is None:
        raise ValueError("rigid-group unwrap requires chemical symbols")
    group_symbols = [symbols[int(index)] for index in indices]
    mode = group.unwrap_mode
    if mode != "as_provided":
        try:
            group_a, reactant_atom_image_shifts = _unwrap_along_bonds(
                positions_a[indices], group_symbols, lattice, pbc, bond_scale
            )
            group_b, atom_image_shifts = _unwrap_along_bonds(
                positions_b[indices], group_symbols, lattice, pbc, bond_scale
            )
            mode_used = "bond_walk"
        except _DisconnectedRigidGroup:
            if mode == "bond_walk":
                raise
            mode_used = "as_provided_disconnected"
    else:
        mode_used = "as_provided"
    if mode_used != "bond_walk":
        group_a = np.asarray(positions_a[indices], dtype=float).copy()
        group_b = np.asarray(positions_b[indices], dtype=float).copy()
        reactant_atom_image_shifts = ((0, 0, 0),) * len(indices)
        atom_image_shifts = ((0, 0, 0),) * len(indices)
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
            unwrap_mode_used=mode_used,
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
        reactant_positions=group_a,
        reactant_atom_image_shifts=reactant_atom_image_shifts,
        atom_image_shifts=atom_image_shifts,
        unwrap_mode_used=mode_used,
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


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)



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
        if not isinstance(change, BondChange):
            raise TypeError("bond_changes must contain BondChange objects")
        atom_i = _require_int(change.atom_i, "BondChange.atom_i")
        atom_j = _require_int(change.atom_j, "BondChange.atom_j")
        if not isinstance(change.reactant_bonded, (bool, np.bool_)) or not isinstance(change.product_bonded, (bool, np.bool_)):
            raise ValueError("BondChange bonded states must be booleans")
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
    Connected groups are unwrapped by a walk along their periodic bonds, so a
    chain spanning more than half a cell is not folded onto the first atom.
    Disconnected groups retain the supplied embedding in the default ``auto``
    mode, with the fallback recorded; callers can require a bond walk or
    explicitly retain the supplied embedding via ``RigidGroup.unwrap_mode``.
    Atoms outside the groups move on straight lines unless ``free_atom_arcs`` gives
    them a polar arc about a pivot. Arcs use the same path parameter as the
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
    tolerances = (config.rigid_fit_tolerance_A, config.endpoint_tolerance_A, config.stationary_tolerance_A)
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Path tolerances must be finite and non-negative")
    if not np.isfinite(config.bond_scale) or config.bond_scale <= 0:
        raise ValueError("bond_scale must be positive and finite")

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
    periodic = np.asarray(reactant.pbc, dtype=bool)
    if any(shift is not None and np.any(np.asarray(shift)[~periodic] != 0) for shift in group_shifts):
        raise ValueError("rigid group image shifts must be zero on non-periodic axes")
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
            stationary_tolerance_A=min(config.stationary_tolerance_A, config.endpoint_tolerance_A),
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
                "unwrap_mode_requested": pose.group.unwrap_mode,
                "unwrap_mode_used": pose.unwrap_mode_used,
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
