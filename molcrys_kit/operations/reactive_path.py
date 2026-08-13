"""Atom-mapped reactive initial paths with independently rigid groups.

The routines in this module generate geometric initial guesses.  They do not
perform energy minimization or a nudged elastic band calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np

from ..analysis.interactions.bonding import get_bonding_threshold
from ..constants import get_atomic_radius, has_atomic_radius, is_metal_element
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import (
    cart_to_frac,
    frac_to_cart,
    kabsch_align,
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_to_axis_angle,
    se3_exp,
    se3_log,
)
from .interpolation import InterpolationMethod


ImageShift = tuple[int, int, int]


@dataclass(frozen=True)
class RigidGroup:
    """A set of reactant-global atom identities that moves as one rigid body."""

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
    """Generated images and the atom/image correspondence used to build them."""

    images: list[MolecularCrystal]
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


def _coerce_method(method: InterpolationMethod | str) -> InterpolationMethod:
    if isinstance(method, InterpolationMethod):
        return method
    try:
        return InterpolationMethod(str(method))
    except ValueError as exc:
        raise ValueError(f"Unknown interpolation method: {method!r}") from exc


def _lambda_values(n_images: int, include_endpoints: bool) -> np.ndarray:
    if n_images < 1:
        raise ValueError("n_images must be >= 1")
    if include_endpoints:
        if n_images == 1:
            return np.array([0.0])
        return np.linspace(0.0, 1.0, int(n_images))
    return np.linspace(0.0, 1.0, int(n_images) + 2)[1:-1]


def _minimum_image_displacement(
    delta_cart: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
) -> tuple[np.ndarray, ImageShift]:
    """Return the shortest legal image of a Cartesian displacement."""
    delta_cart = np.asarray(delta_cart, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    frac_delta = cart_to_frac(delta_cart, lattice)
    base = np.zeros(3, dtype=int)
    choices: list[tuple[int, ...]] = []
    for axis, periodic in enumerate(pbc):
        if periodic:
            base[axis] = -int(np.rint(frac_delta[axis]))
            choices.append((-1, 0, 1))
        else:
            choices.append((0,))

    best_vector: np.ndarray | None = None
    best_shift: np.ndarray | None = None
    best_norm2 = np.inf
    for offset in product(*choices):
        shift = base + np.asarray(offset, dtype=int)
        candidate = delta_cart + frac_to_cart(shift, lattice)
        norm2 = float(np.dot(candidate, candidate))
        if norm2 < best_norm2:
            best_norm2 = norm2
            best_vector = candidate
            best_shift = shift

    assert best_vector is not None and best_shift is not None
    return best_vector, tuple(int(value) for value in best_shift)


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
        raise ValueError("product_index_by_reactant must be a permutation of all atom indices")
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
            raise ValueError(f"Rigid group {group_index} contains duplicate atom indices")
        invalid = [index for index in indices if index < 0 or index >= n_atoms]
        if invalid:
            raise ValueError(f"Rigid group {group_index} has out-of-range indices: {invalid}")
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
) -> tuple[_RigidGroupPose, np.ndarray]:
    indices = np.asarray(group.atom_indices, dtype=int)
    group_a = positions_a[indices]
    group_b_raw = positions_b[indices]
    center_a = np.mean(group_a, axis=0)
    center_b_raw = np.mean(group_b_raw, axis=0)
    translation, image_shift = _minimum_image_displacement(
        center_b_raw - center_a, lattice, pbc
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
    centered = positions_a - pose.center_a
    if method is InterpolationMethod.SE3_SCREW:
        xi = se3_log(pose.rotation, pose.translation)
        rotation, translation = se3_exp(float(lam) * xi)
    elif method is InterpolationMethod.COM_SO3:
        rotation = se3_exp(
            np.concatenate([pose.axis * pose.angle_rad * float(lam), np.zeros(3)])
        )[0]
        translation = float(lam) * pose.translation
    elif method is InterpolationMethod.SLERP:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = rotation_matrix_to_quaternion(pose.rotation)
        rotation = quaternion_to_rotation_matrix(quaternion_slerp(q0, q1, lam))
        translation = float(lam) * pose.translation
    else:  # pragma: no cover - guarded by _coerce_method
        raise ValueError(f"Unhandled interpolation method: {method}")
    return centered @ rotation.T + pose.center_a + translation


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
        raise ValueError(f"Missing atomic radius for bond validation: {symbol_i}-{symbol_j}")
    displacement, _ = _minimum_image_displacement(
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
            raise ValueError(f"BondChange has out-of-range atom indices: {(atom_i, atom_j)}")
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
            if actual_a != bool(change.reactant_bonded) or actual_b != bool(change.product_bonded):
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


def interpolate_reactive_path(
    reactant: MolecularCrystal,
    product_crystal: MolecularCrystal,
    *,
    rigid_groups: Sequence[RigidGroup],
    product_index_by_reactant: Sequence[int] | None = None,
    bond_changes: Sequence[BondChange] = (),
    config: ReactivePathConfig | None = None,
) -> ReactivePathResult:
    """Generate a fixed-cell atom-mapped reactive initial path.

    Rigid groups move independently but share the same interpolation parameter.
    Atoms outside all rigid groups interpolate linearly between endpoint images.
    """
    config = config or ReactivePathConfig()
    method = _coerce_method(config.method)
    lambdas = _lambda_values(config.n_images, config.include_endpoints)
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
    if not np.array_equal(np.asarray(reactant.pbc, dtype=bool), np.asarray(product_crystal.pbc, dtype=bool)):
        raise ValueError("Reactive interpolation requires identical PBC flags")

    symbols_a = atoms_a.get_chemical_symbols()
    symbols_b = atoms_b.get_chemical_symbols()
    mapping = _validate_mapping(symbols_a, symbols_b, product_index_by_reactant)
    groups, rigid_indices = _validate_groups(rigid_groups, len(atoms_a))
    positions_a = np.asarray(atoms_a.positions, dtype=float)
    positions_b = np.asarray(atoms_b.positions, dtype=float)[np.asarray(mapping, dtype=int)]

    target_positions = positions_b.copy()
    shifts = np.zeros((len(atoms_a), 3), dtype=int)
    poses: list[_RigidGroupPose] = []
    for group in groups:
        pose, group_target = _fit_rigid_group(
            group,
            positions_a,
            positions_b,
            lattice_a,
            reactant.pbc,
            config.rigid_fit_tolerance_A,
        )
        indices = np.asarray(group.atom_indices, dtype=int)
        target_positions[indices] = group_target
        shifts[indices] = np.asarray(pose.image_shift, dtype=int)
        poses.append(pose)

    free_indices = sorted(set(range(len(atoms_a))) - rigid_indices)
    for atom_index in free_indices:
        _, image_shift = _minimum_image_displacement(
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

    images: list[MolecularCrystal] = []
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
                positions[indices] = (
                    (1.0 - lam) * positions_a[indices]
                    + lam * target_positions[indices]
                )

        frame_atoms = atoms_a.copy()
        frame_atoms.calc = None
        frame_atoms.positions[:] = positions
        frame_atoms.info = dict(frame_atoms.info)
        frame_atoms.info.update(
            {
                "path_kind": "reactive",
                "path_image_index": int(image_index),
                "path_lambda": lam,
                "path_method": method.value,
            }
        )
        images.append(MolecularCrystal.from_ase_atoms(frame_atoms))

    endpoint_error = 0.0
    if len(lambdas) and float(lambdas[-1]) == 1.0:
        endpoint_error = float(
            np.max(np.linalg.norm(images[-1].to_ase().positions - target_positions, axis=1))
        )
        if endpoint_error > config.endpoint_tolerance_A:
            raise RuntimeError(
                f"Reactive path endpoint error {endpoint_error:.6g} Å exceeds "
                f"{config.endpoint_tolerance_A:.6g} Å"
            )

    reference_molecule_index = atoms_a.arrays.get("molecule_index")
    for frame in images:
        atoms = frame.to_ase()
        if atoms.get_chemical_symbols() != symbols_a:
            raise RuntimeError("Reactive path changed global atom order")
        if reference_molecule_index is not None and not np.array_equal(
            atoms.arrays.get("molecule_index"), reference_molecule_index
        ):
            raise RuntimeError("Reactive path changed molecule partitioning")
        if not np.all(np.isfinite(atoms.positions)):
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
        "schema": "molcrys_kit.reactive_path.v1",
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
