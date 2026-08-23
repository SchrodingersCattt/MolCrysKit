"""Private primitives shared by path-generation adapters.

This module owns geometric mechanics only. Endpoint correspondence and
domain-specific validation remain in the public interpolation adapters.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from ..constants.config import KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import _strip_stale_frac_arrays
from ..utils.geometry import (
    cart_to_frac,
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_to_axis_angle,
    se3_exp,
    se3_log,
)


class InterpolationMethod(str, Enum):
    """Supported rigid-body interpolation metrics."""

    SE3_SCREW = "se3_screw"
    COM_SO3 = "com_so3"
    SLERP = "slerp"


def coerce_interpolation_method(
    method: InterpolationMethod | str,
) -> InterpolationMethod:
    """Resolve the public method names and their compatibility aliases."""
    if isinstance(method, InterpolationMethod):
        return method
    try:
        return InterpolationMethod(str(method))
    except ValueError:
        normalized = str(method).lower().replace("-", "_")
        aliases = {
            "screw_rotation": InterpolationMethod.SE3_SCREW,
            "screw": InterpolationMethod.SE3_SCREW,
            "se3": InterpolationMethod.SE3_SCREW,
            "se3_geodesic": InterpolationMethod.SE3_SCREW,
            "com_alignment": InterpolationMethod.COM_SO3,
            "com": InterpolationMethod.COM_SO3,
            "so3_com": InterpolationMethod.COM_SO3,
            "quaternion_slerp": InterpolationMethod.SLERP,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown interpolation method: {method!r}") from None


def path_lambda_values(n_images: int, include_endpoints: bool) -> np.ndarray:
    """Return the shared frame-count convention for path adapters."""
    if isinstance(n_images, bool) or int(n_images) != n_images or n_images < 1:
        raise ValueError("n_images must be an integer >= 1")
    count = int(n_images)
    if include_endpoints:
        if count == 1:
            return np.array([0.0])
        return np.linspace(0.0, 1.0, count)
    return np.linspace(0.0, 1.0, count + 2)[1:-1]


def minimum_image_displacement(
    delta_cart: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a shortest legal image and its integer lattice shift."""
    delta = np.asarray(delta_cart, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    if delta.shape != (3,) or cell.shape != (3, 3) or periodic.shape != (3,):
        raise ValueError("Expected delta (3,), lattice (3, 3), and pbc (3,)")
    mic_vector, _ = find_mic(delta, cell=cell, pbc=periodic)
    shift_fractional = cart_to_frac(np.asarray(mic_vector) - delta, cell)
    shift = np.rint(shift_fractional).astype(int)
    shift[~periodic] = 0
    return np.asarray(mic_vector, dtype=float), tuple(int(value) for value in shift)


def interpolate_rigid_positions(
    positions: np.ndarray,
    *,
    center: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    lam: float,
    method: InterpolationMethod | str,
) -> np.ndarray:
    """Interpolate one rigid set from the identity pose to a target pose."""
    resolved_method = coerce_interpolation_method(method)
    fraction = float(lam)
    centered = np.asarray(positions, dtype=float) - np.asarray(center, dtype=float)
    target_rotation = np.asarray(rotation, dtype=float)
    target_translation = np.asarray(translation, dtype=float)

    if resolved_method is InterpolationMethod.SE3_SCREW:
        xi = se3_log(target_rotation, target_translation)
        rotation_i, translation_i = se3_exp(fraction * xi)
    elif resolved_method is InterpolationMethod.COM_SO3:
        axis, angle = rotation_to_axis_angle(target_rotation)
        rotation_i = se3_exp(np.concatenate([axis * angle * fraction, np.zeros(3)]))[0]
        translation_i = fraction * target_translation
    elif resolved_method is InterpolationMethod.SLERP:
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        target = rotation_matrix_to_quaternion(target_rotation)
        rotation_i = quaternion_to_rotation_matrix(
            quaternion_slerp(identity, target, fraction)
        )
        translation_i = fraction * target_translation
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unhandled interpolation method: {resolved_method}")
    return centered @ rotation_i.T + np.asarray(center, dtype=float) + translation_i


def copy_crystal_with_molecule_positions(
    crystal: MolecularCrystal,
    positions_by_index: Mapping[int, np.ndarray],
) -> MolecularCrystal:
    """Copy a crystal frame, preserving payloads and invalidating stale data."""
    frame = crystal.copy()
    for index, positions in positions_by_index.items():
        molecule = frame.molecules[int(index)]
        molecule.set_positions(np.asarray(positions, dtype=float))
        _strip_stale_frac_arrays(molecule)
    for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
        frame.extra_arrays.pop(key, None)
    frame._calc_results = None
    return frame


def materialize_atoms_frame(
    reference: Atoms,
    positions: np.ndarray,
    *,
    info_updates: Mapping[str, object] | None = None,
    drop_arrays: Sequence[str] = (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z),
) -> Atoms:
    """Copy a flat atom-ordered frame and replace geometry-dependent payloads."""
    frame = reference.copy()
    frame.calc = None
    frame.positions[:] = np.asarray(positions, dtype=float)
    for key in drop_arrays:
        if key in frame.arrays:
            del frame.arrays[key]
    frame.info = copy.deepcopy(frame.info)
    if info_updates:
        frame.info.update(copy.deepcopy(dict(info_updates)))
    return frame
