"""Exact minimum-image and lattice shell enumeration on general cells.

This module is core-install safe (NumPy only). It implements two separate
contracts:

* :func:`minimum_image_cartesian` returns one shortest lattice image of a
  Cartesian displacement (ties break on lexicographically smallest shift).
* :func:`translations_within_cartesian` lists every integer cell shift whose
  image lies strictly inside an open Euclidean cutoff ball.

The Fincke–Pohst walk operates on the Gram–Schmidt form of the active lattice
rows. Non-periodic axes and zero-length rows are skipped. When the 3×3 cell
determinant is below :data:`~molcrys_kit.constants.config.LATTICE_MIC_CELL_DET_EPS`,
fractional rounding is not used as a search bound, but enumeration on usable
rows still runs when periodic axes remain.
"""

from __future__ import annotations

import numpy as np

from ..constants.config import (
    LATTICE_MIC_CELL_DET_EPS,
    LATTICE_MIC_ENUM_EPS,
    LATTICE_MIC_ENUM_MARGIN,
    LATTICE_MIC_GRAM_SCHMIDT_EPS,
    LATTICE_MIC_LENGTH_EPS,
    LATTICE_MIC_ROW_NORM_EPS,
    LATTICE_MIC_SHELL_PADDING_A,
)


def periodic_mask(pbc) -> np.ndarray:
    """Normalize an ASE-style ``pbc`` value to three booleans."""
    if pbc is None:
        return np.zeros(3, dtype=bool)
    if isinstance(pbc, (bool, np.bool_)):
        return np.full(3, bool(pbc), dtype=bool)
    values = np.asarray(pbc, dtype=bool).reshape(-1)
    if values.size == 1:
        return np.full(3, bool(values[0]), dtype=bool)
    if values.size != 3:
        return np.zeros(3, dtype=bool)
    return values.copy()


def _gram_schmidt_rows(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mu, orthogonal, norm2)`` for row vectors."""
    count = basis.shape[0]
    mu = np.zeros((count, count), dtype=float)
    orthogonal = np.zeros((count, 3), dtype=float)
    norm2 = np.zeros(count, dtype=float)
    for i in range(count):
        vector = np.array(basis[i], dtype=float, copy=True)
        for j in range(i):
            if norm2[j] <= LATTICE_MIC_GRAM_SCHMIDT_EPS:
                mu[i, j] = 0.0
                continue
            mu[i, j] = float(np.dot(basis[i], orthogonal[j]) / norm2[j])
            vector -= mu[i, j] * orthogonal[j]
        orthogonal[i] = vector
        norm2[i] = float(np.dot(vector, vector))
    return mu, orthogonal, norm2


def translations_within_cartesian(
    delta,
    cell,
    pbc,
    radius: float,
) -> list[np.ndarray]:
    """Integer shifts ``S`` with ``||delta + S @ cell|| < radius``."""
    delta = np.asarray(delta, dtype=float).reshape(3)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    mask = periodic_mask(pbc)
    if radius <= 0.0 or not np.any(mask):
        if float(np.linalg.norm(delta)) < radius:
            return [np.zeros(3, dtype=int)]
        return []
    active = [
        axis
        for axis in range(3)
        if mask[axis] and float(np.linalg.norm(cell[axis])) > LATTICE_MIC_ROW_NORM_EPS
    ]
    if not active:
        if float(np.linalg.norm(delta)) < radius:
            return [np.zeros(3, dtype=int)]
        return []
    basis = cell[np.asarray(active, dtype=int)]
    mu, orthogonal, norm2 = _gram_schmidt_rows(basis)
    sigma = np.zeros(len(active), dtype=float)
    parallel2 = 0.0
    for level, length2 in enumerate(norm2):
        if length2 <= LATTICE_MIC_GRAM_SCHMIDT_EPS:
            continue
        sigma[level] = float(np.dot(delta, orthogonal[level]) / length2)
        parallel2 += sigma[level] ** 2 * length2
    outside2 = max(0.0, float(np.dot(delta, delta)) - parallel2)
    limit2 = float(radius) * float(radius)
    if outside2 >= limit2:
        return []
    found: list[np.ndarray] = []
    coeffs = np.zeros(len(active), dtype=int)

    def walk(level: int, used2: float) -> None:
        if norm2[level] <= LATTICE_MIC_GRAM_SCHMIDT_EPS:
            coeffs[level] = 0
            if level == 0:
                found.append(coeffs.copy())
            else:
                walk(level - 1, used2)
            return
        extra = float(sigma[level])
        for higher in range(level + 1, len(active)):
            extra += float(coeffs[higher]) * float(mu[higher, level])
        budget = limit2 - used2
        if budget < 0.0:
            return
        reach = float(np.sqrt(budget / norm2[level]))
        low = int(np.ceil(-reach - extra - LATTICE_MIC_ENUM_MARGIN))
        high = int(np.floor(reach - extra + LATTICE_MIC_ENUM_MARGIN))
        for coeff in range(low, high + 1):
            coeffs[level] = coeff
            used_next = used2 + (coeff + extra) ** 2 * float(norm2[level])
            if used_next > limit2 + LATTICE_MIC_ENUM_EPS:
                continue
            if level == 0:
                found.append(coeffs.copy())
            else:
                walk(level - 1, used_next)

    walk(len(active) - 1, outside2)
    shifts = []
    for coeff in found:
        shift = np.zeros(3, dtype=int)
        for local, axis in enumerate(active):
            shift[axis] = int(coeff[local])
        vector = delta + shift @ cell
        if float(np.dot(vector, vector)) < limit2:
            shifts.append(shift)
    shifts.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    return shifts


def minimum_image_cartesian(delta, cell, pbc=None) -> np.ndarray:
    """Shortest Cartesian image of ``delta`` under the periodic lattice."""
    delta = np.asarray(delta, dtype=float).reshape(3)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    mask = periodic_mask(True if pbc is None else pbc)
    if not np.any(mask):
        return delta.copy()
    best_shift = np.zeros(3, dtype=int)
    best_vector = delta.copy()
    best_length = float(np.dot(best_vector, best_vector))
    if abs(float(np.linalg.det(cell))) > LATTICE_MIC_CELL_DET_EPS:
        fractional = delta @ np.linalg.inv(cell)
        rounded = np.zeros(3, dtype=int)
        rounded[mask] = -np.rint(fractional[mask]).astype(int)
        rounded_vector = delta + rounded @ cell
        rounded_length = float(np.dot(rounded_vector, rounded_vector))
        if rounded_length < best_length - LATTICE_MIC_LENGTH_EPS:
            best_shift = rounded
            best_vector = rounded_vector
            best_length = rounded_length
    radius = float(np.sqrt(max(best_length, 0.0))) + LATTICE_MIC_SHELL_PADDING_A
    for shift in translations_within_cartesian(delta, cell, mask, radius):
        vector = delta + shift @ cell
        length = float(np.dot(vector, vector))
        if length < best_length - LATTICE_MIC_LENGTH_EPS or (
            abs(length - best_length) <= LATTICE_MIC_LENGTH_EPS
            and tuple(int(value) for value in shift)
            < tuple(int(value) for value in best_shift)
        ):
            best_shift = shift
            best_vector = vector
            best_length = length
    return best_vector