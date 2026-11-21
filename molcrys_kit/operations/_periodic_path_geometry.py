"""Periodic geometry primitives for atom-mapped paths.

Cells use row vectors.  Edge image shifts are integers; a shift ``s`` means
``s @ cell``.  Geometric embedding and periodic graph topology remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd

import numpy as np
from ase.geometry import find_mic, minkowski_reduce
from scipy.linalg import polar
from scipy.spatial.transform import Rotation

from ..constants.periodic_path import (
    PERIODIC_PATH_BASIS_ROUND_DECIMALS,
    PERIODIC_PATH_BASIS_SEARCH_RADIUS,
    PERIODIC_PATH_CELL_TOLERANCE,
)
from ..periodic import PeriodicQuotientGraph


@dataclass(frozen=True)
class ImageEdge:
    """An oriented atom pair and the image of its second endpoint."""

    left: int
    right: int
    shift: tuple[int, int, int]
    role: str = "covalent"


def _integer_inverse(matrix: np.ndarray) -> np.ndarray:
    inverse = np.rint(np.linalg.inv(matrix)).astype(int)
    if not np.array_equal(inverse @ matrix, np.eye(3, dtype=int)):
        raise ValueError("periodic basis transform must be unimodular")
    return inverse


def _periodic_flags(pbc) -> np.ndarray:
    flags = np.asarray(pbc, dtype=bool)
    if flags.shape != (3,):
        raise ValueError("periodicity flags must have shape (3,)")
    return flags


def _validate_cell(cell, pbc) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(cell, dtype=float)
    flags = _periodic_flags(pbc)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("cell must be a finite 3 by 3 matrix")
    rows = matrix[flags]
    if len(rows) and np.linalg.matrix_rank(
        rows, tol=PERIODIC_PATH_CELL_TOLERANCE
    ) != len(rows):
        raise ValueError("periodic cell vectors must be linearly independent")
    return matrix, flags


def complete_periodic_cell(cell, pbc) -> np.ndarray:
    """Complete non-periodic rows without altering the periodic subspace."""

    matrix, flags = _validate_cell(cell, pbc)
    if flags.all():
        return matrix.copy()
    out = matrix.copy()
    basis: list[np.ndarray] = []
    for row in matrix[flags]:
        vector = row.copy()
        for existing in basis:
            vector -= np.dot(vector, existing) * existing
        basis.append(vector / np.linalg.norm(vector))
    for axis in np.flatnonzero(~flags):
        choices = []
        for index, candidate in enumerate(np.eye(3)):
            vector = candidate.copy()
            for existing in basis:
                vector -= np.dot(vector, existing) * existing
            choices.append((float(np.linalg.norm(vector)), -index, vector))
        _, _, vector = max(
            choices,
            key=lambda item: (
                round(item[0], PERIODIC_PATH_BASIS_ROUND_DECIMALS),
                item[1],
            ),
        )
        vector /= np.linalg.norm(vector)
        out[axis] = vector
        basis.append(vector)
    if np.linalg.det(out) < 0:
        out[np.flatnonzero(~flags)[-1]] *= -1
    return out


def canonical_periodic_basis(cell, pbc) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic short basis and its integer change of basis.

    The bounded enumeration follows a Minkowski reduction.  It resolves common
    symmetry ties using Cartesian vectors, rather than the supplied cell order.
    """

    matrix, flags = _validate_cell(cell, pbc)
    axes = np.flatnonzero(flags)
    dim = len(axes)
    if dim == 0:
        return matrix.copy(), np.eye(3, dtype=int)
    reduced, transform = minkowski_reduce(matrix, pbc=flags)
    transform = np.asarray(transform, dtype=int)
    reduced = np.asarray(reduced, dtype=float)
    choices = []
    radius = PERIODIC_PATH_BASIS_SEARCH_RADIUS
    for raw in product(range(-radius, radius + 1), repeat=dim):
        if not any(raw):
            continue
        common = 0
        for value in raw:
            common = gcd(common, abs(value))
        if common != 1:
            continue
        coefficients = np.asarray(raw, dtype=int)
        vector = coefficients @ reduced[axes]
        decimals = PERIODIC_PATH_BASIS_ROUND_DECIMALS
        key = (round(float(vector @ vector), decimals),) + tuple(
            float(round(value, decimals)) for value in vector
        )
        choices.append((key, coefficients))
    choices.sort(key=lambda item: item[0])

    selected: list[np.ndarray] = []
    for _, candidate in choices:
        trial = selected + [candidate]
        if len(trial) == dim:
            if abs(round(float(np.linalg.det(np.asarray(trial))))) != 1:
                continue
        elif dim == 3 and len(trial) == 2:
            cross = np.cross(trial[0], trial[1])
            if (
                gcd(gcd(abs(int(cross[0])), abs(int(cross[1]))), abs(int(cross[2])))
                != 1
            ):
                continue
        elif np.linalg.matrix_rank(np.asarray(trial, dtype=float)) != len(trial):
            continue
        selected.append(candidate)
        if len(selected) == dim:
            break
    if len(selected) != dim:
        raise ValueError("could not choose a complete periodic lattice basis")
    coefficients = np.asarray(selected, dtype=int)
    canonical_transform = np.eye(3, dtype=int)
    canonical_transform[np.ix_(axes, axes)] = coefficients
    canonical_transform = canonical_transform @ transform
    result = canonical_transform @ matrix
    if dim == 3 and np.linalg.det(result) < 0:
        canonical_transform[axes[-1]] *= -1
        result = canonical_transform @ matrix
    _integer_inverse(canonical_transform)
    return result, canonical_transform


def interpolate_cells(start, end, pbc, fractions):
    """Interpolate physically aligned cells and preserve independent axes.

    Returns cells in the start basis, and the integer transformation that
    expresses the end cell in that basis. Non-periodic rows remain ordinary
    endpoint-interpolated vectors; they may be zero.
    """

    start, flags = _validate_cell(start, pbc)
    end, _ = _validate_cell(end, flags)
    first, first_op = canonical_periodic_basis(start, flags)
    last, last_op = canonical_periodic_basis(end, flags)
    first_inverse = _integer_inverse(first_op)
    end_transform = first_inverse @ last_op
    first_full = complete_periodic_cell(first, flags)
    last_full = complete_periodic_cell(last, flags)
    deformation = np.linalg.solve(first_full, last_full)
    rotation, stretch = polar(deformation)
    if np.linalg.det(rotation) <= 0:
        raise ValueError("periodic cell alignment does not have a proper rotation")
    eigenvalues, eigenvectors = np.linalg.eigh(stretch)
    if np.any(eigenvalues <= PERIODIC_PATH_CELL_TOLERANCE):
        raise ValueError("periodic cell interpolation has a singular deformation")
    rotation_vector = Rotation.from_matrix(rotation).as_rotvec()
    cells = []
    for value in fractions:
        fraction = float(value)
        current_rotation = Rotation.from_rotvec(fraction * rotation_vector).as_matrix()
        current_stretch = (eigenvectors * eigenvalues**fraction) @ eigenvectors.T
        canonical = first_full @ current_rotation @ current_stretch
        canonical[~flags] = (1.0 - fraction) * first[~flags] + fraction * last[~flags]
        result = first_inverse @ canonical
        _validate_cell(result, flags)
        cells.append(result)
    return cells, end_transform


def nearest_image(delta, cell, pbc) -> tuple[np.ndarray, np.ndarray]:
    """Shortest periodic displacement and exact integer image shift."""

    matrix, flags = _validate_cell(cell, pbc)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("displacement must be a finite 3-vector")
    if not flags.any():
        return delta.copy(), np.zeros(3, dtype=int)
    vector, _ = find_mic(delta, cell=complete_periodic_cell(matrix, flags), pbc=flags)
    coefficients = np.linalg.lstsq(
        matrix[flags].T, np.asarray(vector) - delta, rcond=None
    )[0]
    shift = np.zeros(3, dtype=int)
    shift[flags] = np.rint(coefficients).astype(int)
    if not np.allclose(
        delta + shift @ matrix, vector, atol=PERIODIC_PATH_CELL_TOLERANCE
    ):
        raise ValueError("could not recover an integer periodic image shift")
    return np.asarray(vector, dtype=float), shift


def records_to_edges(crystal, positions, index_to_reference, cell, pbc, records=None):
    """Adapt existing structure records to the shared periodic quotient graph."""

    positions = np.asarray(positions, dtype=float)
    flags = _periodic_flags(pbc)
    inverse_map = np.asarray(index_to_reference, dtype=int)
    edges = []
    if records is None:
        records = crystal.get_bond_records()
    for record in records:
        source = int(record.left_global_index)
        target = int(record.right_global_index)
        vector = np.asarray(record.vector_A, dtype=float)
        delta = positions[target] - positions[source]
        residual = vector - delta
        shift = np.zeros(3, dtype=int)
        if flags.any():
            coeff = np.linalg.lstsq(cell[flags].T, residual, rcond=None)[0]
            shift[flags] = np.rint(coeff).astype(int)
        if not np.allclose(
            delta + shift @ cell, vector, atol=PERIODIC_PATH_CELL_TOLERANCE * 10
        ):
            raise ValueError("bond record is inconsistent with the endpoint geometry")
        left, right = int(inverse_map[source]), int(inverse_map[target])
        if left > right or (left == right and tuple(shift) < tuple(-shift)):
            left, right, shift = right, left, -shift
        edges.append(ImageEdge(left, right, tuple(int(v) for v in shift), record.role))
    graph = PeriodicQuotientGraph.from_edges(
        [str(index) for index in range(len(positions))],
        [(str(edge.left), str(edge.right), edge.shift) for edge in edges],
        pbc=tuple(bool(flag) for flag in flags),
    )
    return edges, graph


def transform_edge(
    edge: ImageEdge, basis_inverse: np.ndarray, atom_shifts=None
) -> ImageEdge:
    """Express an edge in a new basis and, optionally, a new atom gauge."""

    shift = np.asarray(edge.shift, dtype=int) @ basis_inverse
    if atom_shifts is not None:
        shift = shift - atom_shifts[edge.right] + atom_shifts[edge.left]
    return ImageEdge(edge.left, edge.right, tuple(int(v) for v in shift), edge.role)
