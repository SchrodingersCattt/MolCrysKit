"""Bond pairs and trajectory-aware bond inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..constants import (
    METAL_ELEMENTS,
    get_atomic_radius,
    has_atomic_radius,
)
from ..constants.config import BONDING_CONFIG, PERIODIC_NEIGHBOR_TOLERANCE_A

_DEFAULT_ATOMIC_RADIUS = float(BONDING_CONFIG["DEFAULT_ATOMIC_RADIUS"])


@dataclass(frozen=True, slots=True)
class BondPairs:
    """Contiguous bond pairs, minimum-image vectors, and distances."""

    pairs: np.ndarray
    vectors: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        pairs = np.ascontiguousarray(self.pairs, dtype=np.int32)
        vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        distances = np.ascontiguousarray(self.distances, dtype=np.float32)
        if pairs.ndim != 2 or pairs.shape[1:] != (2,):
            raise ValueError("pairs must have shape (M, 2)")
        if vectors.shape != (len(pairs), 3):
            raise ValueError("vectors must have shape (M, 3)")
        if distances.shape != (len(pairs),):
            raise ValueError("distances must have shape (M,)")
        pairs.setflags(write=False)
        vectors.setflags(write=False)
        distances.setflags(write=False)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "distances", distances)


@dataclass(frozen=True, slots=True)
class BondCandidates:
    """Pairs that can become bonded before the Verlet skin is exhausted."""

    pairs: np.ndarray
    reference_positions: np.ndarray
    atomic_numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    skin: float
    search_cutoff: float
    inverse_cell: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pairs = np.ascontiguousarray(self.pairs, dtype=np.int32)
        positions = np.ascontiguousarray(self.reference_positions, dtype=np.float32)
        numbers = np.ascontiguousarray(self.atomic_numbers, dtype=np.uint8)
        cell = np.ascontiguousarray(self.cell, dtype=np.float64)
        pbc = np.ascontiguousarray(self.pbc, dtype=bool)
        if pairs.ndim != 2 or pairs.shape[1:] != (2,):
            raise ValueError("candidate pairs must have shape (K, 2)")
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("reference_positions must have shape (N, 3)")
        if numbers.shape != (len(positions),):
            raise ValueError("atomic_numbers must have shape (N,)")
        if cell.shape != (3, 3) or pbc.shape != (3,):
            raise ValueError("cell and pbc must have shapes (3, 3) and (3,)")
        inverse_cell = np.linalg.inv(cell) if np.any(pbc) else np.eye(3)
        inverse_cell = np.ascontiguousarray(inverse_cell, dtype=np.float64)
        for array in (pairs, positions, numbers, cell, pbc, inverse_cell):
            array.setflags(write=False)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "reference_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "inverse_cell", inverse_cell)


def _element_tables() -> tuple[np.ndarray, np.ndarray]:
    from ase.data import chemical_symbols

    radii = np.full(len(chemical_symbols), _DEFAULT_ATOMIC_RADIUS, dtype=np.float64)
    metals = np.zeros(len(chemical_symbols), dtype=bool)
    for number, symbol in enumerate(chemical_symbols):
        if number == 0 or not symbol:
            continue
        radii[number] = (
            get_atomic_radius(symbol)
            if has_atomic_radius(symbol)
            else _DEFAULT_ATOMIC_RADIUS
        )
        metals[number] = symbol in METAL_ELEMENTS
    return radii, metals


_RADII, _METALS = _element_tables()


def _validate_inputs(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray | None,
    pbc: np.ndarray | tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    numbers = np.ascontiguousarray(atomic_numbers, dtype=np.int64)
    matrix = (
        np.zeros((3, 3), dtype=np.float64)
        if cell is None
        else np.ascontiguousarray(cell, dtype=np.float64)
    )
    periodic = np.ascontiguousarray(pbc, dtype=bool)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if numbers.shape != (len(positions),):
        raise ValueError("atomic_numbers must have shape (N,)")
    if np.any(numbers <= 0) or np.any(numbers >= len(_RADII)):
        raise ValueError("atomic_numbers contains an unsupported atomic number")
    if matrix.shape != (3, 3) or periodic.shape != (3,):
        raise ValueError("cell and pbc must have shapes (3, 3) and (3,)")
    if np.any(periodic) and abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise ValueError("a nonsingular cell is required for periodic bond inference")
    return positions, numbers, matrix, periodic


def _thresholds(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    from ..analysis.interactions.bonding import get_bonding_thresholds

    return get_bonding_thresholds(
        _RADII[first],
        _RADII[second],
        _METALS[first],
        _METALS[second],
    )


def _maximum_threshold(numbers: np.ndarray) -> float:
    present = np.unique(numbers)
    first, second = np.triu_indices(len(present))
    return float(np.max(_thresholds(present[first], present[second])))


def _is_axis_aligned_orthogonal(cell: np.ndarray) -> bool:
    """Return whether ``cell`` is compatible with cKDTree's axis-aligned box."""
    off_diagonal = cell - np.diag(np.diag(cell))
    return bool(np.max(np.abs(off_diagonal)) <= 1.0e-10)


def _orthogonal_candidates(
    positions: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    if not np.any(pbc):
        pairs = cKDTree(positions).query_pairs(cutoff, output_type="ndarray")
        return np.ascontiguousarray(pairs, dtype=np.int32)
    lengths = np.abs(np.diag(cell))
    data = positions.copy()
    box = np.empty(3, dtype=np.float64)
    for axis in range(3):
        if pbc[axis]:
            if lengths[axis] <= 1.0e-12:
                raise ValueError("periodic orthogonal cell lengths must be positive")
            box[axis] = lengths[axis]
            data[:, axis] = np.mod(data[:, axis], lengths[axis])
        else:
            minimum = float(data[:, axis].min(initial=0.0))
            maximum = float(data[:, axis].max(initial=0.0))
            data[:, axis] = data[:, axis] - minimum + cutoff
            # cKDTree requires every coordinate to lie strictly inside boxsize;
            # 2*cutoff also gives zero-width non-periodic data a valid box.
            box[axis] = max(maximum - minimum + 2.0 * cutoff, 2.0 * cutoff)
            box[axis] += max(1.0e-8, cutoff * 1.0e-8)
    pairs = cKDTree(data, boxsize=box).query_pairs(cutoff, output_type="ndarray")
    return np.ascontiguousarray(pairs, dtype=np.int32)


def _triclinic_candidates(
    positions: np.ndarray,
    _atomic_numbers: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    from pymatgen.optimization.neighbors import find_points_in_spheres

    first, second, _, _ = find_points_in_spheres(
        positions,
        positions,
        cutoff,
        pbc.astype(np.int64, copy=False),
        cell,
        tol=PERIODIC_NEIGHBOR_TOLERANCE_A,
    )
    keep = first < second
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.column_stack((first[keep], second[keep])).astype(np.int32, copy=False)
    return np.ascontiguousarray(np.unique(pairs, axis=0), dtype=np.int32)


def build_bond_candidates(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | tuple[bool, bool, bool] = (False, False, False),
    skin: float = 0.5,
) -> BondCandidates:
    """Build a candidate list containing every pair that can become bonded."""
    if not np.isfinite(skin) or skin < 0.0:
        raise ValueError("skin must be finite and non-negative")
    positions, numbers, matrix, periodic = _validate_inputs(
        positions, atomic_numbers, cell, pbc
    )
    search_cutoff = _maximum_threshold(numbers) + float(skin)
    if _is_axis_aligned_orthogonal(matrix):
        pairs = _orthogonal_candidates(
            positions,
            cell=matrix,
            pbc=periodic,
            cutoff=search_cutoff,
        )
    else:
        pairs = _triclinic_candidates(
            positions,
            numbers,
            cell=matrix,
            pbc=periodic,
            cutoff=search_cutoff,
        )
    return BondCandidates(
        pairs=pairs,
        reference_positions=positions,
        atomic_numbers=numbers,
        cell=matrix,
        pbc=periodic,
        skin=float(skin),
        search_cutoff=search_cutoff,
    )


def _minimum_image_vectors(
    positions: np.ndarray,
    pairs: np.ndarray,
    cell: np.ndarray,
    inverse_cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    if len(vectors) and np.any(pbc):
        fractional = vectors @ inverse_cell
        fractional[:, pbc] -= np.rint(fractional[:, pbc])
        vectors = fractional @ cell
    return vectors


def evaluate_bond_candidates(
    candidates: BondCandidates,
    positions: np.ndarray,
    atomic_numbers: np.ndarray | None = None,
) -> BondPairs:
    """Recompute distances and threshold decisions for one trajectory frame."""
    numbers = (
        candidates.atomic_numbers
        if atomic_numbers is None
        else np.asarray(atomic_numbers, dtype=np.int64)
    )
    positions, numbers, matrix, periodic = _validate_inputs(
        positions,
        numbers,
        candidates.cell,
        candidates.pbc,
    )
    if not np.array_equal(numbers, candidates.atomic_numbers):
        raise ValueError("atomic_numbers changed relative to the candidate list")
    pairs = candidates.pairs
    vectors = _minimum_image_vectors(
        positions,
        pairs,
        matrix,
        candidates.inverse_cell,
        periodic,
    )
    distances = np.linalg.norm(vectors, axis=1)
    limits = _thresholds(numbers[pairs[:, 0]], numbers[pairs[:, 1]])
    keep = distances < limits
    return BondPairs(
        pairs=pairs[keep],
        vectors=vectors[keep],
        distances=distances[keep],
    )


def candidate_list_needs_rebuild(
    candidates: BondCandidates,
    positions: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | tuple[bool, bool, bool] | None = None,
) -> bool:
    """Return whether cell or displacement invalidates the Verlet guarantee."""
    matrix = candidates.cell if cell is None else np.asarray(cell, dtype=np.float64)
    periodic = candidates.pbc if pbc is None else np.asarray(pbc, dtype=bool)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != candidates.reference_positions.shape:
        return True
    if matrix.shape != (3, 3) or periodic.shape != (3,):
        return True
    if not np.array_equal(periodic, candidates.pbc):
        return True
    if not np.allclose(matrix, candidates.cell, rtol=1.0e-10, atol=1.0e-12):
        return True
    displacement = positions - candidates.reference_positions
    if np.any(periodic):
        fractional = displacement @ candidates.inverse_cell
        fractional[:, periodic] -= np.rint(fractional[:, periodic])
        displacement = fractional @ matrix
    maximum = float(np.linalg.norm(displacement, axis=1).max(initial=0.0))
    return maximum > candidates.skin * 0.5


class VerletBondTracker:
    """Stateful bond inference that safely reuses candidate pairs.

    Every frame re-evaluates candidate distances, so both bond formation and
    breaking are detected immediately. Rebuilding protects against pairs that
    were outside the search shell entering bonding range.
    """

    def __init__(self, *, skin: float = 0.5):
        if not np.isfinite(skin) or skin <= 0.0:
            raise ValueError("VerletBondTracker skin must be finite and positive")
        self.skin = float(skin)
        self.candidates: BondCandidates | None = None
        self.rebuild_count = 0

    def clear(self) -> None:
        """Release the retained candidate pairs and reference frame."""
        self.candidates = None

    def update(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        cell: np.ndarray | None = None,
        pbc: np.ndarray | tuple[bool, bool, bool] = (False, False, False),
    ) -> BondPairs:
        """Infer current bonds, rebuilding candidates only when required."""
        if self.candidates is None or candidate_list_needs_rebuild(
            self.candidates,
            positions,
            cell=cell,
            pbc=pbc,
        ):
            self.candidates = build_bond_candidates(
                positions,
                atomic_numbers,
                cell=cell,
                pbc=pbc,
                skin=self.skin,
            )
            self.rebuild_count += 1
        return evaluate_bond_candidates(
            self.candidates,
            positions,
            atomic_numbers,
        )


def infer_bond_pairs(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | tuple[bool, bool, bool] = (False, False, False),
) -> BondPairs:
    """Infer bond pairs without constructing a graph or bond records."""
    candidates = build_bond_candidates(
        positions,
        atomic_numbers,
        cell=cell,
        pbc=pbc,
        skin=0.0,
    )
    return evaluate_bond_candidates(candidates, positions, atomic_numbers)


__all__ = [
    "BondPairs",
    "BondCandidates",
    "VerletBondTracker",
    "build_bond_candidates",
    "candidate_list_needs_rebuild",
    "evaluate_bond_candidates",
    "infer_bond_pairs",
]
