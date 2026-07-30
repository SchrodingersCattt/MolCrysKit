"""General N-membered ring puckering coordinates and cycle analysis.

This module implements the Cremer–Pople (CP) ring-puckering coordinate system
for arbitrary N-membered rings (N >= 3).  The CP formalism decomposes out-of-
plane displacements into orthogonal Fourier modes, providing a topology-
independent descriptor of ring conformation.

References
----------
.. [1] D. Cremer, J. A. Pople, "General definition of ring puckering
   coordinates," J. Am. Chem. Soc. 97 (6), 1354–1358 (1975).
   DOI: 10.1021/ja00839a011

.. [2] M. Kessler, J. Pérez, "Equivalence properties of the Cremer & Pople
   puckering coordinates for N-membered rings," J. Math. Chem. 50, 187–209
   (2012). DOI: 10.1007/s10910-011-9905-5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..structures.molecule import CrystalMolecule


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PuckeringCoordinates:
    """Cremer–Pople puckering coordinates for an N-membered ring.

    Attributes
    ----------
    ring_size : int
        Number of atoms in the ring (N).
    z_displacements : np.ndarray
        Out-of-plane displacements (N,) relative to the mean ring plane.
    amplitudes : np.ndarray
        Puckering amplitudes q_m for each mode.
    phases : np.ndarray
        Puckering phases φ_m for each paired mode (NaN for unpaired terminal).
    total_amplitude : float
        Total puckering amplitude Q = sqrt(sum(z_j^2)).
    mean_plane_normal : np.ndarray
        Unit normal to the least-squares mean ring plane.
    mean_plane_center : np.ndarray
        Geometric center of the ring atoms projected onto the mean plane.
    """

    ring_size: int
    z_displacements: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    total_amplitude: float
    mean_plane_normal: np.ndarray
    mean_plane_center: np.ndarray


@dataclass(frozen=True)
class RingSystem:
    """Detected ring system from a molecular graph.

    Attributes
    ----------
    ring_atoms : tuple of int
        Ordered atom indices forming the ring cycle.
    ring_size : int
        Number of atoms.
    is_simple : bool
        True if no atom appears in another detected ring (monocyclic).
    classification : str
        One of 'simple', 'fused', 'spiro', 'bridged'.
    """

    ring_atoms: tuple[int, ...]
    ring_size: int
    is_simple: bool
    classification: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RingConformationError(ValueError):
    """Base error for ring conformation operations."""


class InvalidRingOrderError(RingConformationError):
    """Raised when the supplied ring atom ordering is invalid."""


class DegenerateRingGeometryError(RingConformationError):
    """Raised when the ring geometry is too degenerate for plane fitting."""


# ---------------------------------------------------------------------------
# General N-membered Cremer–Pople puckering coordinates
# ---------------------------------------------------------------------------


def _validate_ring_atoms(
    molecule: CrystalMolecule,
    ring_atoms: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize ring atom specification."""
    ring = tuple(int(idx) for idx in ring_atoms)
    n = len(ring)
    if n < 3:
        raise InvalidRingOrderError(
            f"A ring must have at least 3 atoms, got {n}."
        )
    if len(set(ring)) != n:
        raise InvalidRingOrderError(
            f"Ring atoms must be unique; duplicates found in {ring}."
        )
    n_atoms = len(molecule)
    for idx in ring:
        if idx < 0 or idx >= n_atoms:
            raise InvalidRingOrderError(
                f"Atom index {idx} out of range for molecule with {n_atoms} atoms."
            )
    # Verify consecutive atoms and closure are graph edges
    graph = molecule.get_graph()
    for k in range(n):
        i, j = ring[k], ring[(k + 1) % n]
        if not graph.has_edge(i, j):
            raise InvalidRingOrderError(
                f"Atoms {i} and {j} (positions {k} and {(k+1)%n} in ring) "
                "are not bonded in the molecular graph."
            )
    return ring


def _mean_ring_plane(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Cremer–Pople reference plane for an N-membered ring.

    The CP reference plane is defined (Eq. 3–5 in [1]) by the two vectors:

        R1 = Σ_j (r_j - R') sin(2πj/N)
        R2 = Σ_j (r_j - R') cos(2πj/N)

    where R' is the geometric centroid. The plane normal is n = R1 × R2
    (normalized). This specific construction guarantees that the resulting
    out-of-plane displacements z_j satisfy the three CP constraints:
        Σ z_j = 0,  Σ z_j cos(2πj/N) = 0,  Σ z_j sin(2πj/N) = 0.

    Returns (normal, center) where normal is a unit vector and center is
    the geometric centroid of the ring atoms.
    """
    n = len(positions)
    center = positions.mean(axis=0)
    centered = positions - center

    # CP reference vectors (Eq. 4 in Cremer & Pople 1975)
    j_arr = np.arange(n)
    sin_coeffs = np.sin(2.0 * np.pi * j_arr / n)
    cos_coeffs = np.cos(2.0 * np.pi * j_arr / n)

    r1 = sin_coeffs @ centered  # shape (3,)
    r2 = cos_coeffs @ centered  # shape (3,)

    normal = np.cross(r1, r2)
    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        # Fallback: check if atoms are collinear or coincident
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if s[0] < 1e-12:
            raise DegenerateRingGeometryError(
                "All ring atoms are coincident; cannot define a ring plane."
            )
        raise DegenerateRingGeometryError(
            "Ring atoms are collinear; cannot define a ring plane."
        )

    normal = normal / norm
    # Ensure consistent orientation (positive z convention)
    if normal[2] < 0:
        normal = -normal
    return normal, center


def puckering_coordinates(
    molecule: CrystalMolecule,
    ring_atoms: Sequence[int],
) -> PuckeringCoordinates:
    """Compute general Cremer–Pople puckering coordinates for an N-membered ring.

    Parameters
    ----------
    molecule : CrystalMolecule
        Molecule with contiguous (unwrapped) coordinates.
    ring_atoms : sequence of int
        Ordered atom indices forming the ring cycle. Consecutive pairs and the
        closing pair must be bonded in the molecular graph.

    Returns
    -------
    PuckeringCoordinates
        Fourier-decomposed out-of-plane displacements.

    Raises
    ------
    InvalidRingOrderError
        If any consecutive pair or the closure is not a graph edge, or if
        atoms are repeated or out of range.
    DegenerateRingGeometryError
        If the ring atoms are collinear or coincident.
    """
    ring = _validate_ring_atoms(molecule, ring_atoms)
    n = len(ring)
    positions = molecule.get_positions()[list(ring)]

    normal, center = _mean_ring_plane(positions)

    # Out-of-plane displacements
    z = (positions - center) @ normal

    # Total puckering amplitude
    total_q = float(np.sqrt(np.sum(z ** 2)))

    # Fourier decomposition into modes m = 2, 3, ..., floor((N-1)/2)
    # plus the terminal alternating mode for even N (m = N/2)
    amplitudes = []
    phases = []

    # Paired modes: m = 2, ..., floor((N-1)/2)
    max_paired_m = (n - 1) // 2
    for m in range(2, max_paired_m + 1):
        cos_sum = np.sqrt(2.0 / n) * np.sum(
            z * np.cos(2.0 * np.pi * m * np.arange(n) / n)
        )
        sin_sum = -np.sqrt(2.0 / n) * np.sum(
            z * np.sin(2.0 * np.pi * m * np.arange(n) / n)
        )
        q_m = np.sqrt(cos_sum ** 2 + sin_sum ** 2)
        phi_m = np.arctan2(sin_sum, cos_sum)
        amplitudes.append(float(q_m))
        phases.append(float(phi_m))

    # Terminal alternating mode for even N: m = N/2
    # Store signed amplitude since there is no phase degree of freedom
    if n % 2 == 0:
        q_term = (1.0 / np.sqrt(n)) * np.sum(
            z * np.array([(-1) ** j for j in range(n)])
        )
        amplitudes.append(float(q_term))
        phases.append(float(np.nan))  # No phase for this mode

    return PuckeringCoordinates(
        ring_size=n,
        z_displacements=z.copy(),
        amplitudes=np.array(amplitudes, dtype=float),
        phases=np.array(phases, dtype=float),
        total_amplitude=total_q,
        mean_plane_normal=normal.copy(),
        mean_plane_center=center.copy(),
    )


def reconstruct_z_from_modes(
    ring_size: int,
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Reconstruct out-of-plane displacements from Fourier puckering modes.

    This is the inverse of the forward CP transform. Combined with the mean
    ring plane, it allows reconstruction of 3D ring atom positions from
    puckering coordinates.

    Parameters
    ----------
    ring_size : int
        Number of atoms N.
    amplitudes, phases : np.ndarray
        Mode amplitudes and phases as returned by :func:`puckering_coordinates`.

    Returns
    -------
    np.ndarray
        Out-of-plane displacements z_j, shape (N,).
    """
    n = ring_size
    if n < 3:
        raise InvalidRingOrderError(f"ring_size must be >= 3, got {n}")

    z = np.zeros(n)
    j_arr = np.arange(n)

    # Paired modes
    max_paired_m = (n - 1) // 2
    mode_idx = 0
    for m in range(2, max_paired_m + 1):
        q_m = amplitudes[mode_idx]
        phi_m = phases[mode_idx]
        z += q_m * np.sqrt(2.0 / n) * np.cos(
            2.0 * np.pi * m * j_arr / n + phi_m
        )
        mode_idx += 1

    # Terminal mode for even N (signed amplitude, no phase)
    if n % 2 == 0 and mode_idx < len(amplitudes):
        q_term = amplitudes[mode_idx]
        z += q_term * (1.0 / np.sqrt(n)) * np.array([(-1) ** j for j in range(n)])

    return z


# ---------------------------------------------------------------------------
# Ring system detection
# ---------------------------------------------------------------------------


def find_ring_systems(
    molecule: CrystalMolecule,
    *,
    max_ring_size: int = 20,
) -> List[RingSystem]:
    """Detect and classify ring systems in a molecular graph.

    Uses the NetworkX cycle basis to find candidate simple cycles, then
    classifies them based on atom sharing. The returned ring orderings are
    deterministic but not guaranteed to be chemically canonical; callers
    should treat them as **proposals** and verify or override for their
    specific application.

    Parameters
    ----------
    molecule : CrystalMolecule
        Molecule to analyze.
    max_ring_size : int
        Ignore cycles larger than this to avoid combinatorial explosion.

    Returns
    -------
    list of RingSystem
        Detected rings sorted by size then atom indices.
    """
    graph = molecule.get_graph()
    cycles = nx.cycle_basis(graph)

    # Filter by size and convert to ordered sequences
    ordered_cycles: list[tuple[int, ...]] = []
    for cycle in cycles:
        if len(cycle) > max_ring_size:
            continue
        # Order: start from smallest index, direction chosen to make second
        # element smaller than the last element
        cycle_sorted = _canonicalize_cycle(cycle)
        ordered_cycles.append(cycle_sorted)

    # Classify based on atom sharing
    atom_memberships: dict[int, list[int]] = {}
    for idx, cycle in enumerate(ordered_cycles):
        for atom in cycle:
            atom_memberships.setdefault(atom, []).append(idx)

    results: list[RingSystem] = []
    for idx, cycle in enumerate(ordered_cycles):
        shared_atoms = sum(
            1 for atom in cycle if len(atom_memberships[atom]) > 1
        )
        if shared_atoms == 0:
            classification = "simple"
            is_simple = True
        elif shared_atoms == 1:
            classification = "spiro"
            is_simple = False
        elif shared_atoms == 2:
            classification = "fused"
            is_simple = False
        else:
            classification = "bridged"
            is_simple = False

        results.append(
            RingSystem(
                ring_atoms=cycle,
                ring_size=len(cycle),
                is_simple=is_simple,
                classification=classification,
            )
        )

    return sorted(results, key=lambda r: (r.ring_size, r.ring_atoms))


def _canonicalize_cycle(cycle: list[int]) -> tuple[int, ...]:
    """Produce a deterministic ordering for a cycle.

    Starts at the smallest atom index and chooses the direction that makes
    the second element smaller than the last.
    """
    n = len(cycle)
    min_idx = cycle.index(min(cycle))
    # Two directions from the minimum
    forward = tuple(cycle[(min_idx + k) % n] for k in range(n))
    backward = tuple(cycle[(min_idx - k) % n] for k in range(n))
    # Choose lexicographically smaller
    return min(forward, backward)
