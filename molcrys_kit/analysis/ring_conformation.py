"""General N-membered ring puckering descriptors and cycle analysis.

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
from numbers import Integral
from typing import Sequence

import networkx as nx
import numpy as np

from ..constants.config import RING_CONFORMATION_TOLERANCE_FACTOR
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
        Out-of-plane displacements in the input Cartesian length unit, shape
        ``(N,)``, relative to the CP reference plane and in ``ring_atoms``
        order.
    amplitudes : np.ndarray
        Puckering amplitudes in the input Cartesian length unit. Paired modes
        are ordered ``m = 2, ..., floor((N - 1) / 2)``; an even ring has one
        final signed alternating mode ``m = N / 2``.
    phases : np.ndarray
        Puckering phases φ_m in radians and in the same mode order. Forward
        phases lie in ``[-pi, pi]``. The unpaired even-ring terminal mode has
        a NaN phase. An exactly zero paired mode has phase zero by convention;
        finite nonzero modes are never thresholded away.
    total_amplitude : float
        Total puckering amplitude Q = sqrt(sum(z_j^2)) in the input Cartesian
        length unit.
    mean_plane_normal : np.ndarray
        Unit normal to the Cremer–Pople reference plane. Its sign is determined
        by the ordered ring; reversing or cyclically relabeling that order can
        change signed coordinates.
    mean_plane_center : np.ndarray
        Geometric center of the ring atoms in the input Cartesian length unit.

    Notes
    -----
    Paired phases use ``z_j = q_m sqrt(2/N) cos(2πmj/N + φ_m)``, equivalently
    ``φ_m = atan2(-Σ_j z_j sin(2πmj/N), Σ_j z_j cos(2πmj/N))``. This explicit
    sign convention fixes comparisons with other Cremer–Pople implementations.
    For a cyclic shift ``r'_j = r_(j+k)``, ``z'_j = z_(j+k)``, paired-mode
    phases transform as ``φ'_m = wrap(φ_m + 2πmk/N)``, and the even terminal
    mode transforms as ``q'_(N/2) = (-1)^k q_(N/2)``. For a reversal
    ``r'_j = r_(k-j)``, the normal changes sign, ``z'_j = -z_(k-j)``, paired
    phases transform as ``φ'_m = wrap(π - φ_m - 2πmk/N)``, and the terminal
    mode transforms as ``q'_(N/2) = (-1)^(k+1) q_(N/2)``. Here ``wrap`` maps
    angles to ``[-π, π]``. A zero-amplitude paired mode has undefined phase;
    this API canonicalizes it to zero.
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
        True when the ring is classified as simple from the complete molecular
        graph, including cyclic partners omitted by ``max_ring_size``.
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


class RingCycleLimitError(RingConformationError):
    """Raised when chordless-cycle enumeration exceeds a configured budget."""


# ---------------------------------------------------------------------------
# General N-membered Cremer–Pople puckering coordinates
# ---------------------------------------------------------------------------


def _validate_ring_atoms(
    molecule: CrystalMolecule,
    ring_atoms: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize ring atom specification."""
    try:
        raw_ring = tuple(ring_atoms)
    except TypeError as exc:
        raise InvalidRingOrderError(
            "ring_atoms must be a sequence of integer atom indices."
        ) from exc

    if any(
        isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral)
        for idx in raw_ring
    ):
        raise InvalidRingOrderError(
            "ring_atoms must contain only integer atom indices; boolean, "
            "floating-point, and string values are not accepted."
        )
    ring = tuple(int(idx) for idx in raw_ring)
    n = len(ring)
    if n < 3:
        raise InvalidRingOrderError(f"A ring must have at least 3 atoms, got {n}.")
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
                f"Atoms {i} and {j} (positions {k} and {(k + 1) % n} in ring) "
                "are not bonded in the molecular graph."
            )
    return ring


def _mean_ring_plane(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Cremer–Pople reference plane for an N-membered ring.

    The CP reference plane is defined (Eq. 3–5 in [1]) by the two vectors:

        R1 = Σ_j (r_j - R') sin(2πj/N)
        R2 = Σ_j (r_j - R') cos(2πj/N)

    where R' is the geometric centroid. The plane normal is n = R1 × R2
    (normalized). This specific construction guarantees that the resulting
    out-of-plane displacements z_j satisfy the three CP constraints:
        Σ z_j = 0,  Σ z_j cos(2πj/N) = 0,  Σ z_j sin(2πj/N) = 0.

    Returns ``(normal, center, centered)`` where normal is a unit vector,
    center is the geometric centroid, and centered contains translation-stable
    coordinates relative to that centroid.
    """
    n = len(positions)
    relative = positions - positions[0]
    center_offset = relative.mean(axis=0)
    center = positions[0] + center_offset
    centered = relative - center_offset
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        raise DegenerateRingGeometryError(
            "All ring atoms are coincident; cannot define a ring plane."
        )
    normalized = centered / scale
    tolerance = RING_CONFORMATION_TOLERANCE_FACTOR * np.finfo(float).eps * max(n, 3)

    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values[0] == 0.0:
        raise DegenerateRingGeometryError(
            "All ring atoms are coincident; cannot define a ring plane."
        )
    if singular_values[1] <= tolerance * singular_values[0]:
        raise DegenerateRingGeometryError(
            "Ring atoms are collinear; cannot define a ring plane."
        )

    # CP reference vectors (Eq. 4 in Cremer & Pople 1975)
    j_arr = np.arange(n)
    sin_coeffs = np.sin(2.0 * np.pi * j_arr / n)
    cos_coeffs = np.cos(2.0 * np.pi * j_arr / n)

    r1 = sin_coeffs @ normalized  # shape (3,)
    r2 = cos_coeffs @ normalized  # shape (3,)

    normal = np.cross(r1, r2)
    norm = np.linalg.norm(normal)
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    if (
        r1_norm <= tolerance
        or r2_norm <= tolerance
        or norm <= tolerance * r1_norm * r2_norm
    ):
        raise DegenerateRingGeometryError(
            "Cremer-Pople reference plane is undefined for this ordered geometry."
        )

    return normal / norm, center, centered


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
        closing pair must be bonded in the molecular graph. The order fixes the
        sign of the plane normal, displacements, and even-ring terminal mode.

    Returns
    -------
    PuckeringCoordinates
        Fourier-decomposed out-of-plane displacements. Displacements and
        amplitudes inherit the Cartesian coordinate unit; phases are radians.

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

    normal, center, centered = _mean_ring_plane(positions)

    # Out-of-plane displacements
    z = centered @ normal

    # Total puckering amplitude
    total_q = float(np.sqrt(np.sum(z**2)))

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
        q_m = np.sqrt(cos_sum**2 + sin_sum**2)
        if q_m == 0.0:
            phi_m = 0.0
        else:
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

    amplitudes_array = np.array(amplitudes, dtype=float)
    phases_array = np.array(phases, dtype=float)
    return PuckeringCoordinates(
        ring_size=n,
        z_displacements=z.copy(),
        amplitudes=amplitudes_array,
        phases=phases_array,
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

    This is the inverse of the forward CP transform for normal displacements.
    Reconstructing full Cartesian positions additionally requires the in-plane
    coordinates of every ring atom.

    Parameters
    ----------
    ring_size : int
        Number of atoms N.
    amplitudes, phases : np.ndarray
        One-dimensional arrays in the order returned by
        :func:`puckering_coordinates`: paired modes ``m = 2, ...,
        floor((N - 1) / 2)``, followed for even N by the signed terminal mode
        ``m = N / 2``. Amplitudes must be finite. Paired phases must be finite
        radians; the terminal phase must be NaN.

    Returns
    -------
    np.ndarray
        Out-of-plane displacements z_j, shape (N,).
    """
    if isinstance(ring_size, (bool, np.bool_)) or not isinstance(ring_size, Integral):
        raise InvalidRingOrderError(
            f"ring_size must be an integer >= 3, got {ring_size!r}"
        )
    n = int(ring_size)
    if n < 3:
        raise InvalidRingOrderError(f"ring_size must be >= 3, got {n}")

    try:
        amplitudes = np.asarray(amplitudes, dtype=float)
        phases = np.asarray(phases, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RingConformationError(
            "amplitudes and phases must be numeric one-dimensional arrays."
        ) from exc

    paired_count = max(0, (n - 1) // 2 - 1)
    expected_modes = paired_count + int(n % 2 == 0)
    expected_shape = (expected_modes,)
    if amplitudes.shape != expected_shape or phases.shape != expected_shape:
        raise RingConformationError(
            "amplitudes and phases must both have shape "
            f"{expected_shape} for ring_size={n}; got "
            f"{amplitudes.shape} and {phases.shape}."
        )
    if not np.all(np.isfinite(amplitudes)):
        raise RingConformationError("amplitudes must contain only finite values.")
    if not np.all(np.isfinite(phases[:paired_count])):
        raise RingConformationError(
            "phases for paired modes must contain only finite values."
        )
    if n % 2 == 0 and not np.isnan(phases[-1]):
        raise RingConformationError(
            "The even-ring terminal mode has no phase; its phase must be NaN."
        )

    z = np.zeros(n)
    j_arr = np.arange(n)

    # Paired modes
    max_paired_m = (n - 1) // 2
    mode_idx = 0
    for m in range(2, max_paired_m + 1):
        q_m = amplitudes[mode_idx]
        phi_m = phases[mode_idx]
        z += q_m * np.sqrt(2.0 / n) * np.cos(2.0 * np.pi * m * j_arr / n + phi_m)
        mode_idx += 1

    # Terminal mode for even N (signed amplitude, no phase)
    if n % 2 == 0:
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
    max_cycles: int = 10_000,
    max_search_states: int = 1_000_000,
) -> list[RingSystem]:
    """Detect and classify ring systems in a molecular graph.

    Enumerates all chordless cycles up to ``max_ring_size``, then classifies each
    cycle from attachment paths in the full molecular graph. An alternative
    path between adjacent ring atoms indicates fusion; a path between
    non-adjacent ring atoms indicates a bridged system; and a cyclic block
    attached at one ring atom indicates a spiro system. Classification therefore
    includes rings omitted by ``max_ring_size`` and does not depend on counting
    overlaps in an arbitrary cycle basis. Cycles are canonicalized by rotation
    and reversal and returned deterministically. Highly polycyclic graphs can
    have many chordless cycles; ``max_ring_size`` bounds their length,
    ``max_cycles`` bounds materialized output, and ``max_search_states`` bounds
    traversal work.

    Parameters
    ----------
    molecule : CrystalMolecule
        Molecule to analyze.
    max_ring_size : int
        Maximum chordless-cycle length to enumerate.
    max_cycles : int
        Maximum number of distinct chordless cycles to materialize.
    max_search_states : int
        Maximum number of induced path states to examine.

    Returns
    -------
    list of RingSystem
        Detected rings sorted by size then atom indices.
    """
    for name, value, minimum in (
        ("max_ring_size", max_ring_size, 3),
        ("max_cycles", max_cycles, 1),
        ("max_search_states", max_search_states, 1),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise RingConformationError(
                f"{name} must be an integer >= {minimum}, got {value!r}"
            )
        if value < minimum:
            raise RingConformationError(f"{name} must be >= {minimum}, got {value}")

    graph = molecule.get_graph()
    ordered_cycles = _chordless_cycles(
        graph,
        int(max_ring_size),
        max_cycles=int(max_cycles),
        max_search_states=int(max_search_states),
    )

    results: list[RingSystem] = []
    for cycle in ordered_cycles:
        classification = _classify_cycle(graph, cycle)
        results.append(
            RingSystem(
                ring_atoms=cycle,
                ring_size=len(cycle),
                is_simple=classification == "simple",
                classification=classification,
            )
        )

    return sorted(results, key=lambda r: (r.ring_size, r.ring_atoms))


def _chordless_cycles(
    graph: nx.Graph,
    max_ring_size: int,
    *,
    max_cycles: int,
    max_search_states: int,
) -> list[tuple[int, ...]]:
    """Enumerate bounded chordless cycles with explicit resource budgets."""
    cycles: set[tuple[int, ...]] = set()
    search_states = 0

    for component_nodes in nx.biconnected_components(graph):
        if len(component_nodes) < 3:
            continue
        component = graph.subgraph(component_nodes)
        for start in sorted(component.nodes):
            stack = [[int(start)]]
            while stack:
                path = stack.pop()
                search_states += 1
                if search_states > max_search_states:
                    raise RingCycleLimitError(
                        "Chordless-cycle search exceeded max_search_states="
                        f"{max_search_states}."
                    )
                current = path[-1]
                for candidate in reversed(sorted(component.neighbors(current))):
                    if candidate < start or candidate in path:
                        continue
                    earlier_neighbors = [
                        node
                        for node in path[:-1]
                        if component.has_edge(candidate, node)
                    ]
                    if earlier_neighbors == [start]:
                        if 3 <= len(path) + 1 <= max_ring_size:
                            cycle = _canonicalize_cycle([*path, candidate])
                            if cycle not in cycles and len(cycles) >= max_cycles:
                                raise RingCycleLimitError(
                                    "Chordless-cycle output exceeded max_cycles="
                                    f"{max_cycles}."
                                )
                            cycles.add(cycle)
                        continue
                    if earlier_neighbors:
                        continue
                    if len(path) < max_ring_size:
                        stack.append([*path, int(candidate)])
    return sorted(cycles, key=lambda cycle: (len(cycle), cycle))


def _classify_cycle(graph: nx.Graph, cycle: tuple[int, ...]) -> str:
    """Classify one cycle from full-graph paths and cyclic blocks."""
    cycle_set = set(cycle)
    cycle_edges = {
        frozenset((cycle[index], cycle[(index + 1) % len(cycle)]))
        for index in range(len(cycle))
    }

    has_fused_path = False
    has_spiro_block = False
    external_graph = graph.subgraph(set(graph.nodes) - cycle_set)
    attachment_groups: list[tuple[set[int], set[int]]] = []
    for component_nodes in nx.connected_components(external_graph):
        attachments = {
            neighbor
            for node in component_nodes
            for neighbor in graph.neighbors(node)
            if neighbor in cycle_set
        }
        attachment_groups.append((set(component_nodes), attachments))

    for atom_i, atom_j in graph.subgraph(cycle_set).edges:
        if frozenset((atom_i, atom_j)) not in cycle_edges:
            attachment_groups.append((set(), {atom_i, atom_j}))

    for component_nodes, attachments in attachment_groups:
        if len(attachments) >= 2:
            if len(attachments) > 2:
                return "bridged"
            attachment_edge = frozenset(attachments)
            if attachment_edge not in cycle_edges:
                return "bridged"
            has_fused_path = True
        elif len(attachments) == 1:
            attached_block = graph.subgraph(component_nodes | attachments)
            attachment = next(iter(attachments))
            if any(attachment in cycle for cycle in nx.cycle_basis(attached_block)):
                has_spiro_block = True

    if has_fused_path:
        return "fused"
    if has_spiro_block:
        return "spiro"
    return "simple"


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
