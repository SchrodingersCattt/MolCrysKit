"""Periodic-cover invariants of a quotient graph.

Coordination sequences count distinct vertices of the infinite cover. They are
descriptive. A matching sequence is not an isomorphism proof. The shortest
rings through pairs of half-edges are recorded only when the bounded search
finishes; a missing ring is null rather than a guessed size.

Quotient cycle rank is computed by the Stage 1 graph analysis and is not
repeated here as a periodic rank.
"""

from __future__ import annotations

from collections import Counter, deque

from molcrys_kit.periodic._lattice import ImageShift, add
from molcrys_kit.periodic.graph import PeriodicQuotientGraph

from .limits import NetAnalysisLimits


def directed_adjacency(
    graph: PeriodicQuotientGraph,
) -> dict[str, list[tuple[str, ImageShift]]]:
    """Undirected stubs as directed ``(neighbor, shift)`` steps."""
    adjacency: dict[str, list[tuple[str, ImageShift]]] = {
        node_id: [] for node_id in graph.node_ids
    }
    for edge in graph.edges:
        adjacency[edge.source].append((edge.target, edge.image_shift))
        adjacency[edge.target].append(
            (
                edge.source,
                (-edge.image_shift[0], -edge.image_shift[1], -edge.image_shift[2]),
            )
        )
    for node_id in adjacency:
        adjacency[node_id].sort()
    return adjacency


def coordination_sequence(
    graph: PeriodicQuotientGraph,
    start: str,
    limits: NetAnalysisLimits,
) -> tuple[int, ...] | None:
    """Return cover distances ``1..depth`` from ``start``, or ``None`` if capped.

    The state ``(node, image)`` is a vertex of the periodic cover. Translates
    that are not joined by a walk are not visited, so a second interpenetrating
    copy does not inflate the sequence of the first.
    """
    adjacency = directed_adjacency(graph)
    origin: tuple[str, ImageShift] = (start, (0, 0, 0))
    seen = {origin}
    frontier = [origin]
    counts: list[int] = []
    for _depth in range(limits.coordination_depth):
        nxt: list[tuple[str, ImageShift]] = []
        for node_id, image in frontier:
            for neighbor, step in adjacency[node_id]:
                state = (neighbor, add(image, step))
                if state in seen:
                    continue
                seen.add(state)
                if len(seen) > limits.cover_state_limit:
                    return None
                nxt.append(state)
        counts.append(len(nxt))
        frontier = nxt
    return tuple(counts)


def coordination_census(
    graph: PeriodicQuotientGraph,
    limits: NetAnalysisLimits,
) -> tuple[tuple[tuple[int, ...], int], ...] | None:
    """Return the gcd-normalized multiset of coordination sequences.

    A supercell repeats every sequence the same number of times. Dividing the
    counts by their gcd makes that repetition compare equal. ``None`` means at
    least one sequence hit the cover-state cap, so the census is not a proof.
    """
    sequences = [
        coordination_sequence(graph, node_id, limits) for node_id in graph.node_ids
    ]
    if any(sequence is None for sequence in sequences):
        return None
    counts: Counter[tuple[int, ...]] = Counter(sequences)  # type: ignore[arg-type]
    if not counts:
        return ()
    divisor = 0
    for count in counts.values():
        divisor = math_gcd(divisor, count)
    if divisor < 1:
        return ()
    return tuple(sorted((sequence, counts[sequence] // divisor) for sequence in counts))


def math_gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return abs(left)


def degree_sequence(graph: PeriodicQuotientGraph) -> tuple[int, ...]:
    """Cover degrees of the quotient nodes, in node-id order."""
    adjacency = directed_adjacency(graph)
    return tuple(len(adjacency[node_id]) for node_id in graph.node_ids)


def angle_ring_sizes(
    graph: PeriodicQuotientGraph,
    limits: NetAnalysisLimits,
) -> tuple[tuple[int | None, ...], ...] | None:
    """Shortest cover ring through each pair of distinct half-edges.

    The outer tuple follows node-id order. An entry is ``None`` when the ring
    search reaches its length or state cap before a ring is confirmed. The
    whole result is ``None`` when the graph is larger than the ring search is
    willing to start.
    """
    if len(graph.nodes) > limits.canonical_node_limit:
        return None
    adjacency = directed_adjacency(graph)
    rows: list[tuple[int | None, ...]] = []
    for node_id in graph.node_ids:
        stubs = adjacency[node_id]
        sizes: list[int | None] = []
        for left in range(len(stubs)):
            for right in range(left + 1, len(stubs)):
                sizes.append(
                    _angle_ring(
                        adjacency,
                        node_id,
                        stubs[left],
                        stubs[right],
                        limits,
                    )
                )
        rows.append(tuple(sizes))
    return tuple(rows)


def _angle_ring(
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    origin: str,
    first: tuple[str, ImageShift],
    second: tuple[str, ImageShift],
    limits: NetAnalysisLimits,
) -> int | None:
    """Length of the shortest cover cycle using two half-edges at ``origin``."""
    start = (first[0], first[1])
    target = (second[0], second[1])
    if start == target:
        return None
    maximum = limits.ring_length_limit - 2
    if maximum < 0:
        return None
    queue: deque[tuple[str, ImageShift, int]] = deque([(start[0], start[1], 0)])
    seen = {start, (origin, (0, 0, 0))}
    states = 1
    while queue:
        node_id, image, distance = queue.popleft()
        if distance > maximum:
            continue
        if (node_id, image) == target and distance > 0:
            return distance + 2
        if distance == maximum:
            continue
        for neighbor, step in adjacency[node_id]:
            state = (neighbor, add(image, step))
            if state in seen:
                continue
            seen.add(state)
            states += 1
            if states > limits.ring_state_limit:
                return None
            queue.append((neighbor, state[1], distance + 1))
    return None


__all__ = [
    "angle_ring_sizes",
    "coordination_census",
    "coordination_sequence",
    "degree_sequence",
    "directed_adjacency",
]
