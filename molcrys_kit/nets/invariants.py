"""Periodic-cover invariants of a quotient graph.

Coordination sequences count distinct vertices of the infinite cover. They are
descriptive. A matching sequence is not an isomorphism proof. The shortest
rings through pairs of half-edges are recorded only when the bounded search
finishes; a missing ring is null rather than a guessed size.

Point symbols and vertex symbols are the Wells / O'Keeffe combinatorial
symbols of those rings. They are descriptive in the same way. A shared symbol
is not an isomorphism proof, and a search that stops inside its cap does not
invent a symbol.

Quotient cycle rank is computed by the Stage 1 graph analysis and is not
repeated here as a periodic rank.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

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


@dataclass(frozen=True)
class NodeSymbols:
    """Point and vertex symbols of one quotient node, or an explicit gap."""

    node_id: str
    status: str
    reason: str
    degree: int
    point_symbol: str | None
    vertex_symbol: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "reason": self.reason,
            "degree": self.degree,
            "point_symbol": self.point_symbol,
            "vertex_symbol": self.vertex_symbol,
        }


@dataclass(frozen=True)
class NetSymbols:
    """Combinatorial ring symbols of a quotient graph.

    ``signature`` repeats each distinct ``(point, vertex)`` pair once per
    translational copy and then divides those counts by their gcd. An
    equivalent supercell therefore compares equal. The signature is not a
    canonical key and is not consulted by isomorphism.
    """

    status: str
    reason: str
    signature: tuple[tuple[str, str, int], ...] | None
    nodes: tuple[NodeSymbols, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "signature": (
                None
                if self.signature is None
                else [
                    {
                        "point_symbol": point,
                        "vertex_symbol": vertex,
                        "count": count,
                    }
                    for point, vertex, count in self.signature
                ]
            ),
            "nodes": [node.to_dict() for node in self.nodes],
        }


def periodic_net_symbols(
    graph: PeriodicQuotientGraph,
    limits: NetAnalysisLimits | None = None,
) -> NetSymbols:
    """Return point and vertex symbols, or an explicit ``not_computed`` result.

    At a vertex of degree ``n`` there are ``n(n - 1) / 2`` unordered pairs of
    distinct incident half-edges. The point symbol records the shortest simple
    cover cycle through each pair. The vertex symbol records that length and
    how many distinct shortest cycles use the pair. Both strings sort those
    angles, so the edge order at the vertex does not change them.

    A pair with no finished ring inside ``ring_length_limit`` is not written
    as infinity. The node is ``not_computed``. A quotient larger than
    ``canonical_node_limit`` is outside this regime and is not given a symbol.
    """
    limits = limits or NetAnalysisLimits.from_config()
    if len(graph.nodes) > limits.canonical_node_limit:
        return NetSymbols("not_computed", "canonical_node_limit", None, ())
    if not graph.nodes:
        return NetSymbols("computed", "empty_graph", (), ())
    adjacency = directed_adjacency(graph)
    nodes: list[NodeSymbols] = []
    for node_id in graph.node_ids:
        nodes.append(_node_symbols(node_id, adjacency, limits))
    failed = next((node for node in nodes if node.status != "computed"), None)
    if failed is not None:
        return NetSymbols("not_computed", failed.reason, None, tuple(nodes))
    return NetSymbols(
        "computed",
        "computed",
        _symbol_signature(nodes),
        tuple(nodes),
    )


def _symbol_signature(
    nodes: list[NodeSymbols],
) -> tuple[tuple[str, str, int], ...]:
    counts: Counter[tuple[str, str]] = Counter(
        (node.point_symbol or "", node.vertex_symbol or "") for node in nodes
    )
    if not counts:
        return ()
    divisor = 0
    for count in counts.values():
        divisor = math_gcd(divisor, count)
    if divisor < 1:
        return ()
    return tuple(
        sorted(
            (point, vertex, counts[(point, vertex)] // divisor)
            for point, vertex in counts
        )
    )


def _node_symbols(
    node_id: str,
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    limits: NetAnalysisLimits,
) -> NodeSymbols:
    stubs = adjacency[node_id]
    degree = len(stubs)
    if degree < 2:
        return NodeSymbols(node_id, "computed", "no_angles", degree, "", "")
    angles: list[tuple[int, int]] = []
    for left in range(degree):
        for right in range(left + 1, degree):
            size, multiplicity, reason = _ring_census(
                adjacency,
                node_id,
                stubs[left],
                stubs[right],
                limits,
            )
            if reason != "computed" or size is None or multiplicity is None:
                return NodeSymbols(node_id, "not_computed", reason, degree, None, None)
            angles.append((size, multiplicity))
    return NodeSymbols(
        node_id,
        "computed",
        "computed",
        degree,
        _point_symbol(angles),
        _vertex_symbol(angles),
    )


def _point_symbol(angles: list[tuple[int, int]]) -> str:
    counts = Counter(size for size, _multiplicity in angles)
    parts = []
    for size in sorted(counts):
        exponent = counts[size]
        parts.append(f"{size}^{exponent}" if exponent > 1 else str(size))
    return ".".join(parts)


def _vertex_symbol(angles: list[tuple[int, int]]) -> str:
    parts = []
    for size, multiplicity in sorted(angles):
        parts.append(f"{size}_{multiplicity}" if multiplicity > 1 else str(size))
    return ".".join(parts)


def _ring_census(
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    origin: str,
    first: tuple[str, ImageShift],
    second: tuple[str, ImageShift],
    limits: NetAnalysisLimits,
) -> tuple[int | None, int | None, str]:
    """Shortest cover cycles through two half-edges, with a multiplicity."""
    if first == second:
        # Two parallel edges end at the same cover vertex. That pair is one
        # 2-cycle. It is not reported as a missing ring.
        return 2, 1, "computed"
    found = _shortest_ring_length(adjacency, origin, first, second, limits)
    if isinstance(found, str):
        return None, None, found
    path_length = found - 2
    multiplicity = _count_simple_paths(
        adjacency,
        origin,
        (first[0], first[1]),
        (second[0], second[1]),
        path_length,
        limits,
    )
    if isinstance(multiplicity, str):
        return None, None, multiplicity
    if multiplicity < 1:
        return None, None, "ring_not_found_within_bounds"
    return found, multiplicity, "computed"


def _shortest_ring_length(
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    origin: str,
    first: tuple[str, ImageShift],
    second: tuple[str, ImageShift],
    limits: NetAnalysisLimits,
) -> int | str:
    start = (first[0], first[1])
    target = (second[0], second[1])
    maximum = limits.ring_length_limit - 2
    if maximum < 0:
        return "ring_not_found_within_bounds"
    queue: deque[tuple[str, ImageShift, int]] = deque([(start[0], start[1], 0)])
    seen = {start, (origin, (0, 0, 0))}
    states = 1
    capped = False
    while queue:
        node_id, image, distance = queue.popleft()
        if distance > maximum:
            continue
        if (node_id, image) == target and distance > 0:
            return distance + 2
        if distance == maximum:
            capped = True
            continue
        for neighbor, step in adjacency[node_id]:
            state = (neighbor, add(image, step))
            if state in seen:
                continue
            seen.add(state)
            states += 1
            if states > limits.ring_state_limit:
                return "ring_search_capped"
            queue.append((neighbor, state[1], distance + 1))
    if capped:
        return "ring_not_found_within_bounds"
    return "ring_not_found_within_bounds"


def _count_simple_paths(
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    origin: str,
    start: tuple[str, ImageShift],
    target: tuple[str, ImageShift],
    length: int,
    limits: NetAnalysisLimits,
) -> int | str:
    """Count simple cover paths of exactly ``length`` steps from ``start`` to ``target``."""
    if length < 0:
        return "ring_not_found_within_bounds"
    count = 0
    states = 0
    origin_state = (origin, (0, 0, 0))

    def walk(
        node_id: str,
        image: ImageShift,
        steps_left: int,
        seen: set[tuple[str, ImageShift]],
    ) -> str | None:
        nonlocal count, states
        states += 1
        if states > limits.ring_state_limit:
            return "ring_search_capped"
        if steps_left == 0:
            if (node_id, image) == target:
                count += 1
                if count > limits.ring_count_limit:
                    return "ring_multiplicity_capped"
            return None
        for neighbor, step in adjacency[node_id]:
            state = (neighbor, add(image, step))
            if state in seen or state == origin_state:
                continue
            if state == target:
                if steps_left == 1:
                    count += 1
                    if count > limits.ring_count_limit:
                        return "ring_multiplicity_capped"
                continue
            seen.add(state)
            failure = walk(state[0], state[1], steps_left - 1, seen)
            seen.remove(state)
            if failure is not None:
                return failure
        return None

    failure = walk(start[0], start[1], length, {start})
    if failure is not None:
        return failure
    return count


__all__ = [
    "NetSymbols",
    "NodeSymbols",
    "angle_ring_sizes",
    "coordination_census",
    "coordination_sequence",
    "degree_sequence",
    "directed_adjacency",
    "periodic_net_symbols",
]
