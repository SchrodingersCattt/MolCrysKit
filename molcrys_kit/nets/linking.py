"""Embedding relations and interpenetration witnesses.

A verified link is an integer linking number of two disjoint cover cycles,
computed with exact coordinates after an orientation-preserving shear. Close
approaches, projection crossings of trees, and a search that stops early are
not links. Failure to find a cycle pair is not proof that none exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from molcrys_kit.periodic._lattice import ImageShift, add
from molcrys_kit.periodic.graph import PeriodicEmbedding, PeriodicQuotientGraph

from ._exact import F3, fraction_vector
from .invariants import directed_adjacency
from .limits import NetAnalysisLimits

_SHEARS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, -2),
    (-2, 1),
    (2, -3),
    (3, -1),
    (1, 4),
    (4, 1),
)


@dataclass(frozen=True)
class LinkingAssessment:
    """Pairwise embedding relation of two nets in one lattice."""

    status: str
    relation: str
    reason: str
    witness: dict[str, object] | None
    translational_offset: tuple[str, str, str] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "relation": self.relation,
            "reason": self.reason,
            "witness": self.witness,
            "translational_offset": (
                None
                if self.translational_offset is None
                else list(self.translational_offset)
            ),
        }


def assess_linking(
    left: PeriodicQuotientGraph,
    left_embedding: PeriodicEmbedding | None,
    right: PeriodicQuotientGraph,
    right_embedding: PeriodicEmbedding | None,
    *,
    limits: NetAnalysisLimits | None = None,
    same_vertex_namespace: bool = False,
) -> LinkingAssessment:
    """Classify coexistence, a verified link, or an inconclusive search."""
    limits = limits or NetAnalysisLimits.from_config()
    if left_embedding is None or right_embedding is None:
        return LinkingAssessment(
            "not_assessed", "unknown", "missing_embedding", None, None
        )
    if left_embedding.lattice_A is None or right_embedding.lattice_A is None:
        return LinkingAssessment(
            "invalid_or_indeterminate", "unknown", "missing_lattice", None, None
        )
    if left_embedding.lattice_A != right_embedding.lattice_A:
        return LinkingAssessment(
            "invalid_or_indeterminate", "unknown", "lattice_mismatch", None, None
        )
    relation, relation_reason = _relation(
        left, right, same_vertex_namespace=same_vertex_namespace
    )
    if relation != "graph_disjoint":
        return LinkingAssessment(
            "invalid_or_indeterminate",
            relation,
            relation_reason,
            {"kind": relation_reason},
            None,
        )
    try:
        left_points = _fractional_points(left, left_embedding, limits)
        right_points = _fractional_points(right, right_embedding, limits)
    except ValueError as exc:
        return LinkingAssessment(
            "invalid_or_indeterminate",
            "graph_disjoint",
            "inexact_geometry",
            {"kind": "inexact_geometry", "detail": str(exc)},
            None,
        )
    offset = _translational_offset(left_points, right_points)
    separation = _separating_interval(left, left_points, right, right_points)
    if separation is not None:
        return LinkingAssessment(
            "coexistence",
            "graph_disjoint",
            "separating_interval",
            separation,
            offset,
        )
    if _coplanar(left_points, right_points) and _curves_disjoint(
        left, left_points, right, right_points
    ):
        return LinkingAssessment(
            "coexistence",
            "graph_disjoint",
            "coplanar_disjoint",
            {"kind": "coplanar_disjoint"},
            offset,
        )
    left_cycles, left_capped = _cover_cycles(left, limits)
    right_cycles, right_capped = _cover_cycles(right, limits)
    if not left_cycles or not right_cycles:
        return _inconclusive(
            left,
            left_points,
            right,
            right_points,
            left_capped
            or right_capped
            or _could_have_longer_cycles(left, right, limits),
            offset,
            reason="no_cycle_in_bounds" if (left_capped or right_capped) else "acyclic",
        )
    best: dict[str, object] | None = None
    best_abs = 0
    pairs = 0
    capped_pairs = False
    for left_cycle in left_cycles:
        for right_cycle in right_cycles:
            if pairs >= limits.link_pair_limit:
                capped_pairs = True
                break
            pairs += 1
            if _cycles_share_vertex(left_cycle, right_cycle):
                continue
            number, detail = _linking_number(
                left_cycle, left_points, right_cycle, right_points
            )
            if isinstance(number, str):
                return LinkingAssessment(
                    "invalid_or_indeterminate",
                    "graph_disjoint",
                    number,
                    detail if isinstance(detail, dict) else {"kind": number},
                    offset,
                )
            assert isinstance(number, int)
            if abs(number) > best_abs:
                best_abs = abs(number)
                best = detail if isinstance(detail, dict) else None
                if best is not None:
                    best = {**best, "linking_number": number}
            if best_abs:
                break
        if best_abs or capped_pairs:
            break
    if best_abs and best is not None:
        return LinkingAssessment(
            "linked", "graph_disjoint", "linking_number", best, offset
        )
    search_incomplete = (
        left_capped
        or right_capped
        or capped_pairs
        or _could_have_longer_cycles(left, right, limits)
    )
    if not search_incomplete:
        return LinkingAssessment(
            "coexistence",
            "graph_disjoint",
            "linking_numbers_vanish",
            {
                "kind": "linking_numbers",
                "pairs_tested": pairs,
                "max_abs_linking_number": 0,
                "complete": True,
            },
            offset,
        )
    return _inconclusive(
        left,
        left_points,
        right,
        right_points,
        True,
        offset,
        reason="bounded_cycle_search",
    )


def _relation(
    left: PeriodicQuotientGraph,
    right: PeriodicQuotientGraph,
    *,
    same_vertex_namespace: bool,
) -> tuple[str, str]:
    if not same_vertex_namespace:
        return "graph_disjoint", "independent_labels"
    shared_nodes = set(left.node_ids) & set(right.node_ids)
    if shared_nodes:
        return "shared_vertices", "shared_node_ids"
    left_edges = {
        _undirected_key(edge.source, edge.target, edge.image_shift)
        for edge in left.edges
    }
    for edge in right.edges:
        if _undirected_key(edge.source, edge.target, edge.image_shift) in left_edges:
            return "shared_edges", "shared_quotient_edge"
    return "graph_disjoint", "disjoint_subgraphs"


def _undirected_key(
    source: str, target: str, shift: ImageShift
) -> tuple[str, str, ImageShift]:
    if (target, neg_shift(shift), source) < (source, shift, target):
        return (target, source, neg_shift(shift))
    return (source, target, shift)


def neg_shift(shift: ImageShift) -> ImageShift:
    return (-shift[0], -shift[1], -shift[2])


def _fractional_points(
    graph: PeriodicQuotientGraph,
    embedding: PeriodicEmbedding,
    limits: NetAnalysisLimits,
) -> dict[str, F3]:
    points: dict[str, F3] = {}
    for node_id in graph.node_ids:
        raw = embedding.fractional_position(node_id)
        points[node_id] = tuple(
            _exact_component(component, limits, node_id) for component in raw
        )
    return points


def _exact_component(value: float, limits: NetAnalysisLimits, name: str) -> Fraction:
    fractional = Fraction(value).limit_denominator(limits.fraction_denominator_limit)
    if abs(float(fractional) - float(value)) > limits.fraction_residual_A:
        raise ValueError(f"coordinate of {name} is not an exact rational")
    return fractional


def _translational_offset(
    left: dict[str, F3], right: dict[str, F3]
) -> tuple[str, str, str] | None:
    if not left or set(left) != set(right):
        return None
    deltas = []
    for node_id, point in left.items():
        deltas.append(
            (
                right[node_id][0] - point[0],
                right[node_id][1] - point[1],
                right[node_id][2] - point[2],
            )
        )
    if len(set(deltas)) != 1:
        return None
    delta = deltas[0]
    return (str(delta[0]), str(delta[1]), str(delta[2]))


def _separating_interval(
    left: PeriodicQuotientGraph,
    left_points: dict[str, F3],
    right: PeriodicQuotientGraph,
    right_points: dict[str, F3],
) -> dict[str, object] | None:
    for axis in range(3):
        if _winds(left, axis) or _winds(right, axis):
            continue
        left_span = _span(left, left_points, axis)
        right_span = _span(right, right_points, axis)
        if left_span is None or right_span is None:
            continue
        if left_span[1] < right_span[0] or right_span[1] < left_span[0]:
            return {
                "kind": "separating_interval",
                "axis": axis,
                "left": [str(left_span[0]), str(left_span[1])],
                "right": [str(right_span[0]), str(right_span[1])],
            }
    return None


def _winds(graph: PeriodicQuotientGraph, axis: int) -> bool:
    return any(edge.image_shift[axis] for edge in graph.edges)


def _span(
    graph: PeriodicQuotientGraph, points: dict[str, F3], axis: int
) -> tuple[Fraction, Fraction] | None:
    values = [points[node_id][axis] for node_id in graph.node_ids]
    for edge in graph.edges:
        values.append(points[edge.target][axis] + edge.image_shift[axis])
        values.append(points[edge.source][axis])
    if not values:
        return None
    return min(values), max(values)


def _coplanar(left: dict[str, F3], right: dict[str, F3]) -> bool:
    points = list(left.values()) + list(right.values())
    if len(points) < 4:
        return True
    origin = points[0]
    base = None
    for point in points[1:]:
        delta = _sub(point, origin)
        if delta != (Fraction(0), Fraction(0), Fraction(0)):
            base = delta
            break
    if base is None:
        return True
    normal = None
    for point in points:
        cross = _cross(base, _sub(point, origin))
        if cross != (Fraction(0), Fraction(0), Fraction(0)):
            normal = cross
            break
    if normal is None:
        return True
    return all(_dot(normal, _sub(point, origin)) == 0 for point in points)


def _curves_disjoint(
    left: PeriodicQuotientGraph,
    left_points: dict[str, F3],
    right: PeriodicQuotientGraph,
    right_points: dict[str, F3],
) -> bool:
    for left_edge in left.edges:
        for right_edge in right.edges:
            if _edges_meet(left_edge, left_points, right_edge, right_points):
                return False
    return True


def _edges_meet(left_edge, left_points, right_edge, right_points) -> bool:
    a = left_points[left_edge.source]
    b = _add(left_points[left_edge.target], fraction_vector(left_edge.image_shift))
    c = right_points[right_edge.source]
    d = _add(right_points[right_edge.target], fraction_vector(right_edge.image_shift))
    # Bounding-box rejection, then a rational segment test via triple products.
    if _separated(a, b, c, d):
        return False
    volume_ac = _dot(_cross(_sub(b, a), _sub(c, a)), _sub(d, a))
    if volume_ac != 0:
        return False
    return not _separated(a, b, c, d)


def _separated(a: F3, b: F3, c: F3, d: F3) -> bool:
    for axis in range(3):
        if max(a[axis], b[axis]) < min(c[axis], d[axis]):
            return True
        if max(c[axis], d[axis]) < min(a[axis], b[axis]):
            return True
    return False


def _cover_cycles(
    graph: PeriodicQuotientGraph, limits: NetAnalysisLimits
) -> tuple[tuple[tuple[tuple[str, ImageShift], ...], ...], bool]:
    """Return simple cover cycles and whether the search stopped early."""
    adjacency = directed_adjacency(graph)
    found: list[tuple[tuple[str, ImageShift], ...]] = []
    capped = False
    states = 0
    nodes = graph.node_ids
    for start in nodes:
        stack: list[tuple[str, ImageShift, int]] = [(start, (0, 0, 0), 0)]
        path: list[tuple[str, ImageShift]] = []
        visited: set[tuple[str, ImageShift]] = set()
        while stack:
            node_id, image, cursor = stack[-1]
            if cursor == 0:
                if len(path) > limits.ring_length_limit:
                    stack.pop()
                    continue
                path.append((node_id, image))
                visited.add((node_id, image))
                states += 1
                if states > limits.ring_state_limit:
                    capped = True
                    stack.clear()
                    break
            stubs = adjacency[node_id]
            if cursor >= len(stubs):
                path.pop()
                visited.discard((node_id, image))
                stack.pop()
                continue
            stack[-1] = (node_id, image, cursor + 1)
            neighbor, step = stubs[cursor]
            nxt = (neighbor, add(image, step))
            if nxt == (start, (0, 0, 0)) and len(path) >= 3:
                cycle = _canonical_cycle(tuple(path))
                if cycle not in found:
                    found.append(cycle)
                    if len(found) >= limits.ring_count_limit:
                        capped = True
                        stack.clear()
                        break
                continue
            if nxt in visited or len(path) >= limits.ring_length_limit:
                continue
            # Only grow a cycle whose start is the minimum vertex so far.
            if (start, (0, 0, 0)) <= min(path):
                stack.append((neighbor, add(image, step), 0))
        if capped:
            break
    if _could_have_longer_cycles(graph, graph, limits):
        capped = True
    return tuple(found), capped


def _could_have_longer_cycles(
    left: PeriodicQuotientGraph,
    right: PeriodicQuotientGraph,
    limits: NetAnalysisLimits,
) -> bool:
    for graph in (left, right):
        if any(edge.image_shift != (0, 0, 0) for edge in graph.edges):
            return True
        if len(graph.nodes) > limits.ring_length_limit:
            return True
    return False


def _canonical_cycle(
    path: tuple[tuple[str, ImageShift], ...],
) -> tuple[tuple[str, ImageShift], ...]:
    rotations = [path[index:] + path[:index] for index in range(len(path))]
    flipped = tuple(reversed(path))
    rotations.extend(flipped[index:] + flipped[:index] for index in range(len(flipped)))
    return min(rotations)


def _cycles_share_vertex(
    left: tuple[tuple[str, ImageShift], ...],
    right: tuple[tuple[str, ImageShift], ...],
) -> bool:
    return bool(set(left) & set(right))


def _linking_number(
    left: tuple[tuple[str, ImageShift], ...],
    left_points: dict[str, F3],
    right: tuple[tuple[str, ImageShift], ...],
    right_points: dict[str, F3],
) -> tuple[int | str, dict[str, object] | None]:
    left_xyz = [_placed(vertex, left_points) for vertex in left]
    right_xyz = [_placed(vertex, right_points) for vertex in right]
    for sx, sy in _SHEARS:
        shear = (Fraction(sx), Fraction(sy))
        number = _projected_linking(left_xyz, right_xyz, shear)
        if isinstance(number, int):
            return number, {
                "kind": "linking_number",
                "linking_number": number,
                "projection_shear": [sx, sy],
                "left_cycle": [
                    {"node_id": node_id, "image": list(image)}
                    for node_id, image in left
                ],
                "right_cycle": [
                    {"node_id": node_id, "image": list(image)}
                    for node_id, image in right
                ],
            }
    return "projection_degenerate", {"kind": "projection_degenerate"}


def _placed(vertex: tuple[str, ImageShift], points: dict[str, F3]) -> F3:
    node_id, image = vertex
    base = points[node_id]
    return (
        base[0] + image[0],
        base[1] + image[1],
        base[2] + image[2],
    )


def _projected_linking(
    left: list[F3], right: list[F3], shear: tuple[Fraction, Fraction]
) -> int | None:
    left_proj = [_shear(point, shear) for point in left]
    right_proj = [_shear(point, shear) for point in right]
    if not _generic(left_proj, right_proj):
        return None
    total = 0
    for i in range(len(left)):
        a2, b2 = left_proj[i], left_proj[(i + 1) % len(left)]
        az, bz = left[i][2], left[(i + 1) % len(left)][2]
        for j in range(len(right)):
            c2, d2 = right_proj[j], right_proj[(j + 1) % len(right)]
            cz, dz = right[j][2], right[(j + 1) % len(right)][2]
            sign = _crossing_sign(a2, b2, az, bz, c2, d2, cz, dz)
            if sign is None:
                return None
            total += sign
    if total % 2:
        return None
    return total // 2


def _shear(point: F3, shear: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return (point[0] + shear[0] * point[2], point[1] + shear[1] * point[2])


def _generic(
    left: list[tuple[Fraction, Fraction]], right: list[tuple[Fraction, Fraction]]
) -> bool:
    def edges(points):
        return [
            (points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
        ]

    for start, end in edges(left) + edges(right):
        if start == end:
            return False
    return True


def _crossing_sign(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    az: Fraction,
    bz: Fraction,
    c: tuple[Fraction, Fraction],
    d: tuple[Fraction, Fraction],
    cz: Fraction,
    dz: Fraction,
) -> int | None:
    denom = _cross2(_sub2(b, a), _sub2(d, c))
    if denom == 0:
        if _proper_overlap(a, b, c, d):
            return None
        return 0
    t = _cross2(_sub2(c, a), _sub2(d, c)) / denom
    u = _cross2(_sub2(c, a), _sub2(b, a)) / denom
    if not (Fraction(0) < t < 1 and Fraction(0) < u < 1):
        return 0
    height_left = az + t * (bz - az)
    height_right = cz + u * (dz - cz)
    if height_left == height_right:
        return None
    orientation = 1 if denom > 0 else -1
    if height_left > height_right:
        return orientation
    return -orientation


def _proper_overlap(a, b, c, d) -> bool:
    if _cross2(_sub2(b, a), _sub2(c, a)) != 0:
        return False
    return _intervals_overlap(a, b, c, d)


def _intervals_overlap(a, b, c, d) -> bool:
    for axis in range(2):
        if max(a[axis], b[axis]) <= min(c[axis], d[axis]):
            return False
        if max(c[axis], d[axis]) <= min(a[axis], b[axis]):
            return False
    return True


def _inconclusive(
    left: PeriodicQuotientGraph,
    left_points: dict[str, F3],
    right: PeriodicQuotientGraph,
    right_points: dict[str, F3],
    capped: bool,
    offset: tuple[str, str, str] | None,
    *,
    reason: str,
) -> LinkingAssessment:
    if reason == "acyclic" and not capped:
        return LinkingAssessment(
            "coexistence",
            "graph_disjoint",
            "acyclic",
            {"kind": "acyclic"},
            offset,
        )
    interpenetrates = _boxes_overlap(
        left, left_points, right, right_points
    ) or _projection_crosses(left, left_points, right, right_points)
    if interpenetrates:
        return LinkingAssessment(
            "unverified",
            "graph_disjoint",
            reason,
            {"kind": "bounded_search", "interpenetration_without_link_witness": True},
            offset,
        )
    return LinkingAssessment(
        "not_detected_within_bounds",
        "graph_disjoint",
        reason,
        {"kind": "bounded_search", "interpenetration_without_link_witness": False},
        offset,
    )


def _boxes_overlap(left, left_points, right, right_points) -> bool:
    left_box = _box(left, left_points)
    right_box = _box(right, right_points)
    for axis in range(3):
        if (
            left_box[axis][1] < right_box[axis][0]
            or right_box[axis][1] < left_box[axis][0]
        ):
            return False
    return True


def _box(graph, points) -> tuple[tuple[Fraction, Fraction], ...]:
    spans = []
    for axis in range(3):
        span = _span(graph, points, axis)
        spans.append(span if span is not None else (Fraction(0), Fraction(0)))
    return tuple(spans)


def _projection_crosses(left, left_points, right, right_points) -> bool:
    for sx, sy in _SHEARS[:4]:
        shear = (Fraction(sx), Fraction(sy))
        left_edges = _projected_edges(left, left_points, shear)
        right_edges = _projected_edges(right, right_points, shear)
        for a, b in left_edges:
            for c, d in right_edges:
                denom = _cross2(_sub2(b, a), _sub2(d, c))
                if denom == 0:
                    continue
                t = _cross2(_sub2(c, a), _sub2(d, c)) / denom
                u = _cross2(_sub2(c, a), _sub2(b, a)) / denom
                if Fraction(0) < t < 1 and Fraction(0) < u < 1:
                    return True
    return False


def _projected_edges(graph, points, shear):
    edges = []
    for edge in graph.edges:
        start = _shear(points[edge.source], shear)
        target = _add(points[edge.target], fraction_vector(edge.image_shift))
        edges.append((start, _shear(target, shear)))
    return edges


def _add(left: F3, right: F3) -> F3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: F3, right: F3) -> F3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: F3, right: F3) -> Fraction:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: F3, right: F3) -> F3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _sub2(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return (left[0] - right[0], left[1] - right[1])


def _cross2(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


__all__ = ["LinkingAssessment", "assess_linking"]
