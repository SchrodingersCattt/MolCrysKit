"""Link witnesses are geometric. Crossings and capped searches stay inconclusive."""

from __future__ import annotations

from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.nets.linking import assess_linking
from molcrys_kit.periodic import PeriodicEmbedding, PeriodicQuotientGraph


def _graph(nodes, edges):
    return PeriodicQuotientGraph.from_edges(nodes, edges)


def _embed(points):
    return PeriodicEmbedding(
        tuple(
            (node_id, tuple(float(value) for value in point))
            for node_id, point in points
        ),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        fractional=tuple(
            (node_id, tuple(float(value) for value in point))
            for node_id, point in points
        ),
    )


def _ring(prefix, points):
    nodes = [f"{prefix}{index}" for index in range(len(points))]
    edges = [
        (f"{prefix}{index}", f"{prefix}{(index + 1) % len(points)}", (0, 0, 0))
        for index in range(len(points))
    ]
    return _graph(nodes, edges), _embed(list(zip(nodes, points)))


def _square():
    return _ring(
        "a",
        (
            (0.25, 0.25, 0.5),
            (0.75, 0.25, 0.5),
            (0.75, 0.75, 0.5),
            (0.25, 0.75, 0.5),
        ),
    )


def _hopf_partner():
    return _ring(
        "b",
        (
            (0.5, 0.5, 0.25),
            (0.0, 0.5, 0.25),
            (0.0, 0.5, 0.75),
            (0.5, 0.5, 0.75),
        ),
    )


def _translated_partner():
    return _ring(
        "b",
        (
            (0.5, 1.0, 0.25),
            (0.0, 1.0, 0.25),
            (0.0, 1.0, 0.75),
            (0.5, 1.0, 0.75),
        ),
    )


def _limits(**overrides):
    values = NetAnalysisLimits.from_config().to_dict()
    values.update(overrides)
    return NetAnalysisLimits(**values)


def test_hopf_rings_are_linked_and_a_translate_is_not():
    left, left_embedding = _square()
    linked, linked_embedding = _hopf_partner()
    apart, apart_embedding = _translated_partner()
    hopf = assess_linking(left, left_embedding, linked, linked_embedding)
    assert hopf.status == "linked", hopf.reason
    assert hopf.witness is not None
    assert abs(int(hopf.witness["linking_number"])) == 1
    separated = assess_linking(left, left_embedding, apart, apart_embedding)
    assert separated.status == "coexistence"
    assert separated.relation == "graph_disjoint"


def test_integer_image_of_a_hopf_link_stays_linked():
    left, left_embedding = _square()
    right, right_embedding = _hopf_partner()
    shifted = _embed(
        [
            (node_id, (point[0] + 2.0, point[1], point[2]))
            for node_id, point in right_embedding.fractional
        ]
    )
    linked = assess_linking(left, left_embedding, right, shifted)
    assert linked.status == "linked", linked.reason
    assert linked.witness is not None
    assert abs(int(linked.witness["linking_number"])) == 1
    assert linked.witness["relative_image"] == [-2, 0, 0]
    capped = assess_linking(
        left,
        left_embedding,
        right,
        shifted,
        limits=_limits(link_image_limit=0),
    )
    assert capped.status in {"unverified", "not_detected_within_bounds"}
    assert capped.status != "linked"
    assert capped.status != "coexistence"


def test_equal_abstract_rings_can_have_different_embeddings():
    left, left_embedding = _square()
    partner, partner_embedding = _hopf_partner()
    moved, moved_embedding = _translated_partner()
    from molcrys_kit.nets.isomorphism import compare_periodic_nets

    same = compare_periodic_nets(partner, moved)
    assert same.status == "isomorphic"
    assert (
        assess_linking(left, left_embedding, partner, partner_embedding).status
        == "linked"
    )
    assert (
        assess_linking(left, left_embedding, moved, moved_embedding).status
        == "coexistence"
    )


def test_shared_vertices_are_not_a_link():
    graph, embedding = _square()
    assessment = assess_linking(
        graph, embedding, graph, embedding, same_vertex_namespace=True
    )
    assert assessment.status == "invalid_or_indeterminate"
    assert assessment.relation == "shared_vertices"


def test_projection_crossing_of_trees_is_coexistence():
    left = _graph(("a", "b"), [("ab", "a", "b", (0, 0, 0))])
    right = _graph(("c", "d"), [("cd", "c", "d", (0, 0, 0))])
    left_embedding = _embed((("a", (0.0, 0.5, 0.0)), ("b", (1.0, 0.5, 1.0))))
    right_embedding = _embed((("c", (0.5, 0.0, 0.25)), ("d", (0.5, 1.0, 0.75))))
    assessment = assess_linking(left, left_embedding, right, right_embedding)
    assert assessment.status == "coexistence", assessment.reason
    assert assessment.reason in {"acyclic", "coplanar_disjoint"}


def test_link_classification_survives_a_shared_skew_lattice_and_relabeling():
    import numpy as np

    from molcrys_kit.periodic import relabel_nodes

    lattice = ((1.0, 0.25, 0.0), (0.1, 1.0, 0.0), (0.0, 0.05, 1.0))
    matrix = np.asarray(lattice, dtype=float)

    def embed(points):
        rows = []
        fractional = []
        for node_id, point in points:
            frac = tuple(float(value) for value in point)
            cart = tuple(float(value) for value in np.asarray(frac) @ matrix)
            rows.append((node_id, cart))
            fractional.append((node_id, frac))
        return PeriodicEmbedding(tuple(rows), lattice, fractional=tuple(fractional))

    left, _left_embedding = _square()
    right, right_embedding = _hopf_partner()
    renamed = relabel_nodes(
        right, {node_id: f"n-{node_id}" for node_id in right.node_ids}
    ).graph
    square_points = list(_square()[1].fractional)
    partner_points = [
        (f"n-{node_id}", point) for node_id, point in right_embedding.fractional
    ]
    assessment = assess_linking(
        left,
        embed(square_points),
        renamed,
        embed(partner_points),
    )
    assert assessment.status == "linked", assessment.reason
    assert assessment.witness is not None
    assert abs(int(assessment.witness["linking_number"])) == 1


def test_inexact_coordinates_are_not_a_linking_number():
    left = _graph(("a", "b"), [("ab", "a", "b", (0, 0, 0))])
    right = _graph(("c", "d"), [("cd", "c", "d", (0, 0, 0))])
    assessment = assess_linking(
        left,
        _embed((("a", (0.0, 0.0, 0.0)), ("b", (0.1, 0.0, 0.0)))),
        right,
        _embed((("c", (0.0, 0.5, 0.0)), ("d", (0.5, 0.5, 0.0)))),
        limits=_limits(fraction_denominator_limit=2, fraction_residual_A=0.0),
    )
    assert assessment.status == "invalid_or_indeterminate"
    assert assessment.reason == "inexact_geometry"


def _shifted_hopf_partner():
    """Same Hopf geometry, with the return edge stored in the next image."""
    nodes = ("b0", "b1", "b2", "b3")
    points = (
        (0.5, 0.5, 0.25),
        (-1.0, 0.5, 0.25),
        (-1.0, 0.5, 0.75),
        (0.5, 0.5, 0.75),
    )
    edges = [
        ("b0", "b1", (1, 0, 0)),
        ("b1", "b2", (0, 0, 0)),
        ("b2", "b3", (-1, 0, 0)),
        ("b3", "b0", (0, 0, 0)),
    ]
    return _graph(nodes, edges), _embed(list(zip(nodes, points)))


def test_image_window_caps_a_cover_cycle_that_leaves_the_origin():
    left, left_embedding = _square()
    right, right_embedding = _shifted_hopf_partner()
    linked = assess_linking(left, left_embedding, right, right_embedding)
    assert linked.status == "linked", linked.reason
    assert linked.witness is not None
    assert abs(int(linked.witness["linking_number"])) == 1
    capped = assess_linking(
        left,
        left_embedding,
        right,
        right_embedding,
        limits=_limits(link_image_limit=0),
    )
    assert capped.status in {"unverified", "not_detected_within_bounds"}
    assert capped.status != "linked"
    assert capped.status != "coexistence"


def test_overlapping_labels_do_not_hide_a_hopf_link():
    """Independent namespaces may reuse ids. That is not a shared vertex.

    Relabeling the Hopf partner from ``b0``…``b3`` onto ``a0``…``a3`` leaves
    the geometry linked. Skipping the only cycle pair because the labels match
    would report vanishing linking numbers instead.
    """
    from molcrys_kit.periodic import relabel_nodes

    left, left_embedding = _square()
    right, right_embedding = _hopf_partner()
    renamed = relabel_nodes(
        right, {f"b{index}": f"a{index}" for index in range(4)}
    ).graph
    points = [
        (f"a{index}", point)
        for index, (_node_id, point) in enumerate(right_embedding.fractional)
    ]
    assessment = assess_linking(
        left,
        left_embedding,
        renamed,
        _embed(points),
        same_vertex_namespace=False,
    )
    assert assessment.status == "linked", assessment.reason
    assert assessment.witness is not None
    assert abs(int(assessment.witness["linking_number"])) == 1


def _normalize_gauss(codes):
    """Rename crossings by first appearance, keeping over/under signs."""
    mapping = {}
    normalized = []
    for component in codes:
        row = []
        for entry in component:
            label = abs(entry)
            if label not in mapping:
                mapping[label] = len(mapping) + 1
            sign = 1 if entry > 0 else -1
            row.append(sign * mapping[label])
        normalized.append(tuple(row))
    return tuple(normalized)


def _segments(polygon):
    return [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]


def _cross2(left, right):
    return left[0] * right[1] - left[1] * right[0]


def _projection_hit(start, end, other_start, other_end):
    """Interior xy crossing, or ``None``. Heights are the two z values."""
    from fractions import Fraction

    delta = (end[0] - start[0], end[1] - start[1])
    other_delta = (other_end[0] - other_start[0], other_end[1] - other_start[1])
    denominator = _cross2(delta, other_delta)
    if denominator == 0:
        return None
    offset = (other_start[0] - start[0], other_start[1] - start[1])
    tee = Fraction(_cross2(offset, other_delta), denominator)
    you = Fraction(_cross2(offset, delta), denominator)
    if not (0 < tee < 1 and 0 < you < 1):
        return None
    height = start[2] + tee * (end[2] - start[2])
    other_height = other_start[2] + you * (other_end[2] - other_start[2])
    if height == other_height:
        raise AssertionError("projection crossing is not generic")
    return tee, height > other_height


def _segments_meet(start, end, other_start, other_end):
    """True when two segments have a point in both relative interiors."""
    from fractions import Fraction

    delta = tuple(end[index] - start[index] for index in range(3))
    other = tuple(other_end[index] - other_start[index] for index in range(3))
    normal = (
        delta[1] * other[2] - delta[2] * other[1],
        delta[2] * other[0] - delta[0] * other[2],
        delta[0] * other[1] - delta[1] * other[0],
    )
    offset = tuple(start[index] - other_start[index] for index in range(3))
    if sum(offset[index] * normal[index] for index in range(3)) != 0:
        return False
    for drop in range(3):
        axes = [index for index in range(3) if index != drop]

        def project(point, axes=axes):
            return (point[axes[0]], point[axes[1]])

        left, right = project(start), project(end)
        other_left, other_right = project(other_start), project(other_end)
        span = (right[0] - left[0], right[1] - left[1])
        other_span = (
            other_right[0] - other_left[0],
            other_right[1] - other_left[1],
        )
        denominator = _cross2(span, other_span)
        if denominator == 0:
            continue
        begin = (other_left[0] - left[0], other_left[1] - left[1])
        tee = Fraction(_cross2(begin, other_span), denominator)
        you = Fraction(_cross2(begin, span), denominator)
        if not (0 < tee < 1 and 0 < you < 1):
            continue
        point = tuple(start[index] + tee * delta[index] for index in range(3))
        other_point = tuple(
            other_start[index] + you * other[index] for index in range(3)
        )
        if point == other_point:
            return True
    return False


def _whitehead_polygons():
    """Integer model of Knot Atlas ``L5a1`` before the fractional embedding.

    Component A is the square. Component B clasps it. Crossing order and
    over/under signs reproduce the published Gauss code
    ``{1, -4, 5, -3}, {3, -1, 2, -5, 4, -2}`` up to crossing labels.
    A positive scale into the unit cell preserves the diagram.
    """
    from fractions import Fraction

    square = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(4), Fraction(4), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(0)),
    )
    clasp = (
        (Fraction(-2), Fraction(2), Fraction(1)),
        (Fraction(1), Fraction(2), Fraction(1)),
        (Fraction(2), Fraction(-1), Fraction(-1)),
        (Fraction(2), Fraction(-2), Fraction(1)),
        (Fraction(2), Fraction(-5), Fraction(1)),
        (Fraction(-3), Fraction(-5), Fraction(1)),
        (Fraction(-3), Fraction(5), Fraction(1)),
        (Fraction(2), Fraction(5), Fraction(-1)),
        (Fraction(2), Fraction(3), Fraction(-1)),
        (Fraction(5), Fraction(2), Fraction(1)),
        (Fraction(5), Fraction(-3), Fraction(-1)),
        (Fraction(-2), Fraction(-3), Fraction(-1)),
    )
    return square, clasp


def _diagram_gauss(left, right):
    """Signed Gauss code. A positive entry is an overcrossing on that walk."""
    crossings = []
    left_edges = _segments(left)
    right_edges = _segments(right)
    for left_index, (start, end) in enumerate(left_edges):
        for right_index, (other_start, other_end) in enumerate(right_edges):
            hit = _projection_hit(start, end, other_start, other_end)
            if hit is None:
                continue
            tee, left_over = hit
            crossings.append(("inter", left_index, tee, right_index, left_over))
    self_hits = []
    for first in range(len(right_edges)):
        for second in range(first + 1, len(right_edges)):
            adjacent = second == first + 1 or {first, second} == {
                0,
                len(right_edges) - 1,
            }
            if adjacent:
                continue
            start, end = right_edges[first]
            other_start, other_end = right_edges[second]
            hit = _projection_hit(start, end, other_start, other_end)
            if hit is None:
                continue
            tee, first_over = hit
            self_hits.append((first, tee, second, first_over))
    # Number inter-crossings in the order component A meets them, then self-crossings.
    ordered = sorted(crossings, key=lambda item: (item[1], item[2]))
    labels = {}
    next_label = 1
    for _kind, left_index, _tee, right_index, _left_over in ordered:
        labels[("inter", left_index, right_index)] = next_label
        next_label += 1
    for first, _tee, second, _first_over in self_hits:
        labels[("self", first, second)] = next_label
        next_label += 1
    left_code = []
    for left_index, tee, right_index, left_over in (
        (item[1], item[2], item[3], item[4]) for item in ordered
    ):
        label = labels[("inter", left_index, right_index)]
        left_code.append(label if left_over else -label)
    right_events = []
    for left_index, tee, right_index, left_over in (
        (item[1], item[2], item[3], item[4]) for item in ordered
    ):
        # Parameter along the right edge was not stored. Recompute for sorting.
        start, end = left_edges[left_index]
        other_start, other_end = right_edges[right_index]
        hit = _projection_hit(other_start, other_end, start, end)
        assert hit is not None
        right_tee, right_over = hit
        label = labels[("inter", left_index, right_index)]
        right_events.append((right_index, right_tee, label if right_over else -label))
    for first, tee, second, first_over in self_hits:
        label = labels[("self", first, second)]
        start, end = right_edges[first]
        other_start, other_end = right_edges[second]
        right_events.append((first, tee, label if first_over else -label))
        other_hit = _projection_hit(other_start, other_end, start, end)
        assert other_hit is not None
        other_tee, second_over = other_hit
        right_events.append((second, other_tee, label if second_over else -label))
    right_events.sort()
    return tuple(left_code), tuple(entry for _index, _tee, entry in right_events)


def _fractional_whitehead():
    """Place the integer diagram in one cell so only the zero image can meet."""
    from fractions import Fraction

    square, clasp = _whitehead_polygons()

    def place(point):
        # Positive diagonal scaling. Denominators are powers of two.
        return tuple(
            float(Fraction(point[index] + (4, 6, 3)[index], 32)) for index in range(3)
        )

    return _ring("a", tuple(place(point) for point in square)), _ring(
        "b", tuple(place(point) for point in clasp)
    )


def test_whitehead_link_is_not_proved_coexistence():
    """Linking number zero is not an unlinking certificate.

    The polygons realize Knot Atlas ``L5a1``, the Whitehead link: linking
    number zero, and not the unlink. A finished search must stay unverified.
    """
    square, clasp = _whitehead_polygons()
    for left_index, (start, end) in enumerate(_segments(square)):
        for right_index, (other_start, other_end) in enumerate(_segments(clasp)):
            assert not _segments_meet(start, end, other_start, other_end), (
                left_index,
                right_index,
            )
    atlas = ((1, -4, 5, -3), (3, -1, 2, -5, 4, -2))
    assert _normalize_gauss(_diagram_gauss(square, clasp)) == _normalize_gauss(atlas)
    (left, left_embedding), (right, right_embedding) = _fractional_whitehead()
    assessment = assess_linking(
        left,
        left_embedding,
        right,
        right_embedding,
        limits=_limits(
            ring_length_limit=32,
            ring_count_limit=16,
            ring_state_limit=20000,
            link_pair_limit=64,
            link_image_limit=2,
        ),
    )
    assert assessment.status == "unverified", assessment.reason
    assert assessment.status != "coexistence"
    assert assessment.status != "linked"
    assert assessment.reason == "linking_numbers_vanish"
    assert assessment.witness is not None
    assert assessment.witness["max_abs_linking_number"] == 0
    assert assessment.witness["proves_unlinking"] is False


def test_capped_ring_search_does_not_prove_absence():
    left, left_embedding = _square()
    right, right_embedding = _hopf_partner()
    assessment = assess_linking(
        left,
        left_embedding,
        right,
        right_embedding,
        limits=_limits(ring_length_limit=3),
    )
    assert assessment.status in {"unverified", "not_detected_within_bounds"}
    assert assessment.status != "linked"
    assert assessment.status != "coexistence"
