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
