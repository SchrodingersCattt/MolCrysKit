"""Canonical periodic quotient multigraph."""

from __future__ import annotations

import pytest

from molcrys_kit.periodic import (
    PeriodicEmbedding,
    PeriodicQuotientEdge,
    PeriodicQuotientGraph,
    PeriodicQuotientView,
    reversed_edge,
)
from molcrys_kit.periodic._lattice import adjugate, det3, matmul


def _graph(*edges, nodes=("a", "b"), pbc=(True, True, True)):
    return PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)


def test_parallel_and_self_image_edges_keep_distinct_ids():
    graph = _graph(
        ("loop", "a", "a", (2, 0, 0)),
        ("also", "a", "a", (2, 0, 0)),
        ("span", "a", "b", (0, 0, 0)),
        ("span-image", "a", "b", (1, 0, 0)),
        nodes=("a", "b"),
    )
    assert len(graph.edges) == 4
    assert {edge.edge_id for edge in graph.edges} == {
        "loop",
        "also",
        "span",
        "span-image",
    }
    loops = [edge for edge in graph.edges if edge.source == edge.target]
    assert [edge.image_shift for edge in loops] == [(2, 0, 0), (2, 0, 0)]


def test_reversal_and_input_order_do_not_change_the_stored_graph():
    forward = _graph(("e", "a", "b", (1, -2, 3)), ("f", "b", "a", (0, 1, 0)))
    backward = PeriodicQuotientGraph.from_edges(
        ("b", "a"),
        (
            reversed_edge(PeriodicQuotientEdge("e", "a", "b", (1, -2, 3))),
            ("f", "a", "b", (0, -1, 0)),
        ),
    )
    assert backward == forward
    assert isinstance(forward, PeriodicQuotientView)
    stored = forward.edge("e")
    assert stored.source == "a"
    assert stored.target == "b"
    assert stored.image_shift == (1, -2, 3)


def test_self_image_orientation_keeps_nonprimitive_winding():
    graph = _graph(("wind", "a", "a", (-2, 0, 0)), nodes=("a",))
    assert graph.edge("wind").image_shift == (2, 0, 0)
    assert _graph(("wind", "a", "a", (2, 0, 0)), nodes=("a",)) == graph


def test_generated_ids_follow_the_undirected_edge():
    left = PeriodicQuotientGraph.from_edges(("a", "b"), [("b", "a", (-1, 0, 0))])
    right = PeriodicQuotientGraph.from_edges(("b", "a"), [("a", "b", (1, 0, 0))])
    assert left == right
    assert left.edges[0].edge_id == "edge:a->b@1_0_0#0"


def test_rejects_invalid_endpoints_shifts_and_pbc():
    with pytest.raises(ValueError, match="unknown node"):
        PeriodicQuotientGraph.from_edges(("a",), [("a", "b", (0, 0, 0))])
    with pytest.raises(ValueError, match="unique"):
        PeriodicQuotientGraph.from_edges(("a", "a"), [])
    with pytest.raises(ValueError, match="unique"):
        PeriodicQuotientGraph.from_edges(
            ("a",),
            [("e", "a", "a", (0, 0, 0)), ("e", "a", "a", (1, 0, 0))],
        )
    with pytest.raises(TypeError, match="exact integers"):
        PeriodicQuotientGraph.from_edges(("a",), [("a", "a", (1.0, 0, 0))])
    with pytest.raises(TypeError, match="exact integers"):
        PeriodicQuotientGraph.from_edges(("a",), [("a", "a", (True, 0, 0))])
    with pytest.raises(ValueError, match="non-periodic"):
        PeriodicQuotientGraph.from_edges(
            ("a",), [("a", "a", (0, 1, 0))], pbc=(True, False, False)
        )
    with pytest.raises(TypeError, match="booleans"):
        PeriodicQuotientGraph.from_edges(("a",), [], pbc=(1, 0, 0))


def test_embedding_is_independent_of_edge_shifts():
    graph = _graph(("e", "a", "a", (2, 0, 0)), nodes=("a",))
    orthogonal = PeriodicEmbedding(
        (("a", (1.0, 2.0, 3.0)),),
        ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    skew = PeriodicEmbedding(
        (("a", (1.0, 2.0, 3.0)),),
        ((10.0, 0.0, 0.0), (2.0, 9.0, 0.0), (1.0, 0.5, 8.0)),
    )
    assert graph.edge("e").image_shift == (2, 0, 0)
    assert orthogonal.fractional_position("a") == pytest.approx((0.1, 0.2, 0.3))
    assert skew.position("a") == pytest.approx((1.0, 2.0, 3.0))
    assert orthogonal.to_dict()["lattice_A"][0] == [10.0, 0.0, 0.0]


def test_adjugate_matches_the_determinant():
    matrix = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    product = matmul(matrix, adjugate(matrix))
    assert product == (
        (det3(matrix), 0, 0),
        (0, det3(matrix), 0),
        (0, 0, det3(matrix)),
    )
