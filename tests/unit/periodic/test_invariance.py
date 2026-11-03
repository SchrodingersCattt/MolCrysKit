"""Rank and translation lattice under reordering, gauge, cell, and supercell changes."""

from __future__ import annotations

import pytest

from molcrys_kit.chemistry import integer_translation_basis
from molcrys_kit.periodic import (
    PeriodicEmbedding,
    PeriodicQuotientGraph,
    analyze_periodic_graph,
    apply_node_gauge,
    apply_unimodular_basis,
    cover_multiplicity,
    equivalent_supercell,
    periodic_rank,
    relabel_nodes,
    replace_lattice,
    shift_embedding_origin,
)
from molcrys_kit.periodic._lattice import vecmul


def _signature(graph):
    analysis = analyze_periodic_graph(graph)
    return (
        analysis.periodic_rank,
        analysis.quotient_cycle_rank,
        tuple(family.subgroup.basis for family in analysis.cover_families),
        tuple(family.subgroup.multiplicity for family in analysis.cover_families),
    )


def _connected():
    return PeriodicQuotientGraph.from_edges(
        ("a", "b"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("ax", "a", "a", (2, 0, 0)),
            ("by", "b", "b", (0, 3, 0)),
            ("ring", "a", "a", (0, 0, 0)),
        ],
    )


def _embedding():
    return PeriodicEmbedding(
        (("a", (0.0, 0.0, 0.0)), ("b", (4.0, 1.0, 0.0))),
        ((10.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 8.0)),
    )


def _parent_basis(graph, matrix):
    vectors = [
        vecmul(vector, matrix)
        for family in analyze_periodic_graph(graph).cover_families
        for vector in family.subgroup.basis
    ]
    return integer_translation_basis(vectors)


def test_reordering_nodes_and_edges_is_the_same_graph():
    graph = _connected()
    shuffled = PeriodicQuotientGraph.from_edges(
        ("b", "a"),
        [
            ("ring", "a", "a", (0, 0, 0)),
            ("by", "b", "b", (0, -3, 0)),
            ("ab", "b", "a", (0, 0, 0)),
            ("ax", "a", "a", (-2, 0, 0)),
        ],
    )
    assert shuffled == graph
    assert _signature(graph)[0] == 2
    assert _signature(graph)[1] == 3
    assert _signature(graph)[2] == (((2, 0, 0), (0, 3, 0)),)


def test_relabel_and_origin_shift_preserve_the_integer_graph():
    graph = _connected()
    embedding = _embedding()
    relabeled = relabel_nodes(graph, {"a": "n2", "b": "n1"}, embedding)
    assert _signature(relabeled.graph) == _signature(graph)
    assert set(relabeled.graph.node_ids) == {"n1", "n2"}
    shifted = shift_embedding_origin(embedding, (0.25, -0.5, 1.0))
    assert shifted.fractional_position("a") == pytest.approx((-0.25, 0.5, -1.0))
    assert graph == _connected()
    assert _signature(graph) == _signature(_connected())


def test_node_gauge_changes_bridge_shifts_and_keeps_self_image_winding():
    graph = _connected()
    gauged = apply_node_gauge(graph, {"b": (5, 0, 0)}, _embedding())
    assert gauged.graph.edge("ax").image_shift == (2, 0, 0)
    assert gauged.graph.edge("ab").image_shift != (0, 0, 0)
    assert _signature(gauged.graph) == _signature(graph)
    assert gauged.embedding is not None
    assert gauged.embedding.fractional_position("b")[0] == pytest.approx(
        _embedding().fractional_position("b")[0] + 5
    )
    cycled = PeriodicQuotientGraph.from_edges(
        ("a", "b"),
        [("tree", "a", "b", (0, 0, 0)), ("wind", "a", "b", (2, 0, 0))],
    )
    moved = apply_node_gauge(cycled, {"a": (0, 1, 0), "b": (3, -4, 1)})
    assert analyze_periodic_graph(moved.graph).cover_families[0].subgroup.basis == (
        (2, 0, 0),
    )


def test_skew_cell_does_not_change_integer_shifts():
    graph = _connected()
    embedding = _embedding()
    skew = replace_lattice(
        embedding,
        ((10.0, 0.0, 0.0), (3.0, 9.0, 0.0), (1.0, -2.0, 8.0)),
    )
    assert skew.fractional_position("b") == pytest.approx(
        embedding.fractional_position("b")
    )
    assert skew.position("b") != pytest.approx(embedding.position("b"))
    assert _signature(graph) == _signature(_connected())
    assert graph.edge("ax").image_shift == (2, 0, 0)


def test_unimodular_basis_change_preserves_the_parent_lattice():
    graph = _connected()
    change = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    transformed = apply_unimodular_basis(graph, change, _embedding())
    assert transformed.graph != graph
    assert _parent_basis(transformed.graph, change) == _signature(graph)[2][0]
    inverted = apply_unimodular_basis(graph, ((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert _parent_basis(inverted.graph, ((-1, 0, 0), (0, 1, 0), (0, 0, 1))) == (
        (2, 0, 0),
        (0, 3, 0),
    )
    assert transformed.embedding is not None
    assert transformed.embedding.position("a") == pytest.approx((0.0, 0.0, 0.0))


def test_supercell_preserves_nonprimitive_winding():
    nonprimitive = PeriodicQuotientGraph.from_edges(
        ("a",), [("wind", "a", "a", (2, 0, 0))]
    )
    primitive = PeriodicQuotientGraph.from_edges(
        ("a",), [("wind", "a", "a", (1, 0, 0))]
    )
    matrix = ((2, 0, 0), (0, 1, 0), (0, 0, 1))
    coarse = equivalent_supercell(nonprimitive, matrix, _embedding_one())
    fine = equivalent_supercell(primitive, matrix, _embedding_one())
    assert periodic_rank(coarse.graph) == periodic_rank(fine.graph) == 1
    assert len(analyze_periodic_graph(coarse.graph).quotient_components) == 2
    assert len(analyze_periodic_graph(fine.graph).quotient_components) == 1
    assert _parent_basis(coarse.graph, matrix) == ((2, 0, 0),)
    for family in analyze_periodic_graph(coarse.graph).cover_families:
        assert family.subgroup.basis == ((1, 0, 0),)
        assert coarse.replication.parent_shift(family.subgroup.basis[0]) == (2, 0, 0)
    image = "a\x1f1,0,0"
    assert coarse.embedding is not None
    assert coarse.embedding.position(image) == pytest.approx((10.0, 0.0, 0.0))


def test_full_rank_multiplicity_is_invariant_under_equivalent_supercells():
    index_two = PeriodicQuotientGraph.from_edges(
        ("a",),
        [
            ("x", "a", "a", (2, 0, 0)),
            ("y", "a", "a", (0, 1, 0)),
            ("z", "a", "a", (0, 0, 1)),
        ],
    )
    primitive = PeriodicQuotientGraph.from_edges(
        ("a",),
        [
            ("x", "a", "a", (1, 0, 0)),
            ("y", "a", "a", (0, 1, 0)),
            ("z", "a", "a", (0, 0, 1)),
        ],
    )
    matrices = (
        ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((2, 1, 0), (0, 1, 0), (0, 0, 1)),
        ((-2, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    assert cover_multiplicity(index_two) == 2
    assert cover_multiplicity(primitive) == 1
    for matrix in matrices:
        assert cover_multiplicity(equivalent_supercell(index_two, matrix).graph) == 2
        assert cover_multiplicity(equivalent_supercell(primitive, matrix).graph) == 1
        assert periodic_rank(equivalent_supercell(index_two, matrix).graph) == 3


def _embedding_one():
    return PeriodicEmbedding(
        (("a", (0.0, 0.0, 0.0)),),
        ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
