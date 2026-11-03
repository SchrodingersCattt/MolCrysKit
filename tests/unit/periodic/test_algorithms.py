"""Periodic rank is not the quotient cycle rank."""

from __future__ import annotations

from molcrys_kit.chemistry import analyze_periodic_topology, integer_translation_basis
from molcrys_kit.chemistry.models import ChemicalBond
from molcrys_kit.periodic import (
    PeriodicQuotientGraph,
    analyze_periodic_graph,
    cover_multiplicity,
    periodic_rank,
    quotient_cycle_rank,
)


def _graph(nodes, edges, pbc=(True, True, True)):
    return PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)


def test_face_crossing_tree_has_cycle_rank_zero_and_periodic_rank_zero():
    graph = _graph(("a", "b"), [("a", "b", (1, 0, 0))])
    analysis = analyze_periodic_graph(graph)
    assert quotient_cycle_rank(graph) == 0
    assert periodic_rank(graph) == 0
    assert analysis.cover_families[0].closed_walks == ()
    assert analysis.cover_families[0].subgroup.basis == ()


def test_zero_shift_loop_is_a_quotient_cycle_of_rank_zero():
    graph = _graph(("a",), [("ring", "a", "a", (0, 0, 0))])
    analysis = analyze_periodic_graph(graph)
    assert analysis.quotient_cycle_rank == 1
    assert analysis.periodic_rank == 0
    assert analysis.cover_families[0].closed_walks[0].translation == (0, 0, 0)
    assert cover_multiplicity(graph) is None


def test_self_image_edges_distinguish_dimension_without_reducing_winding():
    for expected, shifts in (
        (1, [(2, 0, 0)]),
        (2, [(2, 0, 0), (0, 3, 0)]),
        (3, [(2, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ):
        graph = _graph(
            ("a",), [(f"e{i}", "a", "a", shift) for i, shift in enumerate(shifts)]
        )
        family = analyze_periodic_graph(graph).cover_families[0]
        assert family.subgroup.rank == expected
        assert family.subgroup.basis == integer_translation_basis(shifts)
        assert family.quotient_cycle_rank == len(shifts)
    nonprimitive = _graph(("a",), [("wind", "a", "a", (2, 0, 0))])
    assert analyze_periodic_graph(nonprimitive).cover_families[0].subgroup.basis == (
        (2, 0, 0),
    )


def test_dependent_generators_keep_both_descriptions():
    graph = _graph(
        ("a",),
        [
            ("two", "a", "a", (2, 0, 0)),
            ("three", "a", "a", (3, 0, 0)),
            ("minus", "a", "a", (-3, 0, 0)),
        ],
    )
    subgroup = analyze_periodic_graph(graph).cover_families[0].subgroup
    topology = analyze_periodic_topology(
        ("a",),
        (
            ChemicalBond("a", "a", atom2_image_shift=(2, 0, 0)),
            ChemicalBond("a", "a", atom2_image_shift=(3, 0, 0)),
            ChemicalBond("a", "a", atom2_image_shift=(-3, 0, 0)),
        ),
    )
    assert (
        subgroup.independent_generators
        == topology.translation_generators
        == ((2, 0, 0),)
    )
    assert subgroup.basis == ((1, 0, 0),)
    assert subgroup.rank == 1


def test_disconnected_chains_do_not_pool_their_translations():
    graph = _graph(
        ("a", "b"),
        [("a", "a", (1, 0, 0)), ("b", "b", (0, 1, 0))],
    )
    analysis = analyze_periodic_graph(graph)
    assert analysis.periodic_rank == 1
    assert [family.subgroup.basis for family in analysis.cover_families] == [
        ((1, 0, 0),),
        ((0, 1, 0),),
    ]
    assert analysis.quotient_cycle_rank == 2


def test_cover_multiplicity_counts_finite_index_copies():
    index_two = _graph(
        ("a",),
        [
            ("x", "a", "a", (2, 0, 0)),
            ("y", "a", "a", (0, 1, 0)),
            ("z", "a", "a", (0, 0, 1)),
        ],
    )
    assert cover_multiplicity(index_two) == 2
    primitive = _graph(
        ("a",),
        [
            ("x", "a", "a", (1, 0, 0)),
            ("y", "a", "a", (0, 1, 0)),
            ("z", "a", "a", (0, 0, 1)),
        ],
    )
    assert cover_multiplicity(primitive) == 1
    chain = _graph(("a",), [("x", "a", "a", (2, 0, 0))], pbc=(True, False, False))
    assert periodic_rank(chain) == 1
    assert cover_multiplicity(chain) == 2
    finite = _graph(("a", "b"), [("a", "b", (0, 0, 0))], pbc=(False, False, False))
    assert cover_multiplicity(finite) == 1
    assert analyze_periodic_graph(_graph((), [])).periodic_rank == 0
    assert cover_multiplicity(_graph((), [])) == 0


def test_parallel_edges_span_a_rank_two_cycle_space():
    graph = _graph(
        ("a", "b", "c"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("bc", "b", "c", (0, 0, 0)),
            ("ca", "c", "a", (1, 0, 0)),
            ("ab-y", "a", "b", (0, 1, 0)),
        ],
    )
    analysis = analyze_periodic_graph(graph)
    assert analysis.quotient_cycle_rank == 2
    assert analysis.periodic_rank == 2
    assert analysis.cover_families[0].subgroup.basis == integer_translation_basis(
        [(1, 0, 0), (0, 1, 0)]
    )
