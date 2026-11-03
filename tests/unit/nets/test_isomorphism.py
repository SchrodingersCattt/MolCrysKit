"""Exact periodic-net isomorphism: witnesses, refutations, and capped search."""

from __future__ import annotations

from molcrys_kit.nets.isomorphism import (
    characterize_periodic_net,
    compare_periodic_nets,
    verify_topology_witness,
)
from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.periodic import (
    PeriodicEmbedding,
    PeriodicQuotientGraph,
    apply_node_gauge,
    apply_unimodular_basis,
    equivalent_supercell,
    relabel_nodes,
)
from molcrys_kit.periodic.algorithms import analyze_periodic_graph


def _graph(nodes, edges, pbc=(True, True, True)):
    return PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)


def _sql():
    return _graph(("a",), [("ax", "a", "a", (1, 0, 0)), ("ay", "a", "a", (0, 1, 0))])


def _chain(shift=(1, 0, 0), pbc=(True, True, True)):
    return _graph(("a",), [("wind", "a", "a", shift)], pbc=pbc)


def _dia():
    return _graph(
        ("u", "v"),
        [
            ("e0", "u", "v", (0, 0, 0)),
            ("ex", "u", "v", (1, 0, 0)),
            ("ey", "u", "v", (0, 1, 0)),
            ("ez", "u", "v", (0, 0, 1)),
        ],
    )


def _embedding(node_ids):
    return PeriodicEmbedding(
        tuple((node_id, (0.0, 0.0, 0.0)) for node_id in node_ids),
        ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )


def _limits(**overrides):
    base = NetAnalysisLimits.from_config()
    values = base.to_dict()
    values.update(overrides)
    return NetAnalysisLimits(**values)


def test_square_lattice_matches_itself_and_a_relabeling():
    graph = _sql()
    same = compare_periodic_nets(graph, graph)
    assert same.status == "isomorphic"
    assert same.proof == "explicit_isomorphism"
    assert verify_topology_witness(graph, graph, same)
    renamed = relabel_nodes(graph, {"a": "origin"}).graph
    compared = compare_periodic_nets(graph, renamed)
    assert compared.status == "isomorphic"
    assert compared.left.canonical_key == compared.right.canonical_key
    assert compared.left.periodic_rank == 2
    assert compared.left.quotient_cycle_rank == 2


def test_gauge_unimodular_and_supercell_preserve_square_lattice_identity():
    graph = _sql()
    gauged = apply_node_gauge(graph, {"a": (3, -1, 4)}).graph
    changed = apply_unimodular_basis(graph, ((1, 2, 0), (0, 1, 0), (0, 0, 1))).graph
    supercell = equivalent_supercell(
        graph, ((2, 0, 0), (0, 1, 0), (0, 0, 1)), _embedding(("a",))
    ).graph
    for other in (gauged, changed, supercell):
        compared = compare_periodic_nets(graph, other)
        assert compared.status == "isomorphic", (
            compared.status,
            compared.reason,
            other.to_dict(),
        )
        assert compared.proof == "explicit_isomorphism"
        assert verify_topology_witness(graph, other, compared)


def test_content_two_chain_matches_primitive_chain():
    coarse = _chain((2, 0, 0), pbc=(True, False, False))
    fine = _chain((1, 0, 0), pbc=(True, False, False))
    compared = compare_periodic_nets(coarse, fine)
    assert compared.status == "isomorphic", compared.reason
    assert compared.left.periodic_rank == 1
    assert analyze_periodic_graph(coarse).quotient_cycle_rank == 1
    assert compared.left.components[0].multiplicity == 2
    assert compared.right.components[0].multiplicity == 1


def test_diamond_quotient_is_stable_under_relabel_and_not_the_square_lattice():
    graph = _dia()
    swapped = relabel_nodes(graph, {"u": "b", "v": "a"}).graph
    same = compare_periodic_nets(graph, swapped)
    assert same.status == "isomorphic"
    assert same.proof == "explicit_isomorphism"
    different = compare_periodic_nets(graph, _sql())
    assert different.status == "non_isomorphic"
    assert different.proof == "invariant_mismatch"


def test_locally_similar_rank_two_net_is_not_the_square_lattice():
    other = _graph(
        ("u", "v"),
        [
            ("bridge0", "u", "v", (0, 0, 0)),
            ("bridge1", "u", "v", (1, 0, 0)),
            ("loop_u", "u", "u", (0, 1, 0)),
            ("loop_v", "v", "v", (1, 1, 0)),
        ],
    )
    compared = compare_periodic_nets(_sql(), other)
    assert compared.status == "non_isomorphic"
    assert compared.proof in {"invariant_mismatch", "exhaustive_search"}


def test_capped_search_does_not_promote_a_matching_fingerprint():
    graph = _sql()
    renamed = relabel_nodes(graph, {"a": "b"}).graph
    capped = compare_periodic_nets(
        graph, renamed, _limits(backtrack_limit=0, basis_entry_limit=0)
    )
    assert capped.status == "undetermined"
    assert capped.proof == "none"
    assert capped.left.coordination_census == capped.right.coordination_census


def test_rank_mismatch_is_a_proof_even_when_the_search_budget_is_zero():
    compared = compare_periodic_nets(
        _sql(),
        _chain(),
        _limits(backtrack_limit=0, basis_entry_limit=0),
    )
    assert compared.status == "non_isomorphic"
    assert compared.proof == "invariant_mismatch"
    assert compared.reason == "periodic_rank"


def test_finite_square_cycle_rank_is_not_periodic_rank():
    square = _graph(
        ("a", "b", "c", "d"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("bc", "b", "c", (0, 0, 0)),
            ("cd", "c", "d", (0, 0, 0)),
            ("da", "d", "a", (0, 0, 0)),
        ],
    )
    record = characterize_periodic_net(square)
    assert record.status == "identified"
    assert record.periodic_rank == 0
    assert record.quotient_cycle_rank == 1
    assert record.components[0].quotient_cycle_rank == 1


def test_parallel_chains_do_not_pool_their_ranks():
    graph = _graph(
        ("a", "b"),
        [("ax", "a", "a", (1, 0, 0)), ("by", "b", "b", (0, 1, 0))],
    )
    record = characterize_periodic_net(graph)
    assert record.periodic_rank == 1
    assert sorted(component.periodic_rank for component in record.components) == [1, 1]
    assert record.quotient_cycle_rank == 2


def test_edge_order_does_not_change_the_canonical_key():
    forward = _sql()
    reverse = _graph(("a",), [("ay", "a", "a", (0, 1, 0)), ("ax", "a", "a", (1, 0, 0))])
    compared = compare_periodic_nets(forward, reverse)
    assert compared.status == "isomorphic"
    assert compared.proof == "explicit_isomorphism"
    assert compared.left.canonical_key == compared.right.canonical_key


def test_empty_graph_is_identified():
    empty = _graph((), [])
    record = characterize_periodic_net(empty)
    assert record.status == "identified"
    assert record.canonical_key == "n0|r0|"
    assert record.periodic_rank == 0
    assert record.quotient_cycle_rank == 0
