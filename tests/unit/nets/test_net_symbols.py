"""Point and vertex symbols are cover-ring descriptors, not isomorphism proofs."""

from __future__ import annotations

from molcrys_kit.nets import periodic_net_symbols
from molcrys_kit.nets.isomorphism import compare_periodic_nets
from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.nets.report import analyze_periodic_net_system
from molcrys_kit.periodic import (
    PeriodicEmbedding,
    PeriodicQuotientGraph,
    apply_node_gauge,
    apply_unimodular_basis,
    equivalent_supercell,
    relabel_nodes,
)


def _graph(nodes, edges):
    return PeriodicQuotientGraph.from_edges(nodes, edges)


def _sql():
    return _graph(("a",), [("ax", "a", "a", (1, 0, 0)), ("ay", "a", "a", (0, 1, 0))])


def _pcu():
    return _graph(
        ("a",),
        [
            ("ax", "a", "a", (1, 0, 0)),
            ("ay", "a", "a", (0, 1, 0)),
            ("az", "a", "a", (0, 0, 1)),
        ],
    )


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


def _hcb():
    return _graph(
        ("u", "v"),
        [
            ("e0", "u", "v", (0, 0, 0)),
            ("ex", "u", "v", (1, 0, 0)),
            ("ey", "u", "v", (0, 1, 0)),
        ],
    )


def _limits(**overrides):
    values = NetAnalysisLimits.from_config().to_dict()
    values.update(overrides)
    return NetAnalysisLimits(**values)


def _signature(graph, limits=None):
    symbols = periodic_net_symbols(graph, limits)
    assert symbols.status == "computed", symbols.reason
    return symbols.signature


def test_standard_nets_have_their_combinatorial_symbols():
    assert _signature(_sql()) == (("4^4.6^2", "4.4.4.4.6_2.6_2", 1),)
    assert _signature(_pcu()) == (
        ("4^12.6^3", "4.4.4.4.4.4.4.4.4.4.4.4.6_4.6_4.6_4", 1),
    )
    assert _signature(_dia()) == (("6^6", "6_2.6_2.6_2.6_2.6_2.6_2", 1),)
    assert _signature(_hcb()) == (("6^3", "6.6.6", 1),)
    square = _graph(
        ("a", "b", "c", "d"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("bc", "b", "c", (0, 0, 0)),
            ("cd", "c", "d", (0, 0, 0)),
            ("da", "d", "a", (0, 0, 0)),
        ],
    )
    assert _signature(square) == (("4", "4", 1),)


def test_symbols_are_invariant_and_are_not_an_isomorphism_proof():
    graph = _sql()
    signature = _signature(graph)
    gauged = apply_node_gauge(graph, {"a": (3, -1, 4)}).graph
    changed = apply_unimodular_basis(graph, ((1, 2, 0), (0, 1, 0), (0, 0, 1))).graph
    reordered = _graph(
        ("a",), [("ay", "a", "a", (0, 1, 0)), ("ax", "a", "a", (1, 0, 0))]
    )
    renamed = relabel_nodes(_dia(), {"u": "b", "v": "a"}).graph
    embedding = PeriodicEmbedding(
        (("a", (0.0, 0.0, 0.0)),),
        ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    supercell = equivalent_supercell(
        graph, ((2, 0, 0), (0, 2, 0), (0, 0, 1)), embedding
    ).graph
    assert _signature(gauged) == signature
    assert _signature(changed) == signature
    assert _signature(reordered) == signature
    assert _signature(supercell) == signature
    assert len(supercell.nodes) == 4
    assert _signature(renamed) == _signature(_dia())
    capped = compare_periodic_nets(
        graph, relabel_nodes(graph, {"a": "b"}).graph, _limits(backtrack_limit=0)
    )
    assert capped.status == "undetermined"
    assert capped.proof == "none"
    assert capped.left.symbols.status == "computed"
    assert capped.left.symbols.signature == capped.right.symbols.signature == signature
    assert "4^4.6^2" not in (capped.left.canonical_key or "")


def test_unfinished_or_unsupported_ring_search_does_not_invent_a_symbol():
    chain = _graph(("a",), [("wind", "a", "a", (1, 0, 0))])
    missing = periodic_net_symbols(chain)
    assert missing.status == "not_computed"
    assert missing.reason == "ring_not_found_within_bounds"
    assert missing.signature is None
    assert missing.nodes[0].point_symbol is None
    assert missing.nodes[0].vertex_symbol is None
    short = periodic_net_symbols(_sql(), _limits(ring_length_limit=3))
    assert short.status == "not_computed"
    assert short.reason == "ring_not_found_within_bounds"
    assert short.signature is None
    crowded = periodic_net_symbols(_pcu(), _limits(ring_count_limit=1))
    assert crowded.status == "not_computed"
    assert crowded.reason == "ring_multiplicity_capped"
    assert crowded.nodes[0].point_symbol is None
    isolated = _graph(tuple(f"n{index}" for index in range(9)), [])
    outside = periodic_net_symbols(isolated)
    assert outside.status == "not_computed"
    assert outside.reason == "canonical_node_limit"
    assert outside.nodes == ()
    pendant = _graph(("a", "b"), [("ab", "a", "b", (0, 0, 0))])
    empty = periodic_net_symbols(pendant)
    assert empty.status == "computed"
    assert empty.reason == "computed"
    assert empty.signature == (("", "", 1),)
    assert {node.reason for node in empty.nodes} == {"no_angles"}


def test_report_carries_symbols_beside_the_canonical_key():
    report = analyze_periodic_net_system(_sql())
    topology = report.to_dict()["proposals"][0]["candidates"][0]["topology"]
    assert topology["status"] == "identified"
    assert topology["proof"] == "explicit_isomorphism"
    assert topology["symbols"]["status"] == "computed"
    assert topology["symbols"]["signature"][0]["point_symbol"] == "4^4.6^2"
    assert topology["symbols"]["signature"][0]["vertex_symbol"] == "4.4.4.4.6_2.6_2"
    assert topology["canonical_key"]
    assert "4^4.6^2" not in topology["canonical_key"]
