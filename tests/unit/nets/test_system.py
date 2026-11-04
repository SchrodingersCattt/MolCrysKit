"""Decomposition, proof status, linking, and duality in one report."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.constants.config import (
    KEY_ASYM_ID,
    KEY_ASSEMBLY,
    KEY_ATOM_ID,
    KEY_DISORDER_GROUP,
    KEY_OCCUPANCY,
)
from molcrys_kit.nets.decompose import decompose_periodic_nets
from molcrys_kit.nets.duality import PeriodicTiling, TilingAdjacency, TilingFace
from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.nets.policies import (
    DisorderPolicy,
    EdgePolicy,
    NodePolicy,
    PartitionPolicy,
)
from molcrys_kit.nets.report import (
    SCHEMA,
    analyze_periodic_net_system,
    limits_from_mapping,
)
from molcrys_kit.nets.site_graph import (
    build_periodic_site_graph,
    site_graph_from_quotient,
)
from molcrys_kit.periodic import PeriodicEmbedding, PeriodicQuotientGraph
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def _graph(nodes, edges, pbc=(True, True, True)):
    return PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)


def _embed(points, lattice=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))):
    return PeriodicEmbedding(
        tuple(
            (node_id, tuple(float(value) for value in point))
            for node_id, point in points
        ),
        lattice,
        fractional=tuple(
            (node_id, tuple(float(value) / 10.0 for value in point))
            for node_id, point in points
        ),
    )


def _limits(**overrides):
    values = NetAnalysisLimits.from_config().to_dict()
    values.update(overrides)
    return NetAnalysisLimits(**values)


def _sql():
    return _graph(("a",), [("ax", "a", "a", (1, 0, 0)), ("ay", "a", "a", (0, 1, 0))])


def _self_tiling():
    face = (
        ("a", (0, 0, 0)),
        ("a", (1, 0, 0)),
        ("a", (1, 1, 0)),
        ("a", (0, 1, 0)),
    )
    return PeriodicTiling(
        ("t",),
        (TilingFace("fx", face), TilingFace("fy", face)),
        (
            TilingAdjacency("fx", "t", "t", (1, 0, 0)),
            TilingAdjacency("fy", "t", "t", (0, 1, 0)),
        ),
    )


def _ring(prefix, points):
    nodes = [f"{prefix}{index}" for index in range(len(points))]
    edges = [
        (f"{prefix}e{index}", nodes[index], nodes[(index + 1) % len(nodes)], (0, 0, 0))
        for index in range(len(nodes))
    ]
    cartesian = [
        (node_id, tuple(float(value) * 10.0 for value in point))
        for node_id, point in zip(nodes, points)
    ]
    return _graph(nodes, edges), _embed(cartesian)


def _primary(report):
    payload = report.to_dict()
    for proposal in payload["proposals"]:
        if proposal["proposal_id"] == payload["primary_proposal_id"]:
            return proposal
    raise AssertionError(payload["primary_proposal_id"])


def test_parallel_chains_stay_rank_one_and_are_not_one_forced_partition():
    graph = _graph(
        ("a", "b"),
        [("ax", "a", "a", (1, 0, 0)), ("by", "b", "b", (0, 1, 0))],
        pbc=(True, True, False),
    )
    embedding = _embed((("a", (0.0, 0.0, 0.0)), ("b", (0.0, 5.0, 0.0))))
    site = site_graph_from_quotient(graph, embedding)
    decomposition = decompose_periodic_nets(site)
    by_id = {proposal.proposal_id: proposal for proposal in decomposition.proposals}
    assert "orbits" not in by_id
    assert decomposition.primary_is_not_unique
    assert decomposition.primary.proposal_id == "components"
    assert [
        candidate.periodic_rank for candidate in decomposition.primary.candidates
    ] == [1, 1]
    entire = by_id["entire"].candidates[0]
    assert entire.periodic_rank == 1
    assert entire.quotient_cycle_rank == 2
    assert entire.component_ranks == (1, 1)
    assert entire.to_dict()["translated_copies_are_one_topology"] is True
    report = analyze_periodic_net_system(site)
    payload = report.to_dict()
    assert payload["schema"] == SCHEMA
    assert "primary_partition_is_not_unique" in payload["ambiguities"]
    assert payload["primary_is_not_unique"] is True


def test_multiplicity_counts_translated_copies_of_one_chain():
    graph = _graph(("a",), [("wind", "a", "a", (2, 0, 0))], pbc=(True, False, False))
    site = site_graph_from_quotient(graph)
    decomposition = decompose_periodic_nets(site)
    assert decomposition.primary_is_not_unique is False
    candidate = decomposition.primary.candidates[0]
    assert candidate.periodic_rank == 1
    assert candidate.multiplicity == 2
    assert len(decomposition.primary.candidates) == 1


def test_finite_cycle_rank_is_not_periodic_rank_in_the_report():
    square = _graph(
        ("a", "b", "c", "d"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("bc", "b", "c", (0, 0, 0)),
            ("cd", "c", "d", (0, 0, 0)),
            ("da", "d", "a", (0, 0, 0)),
        ],
    )
    report = analyze_periodic_net_system(square)
    candidate = _primary(report)["candidates"][0]
    assert candidate["periodic_rank"] == 0
    assert candidate["quotient_cycle_rank"] == 1
    assert candidate["topology"]["status"] == "identified"
    assert candidate["topology"]["proof"] == "explicit_isomorphism"
    assert candidate["topology"]["canonical_key"]


def test_explicit_partition_reports_overlap_gaps_and_contradictions():
    graph = _graph(
        ("a", "b"),
        [("ab", "a", "b", (0, 0, 0)), ("ax", "a", "a", (1, 0, 0))],
    )
    site = site_graph_from_quotient(graph)
    overlap = decompose_periodic_nets(
        site,
        PartitionPolicy("explicit", (("a", "one"), ("a", "two"), ("b", "one"))),
    )
    assert overlap.primary.status == "invalid"
    assert overlap.primary.reason == "overlapping_nodes"
    uncovered = decompose_periodic_nets(
        site, PartitionPolicy("explicit", (("a", "one"),))
    )
    assert uncovered.primary.reason == "uncovered_nodes"
    assert uncovered.primary.uncovered_node_ids == ("b",)
    contradictory = decompose_periodic_nets(
        site,
        PartitionPolicy(
            "explicit",
            (("a", "one"), ("b", "one")),
            edge_nets=(("ab", "other"), ("ax", "one")),
        ),
    )
    assert contradictory.primary.reason == "contradictory_edges"
    assert contradictory.primary.contradictory_edge_ids == ("ab",)
    report = analyze_periodic_net_system(
        site,
        partition_policy=PartitionPolicy("explicit", (("a", "one"),)),
    )
    candidate = _primary(report)["candidates"][0]
    assert candidate["topology"]["status"] == "invalid_or_indeterminate"
    assert candidate["topology"]["canonical_key"] is None
    assert candidate["topology"]["proof"] == "none"
    assert _primary(report)["pairwise"] == []
    missing = decompose_periodic_nets(
        site,
        PartitionPolicy(
            "explicit",
            (("a", "one"), ("b", "one")),
            edge_nets=(("missing_edge", "one"),),
        ),
    )
    assert missing.primary.status == "invalid"
    assert missing.primary.reason == "unknown_edge"
    conflict = decompose_periodic_nets(
        site,
        PartitionPolicy(
            "explicit",
            (("a", "one"), ("b", "one")),
            edge_nets=(("ab", "one"), ("ab", "other"), ("ax", "one")),
        ),
    )
    assert conflict.primary.status == "invalid"
    assert conflict.primary.reason == "conflicting_edge_assignment"
    partial = decompose_periodic_nets(
        site,
        PartitionPolicy(
            "explicit",
            (("a", "one"), ("b", "one")),
            edge_nets=(("ab", "one"),),
        ),
    )
    assert partial.primary.status == "invalid"
    assert partial.primary.reason == "unassigned_edges"
    assert partial.primary.candidates[0].edge_ids == ("ab",)


def test_discovery_finds_two_orbit_nets_joined_by_a_cross_edge():
    atoms = Atoms(
        "CCCC",
        positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 4.0, 0.0], [1.4, 4.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    atoms.set_array(KEY_ATOM_ID, np.array(["A", "B", "C", "D"], dtype=object))
    atoms.set_array(KEY_ASYM_ID, np.array([1, 2, 3, 4], dtype=int))
    molecule = CrystalMolecule(atoms, check_pbc=False)
    molecule.info["bond_records"] = [
        {
            "left": 0,
            "right": 1,
            "right_image_shift": [0, 0, 0],
            "vector": [1.4, 0.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 0,
            "right": 1,
            "right_image_shift": [1, 0, 0],
            "vector": [21.4, 0.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 2,
            "right": 3,
            "right_image_shift": [0, 0, 0],
            "vector": [1.4, 0.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 2,
            "right": 3,
            "right_image_shift": [0, 1, 0],
            "vector": [1.4, 20.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 0,
            "right": 2,
            "right_image_shift": [0, 0, 0],
            "vector": [0.0, 4.0, 0.0],
            "role": "covalent",
        },
    ]
    crystal = MolecularCrystal(np.eye(3) * 20.0, [molecule])
    built = build_periodic_site_graph(
        crystal, NodePolicy("atom_sites"), EdgePolicy("supplied")
    )
    decomposition = decompose_periodic_nets(built)
    assert decomposition.primary_is_not_unique
    wanted = {("atom:A", "atom:B"), ("atom:C", "atom:D")}
    matches = [
        proposal
        for proposal in decomposition.proposals
        if {tuple(candidate.node_ids) for candidate in proposal.candidates} == wanted
    ]
    assert matches
    assert matches[0].reason == "ranked_orbit_partition"
    assert all(candidate.periodic_rank >= 1 for candidate in matches[0].candidates)
    assert any(candidate.cross_edge_ids for candidate in matches[0].candidates)


def test_mixed_lattice_or_boundary_flags_are_rejected():
    graph = _graph(("a",), [("ax", "a", "a", (1, 0, 0))])
    opposite = _graph(("b",), [("by", "b", "b", (0, 1, 0))], pbc=(False, True, True))
    unit = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
    scaled = ((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0))

    def embed(node_id, lattice):
        return PeriodicEmbedding(
            ((node_id, (0.0, 0.0, 0.0)),),
            lattice,
            fractional=((node_id, (0.0, 0.0, 0.0)),),
        )

    with pytest.raises(ValueError, match="lattice"):
        analyze_periodic_net_system(
            [(graph, embed("a", unit)), (graph, embed("a", scaled))]
        )
    with pytest.raises(ValueError, match="periodic boundary"):
        analyze_periodic_net_system(
            [(graph, embed("a", unit)), (opposite, embed("b", unit))]
        )


def test_asym_orbits_are_a_second_partition_and_the_search_can_truncate():
    atoms = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[20, 20, 20], pbc=True
    )
    atoms.set_array(KEY_ATOM_ID, np.array(["c", "o"], dtype=object))
    atoms.set_array(KEY_ASYM_ID, np.array([4, 9], dtype=int))
    crystal = MolecularCrystal(
        np.eye(3) * 20.0, [CrystalMolecule(atoms, check_pbc=False)]
    )
    built = build_periodic_site_graph(
        crystal, NodePolicy("atom_sites"), EdgePolicy("supplied")
    )
    decomposition = decompose_periodic_nets(built)
    orbits = next(
        item for item in decomposition.proposals if item.proposal_id == "orbits"
    )
    assert len(orbits.candidates) == 2
    assert decomposition.primary.proposal_id == "components"
    assert decomposition.primary_is_not_unique
    cross = [
        edge_id
        for candidate in orbits.candidates
        for edge_id in candidate.cross_edge_ids
    ]
    assert cross
    truncated = decompose_periodic_nets(
        built, limits=_limits(discovery_partition_limit=1)
    )
    assert truncated.truncated
    assert all(proposal.proposal_id != "orbits" for proposal in truncated.proposals)
    report = analyze_periodic_net_system(
        built, limits=_limits(discovery_partition_limit=1)
    )
    assert "discovery_truncated" in report.to_dict()["ambiguities"]


def test_indeterminate_disorder_has_no_canonical_key_and_conditional_keeps_its_status():
    atoms = Atoms(
        "CC", positions=[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], cell=[20, 20, 20], pbc=True
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([0.7, 0.3]))
    atoms.set_array(KEY_DISORDER_GROUP, np.array([1, 2], dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(["A", "A"], dtype=object))
    crystal = MolecularCrystal(
        np.eye(3) * 20.0, [CrystalMolecule(atoms, check_pbc=False)]
    )
    refined = build_periodic_site_graph(
        crystal, disorder_policy=DisorderPolicy("as_refined")
    )
    refused = analyze_periodic_net_system(refined)
    candidate = _primary(refused)["candidates"][0]
    assert candidate["topology"]["status"] == "invalid_or_indeterminate"
    assert candidate["topology"]["reason"] == "disorder_indeterminate"
    assert candidate["topology"]["canonical_key"] is None
    assert any(
        note.startswith("disorder_indeterminate:")
        for note in refused.to_dict()["ambiguities"]
    )
    major = build_periodic_site_graph(crystal, disorder_policy=DisorderPolicy("major"))
    conditional = analyze_periodic_net_system(major)
    block = _primary(conditional)["candidates"][0]["topology"]
    assert block["status"] == "identified"
    assert block["conditional_on_disorder"] is True
    assert block["reason"].startswith("conditional:")
    assert block["canonical_key"]


def test_hopf_pair_is_linked_in_the_report_and_a_translate_is_not():
    left, left_embedding = _ring(
        "a",
        ((0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.75, 0.75, 0.5), (0.25, 0.75, 0.5)),
    )
    linked, linked_embedding = _ring(
        "b",
        ((0.5, 0.5, 0.25), (0.0, 0.5, 0.25), (0.0, 0.5, 0.75), (0.5, 0.5, 0.75)),
    )
    apart, apart_embedding = _ring(
        "c",
        ((0.5, 1.0, 0.25), (0.0, 1.0, 0.25), (0.0, 1.0, 0.75), (0.5, 1.0, 0.75)),
    )
    limits = _limits(canonical_node_limit=4)
    hopf = analyze_periodic_net_system(
        [(left, left_embedding), (linked, linked_embedding)],
        limits=limits,
    )
    pairs = [
        row
        for proposal in hopf.to_dict()["proposals"]
        if proposal["proposal_id"] == "components"
        for row in proposal["pairwise"]
    ]
    assert len(pairs) == 1
    assert pairs[0]["equivalence"]["status"] == "isomorphic"
    assert pairs[0]["equivalence"]["proof"] == "explicit_isomorphism"
    assert pairs[0]["embedding"]["status"] == "linked"
    assert abs(int(pairs[0]["embedding"]["witness"]["linking_number"])) == 1
    assert pairs[0]["duality"]["status"] == "not_assessed"
    separated = analyze_periodic_net_system(
        [(left, left_embedding), (apart, apart_embedding)],
        limits=limits,
    )
    separated_row = _primary(separated)["pairwise"][0]
    assert separated_row["equivalence"]["status"] == "isomorphic"
    assert separated_row["embedding"]["status"] == "coexistence"
    capped = analyze_periodic_net_system(
        [(left, left_embedding), (linked, linked_embedding)],
        limits=_limits(canonical_node_limit=4, ring_length_limit=3),
    )
    capped_row = _primary(capped)["pairwise"][0]
    assert capped_row["embedding"]["status"] in {
        "unverified",
        "not_detected_within_bounds",
    }


def test_tiling_duality_is_reported_and_a_missing_tiling_is_not_a_dual():
    site = site_graph_from_quotient(_sql(), _embed((("a", (0.0, 0.0, 0.0)),)))
    report = analyze_periodic_net_system(site, duality=_self_tiling())
    row = _primary(report)["pairwise"][0]
    assert row["left"] == row["right"]
    assert row["equivalence"]["status"] == "isomorphic"
    assert row["equivalence"]["proof"] == "explicit_isomorphism"
    assert row["duality"]["status"] == "dual"
    assert row["embedding"]["reason"] == "same_candidate"
    mixed = site_graph_from_quotient(
        _graph(
            ("a", "b"),
            [
                ("ax", "a", "a", (1, 0, 0)),
                ("ay", "a", "a", (0, 1, 0)),
                ("wind", "b", "b", (1, 0, 0)),
            ],
        )
    )
    union = analyze_periodic_net_system(
        mixed,
        duality=_self_tiling(),
        partition_policy=PartitionPolicy("explicit", (("a", "a-net"), ("b", "b-net"))),
    )
    dual_rows = _primary(union)["pairwise"]
    assert len(dual_rows) == 1
    assert dual_rows[0]["left"] == "a-net"
    assert dual_rows[0]["equivalence"]["status"] == "non_isomorphic"
    assert dual_rows[0]["duality"]["status"] == "not_dual"
    plain = analyze_periodic_net_system(_sql())
    assert "duality_not_assessed" in plain.to_dict()["ambiguities"]
    assert _primary(plain)["pairwise"] == []
    broken = PeriodicTiling(
        ("t",),
        (
            TilingFace(
                "bad",
                (
                    ("a", (0, 0, 0)),
                    ("a", (2, 0, 0)),
                    ("a", (2, 1, 0)),
                    ("a", (0, 1, 0)),
                ),
            ),
        ),
        (TilingAdjacency("bad", "t", "t", (1, 0, 0)),),
    )
    invalid = analyze_periodic_net_system(site, duality=broken)
    assert _primary(invalid)["pairwise"][0]["duality"]["status"] == "undetermined"


def test_reanalysis_of_candidates_keeps_embeddings_and_ranks():
    graph = _graph(
        ("a", "b"),
        [("ax", "a", "a", (1, 0, 0)), ("by", "b", "b", (0, 1, 0))],
        pbc=(True, True, False),
    )
    site = site_graph_from_quotient(
        graph, _embed((("a", (0.0, 0.0, 0.0)), ("b", (0.0, 8.0, 0.0))))
    )
    decomposition = decompose_periodic_nets(
        site,
        PartitionPolicy("explicit", (("a", "left"), ("b", "right"))),
    )
    report = analyze_periodic_net_system(list(decomposition.primary.candidates))
    proposal = _primary(report)
    assert proposal["status"] == "admissible"
    assert sorted(
        candidate["periodic_rank"] for candidate in proposal["candidates"]
    ) == [1, 1]
    assert proposal["pairwise"][0]["embedding"]["status"] in {
        "coexistence",
        "unverified",
        "not_detected_within_bounds",
    }


def test_limit_overrides_reject_unknown_names():
    updated = limits_from_mapping(
        NetAnalysisLimits.from_config(), {"ring_length_limit": 3.0}
    )
    assert updated.ring_length_limit == 3
    try:
        limits_from_mapping(updated, {"prototype": 1})
    except ValueError as exc:
        assert "prototype" in str(exc)
    else:
        raise AssertionError("unknown limit was accepted")
