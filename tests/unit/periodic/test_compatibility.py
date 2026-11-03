"""Existing chain, chemistry, and bundle contracts stay in place."""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from molcrys_kit.chemistry import analyze_periodic_topology
from molcrys_kit.chemistry.models import (
    ChemicalAtom,
    ChemicalBond,
    PeriodicChemicalEntity,
)
from molcrys_kit.io.periodic_bundle import read_periodic_bundle, write_periodic_bundle
from molcrys_kit.operations.periodic_chain import build_periodic_chains
from molcrys_kit.periodic import (
    adapt_chemical_entity,
    adapt_fragment_graph,
    analyze_periodic_graph,
    periodic_rank,
    quotient_cycle_rank,
)
from molcrys_kit.structures.periodic_geometry import (
    BoundaryPort,
    ChainSpec,
    ConnectionRule,
    FragmentTemplate,
    PeriodicEdge,
    PeriodicGraph,
)


def test_fragment_cycle_rank_matches_and_heterogeneous_winding_stays_separate():
    graph = PeriodicGraph(
        ("a0", "a1", "b0", "b1"),
        (
            PeriodicEdge("a0", "a1", (0, 0, 0)),
            PeriodicEdge("a1", "a0", (1, 0, 0), closure=True),
            PeriodicEdge("b0", "b1", (0, 0, 0)),
            PeriodicEdge("b1", "b0", (0, 1, 0), closure=True),
        ),
    )
    assert graph.winding_cycles() == ((1, 0, 0), (0, 1, 0))
    assert graph.winding is None
    adapted = adapt_fragment_graph(graph)
    analysis = analyze_periodic_graph(adapted.graph)
    assert quotient_cycle_rank(adapted.graph) == graph.cycle_rank == 2
    assert periodic_rank(adapted.graph) == 1
    assert [family.subgroup.basis for family in analysis.cover_families] == [
        ((1, 0, 0),),
        ((0, 1, 0),),
    ]
    assert graph.winding_cycles() == ((1, 0, 0), (0, 1, 0))


def test_chemical_rank_uses_the_existing_topology_function():
    atoms = (ChemicalAtom("a", "C"),)
    bonds = (
        ChemicalBond("a", "a", atom2_image_shift=(2, 0, 0)),
        ChemicalBond("a", "a", atom2_image_shift=(0, 2, 0)),
    )
    entity = PeriodicChemicalEntity(
        "layer",
        atoms,
        bonds,
        periodic_rank=2,
        translation_generators=((2, 0, 0), (0, 2, 0)),
    )
    topology = analyze_periodic_topology(tuple(atom.atom_id for atom in atoms), bonds)
    adapted = adapt_chemical_entity(entity)
    family = analyze_periodic_graph(adapted.graph).cover_families[0]
    assert family.subgroup.rank == topology.rank == entity.periodic_rank
    assert family.subgroup.independent_generators == topology.translation_generators
    assert family.subgroup.basis == ((2, 0, 0), (0, 2, 0))


def test_chain_sidecar_schema_is_unchanged(tmp_path):
    template = FragmentTemplate(
        "repeat",
        ("C", "C"),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (BoundaryPort("in", (0.0, 0.0, 0.0)), BoundaryPort("out", (1.0, 0.0, 0.0))),
        ((0, 1),),
    )
    rule = ConnectionRule(
        "repeat-close",
        "repeat",
        "out",
        "repeat",
        "in",
        allowed_image_shifts=((0, 0, 0), (1, 0, 0)),
        distance_range=(0.5, 1.5),
    )
    bundle = build_periodic_chains(
        {"repeat": template},
        (rule,),
        np.diag([10.0, 10.0, 10.0]),
        (True, True, True),
        ChainSpec(("repeat",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5),
    )
    _structure, sidecar = write_periodic_bundle(bundle, tmp_path / "bundle")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    graph_payload = payload["periodic_graph"]
    assert set(graph_payload) == {
        "closure",
        "cycle_rank",
        "edges",
        "nodes",
        "winding_cycles",
    }
    assert set(graph_payload["edges"][0]) == {
        "closure",
        "left_node",
        "left_port",
        "right_image_shift",
        "right_node",
        "right_port",
        "rule_id",
    }
    _atoms, metadata = read_periodic_bundle(_structure, sidecar)
    assert (
        metadata["periodic_graph"]["winding_cycles"] == graph_payload["winding_cycles"]
    )
    adapted = adapt_fragment_graph(bundle.graph)
    assert adapted.provenance.closure == bundle.graph.closure
    assert any(edge.rule_id == "repeat-close" for edge in adapted.provenance.edges)


def test_large_cycle_keeps_the_optimized_winding_path():
    count = 4000
    nodes = tuple(f"n{index:05d}" for index in range(count))
    edges = [
        PeriodicEdge(nodes[index], nodes[index + 1], (0, 0, 0))
        for index in range(count - 1)
    ]
    edges.append(PeriodicEdge(nodes[-1], nodes[0], (2, 0, 0), closure=True))
    graph = PeriodicGraph(nodes, tuple(edges))
    assert len(graph.edges) == len(graph.nodes)
    started = perf_counter()
    windings = graph.winding_cycles()
    elapsed = perf_counter() - started
    assert windings == ((2, 0, 0),)
    assert elapsed < 2.0
    adapted = adapt_fragment_graph(graph)
    assert analyze_periodic_graph(adapted.graph).cover_families[0].subgroup.basis == (
        (2, 0, 0),
    )
