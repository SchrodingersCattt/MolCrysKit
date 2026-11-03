"""Domain adapters keep provenance that the quotient graph does not store."""

from __future__ import annotations

import json

import numpy as np
import pytest

from molcrys_kit.analysis.polyhedron_network import (
    CentreId,
    CoordinationShell,
    PolyhedronComponent,
    PolyhedronNetwork,
    SharedContact,
    SharedVertex,
    ShellVertex,
    polyhedron_network_from_shells,
)
from molcrys_kit.chemistry.models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    MulticomponentEntity,
    PeriodicChemicalEntity,
)
from molcrys_kit.periodic import (
    adapt_caller_graph,
    adapt_chemical_entity,
    adapt_fragment_graph,
    adapt_polyhedron_network,
    analyze_periodic_graph,
    fragment_graph_from_provenance,
    periodic_rank,
    to_periodic_quotient_graph,
)
from molcrys_kit.structures.periodic_geometry import PeriodicEdge, PeriodicGraph


def test_fragment_adapter_preserves_ports_rules_and_closure():
    graph = PeriodicGraph(
        ("b", "a"),
        (
            PeriodicEdge(
                "b",
                "a",
                (2, 0, 0),
                rule_id="join",
                closure=True,
                left_port="out",
                right_port="in",
            ),
            PeriodicEdge(
                "a",
                "b",
                (0, 0, 0),
                rule_id="join",
                left_port="out",
                right_port="in",
            ),
        ),
        closure="screw",
    )
    adapted = adapt_fragment_graph(graph)
    assert adapted.provenance.kind == "fragment_graph"
    assert adapted.provenance.closure == "screw"
    rebuilt = fragment_graph_from_provenance(adapted.provenance)
    assert rebuilt.nodes == graph.nodes
    assert rebuilt.closure == "screw"
    assert rebuilt.edges == graph.edges
    assert rebuilt.winding_cycles() == graph.winding_cycles()
    assert periodic_rank(adapted.graph) == 1
    assert analyze_periodic_graph(adapted.graph).cover_families[0].subgroup.basis == (
        (2, 0, 0),
    )
    payload = json.loads(json.dumps(adapted.to_dict()))
    stored = payload["provenance"]["edges"][0]
    assert stored["rule_id"] == "join"
    assert stored["left_port"] == "out"
    assert stored["closure"] is True
    assert "ports" not in payload["graph"]["edges"][0]


def test_chemical_adapter_keeps_bond_records_and_declared_rank():
    evidence = Evidence(EvidenceSource.USER_CONFIRMED, "manual")
    atom = ChemicalAtom("a", "C", evidence=(evidence,))
    bond = ChemicalBond(
        "a",
        "a",
        order=1.0,
        kind=BondKind.COVALENT,
        atom2_image_shift=(2, 0, 0),
        evidence=(evidence,),
    )
    entity = PeriodicChemicalEntity(
        "chain",
        (atom,),
        (bond,),
        periodic_rank=1,
        translation_generators=((2, 0, 0),),
        embedding=Embedding(
            (("a", (0.0, 0.0, 0.0)),),
            cell_A=((4.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 6.0)),
        ),
        evidence=(evidence,),
    )
    adapted = adapt_chemical_entity(entity)
    assert adapted.provenance.kind == "chemical_entity"
    assert adapted.provenance.bonds[0] is bond
    assert adapted.provenance.declared_rank == 1
    assert adapted.provenance.atoms[0].evidence[0].method == "manual"
    assert adapted.embedding is not None
    assert adapted.embedding.position("a") == (0.0, 0.0, 0.0)
    assert periodic_rank(adapted.graph) == 1
    payload = json.loads(json.dumps(adapted.to_dict()))
    assert payload["provenance"]["bonds"][0]["kind"] == "covalent"
    assert payload["provenance"]["bonds"][0]["atom2_image_shift"] == [2, 0, 0]
    assert payload["provenance"]["evidence"][0]["source"] == "user_confirmed"


def test_unwrapped_molecule_stays_rank_zero_and_finite_entities_adapt():
    entity = FiniteChemicalEntity(
        "molecule",
        (ChemicalAtom("a", "C"), ChemicalAtom("b", "C")),
        (ChemicalBond("a", "b", atom2_image_shift=(1, 0, 0)),),
    )
    adapted = adapt_chemical_entity(entity)
    assert adapted.provenance.declared_dimension == 0
    assert adapted.provenance.declared_rank is None
    assert periodic_rank(adapted.graph) == 0
    with pytest.raises(TypeError, match="FiniteChemicalEntity"):
        to_periodic_quotient_graph(MulticomponentEntity("salt", ((entity, 1),)))


def test_polyhedron_adapter_preserves_contacts_and_nonprimitive_shift():
    shell = CoordinationShell(
        molecule_index=0,
        realization_index=1,
        formula="N",
        species_id="N_1",
        center_position_A=(0.0, 0.0, 0.0),
        vertices=(
            ShellVertex(
                1,
                0,
                (0, 0, 0),
                (0.0, 0.0, 0.0),
                0.0,
                "Cl",
                "Cl_1",
            ),
            ShellVertex(
                1,
                0,
                (2, 0, 0),
                (20.0, 0.0, 0.0),
                20.0,
                "Cl",
                "Cl_1",
            ),
        ),
        criterion="provided",
        primary_gap_cn=2,
        enclosure_expanded=False,
        search_cutoff_A=20.0,
        centre_role="metal",
        center_kind="centroid",
        centre_disorder_policy="major",
        ligand_disorder_policy="exclusive",
    )
    network = polyhedron_network_from_shells(
        [shell],
        np.eye(3) * 10.0,
        include_cycle_translations=True,
        classify_boundary=False,
    )
    adapted = adapt_polyhedron_network(network)
    assert adapted.provenance.kind == "polyhedron_network"
    assert adapted.provenance.ligand_disorder_policy == "exclusive"
    assert adapted.provenance.network is network
    contact = network.contacts[0]
    edge_id, contact_index = adapted.provenance.edge_ids[0]
    assert contact_index == 0
    assert adapted.provenance.network.contacts[contact_index] is contact
    assert contact.shared_vertices
    assert contact.right_image_shift == (2, 0, 0)
    family = analyze_periodic_graph(adapted.graph).cover_families[0]
    assert family.subgroup.basis == network.components[0].translation_generators
    assert family.subgroup.basis == ((2, 0, 0),)
    assert adapted.graph.edge(edge_id).image_shift == (2, 0, 0)
    payload = json.loads(json.dumps(adapted.to_dict()))
    assert payload["provenance"]["network"]["contacts"][0]["relation"]
    assert payload["graph"]["nodes"][0]["node_id"] == "centre:0:1"


def test_manual_polyhedron_records_round_trip_their_roles():
    centre = CentreId(3, 0)
    contact = SharedContact(
        centre,
        centre,
        (0, 4, 0),
        relation="vertex",
        ambiguity=None,
        shared_vertices=(SharedVertex(4, 0, (0, 0, 0)),),
    )
    shell = CoordinationShell(
        molecule_index=3,
        realization_index=0,
        formula="N",
        species_id="N_1",
        center_position_A=(1.0, 2.0, 3.0),
        vertices=(),
        criterion="provided",
        primary_gap_cn=0,
        enclosure_expanded=False,
        search_cutoff_A=None,
    )
    component = PolyhedronComponent((centre,), 1, ((0, 4, 0),))
    network = PolyhedronNetwork(
        shells=(shell,),
        contacts=(contact,),
        components=(component,),
        lattice_A=((8.0, 0.0, 0.0), (1.0, 9.0, 0.0), (0.0, 0.0, 7.0)),
        pbc=(True, True, True),
        boundary=None,
        center_kind="centroid",
        centre_disorder_policy="as_refined",
        ligand_disorder_policy="major",
    )
    adapted = adapt_polyhedron_network(network)
    assert adapted.provenance.centre_disorder_policy == "as_refined"
    assert adapted.provenance.network.contacts[0].relation == "vertex"
    assert adapted.provenance.network.contacts[0].shared_vertices[0].molecule_index == 4
    assert periodic_rank(adapted.graph) == 1
    assert analyze_periodic_graph(adapted.graph).cover_families[0].subgroup.basis == (
        (0, 4, 0),
    )


def test_caller_graph_keeps_parallel_edges_and_attributes():
    adapted = adapt_caller_graph(
        ("b", "a"),
        [
            ("b", "a", (-2, 0, 0)),
            ("span", "a", "b", (0, 0, 0)),
            ("e", "a", "a", (0, 0, 1)),
        ],
        node_attributes={"a": {"site": "origin", "occupied": True}},
        edge_attributes=({"role": "bridge"}, None, None),
    )
    shifts = sorted(edge.image_shift for edge in adapted.graph.edges)
    assert shifts == [(0, 0, 0), (0, 0, 1), (2, 0, 0)]
    # The parallel pair differs by (2, 0, 0); the self-image edge adds c.
    assert periodic_rank(adapted.graph) == 2
    payload = json.loads(json.dumps(adapted.to_dict()))
    attributes = payload["provenance"]["node_attributes"]
    stored = {
        row[0]: {item["key"]: item["value"] for item in row[1]} for row in attributes
    }
    assert stored["a"] == {"occupied": True, "site": "origin"}
    assert payload["provenance"]["edge_attributes"][0][1][0]["value"] == "bridge"
    assert to_periodic_quotient_graph(adapted) is adapted.graph
    with pytest.raises(TypeError, match="adapt_caller_graph"):
        to_periodic_quotient_graph({"nodes": ["a"]})
