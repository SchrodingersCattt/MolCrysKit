"""Crystal policies, provenance, and jointly realizable disorder."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.constants import DEFAULT_NEIGHBOR_CUTOFF
from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_ATOM_ID,
    KEY_DISORDER_GROUP,
    KEY_OCCUPANCY,
)
from molcrys_kit.nets.policies import (
    DisorderPolicy,
    EdgePolicy,
    NodePolicy,
    policies_from_mapping,
)
from molcrys_kit.nets.site_graph import build_periodic_site_graph
from molcrys_kit.periodic import analyze_periodic_graph
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def _molecule(symbols, positions, **arrays):
    atoms = Atoms(symbols, positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)
    for key, value in arrays.items():
        atoms.set_array(key, np.asarray(value))
    return CrystalMolecule(atoms, check_pbc=False)


def _crystal(molecules, *, cell=30.0, pbc=(True, True, True)):
    return MolecularCrystal(np.eye(3) * cell, list(molecules), pbc=pbc)


def _water(origin):
    ox, oy, oz = origin
    return _molecule(
        "OHH",
        [[ox, oy, oz], [ox + 0.96, oy, oz], [ox - 0.24, oy + 0.93, oz]],
    )


def _sites(symbols, positions, **arrays):
    return _crystal([_molecule(symbols, positions, **arrays)])


def test_same_formula_stays_two_molecular_nodes_and_bonds_stay_inside_them():
    crystal = _crystal([_water((0.0, 0.0, 0.0)), _water((0.0, 3.0, 0.0))])
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("molecular_units"),
        EdgePolicy("supplied"),
    )
    assert built.disorder_status == "resolved"
    assert [node.node_id for node in built.nodes] == ["molecule:0", "molecule:1"]
    assert (
        built.nodes[0].symbols != ()
        and built.nodes[0].symbols == built.nodes[1].symbols
    )
    assert set(built.nodes[0].site_ids).isdisjoint(built.nodes[1].site_ids)
    assert built.graph.node_ids == ("molecule:0", "molecule:1")
    assert built.graph.edges == ()
    contracted = [record for record in built.records if record.status == "contracted"]
    assert contracted
    assert all(record.image_shift == (0, 0, 0) for record in contracted)
    assert built.settings["cutoff_A"] == float(DEFAULT_NEIGHBOR_CUTOFF)
    observed = [record for record in built.records if record.status == "observed"]
    assert observed
    assert all(record.kind == "contacts" for record in observed)
    assert "wyckoff" not in built.to_dict()


def test_unwrapping_bookkeeping_is_not_a_periodic_self_image():
    """A finite molecule bonded across a cell face stays rank 0.

    The site-image field can be nonzero while the Cartesian bond vector is the
    short unwrapped contact. The quotient uses the vector, and it keeps the
    recorded field beside that choice.
    """
    atoms = Atoms(
        "CC",
        positions=[[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    molecular = build_periodic_site_graph(
        crystal, NodePolicy("molecular_units"), EdgePolicy("supplied")
    )
    assert molecular.graph.edges == ()
    contracted = molecular.records[0]
    assert contracted.status == "contracted"
    assert contracted.image_shift == (0, 0, 0)
    assert contracted.geometry_mismatch is True
    assert contracted.criteria["recorded_image_shift"] == [1, 0, 0]
    assert analyze_periodic_graph(molecular.graph).periodic_rank == 0


def test_a_vector_that_contains_a_lattice_image_stays_periodic():
    atoms = Atoms(
        "CC",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.set_array(KEY_ATOM_ID, np.array(["left", "right"], dtype=object))
    molecule = CrystalMolecule(atoms, check_pbc=False)
    molecule.info["bond_records"] = [
        {
            "left": 0,
            "right": 1,
            "right_image_shift": [1, 0, 0],
            "vector": [11.5, 0.0, 0.0],
            "role": "covalent",
        }
    ]
    crystal = MolecularCrystal(np.eye(3) * 10.0, [molecule])
    molecular = build_periodic_site_graph(
        crystal, NodePolicy("molecular_units"), EdgePolicy("supplied")
    )
    assert len(molecular.graph.edges) == 1
    assert molecular.graph.edges[0].image_shift == (1, 0, 0)
    assert molecular.graph.edges[0].source == molecular.graph.edges[0].target
    assert molecular.records[0].geometry_mismatch is False
    assert analyze_periodic_graph(molecular.graph).periodic_rank == 1
    sites = build_periodic_site_graph(
        crystal, NodePolicy("atom_sites"), EdgePolicy("supplied")
    )
    assert len(sites.nodes) == 2
    assert sites.graph.edges[0].image_shift == (1, 0, 0)
    assert sites.graph.edges[0].source != sites.graph.edges[0].target


def test_atom_sites_keep_inequivalent_atoms_of_one_element():
    crystal = _sites("CC", [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    built = build_periodic_site_graph(
        crystal, NodePolicy("atom_sites"), EdgePolicy("supplied")
    )
    assert len(built.nodes) == 2
    assert built.nodes[0].symbols == ("C",)
    assert built.nodes[1].symbols == ("C",)
    assert built.nodes[0].node_id != built.nodes[1].node_id


def test_explicit_groups_are_reversible_and_reject_a_repeated_site():
    crystal = _sites(
        "COH",
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [6.0, 0.0, 0.0]],
        **{KEY_ATOM_ID: np.array(["c", "o", "h"], dtype=object)},
    )
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("explicit_nodes", groups=(("c", "o"), ("h",))),
        EdgePolicy("user_edges", user_edges=(("group:0", "group:1", (0, 0, 0)),)),
    )
    assert [node.node_id for node in built.nodes] == ["group:0", "group:1"]
    assert built.nodes[0].site_ids == ("c", "o")
    assert built.settings["unassigned_site_ids"] == []
    assert built.settings["sites_not_in_nodes"] == []
    assert any(
        record.status == "defining" and record.record_id == "user-0"
        for record in built.records
    )
    with pytest.raises(ValueError, match="repeat site"):
        build_periodic_site_graph(
            crystal,
            NodePolicy("explicit_nodes", groups=(("c", "o"), ("o",))),
        )


def test_unlisted_explicit_sites_are_not_nodes():
    crystal = _sites(
        "CO",
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        **{KEY_ATOM_ID: np.array(["c", "o"], dtype=object)},
    )
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("explicit_nodes", groups=(("c",),)),
        EdgePolicy("supplied"),
    )
    assert [node.node_id for node in built.nodes] == ["group:0"]
    assert built.settings["unassigned_site_ids"] == ["o"]
    assert built.settings["sites_not_in_nodes"] == ["o"]
    with pytest.raises(ValueError, match="unknown site"):
        build_periodic_site_graph(
            crystal,
            NodePolicy("explicit_nodes", groups=(("missing",),)),
        )


def test_user_edge_on_a_nonperiodic_axis_is_rejected():
    crystal = _crystal(
        [_water((0.0, 0.0, 0.0)), _water((0.0, 8.0, 0.0))],
        pbc=(True, False, False),
    )
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("molecular_units"),
        EdgePolicy(
            "user_edges",
            user_edges=(
                ("molecule:0", "molecule:1", (0, 1, 0)),
                ("molecule:0", "molecule:1", (1, 0, 0)),
            ),
        ),
    )
    statuses = {record.record_id: record.status for record in built.records}
    assert statuses["user-0"] == "rejected"
    assert statuses["user-1"] == "defining"
    assert len(built.graph.edges) == 1


def test_contact_cutoff_records_defining_and_borderline_pairs():
    crystal = _sites("ArArAr", [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0], [0.0, 3.6, 0.0]])
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("atom_sites"),
        EdgePolicy("contacts", cutoff_A=3.5),
    )
    defining = [record for record in built.records if record.status == "defining"]
    borderline = [record for record in built.records if record.status == "borderline"]
    assert len(defining) == 1
    assert defining[0].criteria["cutoff_A"] == 3.5
    assert defining[0].distance_A == pytest.approx(3.4)
    assert len(borderline) == 1
    assert borderline[0].distance_A == pytest.approx(3.6)
    assert len(built.graph.edges) == 1
    far = _sites("ArAr", [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    quiet = build_periodic_site_graph(
        far,
        NodePolicy("atom_sites"),
        EdgePolicy("contacts", cutoff_A=3.5),
    )
    assert quiet.graph.edges == ()
    assert all(record.status != "defining" for record in quiet.records)


def test_supplied_contacts_stay_out_of_the_net():
    crystal = _sites("ArAr", [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]])
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("atom_sites"),
        EdgePolicy("supplied", cutoff_A=3.5),
    )
    assert built.graph.edges == ()
    assert any(
        record.status == "observed" and record.kind == "contacts"
        for record in built.records
    )


def test_coordination_shell_connects_centres_that_share_a_ligand():
    crystal = _sites(
        "CCC",
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]],
        **{KEY_ATOM_ID: np.array(["left", "bridge", "right"], dtype=object)},
    )
    built = build_periodic_site_graph(
        crystal,
        NodePolicy(
            "coordination_centres",
            centre_ids=("left", "right"),
            bond_role="covalent",
        ),
        EdgePolicy("coordination_shell"),
    )
    assert [node.node_id for node in built.nodes] == ["centre:left", "centre:right"]
    assert built.settings["sites_not_in_nodes"] == ["bridge"]
    assert len(built.graph.edges) == 1
    edge = built.graph.edges[0]
    assert edge.image_shift == (0, 0, 0)
    record = built.records[0]
    assert record.kind == "coordination_shell"
    assert record.criteria["ligand_site_id"] == "bridge"
    assert record.site_ids == ("bridge",)


def test_hydrogen_bond_policy_records_the_criteria_it_used():
    donor = _molecule(
        "OHH",
        [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    acceptor = _molecule("O", [[2.8, 0.0, 0.0]])
    crystal = _crystal([donor, acceptor])
    built = build_periodic_site_graph(
        crystal,
        NodePolicy("molecular_units"),
        EdgePolicy(
            "hydrogen_bonds", max_h_acceptor_distance_A=2.5, min_dha_angle_deg=150
        ),
    )
    bonds = [record for record in built.records if record.kind == "hydrogen_bonds"]
    assert len(bonds) == 1
    assert bonds[0].status == "defining"
    assert bonds[0].criteria["max_h_acceptor_distance_A"] == 2.5
    assert bonds[0].criteria["min_dha_angle_deg"] == 150
    assert bonds[0].source_node == "molecule:0"
    assert bonds[0].target_node == "molecule:1"
    assert len(built.graph.edges) == 1


def test_as_refined_competitors_are_indeterminate_and_major_is_conditional():
    crystal = _sites(
        "CC",
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
        **{
            KEY_OCCUPANCY: [0.7, 0.3],
            KEY_DISORDER_GROUP: [1, 2],
            KEY_ASSEMBLY: np.array(["A", "A"], dtype=object),
            KEY_ATOM_ID: np.array(["major", "minor"], dtype=object),
        },
    )
    refined = build_periodic_site_graph(
        crystal,
        disorder_policy=DisorderPolicy("as_refined"),
    )
    assert refined.disorder_status == "indeterminate"
    assert refined.disorder_reason == "competitors_retained"
    assert len(refined.nodes) == 2
    major = build_periodic_site_graph(crystal, disorder_policy=DisorderPolicy("major"))
    assert major.disorder_status == "conditional"
    assert major.disorder_reason == "major_component"
    assert [node.site_ids for node in major.nodes] == [("major",)]


def test_two_assemblies_are_not_combined_unless_asked():
    crystal = _sites(
        "CCOO",
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.3, 0.0]],
        **{
            KEY_OCCUPANCY: [0.8, 0.2, 0.6, 0.4],
            KEY_DISORDER_GROUP: [1, 2, 1, 2],
            KEY_ASSEMBLY: np.array(["A", "A", "B", "B"], dtype=object),
            KEY_ATOM_ID: np.array(["a1", "a2", "b1", "b2"], dtype=object),
        },
    )
    blocked = build_periodic_site_graph(
        crystal, disorder_policy=DisorderPolicy("major")
    )
    assert blocked.disorder_status == "indeterminate"
    assert blocked.disorder_reason == "joint_realization_not_unique"
    assert len(blocked.nodes) == 4
    chosen = build_periodic_site_graph(
        crystal,
        disorder_policy=DisorderPolicy("major", independent_assemblies=True),
    )
    assert chosen.disorder_status == "conditional"
    assert {node.site_ids[0] for node in chosen.nodes} == {"a1", "b1"}


def test_occupancy_tie_and_explicit_selection():
    tied = _sites(
        "CC",
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
        **{
            KEY_OCCUPANCY: [0.5, 0.5],
            KEY_DISORDER_GROUP: [1, 2],
            KEY_ASSEMBLY: np.array(["A", "A"], dtype=object),
        },
    )
    result = build_periodic_site_graph(tied, disorder_policy=DisorderPolicy("major"))
    assert result.disorder_status == "indeterminate"
    assert result.disorder_reason == "major_occupancy_tie"
    crystal = _sites(
        "CCC",
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]],
        **{
            KEY_OCCUPANCY: [0.25, 0.75, 1.0],
            KEY_DISORDER_GROUP: [1, 2, 0],
            KEY_ASSEMBLY: np.array(["A", "A", ""], dtype=object),
            KEY_ATOM_ID: np.array(["part1", "part2", "ordered"], dtype=object),
        },
    )
    duplicate = build_periodic_site_graph(
        crystal,
        disorder_policy=DisorderPolicy(
            "explicit", selected_groups=(("A", 1), ("A", 2))
        ),
    )
    assert duplicate.disorder_status == "indeterminate"
    assert duplicate.disorder_reason == "explicit_selects_two_groups"
    assert len(duplicate.nodes) == 3
    missing = build_periodic_site_graph(
        crystal,
        disorder_policy=DisorderPolicy("explicit"),
    )
    assert missing.disorder_reason == "explicit_selection_incomplete"
    chosen = build_periodic_site_graph(
        crystal,
        disorder_policy=DisorderPolicy("explicit", selected_groups=(("A", 2),)),
    )
    assert chosen.disorder_status == "conditional"
    assert {node.site_ids[0] for node in chosen.nodes} == {"part2", "ordered"}
    full = _sites(
        "CC",
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        **{
            KEY_OCCUPANCY: [1.0, 1.0],
            KEY_DISORDER_GROUP: [1, 2],
            KEY_ASSEMBLY: np.array(["A", "A"], dtype=object),
            KEY_ATOM_ID: np.array(["keep", "drop"], dtype=object),
        },
    )
    resolved = build_periodic_site_graph(
        full,
        disorder_policy=DisorderPolicy("explicit", selected_groups=(("A", 1),)),
    )
    assert resolved.disorder_status == "resolved"
    assert resolved.disorder_reason == "explicit_full_occupancy"
    assert [node.site_ids for node in resolved.nodes] == [("keep",)]


def test_policy_mapping_rejects_unknown_keys():
    parsed = policies_from_mapping(
        {
            "nodes": {"kind": "atom_sites"},
            "edges": {"kind": "contacts", "cutoff_A": 2.0},
        }
    )
    assert parsed["nodes"].kind == "atom_sites"
    assert parsed["edges"].cutoff_A == 2.0
    with pytest.raises(ValueError, match="unknown policy keys"):
        policies_from_mapping({"prototype": "sql"})
    with pytest.raises(ValueError, match="unknown nodes keys"):
        policies_from_mapping({"nodes": {"element": "Cu"}})
