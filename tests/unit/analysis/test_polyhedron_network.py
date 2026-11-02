"""Synthetic coordination-polyhedron networks.

The crystals in this module are constructed in memory. They are not
experimental structures.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from click.testing import CliRunner

from molcrys_kit.analysis import find_polyhedra
from molcrys_kit.analysis.polyhedron_network import (
    CoordinationShell,
    ShellCandidate,
    ShellVertex,
    molecular_polyhedron_network,
    polyhedron_network_from_shells,
)
from molcrys_kit.chemistry import (
    ChemicalBond,
    analyze_periodic_topology,
    integer_translation_basis,
)
from molcrys_kit.cli import main
from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_OCCUPANCY,
    SHELL_TIER_DISTANCE_TOLERANCE_A,
)
from molcrys_kit.io import write_extxyz
from molcrys_kit.structures import MolecularCrystal

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

LATTICE = np.eye(3) * 30.0


def _crystal(molecules, lattice=None, pbc=(True, True, True)) -> MolecularCrystal:
    return MolecularCrystal(
        np.asarray(LATTICE if lattice is None else lattice, dtype=float),
        molecules,
        pbc=pbc,
    )


def _atom(symbol: str, position, lattice=None) -> Atoms:
    cell = np.eye(3) * 30.0 if lattice is None else np.asarray(lattice, dtype=float)
    return Atoms(symbols=[symbol], positions=[position], cell=cell, pbc=True)


def _chain(cell=10.0):
    lattice = np.eye(3) * cell
    return _crystal(
        [
            _atom("N", [0.0, 0.0, 0.0], lattice),
            _atom("Cl", [cell / 2.0, 0.0, 0.0], lattice),
        ],
        lattice=lattice,
    )


def _vertex(
    molecule: int, position, shift=(0, 0, 0), formula="Cl", species="Cl_1"
) -> ShellVertex:
    point = tuple(float(component) for component in position)
    return ShellVertex(
        molecule_index=molecule,
        realization_index=0,
        image_shift=tuple(shift),
        position_A=point,
        distance_A=float(np.linalg.norm(point)),
        formula=formula,
        species_id=species,
    )


def _candidate(molecule: int, position, shift=(0, 0, 0)) -> ShellCandidate:
    point = tuple(float(component) for component in position)
    return ShellCandidate(
        molecule_index=molecule,
        realization_index=0,
        image_shift=tuple(shift),
        position_A=point,
        distance_A=float(np.linalg.norm(point)),
        formula="Cl",
        species_id="Cl_1",
    )


def _shell(
    molecule: int,
    centre,
    vertices,
    outside=(),
    *,
    search=20.0,
    center_kind="centroid",
    policy="major",
    role=None,
) -> CoordinationShell:
    return CoordinationShell(
        molecule_index=molecule,
        realization_index=0,
        formula="N",
        species_id="N_1",
        center_position_A=tuple(float(component) for component in centre),
        vertices=tuple(vertices),
        criterion="provided",
        primary_gap_cn=len(vertices),
        enclosure_expanded=False,
        search_cutoff_A=search,
        centre_role=role,
        center_kind=center_kind,
        centre_disorder_policy=policy,
        ligand_disorder_policy="major",
        outside_candidates=None if outside is None else tuple(outside),
        hard_cutoff_A=None,
    )


def _contact(network, left: int, right: int):
    matches = [
        contact
        for contact in network.contacts
        if {contact.left.molecule_index, contact.right.molecule_index} == {left, right}
        or (
            left == right
            and contact.left.molecule_index == left
            and contact.right.molecule_index == right
        )
    ]
    assert len(matches) == 1, [
        (item.left, item.right, item.relation) for item in network.contacts
    ]
    return matches[0]


def _partial_nitrogen(origin, group: int, occupancy: float, lattice) -> Atoms:
    atoms = _atom("N", origin, lattice)
    atoms.set_array(KEY_DISORDER_GROUP, np.array([group], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([occupancy], dtype=float))
    atoms.set_array(KEY_ASSEMBLY, np.array([""], dtype=object))
    return atoms


def test_integer_translation_basis_keeps_lattice_generators():
    assert integer_translation_basis([(2, 0, 0), (3, 0, 0)]) == ((1, 0, 0),)
    assert integer_translation_basis([(3, 0, 0), (2, 0, 0)]) == ((1, 0, 0),)
    assert integer_translation_basis([(2, 0, 0)]) == ((2, 0, 0),)
    assert integer_translation_basis([(-2, 0, 0)]) == ((2, 0, 0),)
    assert integer_translation_basis([(2, 4, 6)]) == ((2, 4, 6),)
    assert integer_translation_basis([(4, 0, 0), (6, 0, 0)]) == ((2, 0, 0),)
    assert integer_translation_basis([]) == ()
    assert integer_translation_basis([(0, 0, 0)]) == ()
    rank_two = integer_translation_basis([(2, 0, 0), (0, 4, 0), (2, 4, 0)])
    assert len(rank_two) == 2

    topology = analyze_periodic_topology(
        ("a",),
        (
            ChemicalBond("a", "a", atom2_image_shift=(2, 0, 0)),
            ChemicalBond("a", "a", atom2_image_shift=(3, 0, 0)),
        ),
    )
    assert topology.rank == 1
    assert topology.translation_generators == ((2, 0, 0),)


def test_self_image_edge_preserves_nonprimitive_shift():
    shell = _shell(
        0,
        (10.0, 0.0, 0.0),
        [
            _vertex(1, (0.0, 0.0, 0.0), (0, 0, 0)),
            _vertex(1, (20.0, 0.0, 0.0), (2, 0, 0)),
        ],
    )
    network = polyhedron_network_from_shells(
        [shell],
        np.eye(3) * 10.0,
        include_cycle_translations=True,
    )
    assert len(network.contacts) == 1
    contact = network.contacts[0]
    assert contact.left == contact.right
    assert contact.right_image_shift == (2, 0, 0)
    assert {vertex.molecule_index for vertex in contact.shared_vertices} == {1}
    component = network.components[0]
    assert component.rank == 1
    assert component.translation_generators == ((2, 0, 0),)
    assert component.cycle_translations == ((2, 0, 0),)
    assert "cycle_translations" not in (
        polyhedron_network_from_shells([shell], np.eye(3) * 10.0)
        .components[0]
        .to_dict()
    )


def test_chain_records_distinct_images_of_one_ligand_and_a_self_edge():
    crystal = _chain()
    report = molecular_polyhedron_network(
        crystal,
        "N",
        "Cl",
        cutoff=6.0,
        hard_cutoff=6.0,
        centre_role="A",
    )
    assert report.configuration_status == "resolved"
    assert report.ligand_disorder == "major"
    assert report.warnings == ()
    network = report.networks[0]
    assert network.pbc == (True, True, True)
    assert len(network.shells) == 1
    shell = network.shells[0]
    assert shell.centre_role == "A"
    assert [vertex.molecule_index for vertex in shell.vertices] == [1, 1]
    assert {vertex.image_shift for vertex in shell.vertices} == {(0, 0, 0), (-1, 0, 0)}
    contact = network.contacts[0]
    assert contact.left.molecule_index == contact.right.molecule_index == 0
    assert contact.right_image_shift == (1, 0, 0)
    assert network.components[0].rank == 1
    assert network.components[0].translation_generators == ((1, 0, 0),)

    payload = report.to_dict()
    encoded = payload["networks"][0]
    assert encoded["components"][0]["rank"] == len(
        encoded["components"][0]["translation_generators"]
    )
    assert {
        tuple(vertex["image_shift"]) for vertex in encoded["shells"][0]["vertices"]
    } == {tuple(vertex.image_shift) for vertex in shell.vertices}
    assert encoded["contacts"][0]["right_image_shift"] == [1, 0, 0]
    assert "cycle_translations" not in encoded["components"][0]


def test_disconnected_chains_keep_independent_ranks():
    lattice = np.eye(3) * 10.0
    crystal = _crystal(
        [
            _atom("N", [0.0, 0.0, 0.0], lattice),
            _atom("Cl", [5.0, 0.0, 0.0], lattice),
            _atom("N", [0.0, 0.0, 5.0], lattice),
            _atom("Cl", [0.0, 5.0, 5.0], lattice),
        ],
        lattice=lattice,
    )
    network = molecular_polyhedron_network(
        crystal, "N", "Cl", cutoff=6.0, hard_cutoff=6.0
    ).networks[0]
    assert len(network.components) == 2
    assert sorted(component.rank for component in network.components) == [1, 1]
    assert {component.translation_generators for component in network.components} == {
        ((1, 0, 0),),
        ((0, 1, 0),),
    }
    assert len(network.contacts) == 2


def test_supercell_preserves_rank():
    crystal = _chain()
    primitive = molecular_polyhedron_network(
        crystal, "N", "Cl", cutoff=6.0, hard_cutoff=6.0
    ).networks[0]
    supercell = molecular_polyhedron_network(
        crystal.get_supercell(2, 1, 1),
        "N",
        "Cl",
        cutoff=6.0,
        hard_cutoff=6.0,
    ).networks[0]
    assert primitive.components[0].rank == 1
    assert [component.rank for component in supercell.components] == [1]


def test_isomers_are_split_before_the_shell_is_chosen():
    ethanol = Atoms(
        symbols=["C", "C", "O"],
        positions=[[-1.3, 6.0, 0.0], [0.0, 6.0, 0.0], [1.3, 6.0, 0.0]],
        cell=LATTICE,
        pbc=True,
    )
    ether = Atoms(
        symbols=["C", "O", "C"],
        positions=[[2.7, 0.0, 0.0], [4.0, 0.0, 0.0], [5.3, 0.0, 0.0]],
        cell=LATTICE,
        pbc=True,
    )
    crystal = _crystal([_atom("N", [0.0, 0.0, 0.0]), ethanol, ether])

    def shell_for(**kwargs):
        report = molecular_polyhedron_network(
            crystal,
            "N",
            "C2 H6 O",
            cutoff=8.0,
            enforce_enclosure=False,
            **kwargs,
        )
        assert len(report.networks) == 1
        return report.networks[0].shells[0]

    unfiltered = shell_for()
    assert len(unfiltered.vertices) == 1
    assert unfiltered.vertices[0].molecule_index == 2
    assert unfiltered.vertices[0].distance_A == pytest.approx(4.0)
    ethanol_candidate = unfiltered.outside_candidates[0]
    assert ethanol_candidate.molecule_index == 1
    assert ethanol_candidate.distance_A == pytest.approx(6.0)
    assert ethanol_candidate.species_id != unfiltered.vertices[0].species_id

    selected = shell_for(ligand_species=ethanol_candidate.species_id)
    assert [vertex.molecule_index for vertex in selected.vertices] == [1]
    assert selected.vertices[0].distance_A == pytest.approx(6.0)
    assert selected.outside_candidates == ()

    by_index = shell_for(ligand_indices=[1])
    assert [vertex.molecule_index for vertex in by_index.vertices] == [1]

    charged = shell_for(ligand_species=unfiltered.vertices[0].species_id)
    assert charged.vertices[0].molecule_index == 2
    charged_moiety = molecular_polyhedron_network(
        crystal,
        "N",
        "C2 O 2-",
        cutoff=8.0,
        enforce_enclosure=False,
        ligand_indices=[2],
    )
    assert charged_moiety.networks[0].shells[0].vertices[0].molecule_index == 2
    assert (
        molecular_polyhedron_network(
            crystal,
            "N",
            "C2O",
            cutoff=8.0,
            central_species="N_2",
        ).networks
        == ()
    )


def test_exclusive_centres_are_not_merged_and_as_refined_stays_unresolved():
    lattice = np.eye(3) * 20.0
    crystal = _crystal(
        [
            _partial_nitrogen([0.0, 0.0, 0.0], 1, 0.70, lattice),
            _partial_nitrogen([0.30, 0.0, 0.0], 2, 0.30, lattice),
        ],
        lattice=lattice,
    )
    major = molecular_polyhedron_network(crystal, "N", "Cl", disorder_policy="major")
    assert major.configuration_status == "resolved"
    assert [shell.molecule_index for shell in major.networks[0].shells] == [0]

    tied = _crystal(
        [
            _partial_nitrogen([0.0, 0.0, 0.0], 1, 0.50, lattice),
            _partial_nitrogen([0.30, 0.0, 0.0], 2, 0.50, lattice),
        ],
        lattice=lattice,
    )
    tie = molecular_polyhedron_network(tied, "N", "Cl", disorder_policy="major")
    assert tie.configuration_status == "tie_broken"
    assert len(tie.networks) == 1
    assert len(tie.networks[0].shells) == 1
    assert any("tie-break" in warning for warning in tie.warnings)

    exclusive = molecular_polyhedron_network(
        crystal, "N", "Cl", disorder_policy="exclusive"
    )
    assert exclusive.configuration_status == "conditional_local"
    assert len(exclusive.networks) == 2
    assert [len(network.shells) for network in exclusive.networks] == [1, 1]
    assert {network.shells[0].molecule_index for network in exclusive.networks} == {
        0,
        1,
    }
    assert any("joint" in warning for warning in exclusive.warnings)

    refined = molecular_polyhedron_network(
        crystal, "N", "Cl", disorder_policy="as_refined"
    )
    assert refined.configuration_status == "unresolved"
    assert len(refined.networks) == 1
    assert {shell.molecule_index for shell in refined.networks[0].shells} == {0, 1}
    assert any("simultaneously" in warning for warning in refined.warnings)


def test_add_next_tier_reports_contact_and_coordination_changes():
    original = [
        _shell(
            0,
            (0.0, 0.0, 0.0),
            [_vertex(10, (2.0, 0.0, 0.0))],
            [_candidate(11, (4.0, 0.0, 0.0))],
        ),
        _shell(
            1,
            (8.0, 0.0, 0.0),
            [_vertex(11, (4.0, 0.0, 0.0))],
            [_candidate(12, (8.0, 0.0, 6.0))],
        ),
    ]
    network = polyhedron_network_from_shells(original, LATTICE)
    dropped = network.boundary.drop_farthest_tier
    assert dropped.status == "insufficient"
    assert dropped.reason == "single_distance_tier"
    assert {item.reason for item in dropped.unperturbed_centres} == {
        "single_distance_tier"
    }
    assert dropped.added is None
    assert dropped.removed is None
    assert dropped.changed is None
    assert dropped.coordination is None
    assert dropped.distance_tolerance_A == SHELL_TIER_DISTANCE_TOLERANCE_A

    added = network.boundary.add_next_tier
    assert added.status == "checked"
    assert added.reason is None
    assert len(added.added) == 1
    assert added.removed == ()
    assert {(item.before, item.after) for item in added.coordination} == {(1, 2)}
    assert added.distance_tolerance_A == pytest.approx(1e-4)

    private = polyhedron_network_from_shells(
        [
            _shell(
                0,
                (0.0, 0.0, 0.0),
                [_vertex(10, (2.0, 0.0, 0.0))],
                [_candidate(11, (4.0, 0.0, 0.0))],
            )
        ],
        LATTICE,
    )
    private_add = private.boundary.add_next_tier
    assert private_add.status == "checked"
    assert private_add.added == ()
    assert private_add.coordination[0].before == 1
    assert private_add.coordination[0].after == 2


def test_missing_outside_candidates_block_the_whole_addition():
    network = polyhedron_network_from_shells(
        [
            _shell(
                0,
                (0.0, 0.0, 0.0),
                [_vertex(10, (2.0, 0.0, 0.0)), _vertex(11, (5.0, 0.0, 0.0))],
                [_candidate(12, (8.0, 0.0, 0.0))],
            ),
            _shell(
                1,
                (0.0, 6.0, 0.0),
                [_vertex(13, (0.0, 8.0, 0.0))],
                None,
            ),
        ],
        LATTICE,
    )
    addition = network.boundary.add_next_tier
    assert addition.status == "insufficient"
    assert addition.reason == "outside_candidates_missing"
    assert [
        (item.molecule_index, item.reason) for item in addition.unperturbed_centres
    ] == [(1, "outside_candidates_missing")]
    assert addition.added is None
    assert addition.coordination is None
    assert network.boundary.drop_farthest_tier.status == "insufficient"


def test_next_tier_at_the_search_boundary_is_not_partial():
    network = polyhedron_network_from_shells(
        [
            _shell(
                0,
                (0.0, 0.0, 0.0),
                [_vertex(10, (1.0, 0.0, 0.0))],
                [_candidate(11, (4.99995, 0.0, 0.0))],
                search=5.0,
            )
        ],
        LATTICE,
    )
    addition = network.boundary.add_next_tier
    assert addition.status == "insufficient"
    assert addition.reason == "search_boundary"
    assert addition.coordination is None
    assert addition.to_dict()["added"] is None


def test_corner_edge_face_and_degenerate_shells():
    octahedron = [
        _vertex(1, (1.0, 0.0, 0.0)),
        _vertex(2, (0.0, 1.0, 0.0)),
        _vertex(3, (0.0, 0.0, 1.0)),
        _vertex(4, (-1.0, 0.0, 0.0)),
        _vertex(5, (0.0, -1.0, 0.0)),
        _vertex(6, (0.0, 0.0, -1.0)),
    ]
    network = polyhedron_network_from_shells(
        [
            _shell(0, (0.0, 0.0, 0.0), octahedron),
            _shell(
                7,
                (5.0, 0.0, 0.0),
                [
                    _vertex(1, (1.0, 0.0, 0.0)),
                    _vertex(31, (9.0, 0.0, 0.0)),
                    _vertex(32, (5.0, 1.0, 0.0)),
                    _vertex(33, (5.0, -1.0, 0.0)),
                    _vertex(34, (5.0, 0.0, 1.0)),
                    _vertex(35, (5.0, 0.0, -1.0)),
                ],
            ),
            _shell(
                8,
                (0.0, 5.0, 0.0),
                [
                    _vertex(1, (1.0, 0.0, 0.0)),
                    _vertex(2, (0.0, 1.0, 0.0)),
                    _vertex(80, (0.0, 0.0, 2.0)),
                    _vertex(81, (2.0, 2.0, 2.0)),
                ],
            ),
            _shell(
                9,
                (0.0, 0.0, 5.0),
                [
                    _vertex(1, (1.0, 0.0, 0.0)),
                    _vertex(2, (0.0, 1.0, 0.0)),
                    _vertex(3, (0.0, 0.0, 1.0)),
                    _vertex(90, (2.0, 2.0, 2.0)),
                ],
            ),
            _shell(
                10,
                (0.0, -5.0, 0.0),
                [
                    _vertex(1, (1.0, 0.0, 0.0)),
                    _vertex(4, (-1.0, 0.0, 0.0)),
                    _vertex(101, (0.0, 2.0, 0.0)),
                    _vertex(102, (0.0, -2.0, 0.0)),
                    _vertex(103, (0.0, 0.0, 2.0)),
                    _vertex(104, (0.0, 0.0, -2.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(network, 0, 7).relation == "corner"
    assert _contact(network, 0, 7).ambiguity is None
    assert _contact(network, 0, 8).relation == "edge"
    assert _contact(network, 0, 9).relation == "face"
    opposite = _contact(network, 0, 10)
    assert opposite.relation == "ambiguous"
    assert opposite.ambiguity == "non_edge"

    line = [
        _vertex(201, (-1.0, 0.0, 0.0)),
        _vertex(202, (0.0, 0.0, 0.0)),
        _vertex(203, (1.0, 0.0, 0.0)),
    ]
    other_line = [
        _vertex(204, (-2.0, 0.0, 0.0)),
        _vertex(202, (0.0, 0.0, 0.0)),
        _vertex(205, (2.0, 0.0, 0.0)),
    ]
    midpoint = polyhedron_network_from_shells(
        [_shell(20, (0.0, 3.0, 0.0), line), _shell(21, (0.0, -3.0, 0.0), other_line)],
        LATTICE,
        classify_boundary=False,
    )
    shared_midpoint = _contact(midpoint, 20, 21)
    assert shared_midpoint.relation == "ambiguous"
    assert shared_midpoint.ambiguity == "non_vertex"

    endpoint = polyhedron_network_from_shells(
        [
            _shell(20, (0.0, 3.0, 0.0), line),
            _shell(
                22,
                (4.0, 0.0, 0.0),
                [_vertex(201, (-1.0, 0.0, 0.0)), _vertex(206, (4.0, 1.0, 0.0))],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(endpoint, 20, 22).relation == "corner"

    both_ends = polyhedron_network_from_shells(
        [
            _shell(20, (0.0, 3.0, 0.0), line),
            _shell(
                23,
                (0.0, 8.0, 0.0),
                [
                    _vertex(201, (-1.0, 0.0, 0.0)),
                    _vertex(203, (1.0, 0.0, 0.0)),
                    _vertex(207, (0.0, 4.0, 0.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(both_ends, 20, 23).relation == "edge"

    square = [
        _vertex(41, (0.0, 0.0, 0.0)),
        _vertex(42, (1.0, 0.0, 0.0)),
        _vertex(43, (1.0, 1.0, 0.0)),
        _vertex(44, (0.0, 1.0, 0.0)),
    ]
    diagonal = polyhedron_network_from_shells(
        [
            _shell(30, (0.5, 0.5, 1.0), square),
            _shell(
                31,
                (0.5, 0.5, -1.0),
                [
                    _vertex(41, (0.0, 0.0, 0.0)),
                    _vertex(43, (1.0, 1.0, 0.0)),
                    _vertex(51, (0.0, 1.0, 0.0)),
                    _vertex(52, (1.0, 0.0, 0.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(diagonal, 30, 31).ambiguity == "non_edge"

    whole_face = polyhedron_network_from_shells(
        [
            _shell(30, (0.5, 0.5, 1.0), square),
            _shell(32, (0.5, 0.5, 3.0), square),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(whole_face, 30, 32).relation == "face"

    mismatch = polyhedron_network_from_shells(
        [
            _shell(
                40,
                (0.0, 0.0, 2.0),
                [
                    _vertex(61, (0.0, 0.0, 0.0)),
                    _vertex(62, (1.0, 1.0, 0.0)),
                    _vertex(63, (0.0, 1.0, 1.0)),
                ],
            ),
            _shell(
                41,
                (0.0, 0.0, -2.0),
                [
                    _vertex(61, (0.0, 0.0, 0.0)),
                    _vertex(62, (1.0, 1.0, 0.0)),
                    _vertex(71, (0.0, 1.0, 0.0)),
                    _vertex(72, (1.0, 0.0, 0.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(mismatch, 40, 41).ambiguity == "facet_mismatch"

    inconsistent = polyhedron_network_from_shells(
        [
            _shell(
                50,
                (0.5, 0.5, 2.0),
                [
                    _vertex(81, (-1.0, -1.0, 0.0)),
                    _vertex(82, (0.0, 0.0, 0.0)),
                    _vertex(83, (1.0, 1.0, 0.0)),
                    _vertex(84, (2.0, 2.0, 0.0)),
                ],
            ),
            _shell(
                51,
                (0.5, 0.5, -2.0),
                [
                    _vertex(82, (0.0, 0.0, 0.0)),
                    _vertex(83, (1.0, 1.0, 0.0)),
                    _vertex(91, (0.0, 1.0, 0.0)),
                    _vertex(92, (1.0, 0.0, 0.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(inconsistent, 50, 51).ambiguity == "inconsistent_geometry"


def test_failed_hull_is_a_degenerate_shell(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise RuntimeError("hull failed")

    monkeypatch.setattr("molcrys_kit.analysis.polyhedron_network.ConvexHull", _fail)
    tetrahedron = [
        _vertex(1, (1.0, 1.0, 1.0)),
        _vertex(2, (1.0, -1.0, -1.0)),
        _vertex(3, (-1.0, 1.0, -1.0)),
        _vertex(4, (-1.0, -1.0, 1.0)),
    ]
    network = polyhedron_network_from_shells(
        [
            _shell(0, (0.0, 0.0, 0.0), tetrahedron),
            _shell(
                8,
                (4.0, 0.0, 0.0),
                [
                    _vertex(1, (1.0, 1.0, 1.0)),
                    _vertex(5, (5.0, 1.0, 0.0)),
                    _vertex(6, (5.0, -1.0, 0.0)),
                    _vertex(7, (3.0, 0.0, 2.0)),
                ],
            ),
        ],
        LATTICE,
        classify_boundary=False,
    )
    assert _contact(network, 0, 8).ambiguity == "degenerate_shell"


def test_oblique_cell_keeps_enumerated_image_shift():
    lattice = np.array(
        [
            [10.0, 0.0, 0.0],
            [3.0, 8.0, 0.0],
            [0.2, 0.1, 12.0],
        ],
        dtype=float,
    )
    ligand = 2.0 * lattice[1] + np.array([0.5, 0.0, 0.0])
    crystal = _crystal(
        [_atom("N", [0.0, 0.0, 0.0], lattice), _atom("Cl", ligand, lattice)],
        lattice=lattice,
    )
    records = find_polyhedra(crystal, "N", "Cl", level="molecule", cutoff=2.0)
    assert records[0]["shell_image_shifts"] == [[0, -2, 0]]
    shift = np.asarray(records[0]["shell_image_shifts"][0], dtype=float)
    reference = np.asarray(crystal.molecules[1].get_positions()[0], dtype=float)
    assert np.allclose(records[0]["shell_coords"][0], reference + shift @ lattice)
    lengths = np.linalg.norm(lattice, axis=1)
    naive = np.rint(np.asarray(records[0]["shell_offsets"][0]) / lengths).astype(int)
    assert list(naive) != records[0]["shell_image_shifts"][0]

    records, diagnostics = find_polyhedra(
        crystal,
        "N",
        "Cl",
        level="molecule",
        cutoff=2.0,
        return_diagnostics=True,
    )
    assert records[0]["shell_image_shifts"] == [[0, -2, 0]]
    assert diagnostics[0]["centre_index"] == 0
    network = molecular_polyhedron_network(crystal, "N", "Cl", cutoff=2.0).networks[0]
    assert network.shells[0].vertices[0].image_shift == (0, -2, 0)


def test_find_polyhedra_keeps_existing_molecule_fields_and_atom_records():
    crystal = _chain(cell=5.0)
    records = find_polyhedra(
        crystal, "N", "Cl", level="molecule", cutoff=3.0, hard_cutoff=3.0
    )
    expected = {
        "level",
        "center_molecule_index",
        "center_formula",
        "center_position",
        "center_kind",
        "shell_molecule_indices",
        "disorder_policy",
        "disorder_component_ids",
        "shell_formula",
        "shell_offsets",
        "shell_image_shifts",
        "shell_coords",
        "shell_distances",
        "coordination_number",
        "mode",
        "primary_gap_cn",
        "gap_index",
        "gap_value",
        "enclosed",
        "enclosure_expanded",
        "cutoff",
        "search_cutoff",
        "hard_cutoff",
    }
    assert expected <= set(records[0])
    assert len(records[0]["shell_image_shifts"]) == len(
        records[0]["shell_molecule_indices"]
    )
    assert all(
        isinstance(component, int)
        for shift in records[0]["shell_image_shifts"]
        for component in shift
    )

    atoms = Atoms(
        "NaCl",
        positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atom_records = find_polyhedra(atoms, "Na", "Cl", cutoff=4.0)
    assert atom_records
    assert "shell_image_shifts" not in atom_records[0]
    assert "shell_indices" in atom_records[0]


def test_partial_pbc_and_inconsistent_positions_are_rejected():
    crystal = _crystal(
        [_atom("N", [0.0, 0.0, 0.0]), _atom("Cl", [3.0, 0.0, 0.0])],
        pbc=(True, True, False),
    )
    with pytest.raises(ValueError, match="pbc"):
        molecular_polyhedron_network(crystal, "N", "Cl")
    with pytest.raises(ValueError, match="pbc"):
        polyhedron_network_from_shells([], np.eye(3), pbc=(False, True, True))

    inconsistent = _shell(
        0,
        (0.0, 0.0, 0.0),
        [
            _vertex(1, (0.0, 0.0, 0.0), (0, 0, 0)),
            _vertex(1, (0.0, 0.0, 0.0), (1, 0, 0)),
        ],
    )
    with pytest.raises(ValueError, match="image_shift"):
        polyhedron_network_from_shells([inconsistent], np.eye(3) * 10.0)


def test_network_summary_fields_follow_the_shells():
    shells = [
        _shell(
            0, (0.0, 0.0, 0.0), [_vertex(1, (2.0, 0.0, 0.0))], center_kind="centroid"
        ),
        _shell(
            2,
            (6.0, 0.0, 0.0),
            [_vertex(3, (4.0, 0.0, 0.0))],
            center_kind="com",
            policy="exclusive",
        ),
    ]
    mixed = polyhedron_network_from_shells(shells, LATTICE, classify_boundary=False)
    assert mixed.boundary is None
    assert mixed.center_kind is None
    assert mixed.centre_disorder_policy is None
    assert mixed.ligand_disorder_policy == "major"


def test_polyhedron_network_cli_reports_one_centre_type(tmp_path):
    path = tmp_path / "chain.extxyz"
    write_extxyz(_chain(), path)
    runner = CliRunner()
    rejected = runner.invoke(
        main,
        [
            "analyze",
            "polyhedron-network",
            str(path),
            "--central",
            "N",
            "--ligand",
            "Cl",
            "--cutoff",
            "0",
        ],
    )
    assert rejected.exit_code != 0

    text = runner.invoke(
        main,
        [
            "analyze",
            "polyhedron-network",
            str(path),
            "--central",
            "N",
            "--ligand",
            "Cl",
            "--cutoff",
            "6",
            "--hard-cutoff",
            "6",
        ],
    )
    assert text.exit_code == 0, text.output
    assert "Configuration: resolved" in text.output
    assert "ranks=[1]" in text.output

    rendered = runner.invoke(
        main,
        [
            "analyze",
            "polyhedron-network",
            str(path),
            "--central",
            "N",
            "--ligand",
            "Cl",
            "--cutoff",
            "6",
            "--hard-cutoff",
            "6",
            "--json",
        ],
    )
    assert rendered.exit_code == 0, rendered.output
    payload = json.loads(rendered.output)
    network = payload["networks"][0]
    assert network["components"][0]["rank"] == 1
    assert (
        len(network["shells"][0]["vertices"])
        == len(network["contacts"][0]["shared_vertices"]) + 1
    )
