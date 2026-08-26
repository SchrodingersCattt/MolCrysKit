"""Tests for graph-based bond/fragment rotation operations."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from ase import Atoms

from molcrys_kit.analysis.disorder.provenance import DisorderProvenance
from molcrys_kit.constants.config import KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z, KEY_LABEL
from molcrys_kit.operations.bond_rotation import (
    BondNotFoundError,
    BondRotationError,
    BondRotationSelectionError,
    RingBondRotationError,
    partition_at_bond,
    rotate_fragment_about_bond,
    rotate_fragment_in_crystal,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.io.cif import identify_molecules
from molcrys_kit.utils.geometry import dihedral_angle


def _chain_molecule() -> CrystalMolecule:
    """Four-carbon non-collinear chain with bond graph 0-1-2-3."""
    atoms = Atoms(
        "CCCC",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.5, 1.0, 0.0],
                [3.5, 1.0, 1.0],
            ]
        ),
        pbc=False,
    )
    atoms.set_array(KEY_LABEL, np.array(["C0", "C1", "C2", "C3"]))
    atoms.set_array(KEY_FRAC_X, np.array([0.0, 0.1, 0.2, 0.3]))
    atoms.set_array(KEY_FRAC_Y, np.zeros(4))
    atoms.set_array(KEY_FRAC_Z, np.zeros(4))
    mol = CrystalMolecule(atoms, check_pbc=False)
    graph = mol.get_graph()
    assert set(graph.edges()) == {(0, 1), (1, 2), (2, 3)}
    return mol


def _ring_molecule() -> CrystalMolecule:
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    positions = np.column_stack(
        [1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)]
    )
    mol = CrystalMolecule(Atoms("C6", positions=positions, pbc=False), check_pbc=False)
    graph = mol.get_graph()
    assert graph.number_of_edges() == 6
    return mol


def _pairwise_distances(positions: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    subset = positions[list(indices)]
    return np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=-1)


def test_partition_at_bridge_bond():
    mol = _chain_molecule()
    partition = partition_at_bond(mol, 1, 2)

    assert partition.atom_i == 1
    assert partition.atom_j == 2
    assert partition.fixed_atoms == (0, 1)
    assert partition.moving_atoms == (2, 3)
    assert partition.is_ring_bond is False
    assert partition.component_sizes == (2, 2)


def test_rotate_fragment_about_bridge_preserves_fragments():
    mol = _chain_molecule()
    before = mol.get_positions().copy()
    before_moving_distances = _pairwise_distances(before, (2, 3))

    rotated = rotate_fragment_about_bond(mol, 1, 2, 90.0)
    after = rotated.get_positions()

    np.testing.assert_allclose(after[[0, 1]], before[[0, 1]], atol=1e-12)
    np.testing.assert_allclose(after[2], before[2], atol=1e-12)
    np.testing.assert_allclose(
        _pairwise_distances(after, (2, 3)), before_moving_distances, atol=1e-12
    )
    assert not np.allclose(after[3], before[3])
    np.testing.assert_allclose(mol.get_positions(), before, atol=1e-12)


def test_rotation_changes_dihedral_by_requested_angle():
    mol = _chain_molecule()
    before = mol.get_positions()
    phi_before = np.degrees(dihedral_angle(before[0], before[1], before[2], before[3]))

    rotated = rotate_fragment_about_bond(mol, 1, 2, 60.0)
    after = rotated.get_positions()
    phi_after = np.degrees(dihedral_angle(after[0], after[1], after[2], after[3]))

    delta = (phi_after - phi_before + 180.0) % 360.0 - 180.0
    assert delta == pytest.approx(60.0, abs=1e-8)


def test_positive_rotation_follows_right_hand_rule():
    mol = CrystalMolecule(
        Atoms(
            "CCC",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            pbc=False,
        ),
        check_pbc=False,
    )
    mol._graph = nx.path_graph(3)

    rotated = rotate_fragment_about_bond(mol, 0, 1, 90.0)

    np.testing.assert_allclose(
        rotated.get_positions()[2] - rotated.get_positions()[1],
        [0.0, 0.0, 1.0],
        atol=1e-12,
    )


def test_rotate_other_side():
    mol = _chain_molecule()
    before = mol.get_positions().copy()

    rotated = rotate_fragment_about_bond(mol, 1, 2, 45.0, moving_side="i")
    after = rotated.get_positions()

    np.testing.assert_allclose(after[[2, 3]], before[[2, 3]], atol=1e-12)
    assert not np.allclose(after[0], before[0])
    np.testing.assert_allclose(after[1], before[1], atol=1e-12)


def test_round_trip_and_full_turn():
    mol = _chain_molecule()
    original = mol.get_positions().copy()

    forward = rotate_fragment_about_bond(mol, 1, 2, 73.0)
    backward = rotate_fragment_about_bond(forward, 1, 2, -73.0)
    full_turn = rotate_fragment_about_bond(mol, 1, 2, 360.0)

    np.testing.assert_allclose(backward.get_positions(), original, atol=1e-10)
    np.testing.assert_allclose(full_turn.get_positions(), original, atol=1e-10)


def test_metadata_preserved_and_geometry_metadata_invalidated():
    mol = _chain_molecule()
    assert mol._graph is not None

    rotated = rotate_fragment_about_bond(mol, 1, 2, 30.0)

    np.testing.assert_array_equal(rotated.arrays[KEY_LABEL], mol.arrays[KEY_LABEL])
    for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
        assert key not in rotated.arrays
        assert key in mol.arrays
    assert rotated._graph is not None
    assert rotated._graph is not mol._graph
    assert set(rotated.get_graph().edges()) == {(0, 1), (1, 2), (2, 3)}


def test_rotation_preserves_explicit_topology_despite_close_contact():
    mol = CrystalMolecule(
        Atoms(
            "CCCC",
            positions=[
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            pbc=False,
        ),
        check_pbc=False,
    )
    mol._graph = nx.path_graph(4)

    rotated = rotate_fragment_about_bond(mol, 1, 2, 90.0)

    assert np.linalg.norm(rotated.positions[0] - rotated.positions[3]) < 1e-12
    assert set(rotated.get_graph().edges()) == {(0, 1), (1, 2), (2, 3)}


def test_ring_bond_rejected_with_cycle_details():
    mol = _ring_molecule()

    partition = partition_at_bond(mol, 0, 1)
    assert partition.is_ring_bond is True
    assert set(partition.cycle_atoms) == set(range(6))

    with pytest.raises(RingBondRotationError, match="ring"):
        rotate_fragment_about_bond(mol, 0, 1, 30.0)


def test_invalid_bond_and_selection():
    mol = _chain_molecule()

    with pytest.raises(BondNotFoundError):
        partition_at_bond(mol, 0, 3)
    with pytest.raises(IndexError):
        partition_at_bond(mol, -1, 1)
    with pytest.raises(IndexError):
        partition_at_bond(mol, 1, 9)
    with pytest.raises(BondRotationSelectionError):
        rotate_fragment_about_bond(mol, 1, 2, 30.0, moving_side="invalid")
    with pytest.raises(BondRotationSelectionError):
        rotate_fragment_about_bond(mol, 1, 2, 30.0, moving_atoms=[0, 3])


def test_disconnected_graph_is_rejected_before_ring_classification():
    mol = CrystalMolecule(
        Atoms("CCCC", positions=np.zeros((4, 3)), pbc=False), check_pbc=False
    )
    mol._graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    mol._graph.add_node(3)

    with pytest.raises(BondRotationSelectionError, match="connected"):
        partition_at_bond(mol, 0, 1)


def test_zero_length_axis_is_rejected():
    mol = CrystalMolecule(
        Atoms("CCC", positions=np.zeros((3, 3)), pbc=False), check_pbc=False
    )
    mol._graph = nx.path_graph(3)
    with pytest.raises(BondRotationError, match="zero length"):
        rotate_fragment_about_bond(mol, 0, 1, 30.0)


def test_identified_molecule_keeps_authoritative_local_topology():
    atoms = Atoms(
        "HeCCCC",
        positions=[
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.5, 0.0],
        ],
        pbc=False,
    )
    molecules = identify_molecules(atoms, bond_thresholds={("C", "C"): 2.1})
    chain = next(molecule for molecule in molecules if len(molecule) == 4)

    assert chain.info["atom_indices"] == [1, 2, 3, 4]
    assert chain.info["bond_pairs"] == [(1, 2), (2, 3), (3, 4)]
    assert set(chain.get_graph().nodes()) == {0, 1, 2, 3}
    assert set(chain.get_graph().edges()) == {(0, 1), (1, 2), (2, 3)}
    for atom_i, atom_j, edge_data in chain.get_graph().edges(data=True):
        assert edge_data["distance"] == pytest.approx(
            np.linalg.norm(edge_data["vector"])
        )
        np.testing.assert_allclose(
            edge_data["vector"],
            chain.positions[atom_j] - chain.positions[atom_i],
        )

    rotated = rotate_fragment_about_bond(chain, 1, 2, 30.0)
    for atom_i, atom_j, edge_data in rotated.get_graph().edges(data=True):
        assert edge_data["distance"] == pytest.approx(
            np.linalg.norm(edge_data["vector"])
        )
        np.testing.assert_allclose(
            edge_data["vector"],
            rotated.positions[atom_j] - rotated.positions[atom_i],
        )
        np.testing.assert_array_equal(edge_data["image_shift"], [0, 0, 0])
    assert all(
        record["right_image_shift"] == [0, 0, 0]
        for record in rotated.info["bond_records"]
    )
    for record in rotated.info["bond_records"]:
        np.testing.assert_allclose(
            record["vector"],
            rotated.positions[record["right"]] - rotated.positions[record["left"]],
        )


def test_identification_maps_noncontiguous_global_indices_locally():
    atoms = Atoms(
        "CHeCCHeHeC",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 10.0, 0.0],
            [6.0, 10.0, 0.0],
            [6.0, 0.5, 0.0],
        ],
        pbc=False,
    )
    molecules = identify_molecules(atoms, bond_thresholds={("C", "C"): 2.1})
    chain = next(molecule for molecule in molecules if len(molecule) == 4)

    assert chain.info["atom_indices"] == [0, 2, 3, 6]
    assert chain.info["bond_pairs"] == [(0, 2), (2, 3), (3, 6)]
    assert set(chain.get_graph().edges()) == {(0, 1), (1, 2), (2, 3)}


def test_crystal_wrapper_changes_only_selected_molecule():
    mol0 = _chain_molecule()
    mol1 = _chain_molecule()
    mol1.set_positions(mol1.get_positions() + np.array([0.0, 5.0, 0.0]))
    crystal = MolecularCrystal(np.eye(3) * 15.0, [mol0, mol1], pbc=(True, True, True))
    original0 = crystal.molecules[0].get_positions().copy()
    original1 = crystal.molecules[1].get_positions().copy()

    rotated = rotate_fragment_in_crystal(crystal, 0, 1, 2, 50.0)

    assert not np.allclose(rotated.molecules[0].get_positions(), original0)
    np.testing.assert_allclose(
        rotated.molecules[1].get_positions(), original1, atol=1e-12
    )
    np.testing.assert_allclose(
        crystal.molecules[0].get_positions(), original0, atol=1e-12
    )


def test_crystal_wrapper_preserves_frame_payloads_and_invalidates_results():
    mol0 = _chain_molecule()
    mol1 = _chain_molecule()
    crystal = MolecularCrystal(
        np.eye(3) * 15.0,
        [mol0, mol1],
        metadata={"frame_id": 7, "nested": {"labels": ["source"]}},
        extra_arrays={
            "site_id": np.arange(8),
            KEY_FRAC_X: np.linspace(0.0, 0.7, 8),
        },
        disorder_provenance=DisorderProvenance(
            kept_indices=[0, 1],
            dropped_indices=[2],
            method="test",
            coupled=False,
        ),
        calc_results={"energy": -1.0},
    )

    rotated = rotate_fragment_in_crystal(crystal, 1, 1, 2, 50.0)

    assert rotated.metadata == {
        "frame_id": 7,
        "nested": {"labels": ["source"]},
    }
    assert rotated.metadata is not crystal.metadata
    np.testing.assert_array_equal(rotated.extra_arrays["site_id"], np.arange(8))
    assert rotated.extra_arrays["site_id"] is not crystal.extra_arrays["site_id"]
    assert KEY_FRAC_X not in rotated.extra_arrays
    assert rotated._calc_results is None

    rotated.metadata["nested"]["labels"].append("output")
    rotated.disorder_provenance.kept_indices.append(3)
    assert crystal.metadata["nested"]["labels"] == ["source"]
    assert crystal.disorder_provenance.kept_indices == [0, 1]


def test_molecule_copy_deep_copies_graph_payloads():
    mol = _chain_molecule()
    mol.get_graph()[0][1]["payload"] = {"values": [1]}

    copied = mol.copy()
    copied.get_graph()[0][1]["payload"]["values"].append(2)

    assert mol.get_graph()[0][1]["payload"] == {"values": [1]}


def test_pbc_contiguous_coordinates_rotate_without_wrapping():
    mol = _chain_molecule()
    # Shift the contiguous molecule across a triclinic cell boundary. Coordinates
    # intentionally remain outside the primary cell.
    cell = np.array([[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [0.5, 1.0, 9.0]])
    mol.set_cell(cell)
    mol.set_pbc(True)
    mol.set_positions(mol.get_positions() + np.array([7.5, 1.0, 1.0]))
    before = mol.get_positions().copy()

    rotated = rotate_fragment_about_bond(mol, 1, 2, 35.0)

    np.testing.assert_allclose(
        rotated.get_positions()[[0, 1]], before[[0, 1]], atol=1e-12
    )
    # Operation must not wrap coordinates back into the cell.
    assert np.any(rotated.get_positions()[:, 0] > cell[0, 0])
