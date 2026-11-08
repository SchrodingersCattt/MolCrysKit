"""Tests for composite rigid and internal-DOF interpolation."""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations import (
    interpolate_molecule_with_internal_dofs,
    rotate_fragment_about_bond,
)
from molcrys_kit.operations._path_core import interpolate_rigid_positions
from molcrys_kit.operations.interpolation import best_atom_mapping
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.utils.geometry import get_rotation_matrix


def _butane_like() -> CrystalMolecule:
    return CrystalMolecule(
        Atoms(
            "CCCC",
            positions=[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.1, 1.3, 0.0],
                [3.2, 1.8, 1.0],
            ],
        )
    )


def _ring_with_bridge() -> CrystalMolecule:
    """Planar six-membered ring attached to a non-collinear bridge chain."""
    angles = np.arange(6) * np.pi / 3.0
    ring = np.column_stack((1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)))
    positions = np.vstack(
        [
            ring,
            # Atom 6 is the bridge atom and atom 7 defines its torsion.
            [[2.9, 0.0, 0.0], [4.0, 0.4, 1.0]],
        ]
    )
    # Distinct ring labels make the graph-aware endpoint mapping unique; an
    # all-carbon ring could otherwise be permuted by symmetry and hide a
    # rigid-fragment regression.
    return CrystalMolecule(
        Atoms(
            symbols=["C", "N", "O", "F", "P", "S", "Ne", "Ar"],
            positions=positions,
        )
    )


def test_bridge_torsion_path_preserves_bond_lengths_and_endpoints():
    mol_a = _butane_like()
    mol_b = rotate_fragment_about_bond(mol_a, 1, 2, 60.0)
    frames = interpolate_molecule_with_internal_dofs(mol_a, mol_b, n_images=7)

    assert len(frames) == 7
    order = best_atom_mapping(mol_a, mol_b)
    np.testing.assert_allclose(frames[0].get_positions(), mol_a.get_positions())
    np.testing.assert_allclose(
        frames[-1].get_positions(), mol_b.get_positions()[order], atol=1e-8
    )
    for frame in frames:
        assert frame.get_distance(0, 1) == pytest.approx(1.5, abs=2e-2)
        assert frame.get_distance(1, 2) == pytest.approx(1.432, abs=2e-2)
        assert frame.get_distance(2, 3) == pytest.approx(1.568, abs=2e-2)


def test_explicit_ring_cp_path_reaches_endpoint():
    angles = np.arange(6) * np.pi / 3.0
    base = np.column_stack((1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)))
    raised = base.copy()
    raised[[0, 2, 4], 2] = 0.25
    mol_a = CrystalMolecule(Atoms("C6", positions=base))
    mol_b = CrystalMolecule(Atoms("C6", positions=raised))
    frames = interpolate_molecule_with_internal_dofs(
        mol_a, mol_b, n_images=5, ring_atoms=tuple(range(6)), torsion_bonds=[]
    )

    assert len(frames) == 5
    order = best_atom_mapping(mol_a, mol_b)
    np.testing.assert_allclose(
        frames[-1].get_positions(), mol_b.get_positions()[order], atol=1e-8
    )
    assert np.isfinite(frames[2].get_positions()).all()


def test_invalid_image_count_is_rejected():
    with pytest.raises(ValueError, match="n_images"):
        interpolate_molecule_with_internal_dofs(
            _butane_like(), _butane_like(), n_images=0
        )


def test_noncollinear_torsion_converges_continuously_to_endpoint():
    mol_a = _butane_like()
    mol_b = rotate_fragment_about_bond(mol_a, 1, 2, 60.0)
    frames = interpolate_molecule_with_internal_dofs(mol_a, mol_b, n_images=101)

    previous_step = np.linalg.norm(
        frames[-2].get_positions() - frames[-3].get_positions()
    )
    final_step = np.linalg.norm(frames[-1].get_positions() - frames[-2].get_positions())
    assert final_step < 1.5 * previous_step
    np.testing.assert_allclose(
        frames[-1].get_positions(), mol_b.get_positions(), atol=1e-8
    )


def test_torsion_path_interpolates_residual_endpoint_motion():
    mol_a = _butane_like()
    mol_b = rotate_fragment_about_bond(mol_a, 1, 2, 60.0)
    target = mol_b.get_positions().copy()
    target[3] += np.array([0.7, 0.4, 0.9])
    mol_b = CrystalMolecule(Atoms("CCCC", positions=target))
    frames = interpolate_molecule_with_internal_dofs(mol_a, mol_b, n_images=101)

    assert (
        np.linalg.norm(frames[-2].get_positions()[3] - mol_a.get_positions()[3]) > 0.1
    )
    final_step = np.linalg.norm(frames[-1].get_positions() - frames[-2].get_positions())
    assert final_step < 0.1
    previous_step = np.linalg.norm(
        frames[-2].get_positions() - frames[-3].get_positions()
    )
    assert final_step < 1.5 * previous_step
    np.testing.assert_allclose(
        frames[-1].get_positions(), mol_b.get_positions(), atol=1e-8
    )


def test_ring_path_keeps_side_chain_motion():
    angles = np.arange(6) * np.pi / 3.0
    ring = np.column_stack((1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)))
    positions_a = np.vstack([ring, [[2.8, 0.0, 0.0]]])
    positions_b = positions_a.copy()
    positions_b[[0, 2, 4], 2] = 0.25
    positions_b[6] += np.array([0.4, 0.3, 0.2])
    mol_a = CrystalMolecule(Atoms("C7", positions=positions_a))
    mol_b = CrystalMolecule(Atoms("C7", positions=positions_b))
    frames = interpolate_molecule_with_internal_dofs(
        mol_a,
        mol_b,
        n_images=5,
        ring_atoms=tuple(range(6)),
        torsion_bonds=[],
    )

    assert not np.allclose(frames[2].get_positions()[6], frames[0].get_positions()[6])
    previous_step = np.linalg.norm(
        frames[-2].get_positions() - frames[-3].get_positions()
    )
    final_step = np.linalg.norm(frames[-1].get_positions() - frames[-2].get_positions())
    assert final_step < 1.5 * previous_step


def test_ring_plane_change_converges_continuously_to_endpoint():
    """CP interpolation must not snap when the endpoint plane changes."""
    angles = np.arange(6) * np.pi / 3.0
    ring = np.column_stack((1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)))
    positions_a = np.vstack([ring, [[2.8, 0.0, 0.0], [4.0, 0.4, 1.0]]])
    positions_b = positions_a.copy()
    plane_rotation = get_rotation_matrix(np.array([1.0, 0.3, 0.2]), 0.9)
    positions_b[:6] = ring @ plane_rotation.T
    positions_b[:6, 2] += np.array([0.25, -0.1, 0.2, -0.15, 0.1, -0.05])
    symbols = ["C", "N", "O", "F", "P", "S", "Ne", "Ar"]
    mol_a = CrystalMolecule(Atoms(symbols, positions=positions_a))
    mol_b = CrystalMolecule(Atoms(symbols, positions=positions_b))

    frames = interpolate_molecule_with_internal_dofs(
        mol_a,
        mol_b,
        n_images=101,
        ring_atoms=tuple(range(6)),
        torsion_bonds=[],
    )
    target = mol_b.get_positions()[best_atom_mapping(mol_a, mol_b)]
    previous_step = np.linalg.norm(
        frames[-2].get_positions() - frames[-3].get_positions()
    )
    final_step = np.linalg.norm(frames[-1].get_positions() - frames[-2].get_positions())
    assert final_step < 1.5 * previous_step
    assert np.linalg.norm(frames[-2].get_positions() - target) < 1.5 * final_step
    np.testing.assert_allclose(frames[-1].get_positions(), target, atol=1e-8)


def test_internal_torsion_and_rigid_pose_are_composed_once():
    """A global pose must be applied to the already interpolated torsion."""
    mol_a = _butane_like()
    torsion_target = rotate_fragment_about_bond(mol_a, 1, 2, 60.0)
    rotation = get_rotation_matrix(np.array([0.3, 0.6, 0.2]), 0.7)
    offset = np.array([3.0, -2.0, 1.0])
    mol_b = CrystalMolecule(
        Atoms(
            "CCCC",
            positions=torsion_target.get_positions() @ rotation.T + offset,
        )
    )

    frames = interpolate_molecule_with_internal_dofs(
        mol_a, mol_b, n_images=11, torsion_bonds=[(1, 2)]
    )
    fraction = 0.5
    internal = rotate_fragment_about_bond(mol_a, 1, 2, 30.0).get_positions()
    internal -= (
        np.average(internal, axis=0, weights=mol_a.get_masses())
        - mol_a.get_center_of_mass()
    )
    expected = interpolate_rigid_positions(
        internal,
        center=mol_a.get_center_of_mass(),
        rotation=rotation,
        translation=mol_b.get_center_of_mass() - mol_a.get_center_of_mass(),
        lam=fraction,
        method="se3_screw",
    )
    np.testing.assert_allclose(frames[5].get_positions(), expected, atol=1e-8)


@pytest.mark.parametrize("ring_selection", [None, tuple(range(6))])
def test_bridge_torsion_carries_rigid_ring_for_auto_and_explicit_ring(
    ring_selection,
):
    """A bridge torsion must not distort a rigid ring selected either way."""
    mol_a = _ring_with_bridge()
    # Direct the bond from the chain into the ring so the default moving side
    # is the complete ring component (atoms 0--5).
    mol_b = rotate_fragment_about_bond(mol_a, 6, 0, 120.0)
    frames = interpolate_molecule_with_internal_dofs(
        mol_a,
        mol_b,
        n_images=7,
        ring_atoms=ring_selection,
        torsion_bonds=[(6, 0)],
    )

    expected = np.full(6, 1.4)
    for frame in frames:
        positions = frame.get_positions()
        ring_distances = np.array(
            [np.linalg.norm(positions[i] - positions[(i + 1) % 6]) for i in range(6)]
        )
        np.testing.assert_allclose(ring_distances, expected, atol=1e-10)


def _contact_rows(records):
    return [
        (
            int(record["left"]),
            int(record["right"]),
            tuple(int(value) for value in record["right_image_shift"]),
            tuple(float(value) for value in record["vector"]),
        )
        for record in records
    ]


def test_interpolation_across_cell_face_keeps_periodic_bond_image():
    """Interpolating through a cell face must not rebase an explicit contact.

    Pose B moves one endpoint of a stored bond through the cell face while its
    partner stays inside. Recomputing images from the written coordinates would
    add a lattice translation to that zero-image vector. Endpoints stay
    crystal-global on the non-first molecule, including intermediate frames.
    """
    lattice = np.diag([10.0, 10.0, 10.0])
    spectator = CrystalMolecule(
        Atoms("He", positions=[[8.0, 8.0, 8.0]], cell=lattice, pbc=True),
        check_pbc=False,
    )
    spectator.info["atom_indices"] = [0]
    chain = CrystalMolecule(
        Atoms(
            "CCC",
            positions=[
                [1.0, 0.2, 5.0],
                [2.0, 0.2, 5.0],
                [2.0, 0.2, 5.4],
            ],
            cell=lattice,
            pbc=True,
        ),
        check_pbc=False,
    )
    chain.info["atom_indices"] = [1, 2, 3]
    chain.info["bond_records"] = [
        {
            "left": 1,
            "right": 2,
            "right_image_shift": [0, 0, 0],
            "vector": [1.0, 0.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 2,
            "right": 3,
            "right_image_shift": [0, 0, 0],
            "vector": [0.0, 0.0, 0.4],
            "role": "covalent",
        },
        {
            "left": 2,
            "right": 3,
            "right_image_shift": [0, 1, 0],
            "vector": [0.0, 10.0, 0.4],
            "role": "covalent",
        },
    ]
    chain._graph = nx.Graph([(0, 1), (1, 2)])
    crystal = MolecularCrystal(lattice, [spectator, chain], pbc=(True, True, True))
    molecule = crystal.molecules[1]
    assert [int(index) for index in molecule.info["atom_indices"]] == [1, 2, 3]
    source_contacts = _contact_rows(molecule.info["bond_records"])
    source_positions = np.asarray(molecule.get_positions(), dtype=float)
    inverse = np.linalg.inv(lattice)
    source_image = np.floor(source_positions @ inverse + 1e-12).astype(int)

    pose_b = molecule.copy()
    crossed = np.asarray(pose_b.get_positions(), dtype=float)
    crossed[2, 1] = -0.3
    pose_b.set_positions(crossed)

    frames = interpolate_molecule_with_internal_dofs(
        molecule,
        pose_b,
        n_images=5,
        torsion_bonds=[],
    )

    assert _contact_rows(crystal.molecules[1].info["bond_records"]) == source_contacts
    np.testing.assert_allclose(crystal.molecules[1].get_positions(), source_positions)
    assert len(frames) == 5
    crossed_face = False
    expected_identity = [
        (1, 2, (0, 0, 0)),
        (2, 3, (0, 0, 0)),
        (2, 3, (0, 1, 0)),
    ]
    for frame in frames:
        positions = np.asarray(frame.get_positions(), dtype=float)
        frame_image = np.floor(positions @ inverse + 1e-12).astype(int)
        crossed_face = crossed_face or not np.array_equal(frame_image, source_image)
        stored = [
            (
                int(record["left"]),
                int(record["right"]),
                tuple(int(value) for value in record["right_image_shift"]),
            )
            for record in frame.info["bond_records"]
        ]
        assert stored == expected_identity
        hosted = MolecularCrystal(
            lattice,
            [spectator.copy(), frame],
            pbc=(True, True, True),
        )
        bonds = [bond for bond in hosted.get_bond_records() if bond.molecule_index == 1]
        assert [
            (bond.left_global_index, bond.right_global_index, bond.right_image_shift)
            for bond in bonds
        ] == expected_identity
        for bond in bonds:
            delta = (
                positions[bond.right_local_index] - positions[bond.left_local_index]
            )
            expected = delta + np.asarray(bond.right_image_shift, dtype=float) @ lattice
            np.testing.assert_allclose(bond.vector_A, expected, atol=1e-8)
        zero_image = next(
            bond
            for bond in bonds
            if bond.left_global_index == 2 and bond.right_image_shift == (0, 0, 0)
        )
        intra_cell_delta = (
            positions[zero_image.right_local_index]
            - positions[zero_image.left_local_index]
        )
        if np.linalg.norm(intra_cell_delta) < 5.0:
            assert np.linalg.norm(zero_image.vector_A) < 5.0
    assert crossed_face
    assert frames[-1].positions[2, 1] < 0.0
