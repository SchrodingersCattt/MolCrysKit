"""Tests for composite rigid and internal-DOF interpolation."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations import (
    interpolate_molecule_with_internal_dofs,
    rotate_fragment_about_bond,
)
from molcrys_kit.operations.interpolation import best_atom_mapping
from molcrys_kit.structures.molecule import CrystalMolecule


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
    return CrystalMolecule(Atoms("C8", positions=positions))


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
