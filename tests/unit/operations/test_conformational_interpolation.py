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
                [1.54, 0.0, 0.0],
                [3.08, 0.0, 0.0],
                [4.62, 0.0, 0.0],
            ],
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
        assert frame.get_distance(0, 1) == pytest.approx(1.54, abs=2e-2)
        assert frame.get_distance(1, 2) == pytest.approx(1.54, abs=2e-2)
        assert frame.get_distance(2, 3) == pytest.approx(1.54, abs=2e-2)


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
        interpolate_molecule_with_internal_dofs(_butane_like(), _butane_like(), n_images=0)
