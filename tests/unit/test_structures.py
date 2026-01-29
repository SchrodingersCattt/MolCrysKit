"""
Unit tests for molcrys_kit.structures (MolAtom, CrystalMolecule, MolecularCrystal).
"""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures import MolAtom, MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule


class TestMolAtom:
    """MolAtom dataclass."""

    def test_creation(self):
        atom = MolAtom("C", np.array([0.1, 0.2, 0.3]), 1.0)
        assert atom.symbol == "C"
        np.testing.assert_allclose(atom.frac_coords, [0.1, 0.2, 0.3])
        assert atom.occupancy == 1.0

    def test_copy(self):
        atom = MolAtom("C", [0.1, 0.2, 0.3], 1.0)
        c = atom.copy()
        assert c.symbol == atom.symbol
        np.testing.assert_allclose(c.frac_coords, atom.frac_coords)
        assert c.occupancy == atom.occupancy
        assert c is not atom

    def test_to_cartesian(self, cubic_lattice_10):
        atom = MolAtom("C", [0.5, 0.5, 0.5], 1.0)
        cart = atom.to_cartesian(cubic_lattice_10)
        np.testing.assert_allclose(cart, [5.0, 5.0, 5.0])

    def test_repr(self):
        atom = MolAtom("C", [0.0, 0.0, 0.0], 1.0)
        r = repr(atom)
        assert "MolAtom" in r and "C" in r


class TestCrystalMolecule:
    """CrystalMolecule (ASE Atoms wrapper + graph)."""

    @pytest.fixture
    def water_mol(self, water_atoms):
        return CrystalMolecule(water_atoms)

    def test_composition(self, water_mol):
        assert hasattr(water_mol, "get_positions")
        assert hasattr(water_mol, "get_chemical_symbols")
        assert hasattr(water_mol, "get_chemical_formula")
        positions = water_mol.get_positions()
        assert positions.shape == (3, 3)
        assert water_mol.get_chemical_symbols() == ["O", "H", "H"]
        assert water_mol.get_chemical_formula() == "H2O"

    def test_len_and_formula(self, water_mol):
        assert len(water_mol) == 3
        assert water_mol.get_chemical_symbols()[0] == "O"

    def test_center_of_mass_shape(self, water_mol):
        com = water_mol.get_center_of_mass()
        assert com.shape == (3,)

    def test_graph_nodes_edges(self, water_mol):
        g = water_mol.graph
        assert len(g.nodes()) == 3
        assert len(g.edges()) >= 2
        for n in g.nodes():
            assert "symbol" in g.nodes[n]
        for e in g.edges(data=True):
            assert "distance" in e[2]

    def test_centroid_and_com_shape(self, water_mol):
        c1 = water_mol.get_centroid()
        c2 = water_mol.get_center_of_mass()
        assert c1.shape == c2.shape == (3,)


class TestMolecularCrystal:
    """MolecularCrystal container."""

    def test_creation(self, cubic_lattice_10):
        atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
        mol = CrystalMolecule(atoms)
        crystal = MolecularCrystal(cubic_lattice_10, [mol])
        np.testing.assert_allclose(crystal.lattice, cubic_lattice_10)
        assert len(crystal.molecules) == 1
        assert crystal.pbc == (True, True, True)

    def test_lattice_parameters(self, crystal_single_water):
        params = crystal_single_water.get_lattice_parameters()
        assert len(params) == 6
        a, b, c, alpha, beta, gamma = params
        assert a == b == c == 10.0
        assert alpha == beta == gamma == 90.0

    def test_lattice_vectors(self, crystal_single_water, cubic_lattice_10):
        vectors = crystal_single_water.get_lattice_vectors()
        assert vectors.shape == (3, 3)
        np.testing.assert_allclose(vectors, cubic_lattice_10)

    def test_frac_cart_conversion(self, crystal_single_water):
        frac = np.array([0.5, 0.5, 0.5])
        cart = crystal_single_water.fractional_to_cartesian(frac)
        np.testing.assert_allclose(cart, [5.0, 5.0, 5.0])
        back = crystal_single_water.cartesian_to_fractional(cart)
        np.testing.assert_allclose(back, frac)
