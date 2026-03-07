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

    def test_copy_independence(self):
        atom = MolAtom("N", np.array([0.1, 0.2, 0.3]), 1.0)
        c = atom.copy()
        c.frac_coords[0] = 0.9
        assert atom.frac_coords[0] == pytest.approx(0.1)

    def test_to_cartesian(self, cubic_lattice_10):
        atom = MolAtom("C", [0.5, 0.5, 0.5], 1.0)
        cart = atom.to_cartesian(cubic_lattice_10)
        np.testing.assert_allclose(cart, [5.0, 5.0, 5.0])

    def test_to_cartesian_rectangular(self):
        lattice = np.array([[10.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]])
        atom = MolAtom("O", [0.1, 0.2, 0.3], 0.5)
        cart = atom.to_cartesian(lattice)
        np.testing.assert_allclose(cart, [1.0, 4.0, 9.0])

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
        positions = water_mol.get_positions()
        assert positions.shape == (3, 3)
        assert water_mol.get_chemical_symbols() == ["O", "H", "H"]
        assert water_mol.get_chemical_formula() == "H2O"

    def test_len_and_formula(self, water_mol):
        assert len(water_mol) == 3
        assert water_mol.get_chemical_symbols()[0] == "O"

    def test_repr(self, water_mol):
        r = repr(water_mol)
        assert "CrystalMolecule" in r
        assert "H2O" in r

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

    def test_copy_preserves_data(self, water_mol):
        copy = water_mol.copy()
        assert copy.get_chemical_formula() == water_mol.get_chemical_formula()
        np.testing.assert_allclose(copy.get_positions(), water_mol.get_positions())
        assert copy is not water_mol

    def test_copy_graph_independence(self, water_mol):
        _ = water_mol.graph
        copy = water_mol.copy()
        assert copy.graph.number_of_nodes() == water_mol.graph.number_of_nodes()

    def test_ellipsoid_radii_multi_atom(self, water_mol):
        radii = water_mol.get_ellipsoid_radii()
        assert len(radii) == 3
        assert all(r >= 0 for r in radii)
        assert radii[0] >= radii[1] >= radii[2]

    def test_ellipsoid_radii_single_atom(self):
        mol = CrystalMolecule(Atoms("Cl", positions=[[0, 0, 0]]))
        radii = mol.get_ellipsoid_radii()
        assert len(radii) == 3
        assert radii[0] == radii[1] == radii[2]

    def test_principal_axes_multi_atom(self, water_mol):
        axes = water_mol.get_principal_axes()
        assert len(axes) == 3
        for ax in axes:
            assert ax.shape == (3,)
            assert abs(np.linalg.norm(ax) - 1.0) < 1e-6

    def test_principal_axes_single_atom(self):
        mol = CrystalMolecule(Atoms("Ar", positions=[[0, 0, 0]]))
        axes = mol.get_principal_axes()
        assert len(axes) == 3

    def test_centroid_frac_with_crystal(self, cubic_lattice_10, water_atoms):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(water_atoms)])
        mol = crystal.molecules[0]
        frac = mol.get_centroid_frac()
        assert frac.shape == (3,)

    def test_centroid_frac_without_crystal_raises(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        mol.crystal = None
        with pytest.raises(ValueError, match="crystal"):
            mol.get_centroid_frac()

    def test_to_ase_returns_plain_atoms(self, water_mol):
        ase_atoms = water_mol.to_ase()
        assert isinstance(ase_atoms, Atoms)
        assert not isinstance(ase_atoms, CrystalMolecule)
        assert ase_atoms.get_chemical_formula() == "H2O"
        np.testing.assert_allclose(ase_atoms.get_positions(), water_mol.get_positions())

    def test_get_graph_and_build_graph(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        g1 = mol.get_graph()
        g2 = mol.build_graph()
        assert g1.number_of_nodes() == g2.number_of_nodes() == 3


class TestMolecularCrystal:
    """MolecularCrystal container."""

    def test_creation(self, cubic_lattice_10):
        atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
        mol = CrystalMolecule(atoms)
        crystal = MolecularCrystal(cubic_lattice_10, [mol])
        np.testing.assert_allclose(crystal.lattice, cubic_lattice_10)
        assert len(crystal.molecules) == 1
        assert crystal.pbc == (True, True, True)

    def test_creation_from_raw_atoms(self, cubic_lattice_10):
        atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
        crystal = MolecularCrystal(cubic_lattice_10, [atoms])
        assert len(crystal.molecules) == 1
        assert isinstance(crystal.molecules[0], CrystalMolecule)

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

    def test_repr(self, crystal_single_water):
        r = repr(crystal_single_water)
        assert "MolecularCrystal" in r

    def test_to_ase(self, simple_crystal):
        ase_obj = simple_crystal.to_ase()
        assert isinstance(ase_obj, Atoms)
        total_atoms = sum(len(m) for m in simple_crystal.molecules)
        assert len(ase_obj) == total_atoms

    def test_from_ase(self):
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
            cell=np.eye(3) * 15.0,
            pbc=True,
        )
        crystal = MolecularCrystal.from_ase(atoms)
        assert isinstance(crystal, MolecularCrystal)
        assert len(crystal.molecules) >= 1

    def test_get_supercell(self, cubic_lattice_10, co_molecule):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(co_molecule)])
        supercell = crystal.get_supercell(2, 1, 1)
        np.testing.assert_allclose(supercell.lattice[0], crystal.lattice[0] * 2)
        assert len(supercell.molecules) == 2

    def test_summary(self, simple_crystal):
        s = simple_crystal.summary()
        assert "MolecularCrystal" in s
        assert "Number of molecules" in s
        assert "Total atoms" in s

    def test_total_nodes(self, simple_crystal):
        total = simple_crystal.get_total_nodes()
        expected = sum(len(m) for m in simple_crystal.molecules)
        assert total == expected

    def test_total_edges(self, simple_crystal):
        total = simple_crystal.get_total_edges()
        assert total > 0

    def test_default_atomic_radii(self, crystal_single_water):
        radii = crystal_single_water.get_default_atomic_radii()
        assert isinstance(radii, dict)
        assert "H" in radii
        assert "O" in radii

    def test_pbc_default(self, cubic_lattice_10):
        crystal = MolecularCrystal(cubic_lattice_10, [])
        assert crystal.pbc == (True, True, True)
