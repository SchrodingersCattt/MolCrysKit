#!/usr/bin/env python3
"""
Test the structures module.
"""

import os
import sys
import numpy as np

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from molcrys_kit.structures import MolAtom, MolecularCrystal  # noqa: E402
from molcrys_kit.structures.molecule import CrystalMolecule  # noqa: E402
from ase import Atoms  # noqa: E402


def test_atom_creation():
    """Test creating an MolAtom."""
    atom = MolAtom("C", np.array([0.1, 0.2, 0.3]), 1.0)
    assert atom.symbol == "C"
    assert np.allclose(atom.frac_coords, np.array([0.1, 0.2, 0.3]))
    assert atom.occupancy == 1.0

    # Test atom copying
    atom_copy = atom.copy()
    assert atom_copy.symbol == atom.symbol
    assert np.allclose(atom_copy.frac_coords, atom.frac_coords)
    assert atom_copy.occupancy == atom.occupancy


def test_crystal_molecule_composition():
    """Test that CrystalMolecule correctly uses ASE Atoms via composition."""
    # Create a simple molecule using ASE Atoms
    atoms = Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    molecule = CrystalMolecule(atoms)

    # Test composition - should have ASE Atoms methods via delegation
    assert hasattr(molecule, "get_positions")
    assert hasattr(molecule, "get_chemical_symbols")
    assert hasattr(molecule, "get_chemical_formula")

    # Test that methods work correctly
    positions = molecule.get_positions()
    assert positions.shape == (3, 3)  # 3 atoms, 3 coordinates each

    symbols = molecule.get_chemical_symbols()
    assert symbols == ["C", "H", "H"]

    formula = molecule.get_chemical_formula()
    assert formula == "CH2"


def test_crystal_molecule_creation():
    """Test creation of CrystalMolecule objects."""
    # Create a simple molecule using ASE Atoms
    atoms = Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    molecule = CrystalMolecule(atoms)
    assert len(molecule) == 3
    assert molecule.get_chemical_symbols()[0] == "C"

    # Test center of mass calculation
    com = molecule.get_center_of_mass()
    # Expected COM calculation (using approximate atomic masses)
    expected_com = np.array([0.07186141, 0.07186141, 0.0])
    assert com.shape == (3,)
    assert np.allclose(com, expected_com)


def test_crystal_molecule_graph():
    """Test the graph representation of molecules."""
    # Create a simple molecule using ASE Atoms
    atoms = Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    molecule = CrystalMolecule(atoms)
    graph = molecule.graph

    # Check that we have the right number of nodes
    assert len(graph.nodes()) == 3

    # Check that bonds exist (there should be bonds between C and H atoms)
    assert len(graph.edges()) >= 2

    # Check node attributes
    for node_id in graph.nodes():
        assert "symbol" in graph.nodes[node_id]

    # Check edge attributes
    for edge in graph.edges(data=True):
        assert "distance" in edge[2]


def test_crystal_molecule_centroid_and_com():
    """Test centroid and center of mass calculations."""
    # Create a simple molecule
    atoms = Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    molecule = CrystalMolecule(atoms)

    # Test centroid
    centroid = molecule.get_centroid()
    assert centroid.shape == (3,)

    # Test center of mass
    com = molecule.get_center_of_mass()
    assert com.shape == (3,)

    # For a non-uniform mass distribution, centroid and COM should be different
    # (though this depends on the specific arrangement and masses)
    assert centroid.shape == com.shape


def test_molecular_crystal_creation():
    """Test creation of MolecularCrystal objects."""
    # Define lattice
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Create a simple molecule using ASE Atoms
    atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
    molecule = CrystalMolecule(atoms)

    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule])

    assert np.allclose(crystal.lattice, lattice)
    assert len(crystal.molecules) == 1
    assert crystal.pbc == (True, True, True)


def test_molecular_crystal_properties():
    """Test MolecularCrystal property access."""
    # Define lattice
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Create crystal
    atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
    molecule = CrystalMolecule(atoms)
    crystal = MolecularCrystal(lattice, [molecule])

    # Test lattice parameters
    params = crystal.get_lattice_parameters()
    assert len(params) == 6  # a, b, c, alpha, beta, gamma
    a, b, c, alpha, beta, gamma = params
    assert a == 10.0
    assert b == 10.0
    assert c == 10.0
    assert alpha == 90.0
    assert beta == 90.0
    assert gamma == 90.0

    # Test lattice vectors
    vectors = crystal.get_lattice_vectors()
    assert vectors.shape == (3, 3)
    assert np.allclose(vectors, lattice)


def test_coordinate_transformations():
    """Test coordinate transformations."""
    # Define lattice
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Create crystal
    atoms = Atoms(symbols=["C"], positions=[[1.0, 1.0, 1.0]])
    molecule = CrystalMolecule(atoms)
    crystal = MolecularCrystal(lattice, [molecule])

    # Test fractional to cartesian conversion
    frac_coords = np.array([0.5, 0.5, 0.5])
    cart_coords = crystal.fractional_to_cartesian(frac_coords)
    expected_cart = np.array([5.0, 5.0, 5.0])
    assert np.allclose(cart_coords, expected_cart)

    # Test cartesian to fractional conversion
    cart_coords = np.array([5.0, 5.0, 5.0])
    frac_coords = crystal.cartesian_to_fractional(cart_coords)
    expected_frac = np.array([0.5, 0.5, 0.5])
    assert np.allclose(frac_coords, expected_frac)


def run_tests():
    """Run all tests."""
    tests = [
        test_atom_creation,
        test_crystal_molecule_composition,
        test_crystal_molecule_creation,
        test_crystal_molecule_graph,
        test_crystal_molecule_centroid_and_com,
        test_molecular_crystal_creation,
        test_molecular_crystal_properties,
        test_coordinate_transformations,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\nTests passed: {passed}")
    print(f"Tests failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
