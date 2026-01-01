#!/usr/bin/env python3
"""
Test the species module.
"""

import os
import sys
import numpy as np

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from molcrys_kit.analysis.species import identify_molecules, assign_atoms_to_molecules
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from ase import Atoms


def test_identify_molecules():
    """Test identifying molecules in a crystal."""
    # Create a simple crystal with two water molecules
    h2o1 = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    h2o2 = Atoms(
        symbols=["O", "H", "H"],
        positions=[[3.0, 0.0, 0.0], [3.757, 0.586, 0.0], [2.243, 0.586, 0.0]],
    )

    # Create a crystal with these molecules
    lattice = np.eye(3) * 10.0  # 10x10x10 Angstrom box
    crystal = MolecularCrystal(lattice, [CrystalMolecule(h2o1), CrystalMolecule(h2o2)])

    # Identify molecules
    molecules = identify_molecules(crystal)

    # Should return the list of molecules in the crystal
    assert len(molecules) == 2
    assert isinstance(molecules[0], CrystalMolecule)
    assert isinstance(molecules[1], CrystalMolecule)


def test_assign_atoms_to_molecules():
    """Test assigning atoms to molecules in a crystal."""
    # Create a simple crystal with one water molecule
    h2o = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )

    # Create a crystal with this molecule
    lattice = np.eye(3) * 10.0  # 10x10x10 Angstrom box
    original_crystal = MolecularCrystal(lattice, [CrystalMolecule(h2o)])

    # Assign atoms to molecules (should return the same crystal in current implementation)
    new_crystal = assign_atoms_to_molecules(original_crystal)

    # Check that the returned crystal has the same properties
    assert np.allclose(new_crystal.lattice, original_crystal.lattice)
    assert len(new_crystal.molecules) == len(original_crystal.molecules)
    assert isinstance(new_crystal.molecules[0], CrystalMolecule)

    # The function should return the same crystal in current implementation
    assert new_crystal is original_crystal


def test_identify_molecules_empty_crystal():
    """Test identifying molecules in an empty crystal."""
    # Create an empty crystal
    lattice = np.eye(3) * 10.0
    crystal = MolecularCrystal(lattice, [])

    # Identify molecules
    molecules = identify_molecules(crystal)

    # Should return an empty list
    assert len(molecules) == 0


def test_assign_atoms_to_molecules_empty_crystal():
    """Test assigning atoms to molecules in an empty crystal."""
    # Create an empty crystal
    lattice = np.eye(3) * 10.0
    original_crystal = MolecularCrystal(lattice, [])

    # Assign atoms to molecules
    new_crystal = assign_atoms_to_molecules(original_crystal)

    # Should return the same crystal object
    assert new_crystal is original_crystal
    assert len(new_crystal.molecules) == 0


def run_tests():
    """Run all tests."""
    tests = [
        test_identify_molecules,
        test_assign_atoms_to_molecules,
        test_identify_molecules_empty_crystal,
        test_assign_atoms_to_molecules_empty_crystal,
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
