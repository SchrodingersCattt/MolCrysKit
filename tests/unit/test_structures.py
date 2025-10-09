"""
Unit tests for the structures module.
"""

import numpy as np
import sys
import os

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from molcrys.structures import Atom, Molecule, MolecularCrystal


def test_atom_creation():
    """Test creation of Atom objects."""
    atom = Atom("C", np.array([0.1, 0.2, 0.3]), 1.0)
    assert atom.symbol == "C"
    assert np.allclose(atom.frac_coords, np.array([0.1, 0.2, 0.3]))
    assert atom.occupancy == 1.0
    
    # Test atom copying
    atom_copy = atom.copy()
    assert atom_copy.symbol == atom.symbol
    assert np.allclose(atom_copy.frac_coords, atom.frac_coords)
    assert atom_copy.occupancy == atom.occupancy


def test_molecule_creation():
    """Test creation of Molecule objects."""
    atoms = [
        Atom("C", np.array([0.0, 0.0, 0.0])),
        Atom("H", np.array([1.0, 0.0, 0.0])),
        Atom("H", np.array([0.0, 1.0, 0.0])),
    ]
    
    molecule = Molecule(atoms)
    assert len(molecule.atoms) == 3
    assert molecule.atoms[0].symbol == "C"
    
    # Test center of mass calculation
    com = molecule.compute_center_of_mass()
    expected_com = np.array([1/3, 1/3, 0.0])
    assert np.allclose(com, expected_com)


def test_crystal_creation():
    """Test creation of MolecularCrystal objects."""
    # Define lattice
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create a simple molecule
    atoms = [Atom("C", np.array([0.1, 0.1, 0.1]))]
    molecule = Molecule(atoms)
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule])
    
    assert np.allclose(crystal.lattice, lattice)
    assert len(crystal.molecules) == 1
    assert crystal.pbc == (True, True, True)


def test_coordinate_transformations():
    """Test coordinate transformations."""
    # Define lattice
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create crystal
    atoms = [Atom("C", np.array([0.1, 0.1, 0.1]))]
    molecule = Molecule(atoms)
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
        test_molecule_creation,
        test_crystal_creation,
        test_coordinate_transformations
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