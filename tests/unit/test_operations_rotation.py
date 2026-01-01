#!/usr/bin/env python3
"""
Test the rotation module.
"""

import os
import sys
import numpy as np

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from molcrys_kit.operations.rotation import (
    rotate_molecule_at_center,
    rotate_molecule_at_com,
)
from molcrys_kit.structures.molecule import CrystalMolecule
from ase import Atoms


def test_rotate_molecule_at_center():
    """Test rotating a molecule around its centroid."""
    # Create a simple molecule (water)
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    molecule = CrystalMolecule(atoms)

    # Get original positions and centroid
    original_positions = molecule.get_positions().copy()
    original_centroid = molecule.get_centroid()

    # Rotate 90 degrees around z-axis
    axis = np.array([0.0, 0.0, 1.0])
    angle = 90.0

    rotate_molecule_at_center(molecule, axis, angle)

    # Get new positions and centroid
    new_positions = molecule.get_positions()
    new_centroid = molecule.get_centroid()

    # Centroid should remain the same after rotation
    assert np.allclose(
        original_centroid, new_centroid
    ), f"Centroid changed after rotation: {original_centroid} -> {new_centroid}"

    # Positions should have changed
    assert not np.allclose(
        original_positions, new_positions
    ), "Positions did not change after rotation"

    # Verify that positions have changed significantly after rotation
    assert not np.allclose(
        original_positions, new_positions, atol=0.1
    ), "Positions did not change sufficiently after rotation"


def test_rotate_molecule_at_com():
    """Test rotating a molecule around its center of mass."""
    # Create a simple molecule (water)
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    molecule = CrystalMolecule(atoms)

    # Get original positions and center of mass
    original_positions = molecule.get_positions().copy()
    original_com = molecule.get_center_of_mass()

    # Rotate 90 degrees around z-axis
    axis = np.array([0.0, 0.0, 1.0])
    angle = 90.0

    rotate_molecule_at_com(molecule, axis, angle)

    # Get new positions and center of mass
    new_positions = molecule.get_positions()
    new_com = molecule.get_center_of_mass()

    # Center of mass should remain the same after rotation
    assert np.allclose(
        original_com, new_com
    ), f"Center of mass changed after rotation: {original_com} -> {new_com}"

    # Positions should have changed
    assert not np.allclose(
        original_positions, new_positions
    ), "Positions did not change after rotation"

    # Verify that positions have changed significantly after rotation
    assert not np.allclose(
        original_positions, new_positions, atol=0.1
    ), "Positions did not change sufficiently after rotation"


def test_rotate_molecule_different_angles():
    """Test rotating a molecule by different angles."""
    # Create a simple molecule (water)
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    molecule = CrystalMolecule(atoms)

    # Get original positions
    original_positions = molecule.get_positions().copy()

    # Test rotation by 180 degrees around z-axis
    axis = np.array([0.0, 0.0, 1.0])
    angle = 180.0

    rotate_molecule_at_center(molecule, axis, angle)

    # Get new positions
    new_positions = molecule.get_positions()

    # Verify that positions have changed significantly after 180 degree rotation
    assert not np.allclose(
        new_positions[1], original_positions[1]
    ), f"First atom position did not change after rotation: {original_positions[1]} -> {new_positions[1]}"
    assert not np.allclose(
        new_positions[2], original_positions[2]
    ), f"Second atom position did not change after rotation: {original_positions[2]} -> {new_positions[2]}"


def test_rotate_molecule_at_com_different_axis():
    """Test rotating a molecule around COM with different axis."""
    # Create a simple molecule (water)
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    molecule = CrystalMolecule(atoms)

    # Get original positions
    original_positions = molecule.get_positions().copy()

    # Test rotation by 90 degrees around x-axis
    axis = np.array([1.0, 0.0, 0.0])
    angle = 90.0

    rotate_molecule_at_com(molecule, axis, angle)

    # Get new positions
    new_positions = molecule.get_positions()

    # Verify that positions have changed significantly after rotation
    assert not np.allclose(
        new_positions, original_positions
    ), f"Positions did not change after rotation: {original_positions} -> {new_positions}"


def run_tests():
    """Run all tests."""
    tests = [
        test_rotate_molecule_at_center,
        test_rotate_molecule_at_com,
        test_rotate_molecule_different_angles,
        test_rotate_molecule_at_com_different_axis,
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
