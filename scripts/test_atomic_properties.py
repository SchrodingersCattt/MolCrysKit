#!/usr/bin/env python3
"""
Test script for atomic properties functionality.

This script tests the atomic property calculations for molecules.
"""

import numpy as np

def test_atomic_properties():
    """Test atomic properties calculations."""
    try:
        from ase import Atoms
        ASE_AVAILABLE = True
    except ImportError:
        ASE_AVAILABLE = False
        print("Warning: ASE is not available. Some functionality may be limited.")
        return
    
    if not ASE_AVAILABLE:
        print("This test requires ASE. Please install it with 'pip install ase'")
        return
    
    print("Testing Atomic Properties")
    print("=" * 25)
    
    # Create a water molecule
    water = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0]
        ]
    )
    
    # Import CrystalMolecule after confirming ASE availability
    from molcrys_kit.structures.molecule import CrystalMolecule
    
    # Convert to CrystalMolecule
    molecule = CrystalMolecule(water)
    
    print(f"Molecule: {molecule.get_chemical_formula()}")
    print(f"Number of atoms: {len(molecule)}")
    
    # Test atomic properties
    symbols = molecule.get_chemical_symbols()
    positions = molecule.get_positions()
    masses = molecule.get_masses()
    
    print("\nAtomic Properties:")
    for i in range(len(molecule)):
        print(f"  Atom {i+1}: {symbols[i]}")
        print(f"    Position: ({positions[i][0]:.4f}, {positions[i][1]:.4f}, {positions[i][2]:.4f})")
        print(f"    Mass: {masses[i]:.4f}")
    
    # Test molecular properties
    print("\nMolecular Properties:")
    
    # Centroid
    centroid = molecule.get_centroid()
    print(f"  Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    
    # Center of mass
    center_of_mass = molecule.get_center_of_mass()
    print(f"  Center of mass: ({center_of_mass[0]:.4f}, {center_of_mass[1]:.4f}, {center_of_mass[2]:.4f})")
    
    # Check if they're different (they should be for a non-uniform mass distribution)
    diff = np.linalg.norm(centroid - center_of_mass)
    print(f"  Difference between centroid and COM: {diff:.6f}")


def test_complex_molecule():
    """Test properties of a more complex molecule."""
    try:
        from ase import Atoms
        ASE_AVAILABLE = True
    except ImportError:
        ASE_AVAILABLE = False
        print("Warning: ASE is not available. Some functionality may be limited.")
        return
    
    if not ASE_AVAILABLE:
        print("This test requires ASE. Please install it with 'pip install ase'")
        return
    
    print("\n\nTesting Complex Molecule")
    print("=" * 25)
    
    # Create a methane molecule
    methane = Atoms(
        symbols=['C', 'H', 'H', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],        # Carbon
            [0.631, 0.631, 0.631],  # H1
            [-0.631, -0.631, 0.631], # H2
            [-0.631, 0.631, -0.631], # H3
            [0.631, -0.631, -0.631]  # H4
        ]
    )
    
    # Import CrystalMolecule after confirming ASE availability
    from molcrys_kit.structures.molecule import CrystalMolecule
    
    # Convert to CrystalMolecule
    molecule = CrystalMolecule(methane)
    
    print(f"Molecule: {molecule.get_chemical_formula()}")
    print(f"Number of atoms: {len(molecule)}")
    
    # For a symmetric molecule like methane, centroid and COM should be nearly identical
    centroid = molecule.get_centroid()
    center_of_mass = molecule.get_center_of_mass()
    diff = np.linalg.norm(centroid - center_of_mass)
    
    print(f"  Centroid: ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})")
    print(f"  Center of mass: ({center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f})")
    print(f"  Difference: {diff:.6f}")
    
    # Test ellipsoid properties
    radii = molecule.get_ellipsoid_radii()
    print(f"  Ellipsoid radii: {radii[0]:.3f} × {radii[1]:.3f} × {radii[2]:.3f}")
    
    # Test principal axes
    axes = molecule.get_principal_axes()
    print("  Principal axes:")
    for i, axis in enumerate(axes):
        print(f"    Axis {i+1}: ({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})")


def main():
    """Run all tests."""
    test_atomic_properties()
    test_complex_molecule()
    print("\n\nAll tests completed successfully!")


if __name__ == "__main__":
    main()