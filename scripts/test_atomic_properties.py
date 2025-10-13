#!/usr/bin/env python3
"""
Test script to verify atomic properties usage in MolCrysKit.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Import molcrys modules
    from molcrys.structures import Atom, Molecule, MolecularCrystal
    from molcrys.constants import get_atomic_mass, get_atomic_radius
    from molcrys.analysis import identify_molecules
except ImportError as e:
    print(f"Error importing MolCrysKit: {e}")
    sys.exit(1)


def test_center_of_mass():
    """Test center of mass calculation with real atomic masses."""
    print("Testing Center of Mass Calculation")
    print("=" * 40)
    
    # Create a water molecule
    atoms = [
        Atom("O", np.array([0.0, 0.0, 0.0])),      # Oxygen
        Atom("H", np.array([0.0757, 0.0586, 0.0])),  # Hydrogen
        Atom("H", np.array([-0.0757, 0.0586, 0.0])), # Hydrogen
    ]
    
    water = Molecule(atoms)
    
    print("Water molecule:")
    for atom in water.atoms:
        mass = get_atomic_mass(atom.symbol)
        print(f"  {atom.symbol}: mass = {mass:.3f} amu")
    
    com = water.compute_center_of_mass()
    print(f"\nCenter of mass: [{com[0]:8.5f}, {com[1]:8.5f}, {com[2]:8.5f}]")
    
    # Compare with simple geometric center
    coords = np.array([atom.frac_coords for atom in water.atoms])
    geometric_center = np.mean(coords, axis=0)
    print(f"Geometric center: [{geometric_center[0]:8.5f}, {geometric_center[1]:8.5f}, {geometric_center[2]:8.5f}]")
    
    # They should be different because of mass weighting
    assert not np.allclose(com, geometric_center), "Center of mass should differ from geometric center"
    print("\n✓ Center of mass correctly uses atomic masses\n")


def test_bond_detection():
    """Test bond detection using atomic radii."""
    print("Testing Bond Detection")
    print("=" * 40)
    
    # Create a molecule with known bonds (using more realistic fractional coordinates)
    atoms = [
        Atom("C", np.array([0.0, 0.0, 0.0])),         # Carbon
        Atom("H", np.array([0.109, 0.0, 0.0])),       # Hydrogen (bonded)
        Atom("H", np.array([0.0, 0.109, 0.0])),       # Hydrogen (bonded)
        Atom("H", np.array([0.0, 0.0, 0.109])),       # Hydrogen (bonded)
        Atom("H", np.array([-0.109, 0.0, 0.0])),      # Hydrogen (bonded)
    ]
    
    molecule = Molecule(atoms)
    
    print("Methane-like molecule:")
    for i, atom in enumerate(molecule.atoms):
        radius = get_atomic_radius(atom.symbol)
        print(f"  {i}: {atom.symbol} (radius = {radius:.3f} Å)")
    
    bonds = molecule.get_bonds()
    print(f"\nDetected bonds: {len(bonds)}")
    for i, (atom1_idx, atom2_idx, distance) in enumerate(bonds):
        atom1 = molecule.atoms[atom1_idx]
        atom2 = molecule.atoms[atom2_idx]
        print(f"  {i+1}: {atom1.symbol}({atom1_idx}) - {atom2.symbol}({atom2_idx}), distance = {distance:.3f}")
    
    # Should detect 4 C-H bonds
    print(f"\n✓ Bond detection completed (found {len(bonds)} bonds)\n")


def test_molecular_identification():
    """Test molecular identification using atomic radii."""
    print("Testing Molecular Identification")
    print("=" * 40)
    
    # Create a crystal with two water molecules
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # First water molecule
    atoms1 = [
        Atom("O", np.array([0.1, 0.1, 0.1])),       # Oxygen
        Atom("H", np.array([0.17, 0.1, 0.1])),      # Hydrogen
        Atom("H", np.array([0.1, 0.17, 0.1])),      # Hydrogen
    ]
    
    # Second water molecule (far enough to be separate)
    atoms2 = [
        Atom("O", np.array([0.6, 0.6, 0.6])),       # Oxygen
        Atom("H", np.array([0.67, 0.6, 0.6])),      # Hydrogen
        Atom("H", np.array([0.6, 0.67, 0.6])),      # Hydrogen
    ]
    
    # Put all atoms in one molecule initially
    all_atoms = atoms1 + atoms2
    initial_molecule = Molecule(all_atoms)
    crystal = MolecularCrystal(lattice, [initial_molecule])
    
    print("Initial crystal:")
    print(f"  Lattice: {lattice[0,0]:.1f} × {lattice[1,1]:.1f} × {lattice[2,2]:.1f} Å³")
    print(f"  Atoms: {len(all_atoms)}")
    
    # Identify separate molecules
    identified_molecules = identify_molecules(crystal)
    print(f"\nIdentified molecules: {len(identified_molecules)}")
    
    for i, molecule in enumerate(identified_molecules):
        print(f"  Molecule {i+1}: {len(molecule.atoms)} atoms")
        for atom in molecule.atoms:
            print(f"    {atom.symbol}")
    
    print("\n✓ Molecular identification completed\n")


def main():
    """Main function to run all tests."""
    print("MolCrysKit Atomic Properties Test")
    print("================================\n")
    
    test_center_of_mass()
    test_bond_detection()
    test_molecular_identification()
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()