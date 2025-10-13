#!/usr/bin/env python3
"""
Complete molecule center analyzer with enhanced molecule properties.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_water_molecule(origin):
    """Create a water molecule at a specific origin."""
    if not ASE_AVAILABLE:
        return None
    
    # Create a water molecule (Oxygen at origin, Hydrogens at standard positions)
    positions = np.array([
        origin,                                    # Oxygen
        origin + np.array([0.07, 0.0, 0.0]),      # Hydrogen
        origin + np.array([0.0, 0.07, 0.0])       # Hydrogen
    ])
    
    water = Atoms('OH2', positions=positions)
    return water


def create_sample_crystal():
    """Create a sample crystal with multiple water molecules for demonstration."""
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    if not ASE_AVAILABLE:
        print("ASE not available, cannot create water molecules")
        return None
    
    # Create water molecules at different positions
    water1 = create_water_molecule(np.array([0.1, 0.1, 0.1]))
    water2 = create_water_molecule(np.array([0.5, 0.5, 0.5]))
    water3 = create_water_molecule(np.array([0.2, 0.8, 0.3]))
    
    # Create crystal with multiple molecules
    crystal = MolecularCrystal(lattice, [water1, water2, water3])
    return crystal


def analyze_molecule_centers(crystal):
    """Analyze and print the geometric centers of all molecules in the crystal."""
    print("Molecule Center Analysis")
    print("=" * 30)
    
    print(f"Total number of molecules: {len(crystal.molecules)}")
    print()
    
    # Calculate and print the center of mass for each molecule
    for i, molecule in enumerate(crystal.molecules):
        # Get center of mass (considering atomic masses)
        center_of_mass = molecule.get_center_of_mass()
        
        # Get geometric center (simple average of positions)
        positions = molecule.get_positions()
        geometric_center = np.mean(positions, axis=0)
        
        # Get chemical symbols
        symbols = molecule.get_chemical_symbols()
        
        print(f"Molecule {i+1}:")
        print(f"  Atoms: {len(molecule)} ({', '.join(symbols)})")
        print(f"  Center of mass (Cartesian): [{center_of_mass[0]:8.5f}, {center_of_mass[1]:8.5f}, {center_of_mass[2]:8.5f}]")
        print(f"  Geometric center (Cartesian): [{geometric_center[0]:8.5f}, {geometric_center[1]:8.5f}, {geometric_center[2]:8.5f}]")
        print()


def main():
    """Main function to demonstrate molecule center analysis."""
    print("MolCrysKit Script: Complete Molecule Center Analysis")
    print("=" * 55)
    
    try:
        # Try to import required modules
        from ase import Atoms
        from molcrys_kit.structures import MolecularCrystal
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed the molcrys-kit package:")
        print("pip install -e .")
        return 1