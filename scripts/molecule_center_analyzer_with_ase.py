#!/usr/bin/env python3
"""
Analyze molecular centers in a crystal structure using ASE for molecule identification.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ase import Atoms
    from ase.io import read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Error: ASE not found. Please install the package with 'pip install ase'")
    sys.exit(1)


def create_sample_crystal():
    """Create a sample crystal with multiple water molecules for demonstration."""
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create water molecule 1 at origin
    atoms1 = Atoms('OH2', 
                   positions=[[1.0, 1.0, 1.0],
                              [1.757, 1.586, 1.0],
                              [0.243, 1.586, 1.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create water molecule 2 at another position
    atoms2 = Atoms('OH2', 
                   positions=[[5.0, 5.0, 5.0],
                              [5.757, 5.586, 5.0],
                              [4.243, 5.586, 5.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create water molecule 3 at yet another position
    atoms3 = Atoms('OH2', 
                   positions=[[8.0, 2.0, 7.0],
                              [8.757, 2.586, 7.0],
                              [7.243, 2.586, 7.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create crystal with multiple molecules
    crystal = MolecularCrystal(lattice, [atoms1, atoms2, atoms3])
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
    try:
        # Try to import required modules
        from molcrys_kit.structures import MolecularCrystal
        from molcrys_kit.io import parse_cif_advanced
        from molcrys_kit.analysis import identify_molecules
        
        print("MolCrysKit Example: Molecule Center Analysis with ASE")
        print("=" * 55)
        
        if not ASE_AVAILABLE:
            print("This example requires ASE. Please install it with 'pip install ase'")
            return
        
        # Create sample crystal
        crystal = create_sample_crystal()
        
        # Print crystal summary
        print("Sample crystal created:")
        print(crystal.summary())
        print()
        
        # Analyze molecule centers
        analyze_molecule_centers(crystal)

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed the molcrys-kit package:")
        print("pip install -e .")
        return 1


if __name__ == "__main__":
    main()