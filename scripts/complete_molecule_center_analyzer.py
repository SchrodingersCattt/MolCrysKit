#!/usr/bin/env python3
"""
Complete script to analyze molecular centers in a crystal structure.

This script demonstrates how to manually create molecular structures and 
analyze their centers of mass, which is more appropriate for showcasing the functionality.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Import molcrys modules
    from molcrys.structures import Atom, Molecule, MolecularCrystal
except ImportError:
    print("Error: MolCrysKit not found. Please install the package with 'pip install -e .'")
    sys.exit(1)


def create_water_molecule(origin):
    """
    Create a water molecule at a given origin position.
    
    Parameters
    ----------
    origin : np.ndarray
        Fractional coordinates for the oxygen atom position.
        
    Returns
    -------
    Molecule
        A water molecule.
    """
    # Create water molecule with typical geometry
    atoms = [
        Atom("O", origin),                                    # Oxygen
        Atom("H", origin + np.array([0.07, 0.0, 0.0])),      # Hydrogen
        Atom("H", origin + np.array([0.0, 0.07, 0.0])),      # Hydrogen
    ]
    return Molecule(atoms)


def create_crystal_with_molecules():
    """
    Create a crystal structure with proper molecular units.
    
    Returns
    -------
    MolecularCrystal
        A crystal with proper molecular units.
    """
    # Define lattice vectors
    lattice = np.array([
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ])
    
    # Create water molecules at different positions
    water1 = create_water_molecule(np.array([0.1, 0.1, 0.1]))
    water2 = create_water_molecule(np.array([0.6, 0.6, 0.6]))
    water3 = create_water_molecule(np.array([0.2, 0.8, 0.3]))
    
    # Create crystal with the molecules
    crystal = MolecularCrystal(lattice, [water1, water2, water3])
    return crystal


def analyze_molecule_centers(crystal):
    """
    Analyze and display the centers of mass for all molecules in a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal structure to analyze.
    """
    print(f"Analyzing crystal with {len(crystal.molecules)} molecules")
    print(f"Crystal lattice:\n{crystal.lattice}")
    print()
    
    # Calculate and display center of mass for each molecule
    print("Molecular Centers (Fractional Coordinates):")
    print("-" * 50)
    
    for i, molecule in enumerate(crystal.molecules, 1):
        # Compute center of mass
        center_of_mass = molecule.compute_center_of_mass()
        
        # Display results
        print(f"Molecule {i:3d}: [{center_of_mass[0]:8.5f}, {center_of_mass[1]:8.5f}, {center_of_mass[2]:8.5f}]")
        
        # Show basic info about the molecule
        print(f"  Atoms: {len(molecule.atoms)}")
        atom_symbols = [atom.symbol for atom in molecule.atoms]
        print(f"  Elements: {', '.join(atom_symbols)}")
        
        # Show bonds within the molecule
        bonds = molecule.get_bonds()
        print(f"  Internal bonds: {len(bonds)}")
        print()


def main():
    """
    Main function to run the molecular center analysis.
    """
    print("Complete Molecular Center Analyzer")
    print("=" * 35)
    
    # Create a crystal with proper molecules
    crystal = create_crystal_with_molecules()
    
    # Analyze molecule centers
    analyze_molecule_centers(crystal)
    
    # Demonstrate some operations on the molecules
    print("Demonstrating molecular operations:")
    print("-" * 35)
    
    if crystal.molecules:
        first_molecule = crystal.molecules[0]
        original_center = first_molecule.compute_center_of_mass()
        print(f"Original center of first molecule: [{original_center[0]:8.5f}, {original_center[1]:8.5f}, {original_center[2]:8.5f}]")
        
        # Translate the molecule
        first_molecule.translate(np.array([0.1, 0.1, 0.1]))
        new_center = first_molecule.compute_center_of_mass()
        print(f"New center after translation:    [{new_center[0]:8.5f}, {new_center[1]:8.5f}, {new_center[2]:8.5f}]")


if __name__ == "__main__":
    main()