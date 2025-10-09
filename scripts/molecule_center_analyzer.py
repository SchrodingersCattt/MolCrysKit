#!/usr/bin/env python3
"""
Script to analyze molecular centers in a crystal structure.

This script loads a crystal structure, extracts all molecules, 
and computes the fractional coordinates of each molecule's center of mass.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Import molcrys modules
    from molcrys.io import parse_cif
    from molcrys.analysis import identify_molecules
    from molcrys.structures import MolecularCrystal
except ImportError:
    print("Error: MolCrysKit not found. Please install the package with 'pip install -e .'")
    sys.exit(1)


def analyze_molecule_centers(cif_file_path):
    """
    Load a crystal structure from CIF and analyze molecular centers.
    
    Parameters
    ----------
    cif_file_path : str
        Path to the CIF file containing the crystal structure.
    """
    try:
        # Parse the CIF file
        print(f"Parsing CIF file: {cif_file_path}")
        crystal = parse_cif(cif_file_path)
        print("CIF file parsed successfully.")
        
    except FileNotFoundError:
        print(f"Error: File '{cif_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return
    
    # Identify molecular units
    print("\nIdentifying molecular units...")
    try:
        molecules = identify_molecules(crystal)
        print(f"Found {len(molecules)} molecular units.")
    except Exception as e:
        print(f"Error identifying molecules: {e}")
        return
    
    # Calculate and display center of mass for each molecule
    print("\nMolecular Centers (Fractional Coordinates):")
    print("-" * 50)
    
    if not molecules:
        print("No molecules found in the structure.")
        return
    
    for i, molecule in enumerate(molecules, 1):
        # Compute center of mass
        center_of_mass = molecule.compute_center_of_mass()
        
        # Display results
        print(f"Molecule {i:3d}: [{center_of_mass[0]:8.5f}, {center_of_mass[1]:8.5f}, {center_of_mass[2]:8.5f}]")
        
        # Also show basic info about the molecule
        print(f"  Atoms: {len(molecule.atoms)}")
        atom_symbols = [atom.symbol for atom in molecule.atoms]
        unique_symbols = list(set(atom_symbols))
        print(f"  Elements: {', '.join(sorted(unique_symbols))}")
        print()


def create_sample_crystal():
    """
    Create a sample crystal structure for testing purposes.
    
    Returns
    -------
    MolecularCrystal
        A sample crystal with water molecules.
    """
    from molcrys.structures import Atom, Molecule
    
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create water molecule 1
    atoms1 = [
        Atom("O", np.array([0.1, 0.1, 0.1])),  # Oxygen
        Atom("H", np.array([0.2, 0.1, 0.1])),  # Hydrogen
        Atom("H", np.array([0.1, 0.2, 0.1])),  # Hydrogen
    ]
    molecule1 = Molecule(atoms1)
    
    # Create water molecule 2
    atoms2 = [
        Atom("O", np.array([0.6, 0.6, 0.6])),  # Oxygen
        Atom("H", np.array([0.7, 0.6, 0.6])),  # Hydrogen
        Atom("H", np.array([0.6, 0.7, 0.6])),  # Hydrogen
    ]
    molecule2 = Molecule(atoms2)
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule1, molecule2])
    return crystal


def analyze_sample_structure():
    """
    Analyze a sample structure for demonstration.
    """
    print("Analyzing sample structure...")
    
    # Create a sample crystal
    crystal = create_sample_crystal()
    
    # Identify molecular units
    molecules = identify_molecules(crystal)
    print(f"Found {len(molecules)} molecular units in sample structure.")
    
    # Calculate and display center of mass for each molecule
    print("\nMolecular Centers (Fractional Coordinates):")
    print("-" * 50)
    
    for i, molecule in enumerate(molecules, 1):
        center_of_mass = molecule.compute_center_of_mass()
        print(f"Molecule {i:3d}: [{center_of_mass[0]:8.5f}, {center_of_mass[1]:8.5f}, {center_of_mass[2]:8.5f}]")
        print(f"  Atoms: {len(molecule.atoms)}")


def main():
    """
    Main function to run the molecular center analysis.
    """
    print("Molecular Center Analyzer")
    print("=" * 30)
    
    if len(sys.argv) < 2:
        print("Usage: python molecule_center_analyzer.py <cif_file>")
        print("       python molecule_center_analyzer.py --sample")
        print()
        print("If no arguments provided, analyzing sample structure...")
        analyze_sample_structure()
        return
    
    # Check if user wants to run on sample structure
    if sys.argv[1] == "--sample":
        analyze_sample_structure()
        return
    
    # Process the provided CIF file
    cif_file_path = sys.argv[1]
    analyze_molecule_centers(cif_file_path)


if __name__ == "__main__":
    main()