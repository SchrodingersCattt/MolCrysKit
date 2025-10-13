#!/usr/bin/env python3
"""
Improved molecule center analyzer with better molecule assignment.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Try to import required modules
    from molcrys_kit.io import parse_cif
    from molcrys_kit.analysis import assign_atoms_to_molecules
    from molcrys_kit.structures import Atom, Molecule, MolecularCrystal
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have installed the molcrys-kit package:")
    print("pip install -e .")
    return 1


def create_water_crystal():
    """
    Create a crystal structure with proper water molecules for demonstration.
    
    Returns
    -------
    MolecularCrystal
        A crystal with water molecules.
    """
    # Define lattice vectors
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create atoms for two water molecules
    atoms = [
        # First water molecule
        Atom("O", np.array([0.1, 0.1, 0.1])),  # Oxygen
        Atom("H", np.array([0.17, 0.1, 0.1])),  # Hydrogen
        Atom("H", np.array([0.1, 0.17, 0.1])),  # Hydrogen
        # Second water molecule
        Atom("O", np.array([0.6, 0.6, 0.6])),  # Oxygen
        Atom("H", np.array([0.67, 0.6, 0.6])),  # Hydrogen
        Atom("H", np.array([0.6, 0.67, 0.6])),  # Hydrogen
    ]
    
    # Put all atoms in one molecule initially
    molecule = Molecule(atoms)
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule])
    return crystal


def analyze_molecule_centers_in_crystal(crystal):
    """
    Analyze molecular centers in a crystal structure.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal structure to analyze.
    """
    # Identify molecular units using the assignment function
    print("Identifying molecular units...")
    try:
        crystal_with_molecules = assign_atoms_to_molecules(crystal)
        molecules = crystal_with_molecules.molecules
        print(f"Found {len(molecules)} molecular units.")
    except Exception as e:
        print(f"Error identifying molecules: {e}")
        return
    
    # Calculate and display center of mass for each molecule
    print("\nMolecular Centers (Fractional Coordinates):")
    print("-" * 55)
    
    if not molecules:
        print("No molecules found in the structure.")
        return
    
    for i, molecule in enumerate(molecules, 1):
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
        print(f"  Bonds: {len(bonds)}")
        print()


def analyze_from_cif(cif_file_path):
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
        
        # Analyze molecule centers
        analyze_molecule_centers_in_crystal(crystal)
        
    except FileNotFoundError:
        print(f"Error: File '{cif_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error processing CIF file: {e}")
        return


def main():
    """
    Main function to run the molecular center analysis.
    """
    print("Molecular Center Analyzer (Improved Version)")
    print("=" * 45)
    
    if len(sys.argv) < 2:
        print("Usage: python improved_molecule_center_analyzer.py <cif_file>")
        print("       python improved_molecule_center_analyzer.py --sample")
        print()
        print("Running on sample structure...")
        
        # Create and analyze a sample structure
        crystal = create_water_crystal()
        print("Created sample crystal with water molecules")
        print(f"Initial crystal has {len(crystal.molecules)} molecule(s) with {sum(len(mol.atoms) for mol in crystal.molecules)} total atoms")
        
        analyze_molecule_centers_in_crystal(crystal)
        return
    
    # Check if user wants to run on sample structure
    if sys.argv[1] == "--sample":
        crystal = create_water_crystal()
        print("Created sample crystal with water molecules")
        analyze_molecule_centers_in_crystal(crystal)
        return
    
    # Process the provided CIF file
    cif_file_path = sys.argv[1]
    analyze_from_cif(cif_file_path)


if __name__ == "__main__":
    main()