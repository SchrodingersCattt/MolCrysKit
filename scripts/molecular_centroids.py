#!/usr/bin/env python3
"""
Script to calculate centroids of molecules in a molecular crystal.

This script demonstrates how to use MolCrysKit to parse a CIF file and 
calculate the centroids (center of mass) of each identified molecular unit.
"""

import sys
import os
import glob
import numpy as np
from time import time

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from molcrys_kit.io import read_mol_crystal
    from molcrys_kit.structures.molecule import CrystalMolecule
except ImportError as e:
    print(f"Error importing MolCrysKit: {e}")
    print("Please ensure the package is installed with 'pip install -e .'")
    sys.exit(1)


def calculate_molecular_centroids(cif_file_path):
    """
    Calculate centroids of all molecules in a crystal structure.
    
    Parameters
    ----------
    cif_file_path : str
        Path to the CIF file to analyze.
        
    Returns
    -------
    list
        List of dictionaries containing molecule information.
    """
    try:
        # Parse the CIF file with molecular identification
        crystal = read_mol_crystal(cif_file_path)
    except Exception as e:
        print(f"Error parsing CIF file {cif_file_path}: {e}")
        return []
    
    # Calculate centroids for each molecule (now Molecule objects)
    centroids = []
    for i, molecule in enumerate(crystal.molecules):
        try:
            # Get geometric center using the new method
            geometric_center = molecule.get_centroid()
            
            # Get center of mass
            center_of_mass = molecule.get_center_of_mass()
            
            # Get ellipsoid radii
            ellipsoid_radii = molecule.get_ellipsoid_radii()
            
            # Convert geometric center to fractional coordinates
            geometric_center_frac = molecule.get_centroid_frac()
            
            # Store information about this molecule
            molecule_info = {
                'index': i,
                'num_atoms': len(molecule),
                'elements': list(set(molecule.get_chemical_symbols())),
                'geometric_center': geometric_center,
                'geometric_center_fractional': geometric_center_frac,
                'center_of_mass': center_of_mass,
                'ellipsoid_radii': ellipsoid_radii,
                'principal_axes': molecule.get_principal_axes()
            }
            
            centroids.append(molecule_info)
            
        except Exception as e:
            print(f"Warning: Could not process molecule {i} in {cif_file_path}: {e}")
            continue
    
    return centroids


def print_centroids(centroids, cif_file_path):
    """
    Print formatted output of molecular centroids.
    
    Parameters
    ----------
    centroids : list
        List of molecule information dictionaries.
    cif_file_path : str
        Path to the CIF file analyzed.
    """
    print(f"\nAnalysis of {cif_file_path}")
    print("=" * (len(cif_file_path) + 12))
    
    if not centroids:
        print("No molecules found or error occurred.")
        return
    
    print(f"Found {len(centroids)} molecule(s):")
    
    for mol_info in centroids:
        print(f"\nMolecule {mol_info['index'] + 1}:")
        print(f"  Atoms: {mol_info['num_atoms']}")
        print(f"  Elements: {', '.join(mol_info['elements'])}")
        print(f"  Geometric center (Cartesian): [{mol_info['geometric_center'][0]:.6f}, "
              f"{mol_info['geometric_center'][1]:.6f}, {mol_info['geometric_center'][2]:.6f}]")
        print(f"  Geometric center (Fractional): [{mol_info['geometric_center_fractional'][0]:.6f}, "
              f"{mol_info['geometric_center_fractional'][1]:.6f}, {mol_info['geometric_center_fractional'][2]:.6f}]")
        print(f"  Center of mass: [{mol_info['center_of_mass'][0]:.6f}, "
              f"{mol_info['center_of_mass'][1]:.6f}, {mol_info['center_of_mass'][2]:.6f}]")
        print(f"  Ellipsoid radii: a={mol_info['ellipsoid_radii'][0]:.4f}, "
              f"b={mol_info['ellipsoid_radii'][1]:.4f}, c={mol_info['ellipsoid_radii'][2]:.4f}")


def main():
    """Main function to process CIF files."""
    # Get CIF file path from command line argument or use default
    if len(sys.argv) > 1:
        cif_file_path = sys.argv[1]
    else:
        # Look for test CIF files
        test_cif_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data')
        test_cif_files = glob.glob(os.path.join(test_cif_dir, '*.cif'))
        
        if test_cif_files:
            cif_file_path = test_cif_files[0]
            print(f"No CIF file specified. Using test file: {cif_file_path}")
        else:
            print("Usage: python molecular_centroids.py [cif_file_path]")
            print("Please provide a path to a CIF file or ensure test files are available.")
            return
    
    # Process the CIF file
    print("Processing molecular crystal...")
    start_time = time()
    
    centroids = calculate_molecular_centroids(cif_file_path)
    
    elapsed_time = time() - start_time
    print(f"Processing completed in {elapsed_time:.4f} seconds")
    
    # Print results
    print_centroids(centroids, cif_file_path)


if __name__ == "__main__":
    main()