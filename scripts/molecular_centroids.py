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
    from molcrys_kit.structures.molecule import EnhancedMolecule
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
    
    # Calculate centroids for each molecule (now EnhancedMolecule objects)
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
            frac_geometric_center = molecule.get_centroid_frac()
            
            # Convert center of mass to fractional coordinates
            frac_center_of_mass = crystal.cartesian_to_fractional(center_of_mass)
            
            # Get chemical formula using ASE's built-in method
            formula = molecule.atoms.get_chemical_formula()
            
            centroids.append({
                'index': i+1,
                'geometric_center': geometric_center,
                'center_of_mass': center_of_mass,
                'frac_geometric_center': frac_geometric_center,
                'frac_center_of_mass': frac_center_of_mass,
                'ellipsoid_radii': ellipsoid_radii,
                'num_atoms': len(molecule.atoms),
                'formula': formula,
                'filename': os.path.basename(cif_file_path)
            })
        except Exception as e:
            print(f"Error processing molecule {i+1} in {cif_file_path}: {e}")
            continue
    
    return centroids


def find_cif_files(path):
    """
    Find all CIF files in a directory (recursively).
    
    Parameters
    ----------
    path : str
        Path to search for CIF files.
        
    Returns
    -------
    list
        List of CIF file paths.
    """
    if os.path.isfile(path) and path.lower().endswith('.cif'):
        return [path]
    elif os.path.isdir(path):
        cif_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.cif'):
                    cif_files.append(os.path.join(root, file))
        return cif_files
    else:
        return []


def process_and_print_single_file(cif_file):
    """Process a single CIF file and print its results immediately."""
    print(f"\nCrystal: {os.path.basename(cif_file)}")
    try:
        centroids = calculate_molecular_centroids(cif_file)
        print(f"  Molecules: {len(centroids)}")
        for centroid in centroids:
            frac_gc = centroid['frac_geometric_center']
            frac_com = centroid['frac_center_of_mass']
            formula = centroid['formula']
            num_atoms = centroid['num_atoms']
            radii = centroid['ellipsoid_radii']
            print(f"    Molecule {centroid['index']}: {num_atoms} atoms, formula {formula}")
            print(f"      Geometric center: [{frac_gc[0]:8.5f}, {frac_gc[1]:8.5f}, {frac_gc[2]:8.5f}]")
            print(f"      Center of mass:   [{frac_com[0]:8.5f}, {frac_com[1]:8.5f}, {frac_com[2]:8.5f}]")
            print(f"      Ellipsoid radii:  a={radii[0]:6.3f}, b={radii[1]:6.3f}, c={radii[2]:6.3f}")
        return centroids
    except Exception as e:
        print(f"Error processing {cif_file}: {e}")
        return []


def main():
    """Main function to run the molecular centroid calculation."""
    if len(sys.argv) != 2:
        print("Usage: python molecular_centroids.py <cif_file_or_directory_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Find all CIF files
    cif_files = find_cif_files(input_path)
    
    if not cif_files:
        print(f"Error: No CIF files found in '{input_path}'.")
        sys.exit(1)
    
    print(f"Found {len(cif_files)} CIF files to process.")
    
    # Process all CIF files and print results immediately
    all_centroids = []
    for i, cif_file in enumerate(cif_files):
        print(f"\n[{i+1}/{len(cif_files)}] Processing: {cif_file}")
        centroids = process_and_print_single_file(cif_file)
        all_centroids.extend(centroids)
    
    # Print final summary
    print(f"\nFinal Summary:")
    print(f"  Total CIF files processed: {len(cif_files)}")
    print(f"  Total molecules identified: {len(all_centroids)}")


if __name__ == "__main__":
    main()