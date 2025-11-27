#!/usr/bin/env python3
"""
Improved analyzer for molecular centers in a crystal structure.

This script improves upon the basic molecule_center_analyzer by:
1. Using the enhanced CIF parser with molecular identification
2. Providing better output formatting
3. Including error handling and validation
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    try:
        # Try to import required modules
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.analysis import identify_molecules
        from molcrys_kit.structures import MolecularCrystal
        
        def analyze_molecule_centers(cif_file_path):
            """
            Load a crystal structure from CIF and analyze molecular centers.
            
            Parameters
            ----------
            cif_file_path : str
                Path to the CIF file containing the crystal structure.
            """
            print("=" * 60)
            print("MolCrysKit: Improved Molecular Center Analyzer")
            print("=" * 60)
            
            try:
                # Parse the CIF file
                print(f"\nParsing CIF file: {cif_file_path}")
                crystal = read_mol_crystal(cif_file_path)
                print("✓ CIF file parsed successfully.")
                
            except FileNotFoundError:
                print(f"✗ Error: File '{cif_file_path}' not found.")
                return False
            except Exception as e:
                print(f"✗ Error parsing CIF file: {e}")
                return False
            
            # Display basic crystal information
            print(f"\nCrystal Information:")
            print(f"  Lattice vectors:")
            for i, vec in enumerate(crystal.lattice):
                print(f"    a{i+1}: [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
            
            # Identify molecular units
            print(f"\nIdentifying molecular units...")
            try:
                molecules = crystal.molecules  # Already identified by read_mol_crystal
                print(f"✓ Successfully identified {len(molecules)} molecular unit(s)")
                
                # Analyze each molecule
                print(f"\nAnalyzing molecular centers...")
                for i, molecule in enumerate(molecules):
                    print(f"\n  Molecule {i+1}:")
                    
                    # Access the underlying ASE Atoms object
                    atoms = molecule.atoms
                    
                    # Get chemical formula
                    formula = atoms.get_chemical_formula()
                    print(f"    Formula: {formula}")
                    print(f"    Atoms: {len(atoms)}")
                    
                    # Calculate center of mass
                    com = atoms.get_center_of_mass()
                    
                    # Calculate geometric center
                    centroid = atoms.get_positions().mean(axis=0)
                    
                    # Print results in Cartesian coordinates
                    print(f"    Centers (Cartesian):")
                    print(f"      Center of mass:    [{com[0]:8.4f}, {com[1]:8.4f}, {com[2]:8.4f}]")
                    print(f"      Geometric center:  [{centroid[0]:8.4f}, {centroid[1]:8.4f}, {centroid[2]:8.4f}]")
                    
                    # Calculate fractional coordinates
                    frac_com = crystal.cartesian_to_fractional(com)
                    frac_centroid = crystal.cartesian_to_fractional(centroid)
                    
                    # Print results in Fractional coordinates
                    print(f"    Centers (Fractional):")
                    print(f"      Center of mass:    [{frac_com[0]:8.4f}, {frac_com[1]:8.4f}, {frac_com[2]:8.4f}]")
                    print(f"      Geometric center:  [{frac_centroid[0]:8.4f}, {frac_centroid[1]:8.4f}, {frac_centroid[2]:8.4f}]")
                    
            except Exception as e:
                print(f"✗ Error during analysis: {e}")
                return False
            
            print("\n" + "=" * 60)
            print("Analysis completed successfully!")
            print("=" * 60)
            return True
        
        # Check if file path is provided
        if len(sys.argv) != 2:
            print("Usage: python improved_molecule_center_analyzer.py <cif_file_path>")
            sys.exit(1)
            
        cif_file_path = sys.argv[1]
        success = analyze_molecule_centers(cif_file_path)
        sys.exit(0 if success else 1)
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure MolCrysKit is installed properly.")
        sys.exit(1)

if __name__ == "__main__":
    main()