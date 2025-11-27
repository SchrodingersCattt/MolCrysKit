#!/usr/bin/env python3
"""
Analyze molecular centers in a crystal structure.
"""

import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    try:
        # Try to import required modules
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.analysis import identify_molecules
        from molcrys_kit.structures import MolecularCrystal
        
        import numpy as np
        
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
                crystal = read_mol_crystal(cif_file_path)
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
                molecules = crystal.molecules  # Already identified by read_mol_crystal
                print(f"Successfully identified {len(molecules)} molecular unit(s)")
                
                # Analyze each molecule
                print("\nAnalyzing molecular centers...")
                for i, molecule in enumerate(molecules):
                    # Access the underlying ASE Atoms object
                    atoms = molecule.atoms
                    
                    # Calculate center of mass
                    com = atoms.get_center_of_mass()
                    
                    # Calculate geometric center
                    centroid = atoms.get_positions().mean(axis=0)
                    
                    # Print results
                    print(f"\nMolecule {i+1}:")
                    print(f"  Center of mass (Cartesian): [{com[0]:8.4f}, {com[1]:8.4f}, {com[2]:8.4f}]")
                    print(f"  Geometric center (Cartesian): [{centroid[0]:8.4f}, {centroid[1]:8.4f}, {centroid[2]:8.4f}]")
                    
                    # Calculate fractional coordinates
                    frac_com = crystal.cartesian_to_fractional(com)
                    frac_centroid = crystal.cartesian_to_fractional(centroid)
                    
                    print(f"  Center of mass (Fractional): [{frac_com[0]:8.4f}, {frac_com[1]:8.4f}, {frac_com[2]:8.4f}]")
                    print(f"  Geometric center (Fractional): [{frac_centroid[0]:8.4f}, {frac_centroid[1]:8.4f}, {frac_centroid[2]:8.4f}]")
                    
            except Exception as e:
                print(f"Error during analysis: {e}")
                return
            
            print("\nAnalysis completed successfully!")
        
        # Check if file path is provided
        if len(sys.argv) != 2:
            print("Usage: python molecule_center_analyzer.py <cif_file_path>")
            sys.exit(1)
            
        cif_file_path = sys.argv[1]
        analyze_molecule_centers(cif_file_path)
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure MolCrysKit is installed properly.")
        sys.exit(1)

if __name__ == "__main__":
    main()