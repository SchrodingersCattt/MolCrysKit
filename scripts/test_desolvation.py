#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for desolvation functionality.

This script tests the desolvation functionality by:
1. Loading a ZIF-8 CIF file
2. Printing the initial stoichiometry
3. Removing water molecules
4. Printing the final stoichiometry
5. Saving the result to a new CIF file
"""

import os
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.io.output import write_cif
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.operations.desolvation import remove_solvents


def main():
    # Define file paths
    input_path = "examples/ZIF-8.cif"  # Assuming this is where the ZIF-8 file is located
    output_path = "output/ZIF-8_desolvated.cif"

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        print("Please ensure the ZIF-8 CIF file is available at the specified path.")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading ZIF-8 crystal...")
    try:
        crystal = read_mol_crystal(input_path)
        print(f"Successfully loaded crystal: {crystal}")
    except Exception as e:
        print(f"Error loading crystal: {e}")
        return

    print("\nAnalyzing initial stoichiometry...")
    analyzer = StoichiometryAnalyzer(crystal)
    analyzer.print_species_summary()

    print("\nPerforming desolvation (removing water)...")
    try:
        desolvated_crystal = remove_solvents(crystal, targets=["Water"])
        print("Desolvation completed successfully!")
    except ValueError as e:
        print(f"Error during desolvation: {e}")
        return

    print("\nAnalyzing final stoichiometry...")
    final_analyzer = StoichiometryAnalyzer(desolvated_crystal)
    final_analyzer.print_species_summary()

    print(f"\nSaving desolvated crystal to {output_path}...")
    try:
        write_cif(desolvated_crystal, output_path)
        print("Desolvated crystal saved successfully!")
    except Exception as e:
        print(f"Error saving crystal: {e}")

    print("\nDesolvation test completed!")


if __name__ == "__main__":
    main()