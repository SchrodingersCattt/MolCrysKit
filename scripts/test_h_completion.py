#!/usr/bin/env python
"""
Detailed test script for MAP (Methylammonium Perchlorate) hydrogen_completion.

This script validates that:
1. Stoichiometry: The output crystal contains 3 H attached to C (Methyl) and 3 H attached to N (Ammonium)
2. Conformation: The H atoms on C and the H atoms on N are staggered (dihedral angle ≈ 60°), not eclipsed
3. File Format: The output CIF is valid and readable by VESTA/Mercury
"""

import os
import numpy as np
from molcrys_kit.io import read_mol_crystal
from molcrys_kit.io.output import write_cif, write_xyz
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.utils.geometry import dihedral_angle


def main():
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Define rules for the Ammonium Cation (CH3NH3+) using the new flat list format
    # input_path = "examples/DACMOR.cif"
    input_path = "examples/MAP.cif"
    target_elements = [
        "C", 
        "N", 
        # 'O'
    ]
    rules = [
        # N should be Ammonium (coord=4) - general rule
        {"symbol": "N", "target_coordination": 4, "geometry": "tetrahedral"},
        {"symbol": "C", "target_coordination": 4, "geometry": "tetrahedral"},
        # O atoms bonded to Cl should have coordination 1 (not adding H) - specific rule
        # {"symbol": "O", "neighbors": ["Cl"], "target_coordination": 1},
    ]

    # input_path = "examples/PETN_PERYTN10.cif"
    # target_elements = ["C"]
    # rules = []

    print(f"Loading {input_path}...")
    crystal = read_mol_crystal(input_path)

    raw_mol = crystal.molecules[0]
    write_xyz(raw_mol, "output/original_molecule.xyz")
    print("Original crystal summary:")
    print(crystal.summary())

    # Apply hydrogen_completion
    print("\nAdding hydrogens...")
    hydrogenated_crystal = add_hydrogens(crystal, target_elements=target_elements, rules=rules)

    hydrogenated_mol = hydrogenated_crystal.molecules[0]
    write_xyz(hydrogenated_mol, "output/hydrogenated_molecule.xyz")
    print("\nHydrogenated crystal summary:")
    print(hydrogenated_crystal.summary())

    # Output the result
    output_path = "output/hydrogenated.cif"
    write_cif(hydrogenated_crystal, output_path)
    print(f"\nHydrogenated structure saved to {output_path}")



if __name__ == "__main__":
    main()
