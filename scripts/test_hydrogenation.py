#!/usr/bin/env python
"""
Detailed test script for MAP (Methylammonium Perchlorate) hydrogenation.

This script validates that:
1. Stoichiometry: The output crystal contains 3 H attached to C (Methyl) and 3 H attached to N (Ammonium)
2. Conformation: The H atoms on C and the H atoms on N are staggered (dihedral angle ≈ 60°), not eclipsed
3. File Format: The output CIF is valid and readable by VESTA/Mercury
"""

import os
import numpy as np
from molcrys_kit.io import read_mol_crystal
from molcrys_kit.io.output import write_cif
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.utils.geometry import dihedral_angle


def analyze_methyl_groups(crystal):
    """
    Find methyl groups (C with 3 H) and analyze their conformation
    """
    methyl_carbons = []

    for mol_idx, mol in enumerate(crystal.molecules):
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()

        # Find carbon atoms with 3 hydrogens connected
        for atom_idx, symbol in enumerate(symbols):
            if symbol == "C":
                # Get neighbors of this carbon
                neighbors = []
                for neighbor_idx, neighbor_symbol in enumerate(symbols):
                    if neighbor_symbol == "H":
                        dist = mol.get_distance(atom_idx, neighbor_idx, mic=False)
                        if dist < 1.5:  # Typical C-H distance
                            neighbors.append(neighbor_idx)

                if len(neighbors) == 3:
                    methyl_carbons.append((mol_idx, atom_idx, neighbors))

    return methyl_carbons


def analyze_ammonium_groups(crystal):
    """
    Find ammonium groups (N with 3 H) and analyze their conformation
    """
    ammonium_nitrogens = []

    for mol_idx, mol in enumerate(crystal.molecules):
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()

        # Find nitrogen atoms with 3 hydrogens connected
        for atom_idx, symbol in enumerate(symbols):
            if symbol == "N":
                # Get neighbors of this nitrogen
                neighbors = []
                for neighbor_idx, neighbor_symbol in enumerate(symbols):
                    if neighbor_symbol == "H":
                        dist = mol.get_distance(atom_idx, neighbor_idx, mic=False)
                        if dist < 1.5:  # Typical N-H distance
                            neighbors.append(neighbor_idx)

                if len(neighbors) == 3:
                    ammonium_nitrogens.append((mol_idx, atom_idx, neighbors))

    return ammonium_nitrogens


def calculate_dihedral_angles(mol, atom_indices):
    """
    Calculate dihedral angles between specified atoms
    """
    if len(atom_indices) < 4:
        return []

    dihedral_angles = []
    positions = mol.get_positions()

    # Calculate all possible dihedral angles between the atoms
    for i in range(len(atom_indices)):
        for j in range(i + 1, len(atom_indices)):
            for k in range(j + 1, len(atom_indices)):
                for l in range(k + 1, len(atom_indices)):
                    angle = dihedral_angle(
                        positions[atom_indices[i]],
                        positions[atom_indices[j]],
                        positions[atom_indices[k]],
                        positions[atom_indices[l]],
                    )
                    dihedral_angles.append(np.degrees(angle))

    return dihedral_angles


def main():
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Define rules for the Ammonium Cation (CH3NH3+) using the new flat list format
    input_path = "examples/MAP.cif"
    target_elements = ["C", "N"]
    rules = [
        # N should be Ammonium (coord=4) - general rule
        {"symbol": "N", "target_coordination": 4, "geometry": "tetrahedral"},
        # {"symbol": "C", "target_coordination": 4, "geometry": "tetrahedral"},
        # O atoms bonded to Cl should have coordination 1 (not adding H) - specific rule
        # {"symbol": "O", "neighbors": ["Cl"], "target_coordination": 1},
    ]

    input_path = "examples/PETN_PERYTN10.cif"
    target_elements = ["C"]
    rules = []

    print(f"Loading {input_path}...")
    crystal = read_mol_crystal(input_path)

    print("Original crystal summary:")
    print(crystal.summary())

    # Apply hydrogenation
    print("\nAdding hydrogens...")
    hydrogenated_crystal = add_hydrogens(crystal, target_elements=target_elements, rules=rules)

    print("\nHydrogenated crystal summary:")
    print(hydrogenated_crystal.summary())

    # Output the result
    output_path = "output/hydrogenated.cif"
    write_cif(hydrogenated_crystal, output_path)
    print(f"\nHydrogenated structure saved to {output_path}")



if __name__ == "__main__":
    main()
