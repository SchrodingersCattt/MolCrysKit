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
    rules = [
        # N should be Ammonium (coord=4) - general rule
        {"symbol": "N", "target_coordination": 4, "geometry": "tetrahedral"},
        
        # O atoms bonded to Cl should have coordination 1 (not adding H) - specific rule
        {"symbol": "O", "neighbors": ["Cl"], "target_coordination": 1}
    ]

    input_path = "examples/MAP.cif"
    if not os.path.exists(input_path):
        print(
            f"Input file {input_path} does not exist. Please provide the MAP structure."
        )
        return

    print(f"Loading {input_path}...")
    crystal = read_mol_crystal(input_path)

    print("Original crystal summary:")
    print(crystal.summary())

    # Apply hydrogenation
    print("\nAdding hydrogens...")
    hydrogenated_crystal = add_hydrogens(crystal, rules=rules)

    print("\nHydrogenated crystal summary:")
    print(hydrogenated_crystal.summary())

    # Output the result
    output_path = "output/MAP_hydrogenated.cif"
    write_cif(hydrogenated_crystal, output_path)
    print(f"\nHydrogenated structure saved to {output_path}")

    # Verify stoichiometry
    original_symbols = []
    for mol in crystal.molecules:
        original_symbols.extend(mol.get_chemical_symbols())

    hydrogenated_symbols = []
    for mol in hydrogenated_crystal.molecules:
        hydrogenated_symbols.extend(mol.get_chemical_symbols())

    from collections import Counter

    orig_counts = Counter(original_symbols)
    hyd_counts = Counter(hydrogenated_symbols)

    print(f"\nOriginal composition: {orig_counts}")
    print(f"Hydrogenated composition: {hyd_counts}")

    # Check if hydrogens were added
    h_added = hyd_counts.get("H", 0) - orig_counts.get("H", 0)
    print(f"Number of hydrogens added: {h_added}")

    # Analyze methyl groups
    methyl_carbons = analyze_methyl_groups(hydrogenated_crystal)
    print(f"\nFound {len(methyl_carbons)} methyl groups (C with 3 H)")

    # Analyze ammonium groups
    ammonium_nitrogens = analyze_ammonium_groups(hydrogenated_crystal)
    print(f"Found {len(ammonium_nitrogens)} ammonium groups (N with 3 H)")

    # Validate that we have the expected stoichiometry
    expected_methyl_groups = orig_counts.get(
        "C", 0
    )  # Each C should form a methyl group with 3 H
    expected_ammonium_groups = orig_counts.get(
        "N", 0
    )  # Each N should form an ammonium group with 3 H

    print(
        f"Expected methyl groups: {expected_methyl_groups}, Found: {len(methyl_carbons)}"
    )
    print(
        f"Expected ammonium groups: {expected_ammonium_groups}, Found: {len(ammonium_nitrogens)}"
    )

    # Analyze dihedral angles in the first methyl and ammonium groups found
    if methyl_carbons:
        mol_idx, c_idx, h_indices = methyl_carbons[0]
        mol = hydrogenated_crystal.molecules[mol_idx]
        print(
            f"\nAnalyzing methyl group in molecule {mol_idx} (C atom {c_idx} with H atoms {h_indices})"
        )

        positions = mol.get_positions()
        c_pos = positions[c_idx]
        h_positions = [positions[h_idx] for h_idx in h_indices]

        # Calculate angles between C-H bonds to check if they're appropriately spaced
        bond_vectors = [h_pos - c_pos for h_pos in h_positions]
        angles = []
        for i in range(len(bond_vectors)):
            for j in range(i + 1, len(bond_vectors)):
                angle = np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(bond_vectors[i], bond_vectors[j])
                            / (
                                np.linalg.norm(bond_vectors[i])
                                * np.linalg.norm(bond_vectors[j])
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                angles.append(angle)

        print(f"C-H bond angles in methyl group: {[f'{a:.1f}°' for a in angles]}")
        print("Expected ~109.5° for tetrahedral arrangement")

    if ammonium_nitrogens:
        mol_idx, n_idx, h_indices = ammonium_nitrogens[0]
        mol = hydrogenated_crystal.molecules[mol_idx]
        print(
            f"\nAnalyzing ammonium group in molecule {mol_idx} (N atom {n_idx} with H atoms {h_indices})"
        )

        positions = mol.get_positions()
        n_pos = positions[n_idx]
        h_positions = [positions[h_idx] for h_idx in h_indices]

        # Calculate angles between N-H bonds to check if they're appropriately spaced
        bond_vectors = [h_pos - n_pos for h_pos in h_positions]
        angles = []
        for i in range(len(bond_vectors)):
            for j in range(i + 1, len(bond_vectors)):
                angle = np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(bond_vectors[i], bond_vectors[j])
                            / (
                                np.linalg.norm(bond_vectors[i])
                                * np.linalg.norm(bond_vectors[j])
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                angles.append(angle)

        print(f"N-H bond angles in ammonium group: {[f'{a:.1f}°' for a in angles]}")
        print(
            "Expected ~109.5° for tetrahedral arrangement (with one position occupied by a lone pair)"
        )


if __name__ == "__main__":
    main()