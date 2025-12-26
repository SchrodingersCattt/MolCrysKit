#!/usr/bin/env python
"""
Test script for hydrogenation functionality.

This script demonstrates the hydrogenation of methylammonium perchlorate (MAP).
"""

import os
import numpy as np
from ase import Atoms
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.io import read_mol_crystal
from molcrys_kit.io.output import write_cif
from molcrys_kit.operations import add_hydrogens


def main():
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Define rules for the Ammonium Cation (CH3NH3+)
    rules = {
        "global_overrides": {
            "N": {"geometry": "tetrahedral", "target_coordination": 4}
            # Note: C defaults to tetrahedral/coord=4, so it needs 3 H automatically.
            # Note: N defaults to coord=3 usually, so we override it to 4 to get NH3+.
        }
    }

    # Try to load the input file
    input_path = "examples/MAP.cif"
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist. Creating a mock example...")

        # Create a mock molecular crystal for testing
        # Using a simple cubic lattice with C and N atoms
        lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])

        # Create a simple molecule with C and N atoms
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # C atom
                [1.5, 0.0, 0.0],  # N atom
            ]
        )

        symbols = ["C", "N"]
        mol = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
        crystal = MolecularCrystal(lattice, [mol])

        print("Created mock crystal for testing...")
    else:
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


if __name__ == "__main__":
    main()
