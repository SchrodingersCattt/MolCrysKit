#!/usr/bin/env python3
"""
Example: Extract molecular units from a crystal structure.
"""

import numpy as np
from molcrys.structures import Atom, Molecule, MolecularCrystal
from molcrys.analysis import identify_molecules


def create_water_crystal():
    """Create a simple crystal structure with water molecules for demonstration."""
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    
    # Create water molecule 1
    atoms1 = [
        Atom("O", np.array([0.1, 0.1, 0.1])),  # Oxygen
        Atom("H", np.array([0.2, 0.1, 0.1])),  # Hydrogen
        Atom("H", np.array([0.1, 0.2, 0.1])),  # Hydrogen
    ]
    molecule1 = Molecule(atoms1)
    
    # Create water molecule 2
    atoms2 = [
        Atom("O", np.array([0.6, 0.6, 0.6])),  # Oxygen
        Atom("H", np.array([0.7, 0.6, 0.6])),  # Hydrogen
        Atom("H", np.array([0.6, 0.7, 0.6])),  # Hydrogen
    ]
    molecule2 = Molecule(atoms2)
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule1, molecule2])
    return crystal


def main():
    """Main function to demonstrate molecule extraction."""
    print("MolCrysKit Example: Molecular Unit Extraction")
    print("=" * 50)
    
    # Create example crystal
    crystal = create_water_crystal()
    
    # Print initial crystal summary
    print("Initial crystal:")
    print(crystal.summary())
    
    # Identify molecular units
    print("\nIdentifying molecular units...")
    molecules = identify_molecules(crystal)
    
    print(f"\nFound {len(molecules)} molecular units:")
    for i, molecule in enumerate(molecules):
        print(f"  Molecule {i+1}: {len(molecule.atoms)} atoms")
        for atom in molecule.atoms:
            print(f"    {atom.symbol} at {atom.frac_coords}")
    
    # Show bonds within each molecule
    print("\nBonds within molecules:")
    for i, molecule in enumerate(molecules):
        bonds = molecule.get_bonds()
        print(f"  Molecule {i+1}: {len(bonds)} bonds")
        for bond in bonds:
            print(f"    Atom {bond[0]} - Atom {bond[1]}: {bond[2]:.3f}")


if __name__ == "__main__":
    main()