#!/usr/bin/env python3
"""
Example: Disorder enumeration in molecular crystals.
"""

import numpy as np
from molcrys.structures import Atom, Molecule, MolecularCrystal
from molcrys.analysis.disorder import (
    identify_disordered_atoms, 
    group_disordered_atoms, 
    generate_ordered_configurations,
    has_disorder
)


def create_disordered_crystal():
    """Create a crystal structure with partial occupancy atoms for demonstration."""
    # Define lattice vectors
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create a molecule with disordered atoms
    atoms = [
        Atom("C", np.array([0.1, 0.1, 0.1]), occupancy=1.0),     # Regular carbon
        Atom("O", np.array([0.2, 0.1, 0.1]), occupancy=0.7),     # Partial occupancy oxygen
        Atom("N", np.array([0.2, 0.1, 0.1]), occupancy=0.3),     # Partial occupancy nitrogen (same site)
        Atom("H", np.array([0.1, 0.2, 0.1]), occupancy=1.0),     # Regular hydrogen
    ]
    molecule = Molecule(atoms)
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [molecule])
    return crystal


def main():
    """Main function to demonstrate disorder analysis."""
    print("MolCrysKit Example: Disorder Analysis")
    print("=" * 40)
    
    # Create example crystal with disorder
    crystal = create_disordered_crystal()
    
    # Print crystal summary
    print("Crystal with disorder:")
    print(crystal.summary())
    
    # Check if crystal has disorder
    print(f"\nHas disorder: {has_disorder(crystal)}")
    
    # Identify disordered atoms
    disordered_atoms = identify_disordered_atoms(crystal)
    print(f"\nFound {len(disordered_atoms)} disordered atoms:")
    for atom in disordered_atoms:
        print(f"  {atom.symbol} at {atom.frac_coords} (occupancy: {atom.occupancy})")
    
    # Group disordered atoms by site
    disorder_groups = group_disordered_atoms(crystal)
    print(f"\nDisorder groups:")
    for site_id, atoms in disorder_groups.items():
        print(f"  Site {site_id}: {len(atoms)} atoms")
        for atom in atoms:
            print(f"    {atom.symbol} (occupancy: {atom.occupancy})")
    
    # Generate ordered configurations
    print(f"\nGenerating ordered configurations...")
    configurations = list(generate_ordered_configurations(crystal))
    print(f"Generated {len(configurations)} configurations (simplified implementation)")


if __name__ == "__main__":
    main()