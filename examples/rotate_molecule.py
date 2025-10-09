#!/usr/bin/env python3
"""
Example: Rotate a molecule in a crystal structure.
"""

import numpy as np
from molcrys.structures import Atom, Molecule, MolecularCrystal
from molcrys.operations.rotation import rotate_molecule, translate_molecule


def create_benzene_molecule():
    """Create a benzene molecule for demonstration."""
    # Create benzene ring atoms (simplified coordinates)
    atoms = [
        Atom("C", np.array([0.0, 1.0, 0.0])),
        Atom("C", np.array([0.866, 0.5, 0.0])),
        Atom("C", np.array([0.866, -0.5, 0.0])),
        Atom("C", np.array([0.0, -1.0, 0.0])),
        Atom("C", np.array([-0.866, -0.5, 0.0])),
        Atom("C", np.array([-0.866, 0.5, 0.0])),
    ]
    
    return Molecule(atoms)


def main():
    """Main function to demonstrate molecular rotation."""
    print("MolCrysKit Example: Molecular Rotation")
    print("=" * 40)
    
    # Create a benzene molecule
    benzene = create_benzene_molecule()
    print(f"Initial benzene center of mass: {benzene.center_of_mass}")
    
    # Create a simple crystal
    lattice = np.array([
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ])
    
    crystal = MolecularCrystal(lattice, [benzene])
    print(f"Initial crystal summary:")
    print(crystal.summary())
    
    # Rotate the benzene molecule around the z-axis by 90 degrees (π/2 radians)
    print(f"\nRotating benzene molecule around z-axis by 90 degrees...")
    rotate_molecule(benzene, np.array([0, 0, 1]), np.pi/2)
    print(f"New benzene center of mass: {benzene.center_of_mass}")
    
    # Translate the molecule
    print(f"\nTranslating benzene molecule by [0.1, 0.1, 0.1]...")
    translate_molecule(benzene, np.array([0.1, 0.1, 0.1]))
    print(f"New benzene center of mass: {benzene.center_of_mass}")
    
    print(f"\nFinal crystal summary:")
    print(crystal.summary())


if __name__ == "__main__":
    main()