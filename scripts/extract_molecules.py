#!/usr/bin/env python3
"""
Extract molecules from a crystal structure.
"""

from ase.io import read, write

from molcrys_kit.structures import MolecularCrystal
from molcrys_kit.analysis import identify_molecules


def create_water_crystal():
    """Create a simple crystal structure with water molecules for demonstration."""
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    
    if not ASE_AVAILABLE:
        print("ASE not available, cannot create water molecules")
        return None
    
    # Create water molecule 1
    atoms1 = Atoms('OH2', 
                   positions=[[0.1, 0.1, 0.1],
                              [1.2, 0.1, 0.1],
                              [0.1, 1.2, 0.1]])
    
    # Create water molecule 2
    atoms2 = Atoms('OH2', 
                   positions=[[3.1, 3.1, 3.1],
                              [4.2, 3.1, 3.1],
                              [3.1, 4.2, 3.1]])
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [atoms1, atoms2])
    return crystal


def main():
    """Main function to demonstrate molecule extraction."""
    print("MolCrysKit Example: Molecular Unit Extraction")
    print("=" * 50)
    
    if not ASE_AVAILABLE:
        print("This example requires ASE. Please install it with 'pip install ase'")
        return
    
    # Create example crystal
    crystal = create_water_crystal()
    
    if crystal is None:
        return
    
    # Print initial crystal summary
    print("Initial crystal:")
    print(crystal.summary())
    
    # Identify molecular units
    print("\nIdentifying molecular units...")
    molecules = identify_molecules(crystal)
    
    print(f"\nFound {len(molecules)} molecular units:")
    for i, molecule in enumerate(molecules):
        symbols = molecule.get_chemical_symbols()
        print(f"  Molecule {i+1}: {len(molecule)} atoms")
        print(f"    Chemical symbols: {symbols}")
        print(f"    Center of mass: {molecule.get_center_of_mass()}")


if __name__ == "__main__":
    main()