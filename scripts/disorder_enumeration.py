#!/usr/bin/env python3
"""
Disorder enumeration example.
"""

from ase.spacegroup import crystal
from ase.io import write

from molcrys_kit.structures import MolecularCrystal
from molcrys_kit.analysis.disorder import scan_disordered_atoms, enumerate_disorder_configurations, rank_configurations


def create_disordered_crystal():
    """Create a crystal with disordered atoms for demonstration."""
    # Define lattice vectors
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    if not ASE_AVAILABLE:
        print("ASE not available, cannot create molecules")
        return None
    
    # Create a molecule with disordered atoms
    # Note: ASE Atoms doesn't natively support occupancy, so we'll store it as an array
    atoms1 = Atoms('COHN', 
                   positions=[[0.1, 0.1, 0.1],
                              [0.2, 0.1, 0.1],
                              [0.2, 0.1, 0.1],
                              [0.1, 0.2, 0.1]])
    
    # Add occupancy as an array attribute
    occupancies = np.array([1.0, 0.7, 0.3, 1.0])  # O and N share the same site
    atoms1.set_array('occupancies', occupancies)
    
    # Create another regular molecule
    atoms2 = Atoms('H2O', 
                   positions=[[5.0, 5.0, 5.0],
                              [5.1, 5.0, 5.0],
                              [5.0, 5.1, 5.0]])
    
    # Create crystal
    crystal = MolecularCrystal(lattice, [atoms1, atoms2])
    return crystal


def main():
    """Main function to demonstrate disorder enumeration."""
    print("MolCrysKit Example: Disorder Enumeration")
    print("=" * 40)
    
    if not ASE_AVAILABLE:
        print("This example requires ASE. Please install it with 'pip install ase'")
        return
    
    # Create example crystal
    crystal = create_disordered_crystal()
    
    if crystal is None:
        return
    
    # Print initial crystal summary
    print("Initial crystal:")
    print(crystal.summary())
    
    # Scan for disordered atoms
    print("\nScanning for disordered atoms...")
    disordered = scan_disordered_atoms(crystal)
    
    print(f"Found {len(disordered)} molecules with disordered atoms:")
    for i, molecule in enumerate(disordered):
        symbols = molecule.get_chemical_symbols()
        if molecule.has('occupancies'):
            occupancies = molecule.get_array('occupancies')
        else:
            occupancies = [1.0] * len(molecule)
            
        print(f"  Molecule {i+1}: {len(molecule)} atoms")
        for j, (symbol, occ) in enumerate(zip(symbols, occupancies)):
            print(f"    {symbol} (occupancy: {occ:.2f})")


if __name__ == "__main__":
    main()