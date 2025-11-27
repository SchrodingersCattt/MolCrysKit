#!/usr/bin/env python3
"""
Molecule center analyzer with ASE integration.

This script demonstrates how to use MolCrysKit together with ASE for 
enhanced molecular crystal analysis.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ase import Atoms
    from ase.io import read
    from ase.visualize import view
    ASE_AVAILABLE = True
except ImportError as e:
    print(f"Error: ASE not found. Please install it with 'pip install ase'")
    print(f"Import error: {e}")
    ASE_AVAILABLE = False
    sys.exit(1)


def create_sample_crystal():
    """Create a sample crystal with multiple water molecules for demonstration."""
    # Define lattice vectors (simple cubic for demonstration)
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    # Create water molecule 1 at origin
    atoms1 = Atoms('OH2', 
                   positions=[[1.0, 1.0, 1.0],
                              [1.757, 1.586, 1.0],
                              [0.243, 1.586, 1.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create water molecule 2 at another position
    atoms2 = Atoms('OH2', 
                   positions=[[5.0, 5.0, 5.0],
                              [5.757, 5.586, 5.0],
                              [4.243, 5.586, 5.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create water molecule 3 at yet another position
    atoms3 = Atoms('OH2', 
                   positions=[[8.0, 2.0, 7.0],
                              [8.757, 2.586, 7.0],
                              [7.243, 2.586, 7.0]],
                   cell=lattice,
                   pbc=True)
    
    # Create crystal with multiple molecules
    crystal = MolecularCrystal(lattice, [atoms1, atoms2, atoms3])
    return crystal


def analyze_molecule_with_ase(cif_file_path, visualize=False):
    """
    Load a crystal structure from CIF and analyze molecular centers with ASE integration.
    
    Parameters
    ----------
    cif_file_path : str
        Path to the CIF file containing the crystal structure.
    visualize : bool
        Whether to visualize the structure using ASE GUI.
        
    Returns
    -------
    bool
        True if analysis completed successfully, False otherwise.
    """
    print("=" * 60)
    print("MolCrysKit: ASE-Integrated Molecular Center Analyzer")
    print("=" * 60)
    
    try:
        # Try to import required modules
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.structures.molecule import EnhancedMolecule
        
    except ImportError as e:
        print(f"Error importing MolCrysKit modules: {e}")
        print("Make sure you have installed the molcrys-kit package:")
        print("pip install -e .")
        return False
    
    try:
        # Parse the CIF file
        print(f"\nParsing CIF file: {cif_file_path}")
        crystal = read_mol_crystal(cif_file_path)
        print("✓ CIF file parsed successfully.")
        
    except FileNotFoundError:
        print(f"✗ Error: File '{cif_file_path}' not found.")
        return False
    except Exception as e:
        print(f"✗ Error parsing CIF file: {e}")
        return False
    
    # Analyze molecule centers
    print(f"\nAnalyzing molecular centers in {crystal.name or 'unknown'}...")
    print("Molecule Center Analysis")
    print("=" * 30)
    
    if not hasattr(crystal, 'molecules') or not crystal.molecules:
        print("No molecules found in the crystal structure.")
        return False
    
    print(f"Total number of molecules: {len(crystal.molecules)}")
    print()
    
    # Calculate and print the center of mass for each molecule
    for i, molecule in enumerate(crystal.molecules):
        # Get center of mass (considering atomic masses)
        center_of_mass = molecule.get_center_of_mass()
        
        # Get geometric center (simple average of positions)
        positions = molecule.get_positions()
        geometric_center = np.mean(positions, axis=0)
        
        # Get chemical symbols
        symbols = molecule.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        composition = ", ".join([f"{symbol}{symbols.count(symbol)}" for symbol in unique_symbols])
        
        print(f"Molecule {i+1}:")
        print(f"  Composition: {composition}")
        print(f"  Atoms: {len(molecule)} ({', '.join(symbols)})")
        print(f"  Center of mass (Cartesian): [{center_of_mass[0]:8.5f}, {center_of_mass[1]:8.5f}, {center_of_mass[2]:8.5f}]")
        print(f"  Geometric center (Cartesian): [{geometric_center[0]:8.5f}, {geometric_center[1]:8.5f}, {geometric_center[2]:8.5f}]")
        print()

    # Visualize if requested
    if visualize and ASE_AVAILABLE:
        try:
            print("Launching ASE visualization...")
            view(crystal.to_ase_atoms())
        except Exception as e:
            print(f"✗ Error during visualization: {e}")
    
    return True


def main():
    """Main function to run the molecule center analysis."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python molecule_center_analyzer_with_ase.py <cif_file_path> [--visualize]")
        print("Example: python molecule_center_analyzer_with_ase.py data/sample.cif --visualize")
        sys.exit(1)
    
    # Get command line arguments
    cif_file_path = sys.argv[1]
    visualize = "--visualize" in sys.argv or "-v" in sys.argv
    
    # Run the analysis
    success = analyze_molecule_with_ase(cif_file_path, visualize)
    
    if success:
        print("Analysis completed successfully.")
        sys.exit(0)
    else:
        print("Analysis failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()