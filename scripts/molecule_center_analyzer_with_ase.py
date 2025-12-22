#!/usr/bin/env python3
"""
Molecule center analyzer with ASE integration.

This script analyzes molecular centers in a crystal structure using ASE for visualization.
"""

# Check if ASE is available
try:
    from ase import Atoms
    from ase.visualize import view
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE is not available. Visualization functionality will be limited.")

import numpy as np

# Conditional import to handle environments where ASE is not available
def create_test_molecules():
    """Create test molecules for analysis."""
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for this example. Please install it with 'pip install ase'")
    
    # Create water molecules at different positions
    molecules = []
    
    # Water molecule 1
    mol1 = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [1.0, 1.0, 1.0],
            [1.757, 1.586, 1.0],
            [0.243, 1.586, 1.0]
        ]
    )
    molecules.append(mol1)
    
    # Water molecule 2
    mol2 = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [5.0, 5.0, 5.0],
            [5.757, 5.586, 5.0],
            [4.243, 5.586, 5.0]
        ]
    )
    molecules.append(mol2)
    
    # Water molecule 3
    mol3 = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [9.0, 1.0, 9.0],
            [9.757, 1.586, 9.0],
            [8.243, 1.586, 9.0]
        ]
    )
    molecules.append(mol3)
    
    return molecules


def analyze_molecule_centers(molecules):
    """Analyze centers of molecules and visualize if possible."""
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for this analysis. Please install it with 'pip install ase'")
    
    # Import CrystalMolecule after confirming ASE availability
    from molcrys_kit.structures.molecule import CrystalMolecule
    
    print("Molecule Center Analysis")
    print("=" * 25)
    
    # Convert to CrystalMolecule objects and analyze
    crystal_molecules = [CrystalMolecule(mol) for mol in molecules]
    
    # Collect all atoms for visualization
    all_atoms = []
    
    for i, molecule in enumerate(crystal_molecules):
        print(f"\nMolecule {i+1}: {molecule.get_chemical_formula()}")
        
        # Calculate geometric center (centroid)
        centroid = molecule.get_centroid()
        print(f"  Geometric center: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
        
        # Calculate center of mass
        center_of_mass = molecule.get_center_of_mass()
        print(f"  Center of mass: ({center_of_mass[0]:.4f}, {center_of_mass[1]:.4f}, {center_of_mass[2]:.4f})")
        
        # Add atoms to collection for visualization
        all_atoms.extend(molecule)
    
    # Create a combined ASE Atoms object for visualization
    combined_system = Atoms(
        symbols=[atom.symbol for atom in all_atoms],
        positions=[atom.position for atom in all_atoms]
    )
    
    return combined_system


def main():
    """Main function to run the analysis."""
    if not ASE_AVAILABLE:
        print("This example requires ASE. Please install it with 'pip install ase'")
        return
    
    try:
        # Create test molecules
        molecules = create_test_molecules()
        
        # Analyze molecule centers
        combined_system = analyze_molecule_centers(molecules)
        
        # Visualize (uncomment the next line to enable visualization)
        # view(combined_system, viewer='x3d')
        
        print(f"\nTotal atoms in system: {len(combined_system)}")
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
