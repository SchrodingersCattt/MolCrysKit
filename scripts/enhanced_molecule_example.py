#!/usr/bin/env python3
"""
Example showing how to use the CrystalMolecule class.

This script demonstrates the enhanced functionality of the CrystalMolecule class
compared to a plain ASE Atoms object.
"""

import numpy as np

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE is not available. Some functionality may be limited.")

from molcrys_kit.structures.molecule import CrystalMolecule


def main():
    """Run the enhanced molecule example."""
    if not ASE_AVAILABLE:
        print("This example requires ASE. Please install it with 'pip install ase'")
        return
    
    print("Enhanced CrystalMolecule Example")
    print("=" * 40)
    
    # Create a water molecule using ASE
    print("\n1. Creating a water molecule with ASE...")
    water = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],      # Oxygen
            [0.757, 0.586, 0.0],  # Hydrogen 1
            [-0.757, 0.586, 0.0]  # Hydrogen 2
        ]
    )
    print(f"Water molecule formula: {water.get_chemical_formula()}")
    print(f"Number of atoms: {len(water)}")
    
    # Wrap it in a CrystalMolecule
    print("\n2. Wrapping in a CrystalMolecule...")
    molecule_water = CrystalMolecule(water)
    print(f"CrystalMolecule formula: {molecule_water.get_chemical_formula()}")
    print(f"Number of atoms: {len(molecule_water)}")
    
    # Show enhanced properties
    print("\n3. Enhanced properties of CrystalMolecule:")
    
    # Centroid calculation
    centroid = molecule_water.get_centroid()
    print(f"Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    
    # Center of mass calculation
    com = molecule_water.get_center_of_mass()
    print(f"Center of mass: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
    
    # Ellipsoid radii
    radii = molecule_water.get_ellipsoid_radii()
    print(f"Ellipsoid radii: {radii[0]:.3f} × {radii[1]:.3f} × {radii[2]:.3f}")
    
    # Principal axes
    axes = molecule_water.get_principal_axes()
    print("Principal axes:")
    for i, axis in enumerate(axes):
        print(f"  Axis {i+1}: ({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})")
    
    # Graph representation
    print("\n4. Graph representation:")
    graph = molecule_water.graph
    print(f"Graph nodes: {list(graph.nodes(data='symbol'))}")
    print(f"Graph edges: {list(graph.edges(data='distance'))}")
    
    # Create another molecule - methane
    print("\n5. Creating methane molecule...")
    methane = Atoms(
        symbols=['C', 'H', 'H', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],       # Carbon
            [0.631, 0.631, 0.631], # Hydrogen 1
            [-0.631, -0.631, 0.631], # Hydrogen 2
            [-0.631, 0.631, -0.631], # Hydrogen 3
            [0.631, -0.631, -0.631]  # Hydrogen 4
        ]
    )
    
    molecule_methane = CrystalMolecule(methane)
    print(f"Methane formula: {molecule_methane.get_chemical_formula()}")
    
    # Methane properties
    centroid_ch4 = molecule_methane.get_centroid()
    print(f"Methane centroid: ({centroid_ch4[0]:.4f}, {centroid_ch4[1]:.4f}, {centroid_ch4[2]:.4f})")
    
    radii_ch4 = molecule_methane.get_ellipsoid_radii()
    print(f"Methane ellipsoid radii: {radii_ch4[0]:.3f} × {radii_ch4[1]:.3f} × {radii_ch4[2]:.3f}")


if __name__ == "__main__":
    main()
    print("\nExample completed successfully!")