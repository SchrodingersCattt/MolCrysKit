#!/usr/bin/env python3
"""
Example showing how to use the Molecule class.
"""

from ase import Atoms

from molcrys_kit.structures.molecule import Molecule


def main():
    # Create a water molecule as an ASE Atoms object
    water = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.756950, 0.585809, 0.000000],
            [-0.756950, 0.585809, 0.000000]
        ]
    )
    
    # Wrap it in a Molecule
    molecule_water = Molecule(water)
    
    # Demonstrate the new functionality
    print("Water molecule properties:")
    print(f"Chemical formula: {water.get_chemical_formula()}")
    print(f"Number of atoms: {len(water)}")
    
    # Get centroid and center of mass
    centroid = molecule_water.get_centroid()
    center_of_mass = molecule_water.get_center_of_mass()
    print(f"Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    print(f"Center of mass: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]")
    
    # Get ellipsoid radii
    radii = molecule_water.get_ellipsoid_radii()
    print(f"Ellipsoid radii: a={radii[0]:.6f}, b={radii[1]:.6f}, c={radii[2]:.6f}")
    
    # Get principal axes
    ax1, ax2, ax3 = molecule_water.get_principal_axes()
    print(f"Principal axes:")
    print(f"  Axis 1: [{ax1[0]:.6f}, {ax1[1]:.6f}, {ax1[2]:.6f}]")
    print(f"  Axis 2: [{ax2[0]:.6f}, {ax2[1]:.6f}, {ax2[2]:.6f}]")
    print(f"  Axis 3: [{ax3[0]:.6f}, {ax3[1]:.6f}, {ax3[2]:.6f}]")
    
    # Show graph functionality
    print(f"Molecular graph has {len(molecule_water.graph.nodes())} nodes and {len(molecule_water.graph.edges())} edges")
    
    print("\n" + "="*50 + "\n")
    
    # Create a more complex molecule - methane
    methane = Atoms(
        symbols=['C', 'H', 'H', 'H', 'H'],
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [0.629118, -0.629118, -0.629118],
            [-0.629118, 0.629118, -0.629118]
        ]
    )
    
    molecule_methane = Molecule(methane)
    
    print("Methane molecule properties:")
    print(f"Chemical formula: {methane.get_chemical_formula()}")
    print(f"Number of atoms: {len(methane)}")
    
    # Get centroid and center of mass
    centroid = molecule_methane.get_centroid()
    center_of_mass = molecule_methane.get_center_of_mass()
    print(f"Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    print(f"Center of mass: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]")
    
    # Get ellipsoid radii
    radii = molecule_methane.get_ellipsoid_radii()
    print(f"Ellipsoid radii: a={radii[0]:.6f}, b={radii[1]:.6f}, c={radii[2]:.6f}")
    
    # Get principal axes
    ax1, ax2, ax3 = molecule_methane.get_principal_axes()
    print(f"Principal axes:")
    print(f"  Axis 1: [{ax1[0]:.6f}, {ax1[1]:.6f}, {ax1[2]:.6f}]")
    print(f"  Axis 2: [{ax2[0]:.6f}, {ax2[1]:.6f}, {ax2[2]:.6f}]")
    print(f"  Axis 3: [{ax3[0]:.6f}, {ax3[1]:.6f}, {ax3[2]:.6f}]")
    
    # Show graph functionality
    print(f"Molecular graph has {len(molecule_methane.graph.nodes())} nodes and {len(molecule_methane.graph.edges())} edges")


if __name__ == "__main__":
    main()