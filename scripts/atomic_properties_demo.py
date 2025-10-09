#!/usr/bin/env python3
"""
Demo script showing how to use atomic properties (masses and radii) in MolCrysKit with ASE Atoms.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not available. Install with 'pip install ase' for full functionality.")

try:
    # Import molcrys modules
    from molcrys.structures import MolecularCrystal
    from molcrys.constants import (
        get_atomic_mass,
        get_atomic_radius,
        has_atomic_mass,
        has_atomic_radius,
        list_elements_with_data
    )
except ImportError as e:
    print(f"Error importing MolCrysKit: {e}")
    print("Please ensure the package is installed with 'pip install -e .'")
    sys.exit(1)


def demo_atomic_properties():
    """Demonstrate accessing atomic properties."""
    print("Atomic Properties Demo")
    print("=" * 30)
    
    # Test elements
    elements = ["H", "C", "O", "N", "S", "Fe", "Cu", "U"]
    
    print("Element Properties:")
    print("-" * 40)
    print(f"{'Element':<6} {'Mass (amu)':<12} {'Radius (Å)':<12}")
    print("-" * 40)
    
    for element in elements:
        mass = get_atomic_mass(element) if has_atomic_mass(element) else "N/A"
        radius = get_atomic_radius(element) if has_atomic_radius(element) else "N/A"
        print(f"{element:<6} {mass:<12} {radius:<12}")
    
    print()


def demo_molecule_with_real_masses():
    """Demonstrate molecule center of mass calculation with real atomic masses."""
    print("Molecule Center of Mass Demo")
    print("=" * 30)
    
    if not ASE_AVAILABLE:
        print("This demo requires ASE. Please install it with 'pip install ase'")
        return
    
    # Create a water molecule
    water = Atoms('OH2', 
                  positions=[[0.0, 0.0, 0.0],
                            [0.757, 0.586, 0.0],
                            [-0.757, 0.586, 0.0]])
    
    print("Water molecule composition:")
    symbols = water.get_chemical_symbols()
    for symbol in symbols:
        mass = get_atomic_mass(symbol)
        radius = get_atomic_radius(symbol)
        print(f"  {symbol}: mass = {mass:.3f} amu, radius = {radius:.3f} Å")
    
    print()
    print("Atomic coordinates (Cartesian):")
    positions = water.get_positions()
    for i, (symbol, pos) in enumerate(zip(symbols, positions)):
        print(f"  {symbol}: [{pos[0]:8.5f}, {pos[1]:8.5f}, {pos[2]:8.5f}]")
    
    print()
    com = water.get_center_of_mass()
    print(f"Center of mass: [{com[0]:8.5f}, {com[1]:8.5f}, {com[2]:8.5f}]")
    
    # Compare with simple geometric center
    geometric_center = np.mean(positions, axis=0)
    print(f"Geometric center: [{geometric_center[0]:8.5f}, {geometric_center[1]:8.5f}, {geometric_center[2]:8.5f}]")
    
    print("\nNote: The center of mass is different from the geometric center")
    print("      because it takes into account the different atomic masses.")



def demo_data_availability():
    """Demonstrate checking data availability."""
    print("\nData Availability Demo")
    print("=" * 30)
    
    data_info = list_elements_with_data()
    print(f"Elements with mass data: {len(data_info['masses'])}")
    print(f"Elements with radius data: {len(data_info['radii'])}")
    
    # Show some examples
    print("\nFirst 10 elements with mass data:", data_info['masses'][:10])
    print("First 10 elements with radius data:", data_info['radii'][:10])


def main():
    """Main function to run all demos."""
    demo_atomic_properties()
    demo_molecule_with_real_masses()
    demo_data_availability()


if __name__ == "__main__":
    main()