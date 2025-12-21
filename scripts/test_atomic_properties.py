#!/usr/bin/env python3
"""
Test atomic properties functionality.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_center_of_mass():
    """Test center of mass calculation with real atomic masses."""
    print("Testing Center of Mass Calculation")
    print("=" * 40)
    
    try:
        from ase import Atoms
        from molcrys_kit.structures.molecule import Molecule
        from molcrys_kit.constants import get_atomic_mass
        
        # Create a water molecule using ASE
        atoms = Atoms(
            symbols=['O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],          # Oxygen
                [0.0757, 0.0586, 0.0],    # Hydrogen
                [-0.0757, 0.0586, 0.0]    # Hydrogen
            ]
        )
        
        water = Molecule(atoms)
        
        print("Water molecule:")
        symbols = water.get_chemical_symbols()
        for symbol in symbols:
            mass = get_atomic_mass(symbol)
            print(f"  {symbol}: mass = {mass:.3f} amu")
        
        com = water.get_center_of_mass()
        print(f"\nCenter of mass: [{com[0]:8.5f}, {com[1]:8.5f}, {com[2]:8.5f}]")
        
        # Compare with simple geometric center
        geometric_center = water.get_centroid()
        print(f"Geometric center: [{geometric_center[0]:8.5f}, {geometric_center[1]:8.5f}, {geometric_center[2]:8.5f}]")
        
        print("\n✓ Center of mass calculation completed\n")
        
    except ImportError as e:
        print(f"Skipping test due to missing dependencies: {e}\n")


def test_molecular_graph():
    """Test molecular graph representation."""
    print("Testing Molecular Graph Representation")
    print("=" * 40)
    
    try:
        from ase import Atoms
        from molcrys_kit.structures.molecule import Molecule
        
        # Create a water molecule using ASE
        atoms = Atoms(
            symbols=['O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],          # Oxygen
                [0.0757, 0.0586, 0.0],    # Hydrogen
                [-0.0757, 0.0586, 0.0]    # Hydrogen
            ]
        )
        
        water = Molecule(atoms)
        graph = water.graph
        
        print("Water molecule graph:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        for node, data in graph.nodes(data=True):
            print(f"  Node {node}: {data['symbol']}")
            
        for u, v, data in graph.edges(data=True):
            print(f"  Edge {u}-{v}: distance = {data['distance']:.3f}")
        
        print("\n✓ Molecular graph representation completed\n")
        
    except ImportError as e:
        print(f"Skipping test due to missing dependencies: {e}\n")


def main():
    """Main function to run all tests."""
    print("MolCrysKit Atomic Properties Test")
    print("================================\n")
    
    try:
        test_center_of_mass()
        test_molecular_graph()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        print("Make sure you have installed the molcrys-kit package:")
        print("pip install -e .")
        return 1


if __name__ == "__main__":
    main()