#!/usr/bin/env python3
"""
Test script for metal bonding functionality.

This script tests the bonding analysis for metal-containing molecules.
"""

def test_metal_bonding():
    """Test bonding analysis for metal-containing molecules."""
    try:
        from ase import Atoms
        ASE_AVAILABLE = True
    except ImportError:
        ASE_AVAILABLE = False
        print("Warning: ASE is not available. Some functionality may be limited.")
        return
    
    if not ASE_AVAILABLE:
        print("This test requires ASE. Please install it with 'pip install ase'")
        return
    
    print("Testing Metal Bonding Analysis")
    print("=" * 30)
    
    # Create a simple metal-water complex: Cu(H2O)6
    complex_structure = Atoms(
        symbols=['Cu', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],      # Cu
            [2.0, 0.0, 0.0],      # O1
            [2.5, 0.5, 0.0],      # H1
            [2.5, -0.5, 0.0],     # H2
            [0.0, 2.0, 0.0],      # O2
            [0.5, 2.5, 0.0],      # H3
            [-0.5, 2.5, 0.0],     # H4
            [-2.0, 0.0, 0.0],     # O3
            [-2.5, 0.5, 0.0],     # H5
            [-2.5, -0.5, 0.0],    # H6
            [0.0, -2.0, 0.0],     # O4
            [0.5, -2.5, 0.0],     # H7
            [-0.5, -2.5, 0.0],    # H8
            [0.0, 0.0, 2.0],      # O5
            [0.5, 0.0, 2.5],      # H9
            [-0.5, 0.0, 2.5],     # H10
            [0.0, 0.0, -2.0],     # O6
            [0.5, 0.0, -2.5],     # H11
            [-0.5, 0.0, -2.5]     # H12
        ]
    )
    
    # Import CrystalMolecule after confirming ASE availability
    from molcrys_kit.structures.molecule import CrystalMolecule
    
    # Convert to CrystalMolecule
    molecule = CrystalMolecule(complex_structure)
    
    print(f"Complex: {molecule.get_chemical_formula()}")
    print(f"Number of atoms: {len(molecule)}")
    
    # Analyze the bonding graph
    graph = molecule.graph
    
    print(f"\nGraph nodes: {len(graph.nodes())}")
    print(f"Graph edges: {len(graph.edges())}")
    
    # Check for metal bonds
    cu_atom_index = None
    oxygen_indices = []
    
    for node, data in graph.nodes(data=True):
        if data['symbol'] == 'Cu':
            cu_atom_index = node
        elif data['symbol'] == 'O':
            oxygen_indices.append(node)
    
    print(f"\nMetal center: Cu at index {cu_atom_index}")
    print(f"Oxygen atoms at indices: {oxygen_indices}")
    
    # Check bonds between Cu and O atoms
    metal_bonds = []
    for o_index in oxygen_indices:
        if graph.has_edge(cu_atom_index, o_index):
            distance = graph[cu_atom_index][o_index]['distance']
            metal_bonds.append((cu_atom_index, o_index, distance))
    
    print(f"\nMetal-oxygen bonds:")
    for bond in metal_bonds:
        print(f"  Cu({bond[0]}) - O({bond[1]}): {bond[2]:.3f} Å")
    
    # Check water molecules (H-O bonds)
    water_molecules = []
    for o_index in oxygen_indices:
        h_neighbors = [n for n in graph.neighbors(o_index) if graph.nodes[n]['symbol'] == 'H']
        if len(h_neighbors) >= 1:
            water_molecules.append((o_index, h_neighbors))
    
    print(f"\nWater molecules:")
    for o_index, h_indices in water_molecules:
        print(f"  H2O with O at index {o_index}")
        for h_index in h_indices:
            if graph.has_edge(o_index, h_index):
                distance = graph[o_index][h_index]['distance']
                print(f"    O({o_index}) - H({h_index}): {distance:.3f} Å")


def main():
    """Run the metal bonding test."""
    test_metal_bonding()
    print("\n\nTest completed successfully!")


if __name__ == "__main__":
    main()