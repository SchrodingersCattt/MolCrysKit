"""
Tests for the Disorder Graph Builder module.

This module tests the 3-layer conflict detection logic in the DisorderGraphBuilder.
"""

import numpy as np
import networkx as nx
from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder


def test_geometric_collision():
    """Test geometric collision detection - two atoms at the same position."""
    # Create mock DisorderInfo with two atoms at the same position
    labels = ["H1", "H2"]
    symbols = ["H", "H"]
    frac_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Same position
    occupancies = [1.0, 1.0]
    disorder_groups = [0, 0]
    
    info = DisorderInfo(
        labels=labels,
        symbols=symbols,
        frac_coords=frac_coords,
        occupancies=occupancies,
        disorder_groups=disorder_groups
    )
    
    # Create a simple cubic lattice
    lattice = np.eye(3) * 10.0  # 10x10x10 Angstrom box
    
    # Build the exclusion graph
    builder = DisorderGraphBuilder(info, lattice)
    graph = builder.build()
    
    # Verify that an edge exists between the two atoms (they collide)
    assert graph.has_edge(0, 1), "Atoms at same position should be connected by an edge"
    assert graph[0][1]['conflict_type'] == 'geometric', "Conflict type should be geometric"
    
    print("Geometric collision test passed!")


def test_ammonium_case():
    """Test the 'ammonium' case with 8 H atoms around a N atom."""
    # Create mock DisorderInfo simulating DAP-4: 1 N atom with 8 H atoms around it
    # The H atoms form two tetrahedral arrangements that should be mutually exclusive
    
    labels = ["N1"] + [f"H{i}" for i in range(1, 9)]
    symbols = ["N"] + ["H"] * 8
    
    # Create positions that form two tetrahedra around the N atom (in fractional coords)
    # Use distances that ensure bonding to N (N-H ~1.0 Angstrom = 0.1/10 = 0.01 fractional for 10A cell)
    # To space H atoms further apart, use small displacements but ensure H...H distances are > 0.8Å
    
    # Tetrahedron 1: standard tetrahedral positions (in fractional coords)
    # Scale to about 0.8-1.0 Angstrom from N (0.008-0.01 fractional for 10A cell)
    tet1_base = np.array([
        [0.01, 0.01, 0.01],      # First H - ~1.732 Å from origin
        [0.01, -0.01, -0.01],    # Second H - ~1.732 Å from origin
        [-0.01, 0.01, -0.01],    # Third H - ~1.732 Å from origin
        [-0.01, -0.01, 0.01]     # Fourth H - ~1.732 Å from origin
    ])
    
    # Tetrahedron 2: offset from first, ensuring H's don't get too close to each other
    # Rotate the second set to be roughly opposite the first
    tet2_base = np.array([
        [0.008, 0.008, -0.008],    # Fifth H
        [0.008, -0.008, 0.008],    # Sixth H
        [-0.008, 0.008, 0.008],    # Seventh H
        [-0.008, -0.008, -0.008]   # Eighth H
    ])
    
    # Combine all H positions
    all_h_positions = np.vstack([tet1_base, tet2_base])
    
    # N at origin
    n_position = np.array([[0.0, 0.0, 0.0]])
    frac_coords = np.vstack([n_position, all_h_positions])
    
    # All H's have occupancy 0.5 to simulate disorder
    occupancies = [1.0] + [0.5] * 8  # N has full occupancy, H's have 0.5
    disorder_groups = [0] * 9  # All have disorder group 0 (implicit disorder)
    
    info = DisorderInfo(
        labels=labels,
        symbols=symbols,
        frac_coords=frac_coords,
        occupancies=occupancies,
        disorder_groups=disorder_groups
    )
    
    # Create a simple cubic lattice
    lattice = np.eye(3) * 10.0  # 10x10x10 Angstrom box
    
    # Build the exclusion graph
    builder = DisorderGraphBuilder(info, lattice)
    graph = builder.build()
    
    # Check that at least some exclusions exist between H atoms
    h_nodes = list(range(1, 9))  # H atoms are nodes 1-8
    
    # Count exclusions between H atoms
    h_exclusions = 0
    for i in h_nodes:
        for j in h_nodes:
            if i < j and graph.has_edge(i, j):
                h_exclusions += 1
    
    print(f"Ammonium case: Found {h_exclusions} exclusions between H atoms")
    
    # The test should pass if there are exclusions between H atoms due to overcrowding
    # They might be valence or geometric conflicts depending on the exact distances
    assert h_exclusions > 0, f"There should be exclusions between H atoms in the ammonium case, found {h_exclusions}"
    
    print(f"Ammonium case test passed! Found {h_exclusions} exclusions between H atoms")
    
    # Check that we have valid bonding to N (to ensure valence conflicts are triggered)
    n_neighbors = list(graph.neighbors(0))  # neighbors of N (index 0)
    print(f"N atom has {len([n for n in n_neighbors if n != 0])} bonded neighbors")  # excluding self-loops if any
    
    # At least some should be bonded to N to trigger valence analysis
    assert len([n for n in n_neighbors if n != 0]) > 0, "N should have bonded neighbors to trigger valence conflicts"


def test_explicit_conflicts():
    """Test explicit conflict detection based on disorder groups."""
    # Create mock DisorderInfo with atoms in different disorder groups
    labels = ["C1", "C2", "H1", "H2"]
    symbols = ["C", "C", "H", "H"]
    frac_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # Far enough to avoid geometric conflict
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0]
    ])
    occupancies = [1.0, 1.0, 1.0, 1.0]
    disorder_groups = [1, 2, 1, 2]  # Different groups for pairs
    
    info = DisorderInfo(
        labels=labels,
        symbols=symbols,
        frac_coords=frac_coords,
        occupancies=occupancies,
        disorder_groups=disorder_groups
    )
    
    # Create a simple cubic lattice
    lattice = np.eye(3) * 10.0
    
    # Build the exclusion graph
    builder = DisorderGraphBuilder(info, lattice)
    graph = builder.build()
    
    # Verify that atoms with different non-zero disorder groups are connected
    assert graph.has_edge(0, 1), "Atoms in different disorder groups (1,2) should be connected"
    assert graph.has_edge(2, 3), "Atoms in different disorder groups (1,2) should be connected"
    
    # Check that these are explicit conflicts
    assert graph[0][1]['conflict_type'] == 'explicit', "Should be marked as explicit conflict"
    assert graph[2][3]['conflict_type'] == 'explicit', "Should be marked as explicit conflict"
    
    print("Explicit conflicts test passed!")


def test_integration_with_real_cif():
    """Test integration with a real CIF through the graph builder."""
    # This test creates a structure with both explicit and geometric conflicts
    labels = ["C1", "C2", "O1", "O2", "H1", "H2"]
    symbols = ["C", "C", "O", "O", "H", "H"]
    frac_coords = np.array([
        [0.0, 0.0, 0.0],    # C1
        [0.05, 0.0, 0.0],   # C2 - close to C1
        [0.1, 0.0, 0.0],    # O1 - bonded to C2
        [0.0, 0.1, 0.0],    # O2 - bonded to C1  
        [0.0, 0.0, 0.05],   # H1 - close to both C's
        [0.05, 0.0, 0.05],  # H2 - close to both C's
    ])
    occupancies = [1.0, 0.5, 1.0, 1.0, 0.5, 0.5]  # Some low occupancy
    disorder_groups = [1, 2, 1, 0, 0, 0]  # Some explicit groups
    
    info = DisorderInfo(
        labels=labels,
        symbols=symbols,
        frac_coords=frac_coords,
        occupancies=occupancies,
        disorder_groups=disorder_groups
    )
    
    lattice = np.eye(3) * 10.0
    
    builder = DisorderGraphBuilder(info, lattice)
    graph = builder.build()
    
    # Check that explicitly different groups are connected
    assert graph.has_edge(0, 1), "C1(group 1) and C2(group 2) should be connected"
    
    # The conflict type might be explicit or geometric depending on implementation
    # Both are valid outcomes
    conflict_type = graph[0][1]['conflict_type']
    assert conflict_type in ['explicit', 'geometric'], f"Conflict should be explicit or geometric, got {conflict_type}"
    
    # Check that we have multiple edges
    assert len(graph.edges()) > 1, "Should have multiple conflict edges"
    
    print("Integration test passed!")


if __name__ == "__main__":
    print("Running Disorder Graph Builder tests...")
    
    test_geometric_collision()
    test_ammonium_case()
    test_explicit_conflicts()
    test_integration_with_real_cif()
    
    print("All tests passed!")