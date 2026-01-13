import numpy as np
import networkx as nx
import sys
import os

# Ensure we can import molcrys_kit from current directory
sys.path.insert(0, os.getcwd())

from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.constants.config import MAX_COORDINATION_NUMBERS, DEFAULT_MAX_COORDINATION, TRANSITION_METALS

def calculate_effective_coordination(builder, neighbors):
    """
    Simulate the proposed fix: Calculate MIS size of the neighbor subgraph.
    MIS(G) = MaxClique(Complement(G))
    """
    if not neighbors:
        return 0
    
    nb_graph = nx.Graph()
    nb_graph.add_nodes_from(neighbors)
    
    neighbor_list = list(neighbors)
    for i in range(len(neighbor_list)):
        u = neighbor_list[i]
        for j in range(i + 1, len(neighbor_list)):
            v = neighbor_list[j]
            
            # Conflict Check 1: Different Disorder Groups (Logic)
            g_u = builder.info.disorder_groups[u]
            g_v = builder.info.disorder_groups[v]
            if g_u != 0 and g_v != 0 and g_u != g_v:
                nb_graph.add_edge(u, v)
                continue
            
            # Conflict Check 2: Geometry (Too close)
            # Threshold 1.5 A is typical for "too close for non-bonded neighbors"
            if builder.dist_matrix[u, v] < 1.5: 
                nb_graph.add_edge(u, v)

    # --- FIX: Use Complement Graph + Max Clique ---
    # The Maximum Independent Set of G is the Maximum Clique of the Complement of G.
    complement_graph = nx.complement(nb_graph)
    
    # nx.max_weight_clique returns (nodes, total_weight)
    # With weight=None, it counts nodes (Maximum Clique)
    clique_nodes, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight=None)
    
    return len(clique_nodes)

def diagnose_file(cif_path, target_element):
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {cif_path}")
    print(f"{'='*60}")

    try:
        info = scan_cif_disorder(cif_path)
    except Exception as e:
        print(f"Error parsing CIF: {e}")
        return

    # Mock Lattice (Orthogonal for simplicity)
    a, b, c = 15.0, 15.0, 15.0 
    lattice = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])

    builder = DisorderGraphBuilder(info, lattice)
    n_atoms = len(info.labels)
    
    # Build simple connectivity
    connectivity_graph = nx.Graph()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = builder.dist_matrix[i, j]
            if dist < 2.5: 
                connectivity_graph.add_edge(i, j)

    # Check for the Bug
    for i in range(n_atoms):
        sym = info.symbols[i]
        if sym != target_element:
            continue
            
        neighbors = list(connectivity_graph.neighbors(i))
        max_c = MAX_COORDINATION_NUMBERS.get(sym, DEFAULT_MAX_COORDINATION)
        
        # --- THE BUGGY LOGIC ---
        current_count = len(neighbors)
        is_bug_triggered = current_count > max_c
        
        # --- THE CORRECT LOGIC ---
        effective_count = calculate_effective_coordination(builder, neighbors)
        is_physically_valid = effective_count <= max_c

        print(f"Atom: {info.labels[i]} ({sym})")
        print(f"  > Neighbors Found (Total): {current_count}")
        # Print neighbor details for debugging
        nb_details = [f"{info.labels[n]}(Part{info.disorder_groups[n]})" for n in neighbors]
        print(f"    {nb_details}")
        print(f"  > Max Allowed Coordination: {max_c}")
        
        print(f"  [CURRENT CODE JUDGMENT]")
        if is_bug_triggered:
            print(f"  🔴 VIOLATION! {current_count} > {max_c} -> DELETE ATOMS")
        else:
            print(f"  🟢 OK.")

        print(f"  [REALITY CHECK (MIS)]")
        print(f"  > Effective Coexisting Neighbors: {effective_count}")
        if is_physically_valid:
            print(f"  🟢 PHYSICALLY VALID ({effective_count} <= {max_c}) -> SHOULD KEEP")
        else:
            print(f"  🔴 REAL OVERCROWDING")
        
        print("-" * 40)

if __name__ == "__main__":
    # Test ZIF-4 (Target N)
    diagnose_file("examples/ZIF-4.cif", "N")

    # Test ZIF-8 (Target Zn)
    diagnose_file("examples/ZIF-8.cif", "Zn")