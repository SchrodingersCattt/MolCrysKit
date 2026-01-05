#!/usr/bin/env python
"""
Visualize Disorder Exclusion Graphs with MOLECULAR UNWRAPPING.

Features:
1. Infer Chemical Bonds using PBC (Minimum Image Convention).
2. Unwrap/Recenter molecules: Shifts atom coordinates so molecules appear contiguous
   rather than fragmented by the unit cell box.
3. Draws thick chemical bonds (skeleton) + thin colored conflict lines.
"""

import sys
import os
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from molcrys_kit.io.cif import scan_cif_disorder
    from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
except ImportError:
    pass

# --- CONSTANTS ---
# Covalent Radii (Angstroms)
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, 
    "Cl": 1.02, "Br": 1.20, "I": 1.39, "S": 1.05, "P": 1.07,
    "Si": 1.11, "B": 0.84, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22
}

COLOR_MAP = {
    "logical_alternative": "#e74c3c",   # Red
    "symmetry_clash": "#8e44ad",       # Purple
    "conformer_competition": "#e74c3c", # Red
    "explicit": "#2ecc71",             # Green
    "geometric": "#3498db",            # Blue
    "valence": "#f39c12",              # Orange
    "valence_geometry": "#9b59b6",     # Dark Purple
}

BOND_COLOR = "#34495e"
BOND_TOLERANCE = 1.3  # Liberal tolerance to catch bonds

def get_element_radius(label):
    import re
    match = re.match(r"([A-Z][a-z]?)", label)
    if match:
        el = match.group(1)
        return COVALENT_RADII.get(el, 1.1)
    return 1.1

def get_projection_matrix(lattice_matrix):
    """Rotation matrix to align Lattice Vector A to X-axis and B to XY plane."""
    a_vec = lattice_matrix[0]
    b_vec = lattice_matrix[1]
    a_norm = a_vec / np.linalg.norm(a_vec)
    b_proj = b_vec - np.dot(b_vec, a_norm) * a_norm
    if np.linalg.norm(b_proj) < 1e-6:
        b_perp = np.array([0, 1, 0])
    else:
        b_perp = b_proj / np.linalg.norm(b_proj)
    new_z = np.cross(a_norm, b_perp)
    new_z = new_z / np.linalg.norm(new_z)
    return np.array([a_norm, b_perp, new_z])

def mic_displacement(frac_u, frac_v):
    """Calculate Minimum Image displacement vector in fractional coordinates."""
    diff = frac_u - frac_v
    # Round to nearest integer to find the closest image
    image_shift = np.round(diff)
    return diff - image_shift, image_shift

def infer_bonds_pbc(graph, lattice_matrix):
    """
    Detect chemical bonds considering Periodic Boundary Conditions.
    Returns: List of (u, v, shift_vector)
    """
    bonds = []
    nodes = list(graph.nodes(data=True))
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, data_u = nodes[i]
            v, data_v = nodes[j]
            
            if "frac_coord" not in data_u or "frac_coord" not in data_v:
                continue

            # Disorder Group Check (Basic Filter)
            g_u = data_u.get("disorder_group", 0)
            g_v = data_v.get("disorder_group", 0)
            # Valid bond if same group, or one connects to backbone (0)
            if g_u != 0 and g_v != 0 and g_u != g_v:
                continue

            # PBC Distance Calculation
            frac_diff, _ = mic_displacement(data_u["frac_coord"], data_v["frac_coord"])
            cart_dist = np.linalg.norm(np.dot(frac_diff, lattice_matrix))
            
            r_u = get_element_radius(data_u.get("label", ""))
            r_v = get_element_radius(data_v.get("label", ""))
            
            if cart_dist < (r_u + r_v) * BOND_TOLERANCE:
                bonds.append((u, v))
    return bonds

def unwrap_molecular_coordinates(graph, lattice_matrix):
    """
    Traverses the chemical bond network to 'unwrap' coordinates.
    This reconstructs the molecule visually so it is contiguous, 
    fixing atoms that are split across the PBC box edges.
    """
    # 1. Infer bonds to establish connectivity
    bonds = infer_bonds_pbc(graph, lattice_matrix)
    
    # 2. Build a temporary graph for traversal
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(graph.nodes(data=True))
    bond_graph.add_edges_from(bonds)
    
    # 3. Traversal and Unwrapping
    unwrapped_frac_coords = {}
    visited = set()
    
    components = list(nx.connected_components(bond_graph))
    
    for comp in components:
        # Pick a root node for this molecule/fragment
        root = list(comp)[0]
        if "frac_coord" not in graph.nodes[root]:
            continue
            
        unwrapped_frac_coords[root] = graph.nodes[root]["frac_coord"].copy()
        
        # BFS traversal
        queue = [root]
        visited.add(root)
        
        while queue:
            parent = queue.pop(0)
            parent_frac = unwrapped_frac_coords[parent]
            
            for neighbor in bond_graph.neighbors(parent):
                if neighbor not in visited:
                    if "frac_coord" not in graph.nodes[neighbor]:
                        continue
                        
                    # Get raw coords
                    neigh_raw_frac = graph.nodes[neighbor]["frac_coord"]
                    
                    # Find shift required to make neighbor close to parent
                    # diff_mic is the vector pointing from Parent -> Neighbor (shortest path)
                    diff_mic, image_shift = mic_displacement(neigh_raw_frac, parent_frac)
                    
                    # The unwrapped position is simply Parent + Shortest_Vector
                    # Note: mic_displacement(u, v) = u - v - shift
                    # So u_mic = v + (u - v - shift) -> No, simpler:
                    # We want: Neigh_New ~= Parent_Ref
                    # Neigh_New = Neigh_Raw - round(Neigh_Raw - Parent_Ref)
                    
                    shift = np.round(neigh_raw_frac - parent_frac)
                    unwrapped_frac = neigh_raw_frac - shift
                    
                    unwrapped_frac_coords[neighbor] = unwrapped_frac
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    # 4. Project to 2D Cartesian
    rot_matrix = get_projection_matrix(lattice_matrix)
    pos = {}
    
    # Calculate center of mass to center the plot
    if unwrapped_frac_coords:
        all_coords = np.array(list(unwrapped_frac_coords.values()))
        center_frac = np.mean(all_coords, axis=0)
    else:
        center_frac = np.array([0.5, 0.5, 0.5])

    for n in graph.nodes():
        if n in unwrapped_frac_coords:
            # Shift relative to center to keep numbers reasonable
            frac = unwrapped_frac_coords[n] - center_frac
            cart = np.dot(frac, lattice_matrix)
            aligned = np.dot(rot_matrix, cart)
            pos[n] = (aligned[0], aligned[1])
        else:
            # Fallback
            pos[n] = (np.random.rand(), np.random.rand())
            
    return pos, bonds

def plot_unwrapped_graph(graph, lattice_matrix, title, filename):
    plt.figure(figsize=(14, 12))
    
    # --- Step 1: Unwrap Coordinates ---
    # This is the critical fix for "connections across box"
    pos, valid_bonds = unwrap_molecular_coordinates(graph, lattice_matrix)
    
    # --- Step 2: Draw Chemical Bonds (Skeleton) ---
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=valid_bonds,
        edge_color=BOND_COLOR,
        width=3.0,      # Thick lines
        alpha=0.5,      # Semi-transparent
        label="chemical_bond"
    )

    # --- Step 3: Draw Nodes ---
    node_colors = []
    labels = {}
    node_sizes = []
    
    # Dynamic sizing
    base_size = 300 if len(graph.nodes) < 50 else 150
    
    for n in graph.nodes():
        group = graph.nodes[n].get("disorder_group", 0)
        # Clean labels (e.g. C1_part1 -> C1)
        lbl = str(graph.nodes[n].get("label", str(n)))
        labels[n] = lbl.split('_')[0] 
        node_sizes.append(base_size)

        if group == 0:
            node_colors.append("#ecf0f1") # Light Gray
        elif group == 1:
            node_colors.append("#ffffff") # White
        else:
            node_colors.append("#bdc3c7") # Gray

    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        edgecolors=BOND_COLOR,
        linewidths=1.5,
        node_size=node_sizes
    )
    
    # Optional: Draw text if not too crowded
    if len(graph.nodes) < 100:
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')

    # --- Step 4: Draw Conflicts (Logic) ---
    edges = graph.edges(data=True)
    
    for edge_type, color in COLOR_MAP.items():
        specific_edges = []
        for u, v, d in edges:
            if d.get("conflict_type") == edge_type:
                # Since we unwrapped, checking for artifacts is less critical,
                # but we should ensure we don't draw lines between conformers that are far apart visually
                # (though unwrapping usually fixes this for bonded parts).
                specific_edges.append((u, v))
        
        if specific_edges:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=specific_edges,
                edge_color=color,
                width=1.5,
                alpha=0.8,
                style="dashed" if edge_type == "logical_alternative" else "solid"
            )

    # Legend construction
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=BOND_COLOR, lw=3, label='Bond')]
    for et, col in COLOR_MAP.items():
        # Only add to legend if present in graph
        if any(d.get("conflict_type") == et for _, _, d in edges):
            legend_elements.append(Line2D([0], [0], color=col, lw=1.5, label=et))

    plt.title(title, fontsize=16)
    plt.axis('equal')
    plt.axis('off') # Hide axes as coordinates are relative/unwrapped
    plt.legend(handles=legend_elements, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    input_files = glob.glob("examples/*.cif")
    
    output_dir = Path("output/graph_viz_unwrapped")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cif_file in input_files:
        if not os.path.exists(cif_file): continue
        print(f"Processing {cif_file}...")
        
        from pymatgen.io.cif import CifParser
        parser = CifParser(cif_file)
        structure = parser.parse_structures()[0]
        lattice = structure.lattice.matrix
        info = scan_cif_disorder(cif_file)
        
        builder = DisorderGraphBuilder(info, lattice)
        # ... (Run your builder pipeline steps here: identify_conformers, conflicts, etc.)
        builder._identify_conformers()
        builder._add_explicit_conflicts()
        builder._resolve_valence_conflicts()

        plot_unwrapped_graph(
            builder.graph, lattice, 
            f"{Path(cif_file).stem} - Unwrapped Molecular View", 
            output_dir / f"{Path(cif_file).stem}_unwrapped.png"
        )

if __name__ == "__main__":
    main()