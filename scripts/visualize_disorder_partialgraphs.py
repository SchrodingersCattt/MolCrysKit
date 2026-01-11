#!/usr/bin/env python
"""
Visualize Disorder Exclusion Graphs with MOLECULAR UNWRAPPING - Focused View.

Features:
1. Infer Chemical Bonds using PBC (Minimum Image Convention).
2. Unwrap/Recenter molecules: Shifts atom coordinates so molecules appear contiguous.
3. Focus Visualization: Only draws molecules/clusters that contain conflict nodes.
   Filters out irrelevant background molecules to highlight the disorder mechanism.
"""

import sys
import os
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import re
from time import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from molcrys_kit.io.cif import scan_cif_disorder
    from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
except ImportError:
    print("Warning: Could not import from molcrys_kit")

# --- CONSTANTS ---
BOND_COLOR = "#34495e"
BOND_TOLERANCE = 1.3
NODE_COLOR_CYCLE = ["#ecf0f1", "#b2c55f", "#d49b68", "#6fb1dd", "#c981e6", "#f1c40f", "#e67e22", "#78afe7", "#d49b68", "#b2c55f"]

COLOR_MAP = {
    "logical_alternative": "#a17570",   # Red
    "symmetry_clash": "#75658f",       # Purple
    "explicit": "#63aa81",             # Green
    "geometric": "#5c849e",            # Blue
    "valence": "#cfaf7a",              # Orange
    "valence_geometry": "#BD729B",     # Dark Purple
}

VISUAL_CUTOFF = 3

def get_element_radius(label):
    match = re.match(r"([A-Z][a-z]?)", label)
    if match:
        el = match.group(1)
        try:
            from molcrys_kit.constants.config import BONDING_CONFIG
            import json
            with open(os.path.join(os.path.dirname(__file__), '..', 'molcrys_kit', 'constants', 'atomic_radii.json')) as f:
                radii_data = json.load(f)
            return radii_data.get(el, BONDING_CONFIG.get("DEFAULT_ATOMIC_RADIUS", 1.1))
        except (ImportError, FileNotFoundError):
            COVALENT_RADII = {
                "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, 
                "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, 
                "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, 
                "K": 2.03, "Ca": 1.76, "Sc": 1.90, "Ti": 1.75, "V": 1.64, "Cr": 1.54, 
                "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, 
                "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, 
                "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, 
                "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, 
                "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40
            }
            return COVALENT_RADII.get(el, 1.1)
    return 1.1

def cell_parameters_to_matrix(a, b, c, alpha, beta, gamma):
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    v_volume = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    if v_volume < 0:
        raise ValueError("Invalid cell parameters")
    
    val = np.sqrt(v_volume)

    matrix = np.zeros((3, 3))
    matrix[0, :] = [a, 0.0, 0.0]
    matrix[1, :] = [b * cos_gamma, b * np.sin(gamma_rad), 0.0]
    matrix[2, 0] = c * cos_beta
    matrix[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / np.sin(gamma_rad)
    matrix[2, 2] = c * val / np.sin(gamma_rad)
    
    return matrix

def parse_lattice_from_cif(cif_path):
    params = {}
    with open(cif_path, 'r') as f:
        for line in f:
            line = line.strip().split('#')[0]
            if not line: continue
            
            parts = line.split()
            if len(parts) >= 2:
                key, val = parts[0], parts[1]
                val = re.sub(r'\([^)]*\)', '', val)
                
                if '_cell_length_a' in key: params['a'] = float(val)
                elif '_cell_length_b' in key: params['b'] = float(val)
                elif '_cell_length_c' in key: params['c'] = float(val)
                elif '_cell_angle_alpha' in key: params['alpha'] = float(val)
                elif '_cell_angle_beta' in key: params['beta'] = float(val)
                elif '_cell_angle_gamma' in key: params['gamma'] = float(val)
    
    if len(params) < 6:
        raise ValueError(f"Could not parse lattice from {cif_path}")
        
    return cell_parameters_to_matrix(
        params['a'], params['b'], params['c'],
        params['alpha'], params['beta'], params['gamma']
    )

def get_projection_matrix(lattice_matrix, view_axis='a'):
    a_vec, b_vec, c_vec = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]

    if view_axis == 'c':
        u_vec, v_vec = a_vec, b_vec
    elif view_axis == 'b':
        u_vec, v_vec = a_vec, c_vec
    elif view_axis == 'a':
        u_vec, v_vec = b_vec, c_vec
    else:
        raise ValueError("view_axis must be 'a', 'b', or 'c'")

    u_norm = u_vec / np.linalg.norm(u_vec)
    v_proj = v_vec - np.dot(v_vec, u_norm) * u_norm
    if np.linalg.norm(v_proj) < 1e-6:
        v_perp = np.array([0, 1, 0])
    else:
        v_perp = v_proj / np.linalg.norm(v_proj)
        
    w_norm = np.cross(u_norm, v_perp)
    w_norm = w_norm / np.linalg.norm(w_norm)
    
    return np.array([u_norm, v_perp, w_norm])

def infer_bonds_pbc(graph, lattice_matrix):
    nodes = list(graph.nodes(data=True))
    n_nodes = len(nodes)
    if n_nodes == 0: return []

    node_indices = [n for n, _ in nodes]
    frac_coords = np.array([d.get("frac_coord", [0., 0., 0.]) for _, d in nodes])
    disorder_groups = np.array([d.get("disorder_group", 0) for _, d in nodes])
    radii = np.array([get_element_radius(d.get("label", "")) for _, d in nodes])

    frac_diffs = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]
    frac_diffs -= np.round(frac_diffs)
    cart_diffs = np.dot(frac_diffs, lattice_matrix)
    dist_matrix = np.linalg.norm(cart_diffs, axis=2)

    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    dist_mask = dist_matrix < (radii_sum * BOND_TOLERANCE)
    
    g_matrix_i = disorder_groups[:, np.newaxis]
    g_matrix_j = disorder_groups[np.newaxis, :]
    group_mask = (g_matrix_i == 0) | (g_matrix_j == 0) | (g_matrix_i == g_matrix_j)
    
    triu_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    final_mask = dist_mask & group_mask & triu_mask
    
    bond_indices = np.argwhere(final_mask)
    return [(node_indices[i], node_indices[j]) for i, j in bond_indices]

def unwrap_molecular_coordinates(graph, lattice_matrix):
    """
    Unwraps coordinates to make molecules continuous and filters out isolated components
    that do not participate in any conflicts.
    """
    # 1. Infer bonds
    bonds = infer_bonds_pbc(graph, lattice_matrix)
    
    # 2. Build bond graph
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(graph.nodes(data=True))
    bond_graph.add_edges_from(bonds)
    
    # 3. Identify Nodes involved in conflicts
    conflict_nodes = set()
    for u, v, d in graph.edges(data=True):
        if d.get("conflict_type"): # If it's a conflict edge
            conflict_nodes.add(u)
            conflict_nodes.add(v)
            
    # 4. Filter Components based on conflict involvement
    components = list(nx.connected_components(bond_graph))
    relevant_nodes = set()
    
    # Keep components that contain at least one conflict node
    # OR if no conflicts exist at all, keep everything (fallback)
    if not conflict_nodes:
        relevant_nodes = set(graph.nodes())
    else:
        for comp in components:
            if not comp.isdisjoint(conflict_nodes):
                relevant_nodes.update(comp)
    
    # 5. Process unwrapping ONLY for relevant nodes
    unwrapped_frac_coords = {}
    valid_bonds = []
    
    # Filter bonds to only include relevant nodes
    for u, v in bonds:
        if u in relevant_nodes and v in relevant_nodes:
            valid_bonds.append((u, v))
            
    # Unwrap components
    processed_nodes = set()
    
    for comp in components:
        # Skip irrelevant components
        if comp.isdisjoint(relevant_nodes):
            continue
            
        comp_list = list(comp)
        root = comp_list[0]
        
        if "frac_coord" not in graph.nodes[root]: continue
        
        comp_coords = {root: graph.nodes[root]["frac_coord"].copy()}
        queue = [root]
        comp_visited = {root}
        
        while queue:
            parent = queue.pop(0)
            parent_frac = comp_coords[parent]
            
            for neighbor in bond_graph.neighbors(parent):
                if neighbor in comp and neighbor not in comp_visited:
                    if "frac_coord" not in graph.nodes[neighbor]: continue
                    
                    neigh_raw = graph.nodes[neighbor]["frac_coord"]
                    shift = np.round(neigh_raw - parent_frac)
                    comp_coords[neighbor] = neigh_raw - shift
                    
                    comp_visited.add(neighbor)
                    queue.append(neighbor)
        
        unwrapped_frac_coords.update(comp_coords)

    # 6. Project to 2D
    rot_matrix = get_projection_matrix(lattice_matrix)
    pos = {}
    
    if unwrapped_frac_coords:
        center_frac = np.mean(list(unwrapped_frac_coords.values()), axis=0)
    else:
        center_frac = np.array([0.5, 0.5, 0.5])

    # Only generate positions for relevant nodes
    for n in relevant_nodes:
        if n in unwrapped_frac_coords:
            frac = unwrapped_frac_coords[n] - center_frac
            cart = np.dot(frac, lattice_matrix)
            aligned = np.dot(rot_matrix, cart)
            pos[n] = (aligned[0], aligned[1])
        else:
            # Fallback for disconnected nodes that are relevant
            pos[n] = (np.random.rand(), np.random.rand())
            
    return pos, valid_bonds, unwrapped_frac_coords, relevant_nodes

def get_node_color(disorder_group):
    if disorder_group < len(NODE_COLOR_CYCLE):
        return NODE_COLOR_CYCLE[disorder_group]
    return NODE_COLOR_CYCLE[disorder_group % len(NODE_COLOR_CYCLE)]

def get_edge_linestyle(conflict_type):
    if conflict_type == "logical_alternative": return "dashed"
    elif conflict_type == "symmetry_clash": return (0, (1, 1))
    return "solid"

def plot_unwrapped_graph(graph, lattice_matrix, title, filename):
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor('white')
    
    # Step 1: Unwrap and Filter
    start_time = time()
    pos, valid_bonds, unwrapped_3d, visible_nodes = unwrap_molecular_coordinates(graph, lattice_matrix)
    end_time = time()
    print(f"Unwrapping & Filtering took {end_time - start_time:.2f} seconds")
    
    # Create a subgraph of only visible nodes to keep drawing functions happy
    subgraph = graph.subgraph(visible_nodes)
    
    # Step 2: Draw Chemical Bonds
    if valid_bonds:
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=valid_bonds,
            edge_color=BOND_COLOR,
            width=4.0, alpha=0.3, label="Chemical Bond"
        )

    # Step 3: Draw Conflict Edges
    edges = subgraph.edges(data=True)
    edges_by_type = {}
    
    for u, v, d in edges:
        conflict_type = d.get("conflict_type")
        if not conflict_type: continue # Skip chemical bonds stored in graph structure if any
        
        if u in unwrapped_3d and v in unwrapped_3d:
            frac_u = unwrapped_3d[u]
            frac_v = unwrapped_3d[v]
            cart_dist = np.linalg.norm(np.dot(frac_u - frac_v, lattice_matrix))
            if cart_dist > VISUAL_CUTOFF: continue

        if conflict_type not in edges_by_type:
            edges_by_type[conflict_type] = []
        edges_by_type[conflict_type].append((u, v))

    for c_type, e_list in edges_by_type.items():
        if c_type in COLOR_MAP:
            nx.draw_networkx_edges(
                subgraph, pos, edgelist=e_list,
                edge_color=COLOR_MAP[c_type], width=1.0, alpha=0.8,
                style=get_edge_linestyle(c_type), connectionstyle="arc3,rad=0.2",
                label=c_type.replace('_', ' ').title()
            )

    # Step 4: Draw Nodes
    node_colors = []
    node_sizes = []
    
    disorder_groups = set()
    for n in subgraph.nodes():
        group = subgraph.nodes[n].get("disorder_group", 0)
        disorder_groups.add(group)
        node_colors.append(get_node_color(group))
        node_sizes.append(50)

    nx.draw_networkx_nodes(
        subgraph, pos, node_color=node_colors,
        edgecolors=BOND_COLOR, linewidths=0.5, node_size=node_sizes
    )

    # Step 5: Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Node legend
    for group in sorted(list(disorder_groups)):
        label = "Main Scaffold" if group == 0 else f"Disorder Group {group}"
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=get_node_color(group), markersize=10, 
                             label=label, markeredgecolor=BOND_COLOR))
    
    # Edge legend
    legend_elements.append(Line2D([0], [0], color=BOND_COLOR, lw=4, alpha=0.3, label='Chemical Bond'))
    
    for c_type in edges_by_type.keys():
        if c_type in COLOR_MAP:
            legend_elements.append(Line2D([0], [0], color=COLOR_MAP[c_type], lw=2, alpha=0.8, 
                                 label=c_type.replace('_', ' ').title(), 
                                 linestyle=get_edge_linestyle(c_type)))

    plt.title(title, fontsize=16)
    plt.axis('equal')
    plt.axis('off')
    plt.legend(handles=legend_elements, loc="lower center", frameon=False, 
              fontsize=12, bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def main():
    # Only scan specifically requested tricky files to test the focus logic
    input_files = [
        # "examples/DAP-4.cif",
        "examples/PAP-H4.cif",
        "examples/DAN-2.cif",
        # "examples/anhydrousCaffeine2_CGD_2007_7_1406.cif"
    ]
    
    output_dir = Path("output/graph_viz_focused")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cif_file in input_files:
        if not os.path.exists(cif_file): continue
        print(f"Processing {cif_file}...")
        
        try:
            lattice = parse_lattice_from_cif(cif_file)
            info = scan_cif_disorder(cif_file)
            builder = DisorderGraphBuilder(info, lattice)
            builder.build()

            plot_unwrapped_graph(
                builder.graph, lattice, 
                f"{Path(cif_file).stem} - Focused Conflict View", 
                output_dir / f"{Path(cif_file).stem}_focused.png"
            )
            plot_unwrapped_graph(
                builder.graph, lattice, 
                f"{Path(cif_file).stem} - Focused Conflict View", 
                output_dir / f"{Path(cif_file).stem}_focused.pdf"
            )
        except Exception as e:
            print(f"Error processing {cif_file}: {e}")

if __name__ == "__main__":
    main()