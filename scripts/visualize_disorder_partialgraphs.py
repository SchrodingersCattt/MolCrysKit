#!/usr/bin/env python
"""
Visualize Disorder Exclusion Graphs with MOLECULAR UNWRAPPING - Focused View.

Features:
1. Infer Chemical Bonds using PBC (Minimum Image Convention).
2. Unwrap/Recenter molecules: Shifts atom coordinates so molecules appear contiguous.
3. Focus Visualization: Only draws molecules/clusters that contain conflict nodes.
   Filters out irrelevant background molecules to highlight the disorder mechanism.
4. Periodic Replicas: Renders periodic copies to show packing context.
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
BOND_TOLERANCE = 1.2
NODE_COLOR_CYCLE = ["#ecf0f1", "#b2c55f", "#d49b68", "#6fb1dd", "#c981e6", "#f1c40f", "#e67e22", "#78afe7", "#d49b68", "#b2c55f"]

COLOR_MAP = {
    "logical_alternative": "#a17570",   # Red
    "symmetry_clash": "#75658f",       # Purple
    "explicit": "#63aa81",             # Green
    "geometric": "#5c849e",            # Blue
    "valence": "#cfaf7a",              # Orange
    "valence_geometry": "#BD729B",     # Dark Purple
}

VISUAL_CUTOFF = 3.0

def get_element_radius(label):
    match = re.match(r"([A-Z][a-z]?)", label)
    if match:
        el = match.group(1)
        from molcrys_kit.constants.config import BONDING_CONFIG
        import json
        with open(os.path.join(os.path.dirname(__file__), '..', 'molcrys_kit', 'constants', 'atomic_radii.json')) as f:
            radii_data = json.load(f)
        return radii_data.get(el, BONDING_CONFIG.get("DEFAULT_ATOMIC_RADIUS", 1.1))
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
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    # 稍微调大画布以容纳多副本
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor('white')
    
    # --- Step 1: Unwrap & Filter ---
    start_time = time()
    # base_pos: 2D plot coordinates
    # unwrapped_3d: 3D fractional coordinates (continuous molecules)
    base_pos, valid_bonds, unwrapped_3d, visible_nodes = unwrap_molecular_coordinates(graph, lattice_matrix)
    
    # Create subgraph for conflict lookup
    subgraph = graph.subgraph(visible_nodes)
    print(f"Unwrapping took {time() - start_time:.2f} s")

    # --- Step 2: Pre-calculate Screen Vectors (MIC) ---
    rot_matrix = get_projection_matrix(lattice_matrix)
    
    # Helper to calculate 2D vector between two nodes using MIC
    def get_vec(u, v):
        if u not in unwrapped_3d or v not in unwrapped_3d:
            return None
        frac_u = unwrapped_3d[u]
        frac_v = unwrapped_3d[v]
        diff = frac_v - frac_u
        # Minimum Image Convention: force nearest neighbor vector
        mic_diff = diff - np.round(diff)
        cart_vec = np.dot(mic_diff, lattice_matrix)
        proj_vec = np.dot(rot_matrix, cart_vec)
        return proj_vec[:2] # Return (dx, dy)

    # A. Vectors for Chemical Bonds (Inferred)
    # Key: (u, v), Value: (dx, dy)
    bond_vectors_chem = {}
    for u, v in valid_bonds:
        vec = get_vec(u, v)
        if vec is not None:
            bond_vectors_chem[(u, v)] = vec

    # B. Vectors for Conflict Edges (From Graph)
    # Key: (u, v), Value: (dx, dy)
    bond_vectors_conf = {}
    # FIX: Removed keys=True to support standard DiGraph/Graph
    for u, v, d in subgraph.edges(data=True):
        vec = get_vec(u, v)
        if vec is not None:
            bond_vectors_conf[(u, v)] = vec

    # --- Step 3: Drawing Loop (Replicas) ---
    vec_a_proj = np.dot(rot_matrix, lattice_matrix[0])[:2]
    vec_b_proj = np.dot(rot_matrix, lattice_matrix[1])[:2]
    vec_c_proj = np.dot(rot_matrix, lattice_matrix[2])[:2]

    # 3x3 Grid for visualization
    if "ZIF" in title or "MAF" in title:
        x_range = range(-1, 2)
        y_range = range(-1, 2)
        z_range = range(-1, 2)
    elif "DAN" in title:
        x_range = range(-1, 2)
        y_range = range(-1, 2)
        z_range = range(-1, 2)
    else:
        x_range = [0]
        y_range = [0]
        z_range = [0]

    # Collectors for Batch Rendering
    segments_chem = []
    segments_conf = {} # key: conflict_type, value: list of segments

    # Node data collectors
    all_node_x = []
    all_node_y = []
    all_node_colors = []
    all_node_sizes = []
    
    # Pre-fetch node colors
    node_color_map = {}
    node_disorder_groups = {}
    for n in subgraph.nodes():
        group = subgraph.nodes[n].get("disorder_group", 0)
        node_disorder_groups[n] = group
        node_color_map[n] = get_node_color(group)

    for ix in x_range:
        for iy in y_range:
            for iz in z_range:
                screen_shift = (ix * vec_a_proj) + (iy * vec_b_proj) + (iz * vec_c_proj)
                
                # 1. Collect Nodes
                for n, base_xy in base_pos.items():
                    x = base_xy[0] + screen_shift[0]
                    y = base_xy[1] + screen_shift[1]
                    all_node_x.append(x)
                    all_node_y.append(y)
                    all_node_colors.append(node_color_map[n])
                    all_node_sizes.append(50) 
                    
                    # Label center replica only
                    if ix == 0 and iy == 0 and iz == 0 and len(subgraph) < 50:
                        lbl = str(subgraph.nodes[n].get("label", "")).split("_")[0]
                        ax.text(x, y, lbl, fontsize=8, fontweight='bold', 
                                ha='center', va='center', zorder=10)

                # 2. Collect Chemical Bonds
                for (u, v), vec in bond_vectors_chem.items():
                    u_start = np.array(base_pos[u]) + screen_shift
                    v_end = u_start + vec
                    segments_chem.append([u_start, v_end])

                # 3. Collect Conflict Edges
                for u, v, d in subgraph.edges(data=True):
                    if (u, v) not in bond_vectors_conf:
                        continue
                        
                    c_type = d.get("conflict_type")
                    if not c_type: continue # Skip if it's not a conflict edge
                    
                    vec = bond_vectors_conf[(u, v)]
                    u_start = np.array(base_pos[u]) + screen_shift
                    v_end = u_start + vec
                    
                    if c_type not in segments_conf:
                        segments_conf[c_type] = []
                    segments_conf[c_type].append([u_start, v_end])

    # --- Step 4: Batch Rendering ---
    
    # Draw Chemical Bonds
    if segments_chem:
        lc_bonds = LineCollection(segments_chem, colors=BOND_COLOR, 
                                linewidths=4.0, alpha=0.3, zorder=1)
        ax.add_collection(lc_bonds)
        
    # Draw Conflict Edges
    for c_type, segments in segments_conf.items():
        if c_type in COLOR_MAP:
            style = get_edge_linestyle(c_type)
            ls = "solid"
            if style == "dashed": ls = "dashed"
            elif style == (0, (1, 1)): ls = "dotted"
            
            lc_conf = LineCollection(segments, colors=COLOR_MAP[c_type], 
                                   linewidths=1.5, alpha=0.8, linestyles=ls, zorder=2)
            ax.add_collection(lc_conf)

    # Draw Nodes
    ax.scatter(all_node_x, all_node_y, c=all_node_colors, s=all_node_sizes, 
              edgecolors=BOND_COLOR, linewidths=0.5, zorder=5)

    # --- Step 5: Legend ---
    legend_elements = []
    unique_groups = sorted(list(set(node_disorder_groups.values())))
    for group in unique_groups:
        label = "Main Scaffold" if group == 0 else f"Disorder Group {group}"
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=get_node_color(group), 
                                    markersize=10, label=label, 
                                    markeredgecolor=BOND_COLOR))
    
    legend_elements.append(Line2D([0], [0], color=BOND_COLOR, lw=4, alpha=0.3, label='Chemical Bond'))
    
    for c_type in segments_conf.keys():
        if c_type in COLOR_MAP:
            legend_elements.append(Line2D([0], [0], color=COLOR_MAP[c_type], lw=2, alpha=0.8, 
                                        label=c_type.replace('_', ' ').title()))

    ax.set_title(title, fontsize=16)
    ax.axis('equal')
    ax.axis('off')
    ax.legend(handles=legend_elements, loc="lower center", frameon=False, 
             fontsize=12, bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def main():
    # Only scan specifically requested tricky files to test the focus logic
    input_files = glob.glob("examples/MAF*.cif")
    input_files += [
        "examples/ZIF-4.cif",
        "examples/ZIF-8.cif",
        "examples/DAP-4.cif",
        "examples/PAP-H4.cif",
        "examples/DAN-2.cif",
        "examples/anhydrousCaffeine2_CGD_2007_7_1406.cif"
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