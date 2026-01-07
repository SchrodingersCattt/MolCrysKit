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
# Bond color and tolerance
BOND_COLOR = "#34495e"
BOND_TOLERANCE = 1.3  # Liberal tolerance to catch bonds
NODE_COLOR_CYCLE = ["#ecf0f1", "#f1c40f", "#e67e22", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#34495e", "#e67e22", "#f1c40f"]
# Color mapping for different conflict types
COLOR_MAP = {
    "logical_alternative": "#fd2c15",   # Red
    "symmetry_clash": "#a769e0",       # Purple
    "explicit": "#1ea556",             # Green
    "geometric": "#3498db",            # Blue
    "valence": "#f39c12",              # Orange
    "valence_geometry": "#B6347C",     # Dark Purple
}

VISUAL_CUTOFF = 5

def get_element_radius(label):
    match = re.match(r"([A-Z][a-z]?)", label)
    if match:
        el = match.group(1)
        # Try to import from constants, fallback to local definition if needed
        try:
            from molcrys_kit.constants.config import BONDING_CONFIG
            # Use atomic radii from constants if available
            import json
            with open(os.path.join(os.path.dirname(__file__), '..', 'molcrys_kit', 'constants', 'atomic_radii.json')) as f:
                radii_data = json.load(f)
            return radii_data.get(el, BONDING_CONFIG.get("DEFAULT_ATOMIC_RADIUS", 1.1))
        except (ImportError, FileNotFoundError):
            # Fallback covalent radii (Angstroms) - Extended to include common elements in crystal engineering
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
    """
    Standard conversion from cell parameters to 3x3 matrix.
    Aligns vector a along x-axis, b in xy-plane.
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate volume-related term
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    v_volume = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    
    # Check for invalid cell parameters (can happen in incomplete CIFs)
    if v_volume < 0:
        raise ValueError("Invalid cell parameters - volume calculation resulted in negative value")
    
    val = np.sqrt(v_volume)

    # Construct lattice matrix
    matrix = np.zeros((3, 3))
    matrix[0, :] = [a, 0.0, 0.0]
    matrix[1, :] = [b * cos_gamma, b * np.sin(gamma_rad), 0.0]
    matrix[2, 0] = c * cos_beta
    matrix[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / np.sin(gamma_rad)
    matrix[2, 2] = c * val / np.sin(gamma_rad)
    
    return matrix

def parse_lattice_from_cif(cif_path):
    """
    Manually parses cell parameters to avoid Pymatgen normalization issues.
    """
    params = {}
    with open(cif_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Remove comments
            line = line.split('#')[0]
            if not line:
                continue
            
            # Handle simple parameter lines - CIF format has values in the second column
            if '_cell_length_a' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['a'] = float(param_value)
            elif '_cell_length_b' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['b'] = float(param_value)
            elif '_cell_length_c' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['c'] = float(param_value)
            elif '_cell_angle_alpha' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['alpha'] = float(param_value)
            elif '_cell_angle_beta' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['beta'] = float(param_value)
            elif '_cell_angle_gamma' in line:
                values = line.split()
                if len(values) >= 2:
                    param_value = re.sub(r'\([^)]*\)', '', values[1])
                    params['gamma'] = float(param_value)
    
    if len(params) < 6:
        raise ValueError(f"Could not parse all 6 cell parameters from CIF {cif_path}. Found: {list(params.keys())}")
        
    return cell_parameters_to_matrix(
        params['a'], params['b'], params['c'],
        params['alpha'], params['beta'], params['gamma']
    )

def get_projection_matrix(lattice_matrix, view_axis='a'):
    """
    Generate a rotation matrix to project the crystal onto a specific plane.
    
    Parameters:
    -----------
    lattice_matrix : np.ndarray
        3x3 matrix of lattice vectors [a, b, c]
    view_axis : str
        The axis to look along ('a', 'b', or 'c').
        - 'c': View along c (Show ab plane) -> Default behavior
        - 'b': View along b (Show ac plane) -> Your request
        - 'a': View along a (Show bc plane)
    """
    a_vec = lattice_matrix[0]
    b_vec = lattice_matrix[1]
    c_vec = lattice_matrix[2]

    # Decide which vectors align to Screen X and Screen Y
    if view_axis == 'c':
        # View along c: a -> X, b -> Y (approx)
        u_vec = a_vec
        v_vec = b_vec
    elif view_axis == 'b':
        # View along b: a -> X, c -> Y (approx)
        u_vec = a_vec
        v_vec = c_vec
    elif view_axis == 'a':
        # View along a: b -> X, c -> Y (approx)
        u_vec = b_vec
        v_vec = c_vec
    else:
        raise ValueError("view_axis must be 'a', 'b', or 'c'")

    # Construct orthogonal basis for the screen
    # 1. Align first vector (u) to Screen X
    u_norm = u_vec / np.linalg.norm(u_vec)
    
    # 2. Project second vector (v) to Screen XY plane (remove component along u)
    v_proj = v_vec - np.dot(v_vec, u_norm) * u_norm
    
    # Handle degenerate case (should not happen for valid unit cells)
    if np.linalg.norm(v_proj) < 1e-6:
        v_perp = np.array([0, 1, 0])
    else:
        v_perp = v_proj / np.linalg.norm(v_proj)
        
    # 3. Third vector is the viewing direction (Screen Z)
    w_norm = np.cross(u_norm, v_perp)
    w_norm = w_norm / np.linalg.norm(w_norm)
    
    # Return 3x3 rotation matrix
    # Rows are the new basis vectors (X, Y, Z)
    return np.array([u_norm, v_perp, w_norm])

def mic_displacement(frac_u, frac_v):
    """Calculate Minimum Image displacement vector in fractional coordinates."""
    diff = frac_u - frac_v
    # Round to nearest integer to find the closest image
    image_shift = np.round(diff)
    return diff - image_shift, image_shift

def infer_bonds_pbc(graph, lattice_matrix):
    """
    Detect chemical bonds using vectorized numpy operations for high performance.
    (Optimized replacement for the previous O(N^2) loop)
    """
    nodes = list(graph.nodes(data=True))
    n_nodes = len(nodes)
    if n_nodes == 0:
        return []

    # 1. 预处理数据 (Pre-fetch data to avoiding loop lookups)
    node_indices = [n for n, _ in nodes]
    frac_coords = np.array([d.get("frac_coord", [0., 0., 0.]) for _, d in nodes])
    disorder_groups = np.array([d.get("disorder_group", 0) for _, d in nodes])
    radii = np.array([get_element_radius(d.get("label", "")) for _, d in nodes])

    # 2. 向量化计算距离矩阵 (Vectorized Distance Calculation)
    frac_diffs = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]
    frac_diffs -= np.round(frac_diffs)
    cart_diffs = np.dot(frac_diffs, lattice_matrix)
    dist_matrix = np.linalg.norm(cart_diffs, axis=2)

    # 3. 向量化构建判断掩码 (Vectorized Logic Masks)
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    dist_mask = dist_matrix < (radii_sum * BOND_TOLERANCE)
    
    g_matrix_i = disorder_groups[:, np.newaxis]
    g_matrix_j = disorder_groups[np.newaxis, :]
    group_mask = (g_matrix_i == 0) | (g_matrix_j == 0) | (g_matrix_i == g_matrix_j)
    
    triu_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    
    # 4. 综合所有条件
    final_mask = dist_mask & group_mask & triu_mask
    
    # 5. 提取结果
    bond_indices = np.argwhere(final_mask)
    bonds = [(node_indices[i], node_indices[j]) for i, j in bond_indices]
    
    return bonds

def unwrap_molecular_coordinates(graph, lattice_matrix):
    """
    Traverses the chemical bond network to 'unwrap' coordinates AND 
    aligns disjoint components (disorder parts) to the same location.
    Returns: 
        pos: 2D projected coordinates for plotting
        bonds: valid chemical bonds
        unwrapped_frac_coords: 3D fractional coordinates (for distance calc)
    """
    # 1. Infer bonds to establish connectivity
    bonds = infer_bonds_pbc(graph, lattice_matrix)
    
    # 2. Build a temporary graph for traversal
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(graph.nodes(data=True))
    bond_graph.add_edges_from(bonds)
    
    # 3. Component Processing
    unwrapped_frac_coords = {}
    visited = set()
    
    # Get all connected components
    components = list(nx.connected_components(bond_graph))
    
    # Process each component separately without forcing alignment to an anchor
    for comp in components:
        comp_list = list(comp)
        root = comp_list[0]
        
        # --- Internal Unwrapping (BFS) ---
        if "frac_coord" not in graph.nodes[root]:
            continue
            
        comp_coords = {} 
        comp_coords[root] = graph.nodes[root]["frac_coord"].copy()
        
        queue = [root]
        comp_visited = {root}
        
        while queue:
            parent = queue.pop(0)
            parent_frac = comp_coords[parent]
            
            for neighbor in bond_graph.neighbors(parent):
                if neighbor in comp and neighbor not in comp_visited:
                    if "frac_coord" not in graph.nodes[neighbor]:
                        continue
                        
                    neigh_raw_frac = graph.nodes[neighbor]["frac_coord"]
                    
                    shift = np.round(neigh_raw_frac - parent_frac)
                    unwrapped_frac = neigh_raw_frac - shift
                    
                    comp_coords[neighbor] = unwrapped_frac
                    comp_visited.add(neighbor)
                    queue.append(neighbor)

        # Apply the unwrapped coordinates for this component directly
        for node, coords in comp_coords.items():
            unwrapped_frac_coords[node] = coords
            
    # 4. Project to 2D Cartesian
    rot_matrix = get_projection_matrix(lattice_matrix)
    pos = {}
    
    # Center the final plot
    if unwrapped_frac_coords:
        all_coords = np.array(list(unwrapped_frac_coords.values()))
        center_frac = np.mean(all_coords, axis=0)
    else:
        center_frac = np.array([0.5, 0.5, 0.5])

    for n in graph.nodes():
        if n in unwrapped_frac_coords:
            frac = unwrapped_frac_coords[n] - center_frac
            cart = np.dot(frac, lattice_matrix)
            aligned = np.dot(rot_matrix, cart)
            pos[n] = (aligned[0], aligned[1])
        else:
            pos[n] = (np.random.rand(), np.random.rand())
            
    # Return also the 3D coordinates for distance calculation
    return pos, bonds, unwrapped_frac_coords

def get_node_color(disorder_group):
    """Get the color for a node based on its disorder group."""
    if disorder_group < len(NODE_COLOR_CYCLE):
        return NODE_COLOR_CYCLE[disorder_group]
    return NODE_COLOR_CYCLE[disorder_group % len(NODE_COLOR_CYCLE)]

def get_node_label(disorder_group):
    """Get the label for a node based on its disorder group."""
    if disorder_group == 0:
        return "Main Scaffold"
    return f"Disorder Group {disorder_group}"

def get_edge_linestyle(conflict_type):
    """Get the linestyle for an edge based on its conflict type."""
    if conflict_type == "logical_alternative":
        return "dashed"
    elif conflict_type == "symmetry_clash":
        return (0, (1, 1))  # Dotted pattern
    return "solid"


def plot_unwrapped_graph(graph, lattice_matrix, title, filename):
    fig = plt.figure(figsize=(8, 6))  # Assign figure to variable
    fig.patch.set_facecolor('white')  # White background for publication
    
    # --- Step 1: Unwrap Coordinates ---
    start_time = time()
    pos, valid_bonds, unwrapped_3d = unwrap_molecular_coordinates(graph, lattice_matrix)
    end_time = time()
    print(f"Unwrapping took {end_time - start_time:.2f} seconds")
    
    # --- Step 2: Draw Chemical Bonds (Skeleton) in the background ---
    if valid_bonds:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=valid_bonds,
            edge_color=BOND_COLOR,
            width=4.0,      # Thick lines
            alpha=0.3,      # Semi-transparent
            label="Chemical Bond"
        )

    # --- Step 3: Draw Conflict Edges (The graph edges) in the foreground ---
    edges = graph.edges(data=True)
    
    # Group edges by conflict type to draw them separately
    edges_by_type = {}
    for u, v, d in edges:
        conflict_type = d.get("conflict_type", "unknown")
        
        # --- GLOBAL VISUALIZATION TRUNCATION ---
        if u in unwrapped_3d and v in unwrapped_3d:
            frac_u = unwrapped_3d[u]
            frac_v = unwrapped_3d[v]
            cart_dist = np.linalg.norm(np.dot(frac_u - frac_v, lattice_matrix))
            
            if cart_dist > VISUAL_CUTOFF:
                continue

        if conflict_type not in edges_by_type:
            edges_by_type[conflict_type] = []
        edges_by_type[conflict_type].append((u, v))

    # Draw each type of conflict with appropriate styling
    for conflict_type, specific_edges in edges_by_type.items():
        if conflict_type in COLOR_MAP:
            color = COLOR_MAP[conflict_type]
            linestyle = get_edge_linestyle(conflict_type)
            
            # Use curved lines to distinguish from chemical bonds
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=specific_edges,
                edge_color=color,
                width=2.0,
                alpha=0.8,
                style=linestyle,
                connectionstyle="arc3,rad=0.2",  # Curved lines
                label=conflict_type.replace('_', ' ').title()  # Add label for legend
            )

    # --- Step 4: Draw Nodes ---
    node_colors = []
    labels = {}
    node_sizes = []
    
    # Dynamic sizing
    base_size = 50 # 300 if len(graph.nodes) < 50 else 150
    
    # Get unique disorder groups for legend
    disorder_groups = set()
    for n in graph.nodes():
        group = graph.nodes[n].get("disorder_group", 0)
        disorder_groups.add(group)
    
    for n in graph.nodes():
        group = graph.nodes[n].get("disorder_group", 0)
        # Clean labels (e.g. C1_part1 -> C1)
        lbl = str(graph.nodes[n].get("label", str(n)))
        labels[n] = lbl.split('_')[0] 
        node_sizes.append(base_size)

        # Color nodes by their disorder_group
        node_colors.append(get_node_color(group))

    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        edgecolors=BOND_COLOR,
        linewidths=1,
        node_size=node_sizes
    )
    
    # Optional: Draw text if not too crowded
    if len(graph.nodes) < 50:
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')

    # --- Step 5: Legend ---
    # Create custom legend elements for all types
    from matplotlib.lines import Line2D
    
    # Create legend entries for node types (disorder groups)
    node_legend_elements = []
    for group in sorted(list(disorder_groups)):
        color = get_node_color(group)
        label = get_node_label(group)
        
        node_legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=color, 
                   markersize=10, 
                   label=label,
                   markeredgecolor=BOND_COLOR,
                   markeredgewidth=1.5)
        )
    
    # Create legend entries for edge types
    edge_legend_elements = [
        Line2D([0], [0], color=BOND_COLOR, lw=4, alpha=0.3, label='Chemical Bond')
    ]
    # Note: We need to re-scan valid edges, as the loop above might have filtered some
    present_conflicts = set(edges_by_type.keys())
    for conflict_type, col in COLOR_MAP.items():
        # Only add to legend if present in graph
        if conflict_type in present_conflicts:
            linestyle = get_edge_linestyle(conflict_type)
                
            edge_legend_elements.append(
                Line2D([0], [0], color=col, lw=2, alpha=0.8, 
                       label=conflict_type.replace('_', ' ').title(), linestyle=linestyle)
            )
    
    # Combine all legend elements
    all_legend_elements = node_legend_elements + edge_legend_elements
    
    plt.title(title, fontsize=16)
    plt.axis('equal')
    plt.axis('off')  # Hide axes for clean look
    
    # Add legend
    plt.legend(handles=all_legend_elements, loc="center right", frameon=False, fontsize=12, bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def main():
    input_files = []
    input_files.extend([
        "examples/1-HTP.cif",
        "examples/PAP-M5.cif",
        "examples/PAP-H4.cif",
        "examples/DAP-4.cif",
        "examples/EAP-8.cif",
        "examples/DAN-2.cif",
    ])
    input_files.extend(glob.glob("examples/TIL*.cif"))
    input_files.extend(glob.glob("examples/1_*.cif"))
    
    
    output_dir = Path("output/graph_viz_unwrapped")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cif_file in input_files:
        if not os.path.exists(cif_file): continue
        print(f"Processing {cif_file}...")
        
        try:
            # Replace Pymatgen lattice parsing with manual CIF parsing
            lattice = parse_lattice_from_cif(cif_file)
            info = scan_cif_disorder(cif_file)
            
            builder = DisorderGraphBuilder(info, lattice)
            
            # --- BUILDER STRATEGY ---
            builder.build()

            plot_unwrapped_graph(
                builder.graph, lattice, 
                f"{Path(cif_file).stem} - Unwrapped Molecular View", 
                output_dir / f"{Path(cif_file).stem}_unwrapped.png"
            )
        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue

if __name__ == "__main__":
    main()