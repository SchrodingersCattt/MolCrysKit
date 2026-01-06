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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from molcrys_kit.io.cif import scan_cif_disorder
    from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
except ImportError:
    pass

# --- CONSTANTS ---
# Covalent Radii (Angstroms) - Extended to include common elements in crystal engineering
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, 
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, 
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, 
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, 
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, 
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, 
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, 
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, 
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40
    # Can be extended further as needed
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
NODE_COLOR_CYCLE = ["#ecf0f1", "#1abc9c", "#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#34495e", "#16a085", "#8e44ad"]

# 显式冲突的可视化截断半径 (单位: Angstrom)
EXPLICIT_CUTOFF = 3.5

def get_element_radius(label):
    match = re.match(r"([A-Z][a-z]?)", label)
    if match:
        el = match.group(1)
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
            line = line.strip().split('#')[0]  # Remove comments
            
            # Handle simple parameter lines
            if '_cell_length_a' in line:
                values = line.split()
                params['a'] = float(values[1].split('(')[0])
            elif '_cell_length_b' in line:
                values = line.split()
                params['b'] = float(values[1].split('(')[0])
            elif '_cell_length_c' in line:
                values = line.split()
                params['c'] = float(values[1].split('(')[0])
            elif '_cell_angle_alpha' in line:
                values = line.split()
                params['alpha'] = float(values[1].split('(')[0])
            elif '_cell_angle_beta' in line:
                values = line.split()
                params['beta'] = float(values[1].split('(')[0])
            elif '_cell_angle_gamma' in line:
                values = line.split()
                params['gamma'] = float(values[1].split('(')[0])
    
    if len(params) < 6:
        raise ValueError(f"Could not parse all 6 cell parameters from CIF {cif_path}. Found: {list(params.keys())}")
        
    return cell_parameters_to_matrix(
        params['a'], params['b'], params['c'],
        params['alpha'], params['beta'], params['gamma']
    )

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
        # Initialize the root of this component
        if "frac_coord" not in graph.nodes[root]:
            continue
            
        # Temporarily store coords for this component locally
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
                    
                    # Unwrap relative to parent (Internal Continuity)
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
    fig = plt.figure(figsize=(14, 12))  # Assign figure to variable
    fig.patch.set_facecolor('white')  # White background for publication
    
    # --- Step 1: Unwrap Coordinates ---
    # This is the critical fix for "connections across box"
    pos, valid_bonds, unwrapped_3d = unwrap_molecular_coordinates(graph, lattice_matrix)
    
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
        
        # --- 核心修改逻辑：显式冲突距离检查 ---
        if conflict_type == "explicit":
            # 如果节点不在 unwrap 结果里（可能是孤立点），跳过
            if u not in unwrapped_3d or v not in unwrapped_3d:
                continue
                
            # 计算真实的 3D 笛卡尔距离
            frac_u = unwrapped_3d[u]
            frac_v = unwrapped_3d[v]
            # 注意：这里不需要 MIC，因为坐标已经 Unwrap 过了，直接算欧氏距离即可
            cart_dist = np.linalg.norm(np.dot(frac_u - frac_v, lattice_matrix))
            
            # 只有距离小于截断值才绘制
            if cart_dist > EXPLICIT_CUTOFF:
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
    base_size = 300 if len(graph.nodes) < 50 else 150
    
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
        linewidths=1.5,
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
    # 注意：这里需要重新扫描 valid edges，因为上面的 loop 可能过滤掉了所有的 explicit
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
    plt.legend(handles=all_legend_elements, loc="upper right", frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def main():
    input_files = []
    input_files.extend(glob.glob("examples/TIL*.cif"))
    input_files.extend([
        "examples/1-HTP.cif",
        "examples/PAP-M5.cif",
        "examples/PAP-H4.cif",
        "examples/DAP-4.cif",
        "examples/EAP-8.cif",
    ])
    
    
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
            # ... (Run your builder pipeline steps here: identify_conformers, conflicts, etc.)
            builder._identify_conformers()
            builder._add_explicit_conflicts()
            builder._resolve_valence_conflicts()

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