#!/usr/bin/env python
"""
Visualize the construction of Disorder Exclusion Graphs using PHYSICAL COORDINATES.
(Clean Version: Hides Periodic Boundary Crossing Edges)

This script processes the example disordered CIF files and generates 
plots where nodes are placed at their actual projected (XY) coordinates.
It automatically filters out edges that wrap around periodic boundaries 
to avoid visual clutter ("long lines crossing the cell").
"""

import glob
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder

# Define color scheme for different conflict types
COLOR_MAP = {
    "conformer_competition": "#e74c3c",  # Red
    "explicit": "#2ecc71",               # Green
    "geometric": "#3498db",              # Blue
    "valence": "#f39c12",                # Orange
    "valence_geometry": "#9b59b6",       # Purple
}

def get_physical_positions(graph, lattice_matrix):
    """
    Extract physical coordinates from graph nodes and project to 2D.
    Returns a dictionary {node_idx: (x, y)} for nx.draw.
    """
    pos = {}
    for n in graph.nodes():
        # Get fractional coords stored in node
        frac = graph.nodes[n].get("frac_coord")
        if frac is None:
            pos[n] = (0, 0)
            continue
            
        # Convert to Cartesian: Cartesian = Frac * Lattice
        cart = np.dot(frac, lattice_matrix)
        
        # Project to 2D (XY plane)
        pos[n] = (cart[0], cart[1])
    return pos

def is_pbc_artifact(u_idx, v_idx, graph, lattice_matrix, tolerance=2.0):
    """
    Check if an edge is a Periodic Boundary Condition (PBC) artifact.
    
    A PBC artifact occurs when two atoms are close in 3D (across the boundary)
    but far apart in the non-periodic 2D projection, causing a long line 
    to be drawn across the unit cell.
    
    Returns:
        True if the edge is an artifact and should be hidden.
    """
    node_u = graph.nodes[u_idx]
    node_v = graph.nodes[v_idx]
    
    if "frac_coord" not in node_u or "frac_coord" not in node_v:
        return False
        
    frac_u = node_u["frac_coord"]
    frac_v = node_v["frac_coord"]
    
    # 1. Calculate Visual Distance (Non-Periodic Cartesian)
    # This matches exactly how matplotlib draws the line
    delta_frac_visual = frac_u - frac_v
    cart_visual = np.dot(delta_frac_visual, lattice_matrix)
    dist_visual = np.linalg.norm(cart_visual)
    
    # 2. Calculate Physical Distance (Minimum Image Convention)
    # This is the real chemical distance
    delta_frac_mic = delta_frac_visual - np.round(delta_frac_visual)
    cart_mic = np.dot(delta_frac_mic, lattice_matrix)
    dist_mic = np.linalg.norm(cart_mic)
    
    # 3. If Visual Distance is significantly larger than Physical Distance,
    # it means the edge wraps around the boundary.
    if dist_visual > dist_mic + tolerance:
        return True
        
    return False

def plot_graph(graph, lattice_matrix, title, filename):
    """
    Helper function to plot and save the current state of the graph
    using PHYSICAL POSITIONS and filtering PBC artifacts.
    """
    plt.figure(figsize=(10, 10))
    
    # Use physical positions
    pos = get_physical_positions(graph, lattice_matrix)
    
    # Draw nodes
    node_colors = []
    labels = {}
    
    for n in graph.nodes():
        group = graph.nodes[n].get("disorder_group", 0)
        # Label with Element + Index (e.g., C12)
        node_label = graph.nodes[n].get("label", str(n))
        labels[n] = node_label
        
        if group == 0:
            node_colors.append("#bdc3c7")  # Gray for backbone
        elif group == 1:
            node_colors.append("#ffffff")  # White for Part 1
        elif group == 2:
            node_colors.append("#7f8c8d")  # Dark Gray for Part 2
        else:
            node_colors.append("#95a5a6")
            
    # Draw nodes with black borders
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors="black", node_size=300)
    
    # Draw simple labels inside nodes
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

    # Draw edges by type, filtering out PBC artifacts
    edges = graph.edges(data=True)
    if edges:
        for edge_type, color in COLOR_MAP.items():
            # Filter edges of this type AND filter out PBC artifacts
            specific_edges = []
            for u, v, d in edges:
                if d.get("conflict_type") == edge_type:
                    # Check for visual artifact
                    if not is_pbc_artifact(u, v, graph, lattice_matrix):
                        specific_edges.append((u, v))
            
            if specific_edges:
                nx.draw_networkx_edges(
                    graph, pos, 
                    edgelist=specific_edges, 
                    edge_color=color, 
                    width=2.0, 
                    label=edge_type,
                    alpha=0.7
                )
    
    plt.title(title)
    
    # Turn on axis for geometric context (Angstroms)
    plt.axis("on")
    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Ensure aspect ratio is equal
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add legend
    plt.legend(loc="upper right", title="Conflict Types")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def visualize_disorder_graph_pipeline():
    """Run the visualization pipeline on all example files."""
    
    input_files = []
    # Define input files
    input_files += [
        "examples/EAP-8.cif",
        "examples/1-HTP.cif",
        "examples/PAP-M5.cif",
        "examples/PAP-H4.cif",
        "examples/DAP-4.cif",
    ]
    input_files += glob.glob("examples/TIL*.cif")
    input_files += glob.glob("examples/1_*.cif")

    # Create output directory
    output_base = Path("output/graph_visualization_clean")
    output_base.mkdir(parents=True, exist_ok=True)

    for cif_file in input_files:
        if not os.path.exists(cif_file):
            continue

        print(f"Visualizing graph construction for {cif_file}...")
        base_name = Path(cif_file).stem
        file_out_dir = output_base / base_name
        file_out_dir.mkdir(exist_ok=True)

        try:
            # --- Phase 1: Data Extraction ---
            info = scan_cif_disorder(cif_file)

            # Extract lattice
            from pymatgen.io.cif import CifParser
            parser = CifParser(cif_file)
            structure = parser.parse_structures()[0]
            lattice_matrix = structure.lattice.matrix

            # --- Phase 2: Graph Building ---
            builder = DisorderGraphBuilder(info, lattice_matrix)
            
            # SNAPSHOT 0: Initial Nodes
            plot_graph(builder.graph, lattice_matrix,
                      f"{base_name} - Step 0: Initial Atoms (XY Plane)", 
                      file_out_dir / "step_0_nodes.png")

            # Step 1: Conformer Conflicts
            builder._identify_conformers() 
            builder._add_conformer_conflicts()
            plot_graph(builder.graph, lattice_matrix,
                      f"{base_name} - Step 1: Conformer Competition", 
                      file_out_dir / "step_1_conformer_conflicts.png")

            # Step 2: Explicit Conflicts
            builder._add_explicit_conflicts()
            plot_graph(builder.graph, lattice_matrix,
                      f"{base_name} - Step 2: Explicit PART Conflicts", 
                      file_out_dir / "step_2_explicit_conflicts.png")

            # Step 3: Geometric Conflicts
            builder._add_geometric_conflicts()
            plot_graph(builder.graph, lattice_matrix,
                      f"{base_name} - Step 3: Geometric Clashes", 
                      file_out_dir / "step_3_geometric_conflicts.png")

            # Step 4: Valence Conflicts (Final)
            builder._resolve_valence_conflicts()
            plot_graph(builder.graph, lattice_matrix,
                      f"{base_name} - Step 4: Valence/Overcrowding (Final)", 
                      file_out_dir / "step_4_final.png")
            
            print(f"  - Saved clean visualizations to {file_out_dir}")

        except ImportError:
            print("  - Error: pymatgen is required for lattice extraction.")
            break
        except Exception as e:
            print(f"  - Error processing {cif_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\nVisualization complete!")


if __name__ == "__main__":
    visualize_disorder_graph_pipeline()