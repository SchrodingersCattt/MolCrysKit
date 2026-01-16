"""
Main entry point for disorder handling pipeline.

This module orchestrates the full disorder handling workflow:
Phase 1 (Data Extraction) -> Phase 2 (Graph Building) -> Phase 3 (Structure Generation)
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from ...structures.crystal import MolecularCrystal
from .graph import DisorderGraphBuilder
from .solver import DisorderSolver
from ...io.cif import scan_cif_disorder  # Correct import from io module
from ...io.cif import DisorderInfo


def resolve_disorder(info: DisorderInfo) -> DisorderInfo:
    """
    Clean the atom list by resolving disorder according to the Major Component Selection rule.

    Logic:
    1. Group atoms by their disorder_assembly and disorder_group.
    2. ALWAYS keep disorder_group 0 (static backbone).
    3. For all other groups (1, 2, -1, etc.):
       - Treat them as competitive components.
       - Calculate total occupancy for each component.
       - Keep ONLY the component with the highest total occupancy.
       - This fixes the bug where Part -1 (negative) was incorrectly kept because -1 <= 1.

    Parameters:
    -----------
    info : DisorderInfo
        Raw disorder data from scan_cif_disorder

    Returns:
    --------
    DisorderInfo
        Cleaned disorder data with only major components
    """
    # Group atoms by assembly to handle independent disorder sites separately
    # Key: assembly_label (str) -> Value: List of atom indices
    assembly_map: Dict[str, List[int]] = defaultdict(list)

    for i in range(len(info.labels)):
        # Normalize assembly: treat "." or "?" as empty string ""
        assembly = info.assemblies[i] if i < len(info.assemblies) else ""
        if assembly in [".", "?"]:
            assembly = ""
        assembly_map[assembly].append(i)

    atoms_to_keep = []

    for assembly, atom_indices in assembly_map.items():
        # Within each assembly, group atoms by their ACTUAL disorder group
        # Key: disorder_group (int) -> Value: List of atom indices
        group_parts: Dict[int, List[int]] = defaultdict(list)
        
        for idx in atom_indices:
            group = info.disorder_groups[idx]
            group_parts[group].append(idx)

        # 1. Always keep Part 0 (Static Backbone)
        if 0 in group_parts:
            atoms_to_keep.extend(group_parts[0])
            # Remove 0 from the map so it doesn't participate in the competition below
            del group_parts[0]

        # 2. Process competitive parts (Part 1, 2, -1, -2, etc.)
        if group_parts:
            # Calculate total occupancy for each remaining part
            part_occupancies = {}
            for part, indices in group_parts.items():
                total_occ = sum(info.occupancies[i] for i in indices)
                part_occupancies[part] = total_occ
            
            # Select the single best part with the highest total occupancy
            # In case of ties (e.g., 0.5 vs 0.5), max() picks the first one, which is acceptable
            best_part = max(part_occupancies, key=part_occupancies.get)
            
            atoms_to_keep.extend(group_parts[best_part])

    # Sort indices to preserve original CIF order roughly
    atoms_to_keep.sort()

    # Create new DisorderInfo with only the selected atoms
    # (Helper function to slice arrays safely)
    def slice_attr(attr_list, indices):
        return [attr_list[i] for i in indices]

    new_labels = slice_attr(info.labels, atoms_to_keep)
    new_symbols = slice_attr(info.symbols, atoms_to_keep)
    new_frac_coords = info.frac_coords[atoms_to_keep]
    new_occupancies = slice_attr(info.occupancies, atoms_to_keep)
    new_disorder_groups = slice_attr(info.disorder_groups, atoms_to_keep)
    
    # Handle optional fields safely
    new_assemblies = slice_attr(info.assemblies, atoms_to_keep) if info.assemblies else [""] * len(atoms_to_keep)
    new_sym_op_indices = slice_attr(info.sym_op_indices, atoms_to_keep) if info.sym_op_indices else [0] * len(atoms_to_keep)

    return DisorderInfo(
        labels=new_labels,
        symbols=new_symbols,
        frac_coords=new_frac_coords,
        occupancies=new_occupancies,
        disorder_groups=new_disorder_groups,
        assemblies=new_assemblies,
        sym_op_indices=new_sym_op_indices
    )


def generate_ordered_replicas_from_disordered_sites(
    filepath: str, generate_count: int = 1, method: str = "optimal", apply_disorder_resolution: bool = True
) -> List[MolecularCrystal]:
    """
    Process a disordered CIF file through the full disorder handling pipeline.

    Parameters:
    -----------
    filepath : str
        Path to the CIF file
    generate_count : int
        Number of structures to generate (for 'random' method)
    method : str
        'optimal' for single best structure, 'random' for ensemble
    apply_disorder_resolution : bool, default=True
        Whether to apply the new disorder resolution strategy before building the graph

    Returns:
    --------
    List[MolecularCrystal]
        List of ordered molecular crystal structures
    """
    # Phase 1: Extract raw disorder data
    info = scan_cif_disorder(filepath)

    # Apply disorder resolution if requested (using the new strategy)
    if apply_disorder_resolution:
        info = resolve_disorder(info)

    # Extract lattice matrix from CIF file using pymatgen
    try:
        from pymatgen.io.cif import CifParser

        parser = CifParser(filepath)
        structure = parser.parse_structures()[0]  # Get first structure
        lattice_matrix = structure.lattice.matrix
    except ImportError:
        raise ImportError(
            "pymatgen is required for lattice extraction. "
            "Please install it with 'pip install pymatgen'"
        )

    # Phase 2: Build exclusion graph
    builder = DisorderGraphBuilder(info, lattice_matrix)
    graph = builder.build()

    # Phase 3: Generate ordered structures
    solver = DisorderSolver(info, graph, lattice_matrix)
    results = solver.solve(num_structures=generate_count, method=method)

    return results
