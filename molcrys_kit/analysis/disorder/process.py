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
    1. Group atoms by their label root or connectivity, but primarily by disorder_assembly and disorder_group.
    2. Keep atoms with disorder_group 0 or 1 (the static backbone).
    3. For explicit disorder groups (e.g., PART 2 vs PART 3):
       - Calculate the sum of occupancies for atoms in each PART.
       - Keep the PART with the highest total occupancy.
       - Discard the minor PARTs entirely from the list.
    4. Handle negative disorder groups consistently.

    Parameters:
    -----------
    info : DisorderInfo
        Raw disorder data from scan_cif_disorder

    Returns:
    --------
    DisorderInfo
        Cleaned disorder data with only major components
    """
    # Group atoms by (assembly, abs(disorder_group)) to handle positive/negative parts separately
    groups_map: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    
    for i in range(len(info.labels)):
        assembly = info.assemblies[i] if i < len(info.assemblies) else ""
        disorder_group = info.disorder_groups[i]
        
        # Use absolute value for grouping to handle negative parts properly
        group_key = (assembly, abs(disorder_group))
        groups_map[group_key].append(i)

    # Identify atoms to keep
    atoms_to_keep = []
    
    for (assembly, abs_group), atom_indices in groups_map.items():
        # Separate atoms by their actual disorder group (including sign)
        group_parts: Dict[int, List[int]] = defaultdict(list)
        
        for idx in atom_indices:
            actual_group = info.disorder_groups[idx]
            group_parts[actual_group].append(idx)
        
        # Process this group's parts
        if len(group_parts) == 1:
            # Only one part in this group, keep all atoms
            for atom_list in group_parts.values():
                atoms_to_keep.extend(atom_list)
        else:
            # Multiple parts exist, apply selection rules
            part_occupancies = {}
            
            # Calculate total occupancy for each part
            for part, indices in group_parts.items():
                total_occ = sum(info.occupancies[i] for i in indices)
                part_occupancies[part] = total_occ
            
            # Special handling for group 0 and 1 (static backbone)
            if 0 in part_occupancies or 1 in part_occupancies:
                # Always keep parts 0 and 1 (static backbone)
                for part, indices in group_parts.items():
                    if part <= 1:  # Part 0 or 1
                        atoms_to_keep.extend(indices)
                    else:
                        # For other parts, select the one with highest occupancy
                        # among the non-backbone parts
                        max_part = max(part_occupancies.keys(), key=lambda x: part_occupancies[x] if x > 1 else float('-inf'))
                        if part == max_part:
                            atoms_to_keep.extend(indices)
            else:
                # No backbone parts, select the part with the highest total occupancy
                max_part = max(part_occupancies.keys(), key=lambda x: part_occupancies[x])
                atoms_to_keep.extend(group_parts[max_part])
    
    # Sort the indices to maintain order
    atoms_to_keep.sort()
    
    # Create new DisorderInfo with only the selected atoms
    new_labels = [info.labels[i] for i in atoms_to_keep]
    new_symbols = [info.symbols[i] for i in atoms_to_keep]
    new_frac_coords = info.frac_coords[atoms_to_keep]
    new_occupancies = [info.occupancies[i] for i in atoms_to_keep]
    new_disorder_groups = [info.disorder_groups[i] for i in atoms_to_keep]
    new_assemblies = [info.assemblies[i] for i in atoms_to_keep] if info.assemblies else [""] * len(atoms_to_keep)
    new_sym_op_indices = [info.sym_op_indices[i] for i in atoms_to_keep] if info.sym_op_indices else [0] * len(atoms_to_keep)
    
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
