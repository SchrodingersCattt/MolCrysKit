"""
Main entry point for disorder handling pipeline.

This module orchestrates the full disorder handling workflow:
Phase 1 (Data Extraction) -> Phase 2 (Graph Building) -> Phase 3 (Structure Generation)
"""

from typing import List, Optional, Tuple
from ...structures.crystal import MolecularCrystal
from .graph import DisorderGraphBuilder
from .solver import DisorderSolver
from ...io.cif import _pymatgen_cif_parser, scan_cif_disorder  # Correct import from io module


def generate_ordered_replicas_from_disordered_sites(
    filepath: str,
    generate_count: int = 1,
    method: str = "optimal",
    random_seed: Optional[int] = None,
    return_kept_indices: bool = False,
) -> List[MolecularCrystal] | List[Tuple[MolecularCrystal, List[int]]]:
    """
    Process a disordered CIF file through the full disorder handling pipeline.

    Parameters:
    -----------
    filepath : str
        Path to the CIF file
    generate_count : int
        Number of structures to generate for 'random', and an optional
        top-N cap for 'enumerate' when greater than 1.
    method : str
        'optimal' for single best structure, 'random' for occupancy-weighted
        sampling, or 'enumerate' for deterministic PART/SP alternative
        enumeration.
    random_seed : int, optional
        Seed forwarded to 'random' mode for reproducible ensembles.
    return_kept_indices : bool, optional
        When True, return ``(crystal, kept_indices)`` tuples where
        ``kept_indices`` index the source ``DisorderInfo`` arrays selected
        by the solver.

    Returns:
    --------
    List[MolecularCrystal] or List[Tuple[MolecularCrystal, List[int]]]
        List of ordered molecular crystal structures, optionally paired
        with selected source atom indices.
    """
    # Phase 1: Extract raw disorder data
    info = scan_cif_disorder(filepath)

    # Extract lattice matrix from CIF file using pymatgen
    try:
        parser = _pymatgen_cif_parser(filepath)
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
    results = solver.solve(
        num_structures=generate_count,
        method=method,
        random_seed=random_seed,
        return_kept_indices=return_kept_indices,
    )

    return results
