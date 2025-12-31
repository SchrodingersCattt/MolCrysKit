"""
Main entry point for disorder handling pipeline.

This module orchestrates the full disorder handling workflow:
Phase 1 (Data Extraction) -> Phase 2 (Graph Building) -> Phase 3 (Structure Generation)
"""

from typing import List
import numpy as np
from ...structures.crystal import MolecularCrystal
from .graph import DisorderGraphBuilder
from .solver import DisorderSolver
from ...io.cif import scan_cif_disorder  # Correct import from io module


def process_disordered_cif(
    filepath: str, 
    generate_count: int = 1,
    method: str = 'optimal'
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
        
    Returns:
    --------
    List[MolecularCrystal]
        List of ordered molecular crystal structures
    """
    # Phase 1: Extract raw disorder data
    info = scan_cif_disorder(filepath)
    
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