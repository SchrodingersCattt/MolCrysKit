"""
Main entry point for disorder handling pipeline.

This module orchestrates the full disorder handling workflow:
Phase 1 (Data Extraction) -> Phase 2 (Graph Building) -> Phase 3 (Structure Generation)
"""

from typing import List, Optional, Tuple, Union
from ...structures.crystal import MolecularCrystal
from .graph import DisorderGraphBuilder
from .solver import DisorderSolver
from ...io.cif import _pymatgen_cif_parser, scan_cif_disorder, DisorderInfo


def generate_ordered_replicas_from_disordered_sites(
    filepath: Optional[str] = None,
    generate_count: int = 1,
    method: str = "optimal",
    random_seed: Optional[int] = None,
    return_kept_indices: bool = False,
    coupled: bool = False,
    *,
    crystal: Optional[MolecularCrystal] = None,
) -> Union[List[MolecularCrystal], List[Tuple[MolecularCrystal, List[int]]]]:
    """
    Process a disordered structure through the full disorder handling pipeline.

    The disorder metadata can come from either a CIF file (``filepath``) or
    a ``MolecularCrystal`` that already carries the necessary per-atom arrays
    (e.g. loaded from an extxyz file written by :func:`write_extxyz`).
    Exactly one of ``filepath`` and ``crystal`` must be provided.

    Parameters
    ----------
    filepath : str, optional
        Path to the CIF file.  Used when ``crystal`` is *None*.
    generate_count : int
        Number of structures to generate for 'random', and a top-N cap for
        'enumerate' when positive.
    method : str
        'optimal' for single best structure, 'random' for occupancy-weighted
        sampling, or 'enumerate' for deterministic PART/SP alternative
        enumeration.
    random_seed : int, optional
        Seed forwarded to 'random' mode for reproducible ensembles.
    return_kept_indices : bool, optional
        When True, return ``(crystal, kept_indices)`` tuples.
    coupled : bool, optional
        When False (default), symmetry-expanded copies of the same disorder
        assembly make independent PART/orientation decisions.  When True,
        preserve the legacy behaviour that locks symmetry copies together.
    crystal : MolecularCrystal, optional
        Pre-loaded crystal carrying disorder metadata in its per-atom arrays.
        When provided, ``filepath`` is ignored and no CIF is read from disk.

    Returns
    -------
    List[MolecularCrystal] or List[Tuple[MolecularCrystal, List[int]]]
        List of ordered molecular crystal structures, optionally paired
        with selected source atom indices.

    Raises
    ------
    ValueError
        If neither ``filepath`` nor ``crystal`` is provided.
    """
    if crystal is not None:
        # --- in-memory path: reconstruct DisorderInfo from crystal arrays ---
        info = DisorderInfo.from_crystal(crystal)
        lattice_matrix = info.lattice_matrix
    elif filepath is not None:
        # --- CIF path (backward compatible) ---
        info = scan_cif_disorder(filepath)
        lattice_matrix = info.lattice_matrix
        if lattice_matrix is None:
            # Fallback: parse lattice via pymatgen (should not happen with
            # updated scan_cif_disorder, but keeps backward compat).
            parser = _pymatgen_cif_parser(filepath)
            structure = parser.parse_structures()[0]
            lattice_matrix = structure.lattice.matrix
    else:
        raise ValueError(
            "Either 'filepath' (CIF path) or 'crystal' (MolecularCrystal) "
            "must be provided."
        )

    # Phase 2: Build exclusion graph
    builder = DisorderGraphBuilder(info, lattice_matrix, coupled=coupled)
    graph = builder.build()

    # Phase 3: Generate ordered structures
    solver = DisorderSolver(info, graph, lattice_matrix, coupled=coupled)
    results = solver.solve(
        num_structures=generate_count,
        method=method,
        random_seed=random_seed,
        return_kept_indices=return_kept_indices,
    )

    return results
