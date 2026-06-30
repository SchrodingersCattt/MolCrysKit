"""
Structure builders for molecular crystals.

This module provides functionality to build complex structures from simpler units.
"""

from typing import Tuple

import numpy as np

from ..structures.crystal import MolecularCrystal


def create_supercell(
    crystal: MolecularCrystal, scaling_factors: Tuple[int, int, int]
) -> MolecularCrystal:
    """
    Create a supercell by replicating the unit cell.

    Parameters
    ----------
    crystal : MolecularCrystal
        The unit cell.
    scaling_factors : Tuple[int, int, int]
        Scaling factors for each lattice vector.

    Returns
    -------
    MolecularCrystal
        Supercell structure.  Delegates to
        :meth:`MolecularCrystal.get_supercell` which internally copies
        molecules (preserving all per-atom disorder metadata arrays) and
        offsets ``sym_op_index``/``asym_id`` across repeated cells to
        prevent cross-copy merging in the disorder solver.
    """
    return crystal.get_supercell(*scaling_factors)


def create_defect_structure(
    crystal: MolecularCrystal, defect_type: str, defect_position: np.ndarray
) -> MolecularCrystal:
    """
    Create a crystal with a specific defect.

    Parameters
    ----------
    crystal : MolecularCrystal
        The perfect crystal.
    defect_type : str
        Type of defect ('vacancy', 'interstitial', etc.).
    defect_position : np.ndarray
        Position of the defect.

    Returns
    -------
    MolecularCrystal
        Crystal structure with the defect.
    """

    # This is a simplified placeholder implementation
    # A real implementation would modify the crystal according to the defect type

    # For demonstration, we'll just return the original crystal
    return crystal


__all__ = ["create_supercell", "create_defect_structure"]
