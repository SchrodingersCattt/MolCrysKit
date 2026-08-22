"""
Structure builders for molecular crystals.

This module provides functionality to build complex structures from simpler units.
"""

import copy
from typing import Tuple

from ..structures.crystal import MolecularCrystal
from .modeling_readiness import (
    require_complete_topology_units,
    warn_if_unresolved_disorder,
)


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
        molecules while preserving per-atom disorder metadata and the
        original ``sym_op_index``/``asym_id`` source provenance.
    """
    report = warn_if_unresolved_disorder(crystal, operation="create_supercell")
    require_complete_topology_units(report, operation="create_supercell")
    result = crystal.get_supercell(*scaling_factors)
    result.metadata = copy.deepcopy(crystal.metadata)
    result.metadata["modeling_readiness"] = report.to_dict()
    result.metadata["supercell"] = {
        "scaling_factors": [int(value) for value in scaling_factors],
        "source_molecule_count": len(crystal.molecules),
        "source_atom_count": sum(len(molecule) for molecule in crystal.molecules),
    }
    return result


__all__ = ["create_supercell"]
