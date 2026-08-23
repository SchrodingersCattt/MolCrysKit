"""
Structure builders for molecular crystals.

This module provides functionality to build complex structures from simpler units.
"""

import copy
import warnings
from typing import Tuple

import numpy as np

from ..analysis.disorder import UnresolvedDisorderWarning
from ..constants.config import KEY_ASSEMBLY, KEY_DISORDER_GROUP, KEY_OCCUPANCY
from ..structures.crystal import MolecularCrystal


def _warn_if_unresolved_disorder(
    crystal: MolecularCrystal,
) -> dict[str, bool | int]:
    nonunit_occupancies = 0
    active_groups = 0
    active_assemblies = 0
    for molecule in crystal.molecules:
        occupancies = np.asarray(molecule.arrays.get(KEY_OCCUPANCY, []), dtype=float)
        nonunit_occupancies += int(
            np.count_nonzero(~np.isclose(occupancies, 1.0, rtol=0.0, atol=1e-8))
        )
        active_groups += sum(
            str(value).strip() not in {"", ".", "?", "0"}
            for value in molecule.arrays.get(KEY_DISORDER_GROUP, [])
        )
        active_assemblies += sum(
            str(value).strip() not in {"", ".", "?", "0"}
            for value in molecule.arrays.get(KEY_ASSEMBLY, [])
        )
    stats = {
        "all_atom_ordered": not bool(
            nonunit_occupancies or active_groups or active_assemblies
        ),
        "nonunit_occupancy_count": nonunit_occupancies,
        "active_disorder_group_count": active_groups,
        "active_disorder_assembly_count": active_assemblies,
    }
    if not stats["all_atom_ordered"]:
        warnings.warn(
            "create_supercell is continuing with unresolved disorder; resolve an "
            "ordered replica before production MD.",
            UnresolvedDisorderWarning,
            stacklevel=2,
        )
    return stats


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
    input_disorder = _warn_if_unresolved_disorder(crystal)
    result = crystal.get_supercell(*scaling_factors)
    result.metadata = copy.deepcopy(crystal.metadata)
    result.metadata["input_disorder"] = input_disorder
    result.metadata["supercell"] = {
        "scaling_factors": [int(value) for value in scaling_factors],
        "source_molecule_count": len(crystal.molecules),
        "source_atom_count": sum(len(molecule) for molecule in crystal.molecules),
    }
    return result


__all__ = ["create_supercell"]
