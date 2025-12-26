"""
Operations module for molecular crystals.

This module contains various operations that can be performed on molecular crystals
and their constituent molecules.
"""

from .perturbation import (
    apply_gaussian_displacement_molecule,
    apply_gaussian_displacement_crystal,
    apply_directional_displacement,
    apply_random_rotation,
)

from .rotation import rotate_molecule_at_center, rotate_molecule_at_com

from .builders import (
    create_supercell,
    create_defect_structure,
)

from .surface import generate_topological_slab, TopologicalSlabGenerator

from .hydrogenation import Hydrogenator

__all__ = [
    "apply_gaussian_displacement_molecule",
    "apply_gaussian_displacement_crystal",
    "apply_directional_displacement",
    "apply_random_rotation",
    "rotate_molecule_at_center",
    "rotate_molecule_at_com",
    "create_supercell",
    "create_defect_structure",
    "generate_topological_slab",
    "TopologicalSlabGenerator",
    "Hydrogenator",
]


def add_hydrogens(crystal, rules=None, bond_lengths=None):
    """
    Add hydrogen atoms to a molecular crystal based on geometric rules.

    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to hydrogenate.
    rules : Optional[List[Dict]]
        Override rules for coordination geometry. Format:
        [
            {
                "symbol": str,              # Required. E.g., "O", "N"
                "neighbors": List[str],     # Optional. E.g., ["Cl", "S"]. Context condition.
                "target_coordination": int, # Optional. Override coordination.
                "geometry": str             # Optional. Override geometry.
            },
            ...
        ]
        
        Processing Logic:
        1. Specific rules (with neighbors) take priority
        2. General rules (without neighbors) take second priority
        3. Default rules are used if no user rules match
    bond_lengths : Optional[Dict]
        Override bond lengths for specific atom pairs.

    Returns
    -------
    MolecularCrystal
        New crystal with hydrogen atoms added.
    """
    hydrogenator = Hydrogenator(crystal)
    return hydrogenator.add_hydrogens(rules=rules, bond_lengths=bond_lengths)