"""
Operations module for MolCrysKit.

This module provides structure transformation operations for molecular crystals.
"""

from .rotation import rotation_matrix, rotate_molecule, translate_molecule, euler_rotation_matrix
from .perturbation import (
    apply_gaussian_displacement_atom, 
    apply_gaussian_displacement_molecule,
    apply_gaussian_displacement_crystal,
    apply_anisotropic_displacement,
    apply_directional_displacement
)
from .builders import (
    build_supercell,
    build_surface,
    build_defect_crystal,
    create_multilayer_structure
)

__all__ = [
    "rotation_matrix",
    "rotate_molecule",
    "translate_molecule",
    "euler_rotation_matrix",
    "apply_gaussian_displacement_atom",
    "apply_gaussian_displacement_molecule",
    "apply_gaussian_displacement_crystal",
    "apply_anisotropic_displacement",
    "apply_directional_displacement",
    "build_supercell",
    "build_surface",
    "build_defect_crystal",
    "create_multilayer_structure"
]