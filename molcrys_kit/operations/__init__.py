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

from .surface import (
    generate_topological_slab,
    TopologicalSlabGenerator,
    TerminationInfo,
    enumerate_terminations,
    generate_slabs_with_terminations,
    get_surface_basis,
)

from .hydrogen_completion import HydrogenCompleter, add_hydrogens

from .desolvation import Desolvator, remove_solvents

from .defects import VacancyGenerator, generate_vacancy

from .molecule_manipulation import (
    MoleculeManipulator,
    MoleculeClashError,
    translate_molecule,
    rotate_molecule,
    replace_molecule,
)

from .cluster import ClusterCarver, LigandTopologyOverflowError, carve_cluster

from .interpolation import (
    InterpolationConfig,
    InterpolationMethod,
    MoleculeMatch,
    VCMoleculeMatch,
    best_atom_mapping,
    find_flipping_molecules,
    interpolate_crystal,
    interpolate_crystal_vc,
    interpolate_molecule,
    interpolate_pose,
    match_molecules,
    match_molecules_vc,
)

from .path_smoothing import smooth_interpolation_idpp

from .reorientation import reorient_crystal, ReorientationInfo

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
    "TerminationInfo",
    "enumerate_terminations",
    "generate_slabs_with_terminations",
    "HydrogenCompleter",
    "add_hydrogens",
    "Desolvator",
    "remove_solvents",
    "VacancyGenerator",
    "generate_vacancy",
    "MoleculeManipulator",
    "MoleculeClashError",
    "translate_molecule",
    "rotate_molecule",
    "replace_molecule",
    "ClusterCarver",
    "LigandTopologyOverflowError",
    "carve_cluster",
    "InterpolationConfig",
    "InterpolationMethod",
    "MoleculeMatch",
    "best_atom_mapping",
    "find_flipping_molecules",
    "interpolate_crystal",
    "interpolate_crystal_vc",
    "interpolate_molecule",
    "interpolate_pose",
    "match_molecules",
    "match_molecules_vc",
    "VCMoleculeMatch",
    "smooth_interpolation_idpp",
    "get_surface_basis",
    "reorient_crystal",
    "ReorientationInfo",
]
