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

from .defects import VoidCarver, VacancyGenerator, carve_void, generate_vacancy

from .molecule_manipulation import (
    MoleculeManipulator,
    MoleculeClashError,
    translate_molecule,
    rotate_molecule,
    replace_molecule,
)

from .cluster import ClusterCarver, LigandTopologyOverflowError, carve_cluster

from .nanocluster import (
    DEFAULT_NANOCLUSTER_BATCH_SIZE,
    NanoClusterCarver,
    NanoShape,
    carve_nanocluster,
)
from .implicit_shape import DEFAULT_SHAPE_BATCH_SIZE, ImplicitShape

from .symmetry_path import (
    AtomCorrespondence,
    CrystalCorrespondence,
    RigidReachabilityError,
    RigidReachabilityTolerance,
    SymmetryMoleculeMatch,
    SymmetryPathConfig,
    SymmetryPathPlan,
    SymmetryPathProvenance,
    build_symmetry_path_plan,
    generate_collective_symmetry_path,
    interpolate_symmetry_path,
    transform_crystal_fractional,
)

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

from .reorientation import reorient_crystal, ReorientationInfo

__all__ = [
    "apply_gaussian_displacement_molecule",
    "apply_gaussian_displacement_crystal",
    "apply_directional_displacement",
    "apply_random_rotation",
    "rotate_molecule_at_center",
    "rotate_molecule_at_com",
    "create_supercell",
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
    "VoidCarver",
    "carve_void",
    "generate_vacancy",
    "MoleculeManipulator",
    "MoleculeClashError",
    "translate_molecule",
    "rotate_molecule",
    "replace_molecule",
    "ClusterCarver",
    "LigandTopologyOverflowError",
    "carve_cluster",
    "DEFAULT_NANOCLUSTER_BATCH_SIZE",
    "DEFAULT_SHAPE_BATCH_SIZE",
    "ImplicitShape",
    "NanoClusterCarver",
    "NanoShape",
    "carve_nanocluster",
    "AtomCorrespondence",
    "CrystalCorrespondence",
    "RigidReachabilityError",
    "RigidReachabilityTolerance",
    "SymmetryMoleculeMatch",
    "SymmetryPathConfig",
    "SymmetryPathPlan",
    "SymmetryPathProvenance",
    "build_symmetry_path_plan",
    "generate_collective_symmetry_path",
    "interpolate_symmetry_path",
    "transform_crystal_fractional",
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
    "get_surface_basis",
    "reorient_crystal",
    "ReorientationInfo",
]
