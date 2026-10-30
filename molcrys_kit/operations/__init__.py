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

from .bond_rotation import (
    BondNotFoundError,
    BondPartition,
    BondRotationError,
    BondRotationSelectionError,
    RingBondRotationError,
    partition_at_bond,
    rotate_fragment_about_bond,
    rotate_fragment_in_crystal,
)

from .rotation import rotate_molecule_at_center, rotate_molecule_at_com

from .builders import (
    assemble_replica_supercell,
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

from .conformational_interpolation import interpolate_molecule_with_internal_dofs

from .reorientation import reorient_crystal, ReorientationInfo

from .reactive_path import (
    BondChange,
    ReactivePathConfig,
    ReactivePathResult,
    RigidGroup,
    interpolate_reactive_path,
)
from .periodic_chain import build_periodic_chains

from .forward import (
    AverageAtomSite,
    ConstructionValidationError,
    CrystalConstructionContext,
    DISORDER_COMPANION_SCHEMA,
    DISORDER_CONSTRAINT_SCHEMA,
    DisorderCompanionLossError,
    ForwardConstructionContext,
    IncompatibleChemistryError,
    IncompatibleSiteError,
    InvalidOrientationError,
    InvalidSymmetryError,
    MolecularAlternative,
    MolecularPose,
    MolecularTemplate,
    PeriodicBondRecord,
    SiteSymmetryAnalysis,
    StatisticalDisorderModel,
    analyze_molecular_pose,
    analyze_site_symmetry,
    build_molecular_crystal,
    build_orientational_disorder,
    build_symmetry_complete_crystal,
    construct_molecular_crystal,
    construct_statistical_disorder,
    intermolecular_contacts,
    prepare_construction_context,
    prepare_context_from_cif,
    read_statistical_disorder_companion,
    select_ordered_realization,
)

from .chemical_edit import (
    ChemicalAtomMapping,
    ChemicalEdit,
    CrystalEditSpec,
    CrystalInventory,
    PlacementPolicy,
    PackingTransferPlan,
    CounterfactualStatus,
    CounterfactualResult,
    RelaxationProtocol,
    apply_chemical_edit,
    transfer_packing,
    periodic_clash_diagnostics,
    write_cf0,
)

__all__ = [
    "ChemicalAtomMapping",
    "ChemicalEdit",
    "CrystalEditSpec",
    "CrystalInventory",
    "PlacementPolicy",
    "PackingTransferPlan",
    "CounterfactualStatus",
    "CounterfactualResult",
    "RelaxationProtocol",
    "apply_chemical_edit",
    "transfer_packing",
    "periodic_clash_diagnostics",
    "write_cf0",
    "apply_gaussian_displacement_molecule",
    "apply_gaussian_displacement_crystal",
    "apply_directional_displacement",
    "apply_random_rotation",
    "BondNotFoundError",
    "BondPartition",
    "BondRotationError",
    "BondRotationSelectionError",
    "RingBondRotationError",
    "partition_at_bond",
    "rotate_fragment_about_bond",
    "rotate_fragment_in_crystal",
    "rotate_molecule_at_center",
    "rotate_molecule_at_com",
    "assemble_replica_supercell",
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
    "interpolate_molecule_with_internal_dofs",
    "VCMoleculeMatch",
    "get_surface_basis",
    "reorient_crystal",
    "ReorientationInfo",
    "BondChange",
    "ReactivePathConfig",
    "ReactivePathResult",
    "RigidGroup",
    "interpolate_reactive_path",
    "build_periodic_chains",
    "AverageAtomSite",
    "ConstructionValidationError",
    "CrystalConstructionContext",
    "DISORDER_COMPANION_SCHEMA",
    "DISORDER_CONSTRAINT_SCHEMA",
    "DisorderCompanionLossError",
    "ForwardConstructionContext",
    "IncompatibleChemistryError",
    "IncompatibleSiteError",
    "InvalidOrientationError",
    "InvalidSymmetryError",
    "MolecularAlternative",
    "MolecularPose",
    "MolecularTemplate",
    "PeriodicBondRecord",
    "SiteSymmetryAnalysis",
    "StatisticalDisorderModel",
    "analyze_molecular_pose",
    "analyze_site_symmetry",
    "build_molecular_crystal",
    "build_orientational_disorder",
    "build_symmetry_complete_crystal",
    "construct_molecular_crystal",
    "construct_statistical_disorder",
    "intermolecular_contacts",
    "prepare_construction_context",
    "prepare_context_from_cif",
    "read_statistical_disorder_companion",
    "select_ordered_realization",
]
