"""
Structural components for MolCrysKit.

This module contains the basic structural classes for representing atoms,
molecules, and crystals.
"""

from .molecule import CrystalMolecule
from .atom import MolAtom
from .crystal import MolecularCrystal
from .cluster import CrystalCluster, ClusterProvenance
from .polyhedra import all_ideal_polyhedra, convex_hull_payload, ideal_polyhedra_for_cn
from .trajectory import CrystalTrajectory
from .records import BondRecord, SiteRecord
from .bond import (
    BondCandidates,
    BondPairs,
    VerletBondTracker,
    build_bond_candidates,
    candidate_list_needs_rebuild,
    evaluate_bond_candidates,
    infer_bond_pairs,
)
from .symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
    LatticeBasisChange,
    domain_representatives,
    identity_operation,
    left_cosets,
    validate_subgroup,
)

# For backward compatibility
Molecule = CrystalMolecule

__all__ = [
    "MolAtom",
    "CrystalMolecule",
    "MolecularCrystal",
    "CrystalCluster",
    "ClusterProvenance",
    "Molecule",
    "all_ideal_polyhedra",
    "convex_hull_payload",
    "ideal_polyhedra_for_cn",
    "CrystalTrajectory",
    "BondRecord",
    "BondCandidates",
    "BondPairs",
    "SiteRecord",
    "VerletBondTracker",
    "build_bond_candidates",
    "candidate_list_needs_rebuild",
    "evaluate_bond_candidates",
    "infer_bond_pairs",
    "CrystalSymmetry",
    "FractionalAffineOperation",
    "LatticeBasisChange",
    "domain_representatives",
    "identity_operation",
    "left_cosets",
    "validate_subgroup",
]
