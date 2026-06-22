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
]
