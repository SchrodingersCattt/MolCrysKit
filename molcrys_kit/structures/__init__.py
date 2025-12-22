"""
Structural components for MolCrysKit.

This module contains the basic structural classes for representing atoms,
molecules, and crystals.
"""

from .atom import Atom
from .molecule import CrystalMolecule
from .crystal import MolecularCrystal

# For backward compatibility
Molecule = CrystalMolecule