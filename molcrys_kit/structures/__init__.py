"""
Structural components for MolCrysKit.

This module contains the basic structural classes for representing atoms,
molecules, and crystals.
"""

from .molecule import CrystalMolecule
from .atom import Atom
from .crystal import MolecularCrystal

# For backward compatibility
Molecule = CrystalMolecule