"""
Structural components for MolCrysKit.

This module contains the basic structural classes for representing atoms,
molecules, and crystals.
"""

from .molecule import CrystalMolecule

# For backward compatibility
Molecule = CrystalMolecule
