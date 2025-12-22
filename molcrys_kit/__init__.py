"""
MolCrysKit: A Python toolkit for molecular crystal analysis and manipulation.

This toolkit provides functionality for parsing, analyzing, and manipulating
molecular crystal structures, with a particular focus on molecular crystals
where well-defined molecules occupy crystallographic sites.
"""

__version__ = "0.1.0"

from .structures.atom import Atom
from .structures.molecule import CrystalMolecule
from .structures.crystal import MolecularCrystal
from .io.cif import read_mol_crystal

# For backward compatibility
Molecule = CrystalMolecule

__all__ = [
    "Atom",
    "CrystalMolecule",
    "MolecularCrystal",
    "read_mol_crystal",
    # Backward compatibility
    "Molecule",
]
