"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
"""

import numpy as np
from typing import List, Tuple
try:
    from pymatgen.io.cif import CifParser
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

from ..structures.atom import Atom
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


def parse_cif(filepath: str) -> MolecularCrystal:
    """
    Parse a CIF file into a MolecularCrystal object.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure.
        
    Raises
    ------
    ImportError
        If pymatgen is not available.
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for CIF parsing. Please install it with 'pip install pymatgen'")
    
    # Parse the CIF file using pymatgen
    parser = CifParser(filepath)
    structures = parser.get_structures()
    
    # For simplicity, we take the first structure
    structure = structures[0]
    
    # Extract lattice vectors
    lattice = structure.lattice.matrix
    
    # Extract atoms
    atoms = []
    for site in structure.sites:
        atom = Atom(
            symbol=site.species_string,
            frac_coords=site.frac_coords,
            occupancy=site.properties.get('occupation', 1.0)
        )
        atoms.append(atom)
    
    # For now, we put all atoms in a single molecule
    # A more sophisticated implementation would group atoms into molecules
    molecule = Molecule(atoms=atoms)
    
    # Assuming periodic boundary conditions in all directions
    pbc = (True, True, True)
    
    return MolecularCrystal(lattice, [molecule], pbc)


def parse_cif_advanced(filepath: str) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.
    
    This function attempts to identify discrete molecular units within the crystal.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    # This would be implemented in the species.py module
    # For now, we delegate to the basic parser
    return parse_cif(filepath)