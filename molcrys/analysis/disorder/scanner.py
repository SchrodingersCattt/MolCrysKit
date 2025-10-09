"""
Disorder scanning for molecular crystals.

This module identifies atoms with partial occupancy in crystal structures.
"""

from typing import List, Dict
from collections import defaultdict
from ...structures.atom import Atom
from ...structures.molecule import Molecule
from ...structures.crystal import MolecularCrystal


def identify_disordered_atoms(crystal: MolecularCrystal) -> List[Atom]:
    """
    Identify atoms with partial occupancy.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
        
    Returns
    -------
    List[Atom]
        List of atoms with occupancy < 1.0.
    """
    disordered_atoms = []
    
    for molecule in crystal.molecules:
        for atom in molecule.atoms:
            if atom.occupancy < 1.0:
                disordered_atoms.append(atom)
    
    return disordered_atoms


def group_disordered_atoms(crystal: MolecularCrystal) -> Dict[str, List[Atom]]:
    """
    Group disordered atoms by site.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
        
    Returns
    -------
    Dict[str, List[Atom]]
        Dictionary mapping site identifiers to lists of atoms.
    """
    # In a real implementation, we would use CIF disorder labels
    # For now, we group atoms by their fractional coordinates (rounded)
    disordered_groups = defaultdict(list)
    
    for molecule in crystal.molecules:
        for atom in molecule.atoms:
            if atom.occupancy < 1.0:
                # Create a site identifier based on rounded coordinates
                site_id = f"{atom.symbol}_{round(atom.frac_coords[0]*100)}_{round(atom.frac_coords[1]*100)}_{round(atom.frac_coords[2]*100)}"
                disordered_groups[site_id].append(atom)
    
    return dict(disordered_groups)


def has_disorder(crystal: MolecularCrystal) -> bool:
    """
    Check if a crystal structure contains disorder.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to check.
        
    Returns
    -------
    bool
        True if the crystal contains disordered atoms, False otherwise.
    """
    for molecule in crystal.molecules:
        for atom in molecule.atoms:
            if atom.occupancy < 1.0:
                return True
    return False