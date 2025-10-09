"""
Species recognition for molecular crystals.

This module identifies discrete molecular units in periodic crystals.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import List, Tuple
from ..structures.atom import Atom
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


def identify_molecules(crystal: MolecularCrystal) -> List[Molecule]:
    """
    Identify discrete molecular units in a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
        
    Returns
    -------
    List[Molecule]
        List of identified molecular units.
    """
    # Get all atoms from all molecules in the crystal
    all_atoms = []
    for molecule in crystal.molecules:
        all_atoms.extend(molecule.atoms)
    
    if len(all_atoms) == 0:
        return []
    
    # Compute pairwise distances using minimum image convention
    n_atoms = len(all_atoms)
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    
    # Define a simple bonding threshold (in fractional coordinates)
    # This is a simplified approach - a production version would use covalent radii
    bonding_threshold = 0.2
    
    # Build adjacency matrix
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate distance with periodic boundary conditions
            delta = all_atoms[i].frac_coords - all_atoms[j].frac_coords
            
            # Apply minimum image convention
            delta = delta - np.round(delta)
            
            # Calculate distance
            cart_delta = crystal.fractional_to_cartesian(delta)
            distance = np.linalg.norm(cart_delta)
            
            # Check if atoms are bonded
            if distance < bonding_threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    # Find connected components
    sparse_matrix = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(sparse_matrix, directed=False)
    
    # Group atoms by component
    molecules = []
    for component_idx in range(n_components):
        component_atoms = [all_atoms[i] for i in range(n_atoms) if labels[i] == component_idx]
        if component_atoms:
            molecule = Molecule(component_atoms)
            molecules.append(molecule)
    
    return molecules


def assign_atoms_to_molecules(crystal: MolecularCrystal) -> MolecularCrystal:
    """
    Reorganize a crystal by assigning atoms to discrete molecular units.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to reorganize.
        
    Returns
    -------
    MolecularCrystal
        New crystal with atoms organized into molecular units.
    """
    molecules = identify_molecules(crystal)
    return MolecularCrystal(crystal.lattice, molecules, crystal.pbc)