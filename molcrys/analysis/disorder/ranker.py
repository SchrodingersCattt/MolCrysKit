"""
Disorder configuration ranking.

This module provides functions to rank disorder configurations based on physical plausibility.
"""

import numpy as np
from typing import List, Tuple
try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder

from ..structures.atom import Atom
from ..structures.crystal import MolecularCrystal


def compute_interatomic_distances(crystal: MolecularCrystal) -> List[float]:
    """
    Compute all interatomic distances in the crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
        
    Returns
    -------
    List[float]
        List of all interatomic distances.
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for distance computation. Please install it with 'pip install ase'")
    
    distances = []
    
    # Get all atoms
    all_atoms = []
    for molecule in crystal.molecules:
        all_atoms.extend(molecule)
    
    # Compute distances between all pairs
    for i, atom1 in enumerate(all_atoms):
        pos1 = atom1.position
        symbol1 = atom1.symbol
        for atom2 in all_atoms[i+1:]:
            pos2 = atom2.position
            symbol2 = atom2.symbol
            
            # Calculate distance with periodic boundary conditions
            # For now, we use a simplified approach
            delta = pos1 - pos2
            distance = np.linalg.norm(delta)
            distances.append(distance)
    
    return distances


def evaluate_steric_clash(crystal: MolecularCrystal, clash_threshold: float = 0.8) -> float:
    """
    Evaluate steric clashes in the crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to evaluate.
    clash_threshold : float, default=0.8
        Distance threshold for considering a clash (in Angstroms).
        
    Returns
    -------
    float
        Number of steric clashes.
    """
    distances = compute_interatomic_distances(crystal)
    
    # Count distances below the clash threshold
    clashes = [d for d in distances if d < clash_threshold]
    return len(clashes)


def rank_configurations(configurations: List[MolecularCrystal]) -> List[Tuple[MolecularCrystal, float]]:
    """
    Rank disorder configurations by steric clash count.
    
    Parameters
    ----------
    configurations : List[MolecularCrystal]
        List of crystal configurations to rank.
        
    Returns
    -------
    List[Tuple[MolecularCrystal, float]]
        List of (crystal, clash_count) tuples sorted by clash count.
    """
    ranked = []
    for crystal in configurations:
        clashes = evaluate_steric_clash(crystal)
        ranked.append((crystal, clashes))
    
    # Sort by clash count (ascending - fewer clashes is better)
    ranked.sort(key=lambda x: x[1])
    return ranked


def find_best_configuration(configurations: List[MolecularCrystal]) -> MolecularCrystal:
    """
    Find the best disorder configuration based on steric clash count.
    
    Parameters
    ----------
    configurations : List[MolecularCrystal]
        List of crystal configurations to evaluate.
        
    Returns
    -------
    MolecularCrystal
        The configuration with the fewest steric clashes.
    """
    ranked = rank_configurations(configurations)
    return ranked[0][0]