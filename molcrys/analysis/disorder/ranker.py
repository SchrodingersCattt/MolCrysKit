"""
Disorder configuration ranker.

This module evaluates physical plausibility of generated structures and ranks them.
"""

import numpy as np
from typing import List, Tuple
from ...structures.molecule import Molecule
from ...structures.crystal import MolecularCrystal


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
    distances = []
    
    # Get all atoms
    all_atoms = []
    for molecule in crystal.molecules:
        all_atoms.extend(molecule.atoms)
    
    # Compute distances between all pairs
    for i, atom1 in enumerate(all_atoms):
        for atom2 in all_atoms[i+1:]:
            # Calculate distance with periodic boundary conditions
            delta = atom1.frac_coords - atom2.frac_coords
            # Minimum image convention
            delta = delta - np.round(delta)
            # Convert to cartesian
            cart_delta = crystal.fractional_to_cartesian(delta)
            distance = np.linalg.norm(cart_delta)
            distances.append(distance)
    
    return distances


def evaluate_steric_clash(crystal: MolecularCrystal, clash_threshold: float = 0.8) -> float:
    """
    Evaluate the severity of steric clashes in a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to evaluate.
    clash_threshold : float, default=0.8
        Distance threshold for considering a clash (in Angstroms).
        
    Returns
    -------
    float
        Steric clash score (higher means more clashes).
    """
    distances = compute_interatomic_distances(crystal)
    
    # Count distances below the clash threshold
    clashes = [d for d in distances if d < clash_threshold]
    
    # Return a simple clash score
    return len(clashes)


def rank_configurations(configurations: List[MolecularCrystal]) -> List[Tuple[MolecularCrystal, float]]:
    """
    Rank configurations by geometric feasibility.
    
    Parameters
    ----------
    configurations : List[MolecularCrystal]
        List of configurations to rank.
        
    Returns
    -------
    List[Tuple[MolecularCrystal, float]]
        List of (configuration, score) tuples, sorted by score (lower is better).
    """
    scored_configs = []
    
    for config in configurations:
        # Evaluate steric clashes (lower is better)
        clash_score = evaluate_steric_clash(config)
        scored_configs.append((config, clash_score))
    
    # Sort by score (lower scores are better)
    scored_configs.sort(key=lambda x: x[1])
    
    return scored_configs


def find_best_configuration(configurations: List[MolecularCrystal]) -> MolecularCrystal:
    """
    Find the most physically plausible configuration.
    
    Parameters
    ----------
    configurations : List[MolecularCrystal]
        List of configurations to evaluate.
        
    Returns
    -------
    MolecularCrystal
        The best configuration according to ranking criteria.
    """
    ranked = rank_configurations(configurations)
    return ranked[0][0] if ranked else None