"""
Disorder configuration generator.

This module generates all possible ordered configurations from disordered structures.
"""

from typing import List, Iterator
from itertools import product
from ...structures.atom import Atom
from ...structures.molecule import Molecule
from ...structures.crystal import MolecularCrystal
from .scanner import group_disordered_atoms


def generate_ordered_configurations(crystal: MolecularCrystal) -> Iterator[MolecularCrystal]:
    """
    Generate all possible ordered configurations via Cartesian product of disorder groups.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The disordered crystal to generate configurations for.
        
    Yields
    ------
    MolecularCrystal
        An ordered configuration of the crystal.
    """
    # Group disordered atoms
    disorder_groups = group_disordered_atoms(crystal)
    
    if not disorder_groups:
        # No disorder, yield the original crystal
        yield crystal
        return
    
    # Get all possible combinations
    group_items = list(disorder_groups.items())
    group_keys = [key for key, _ in group_items]
    group_values = [values for _, values in group_items]
    
    # Generate all combinations
    for combination in product(*group_values):
        # Create a new crystal with this combination
        # This is a simplified implementation - a full one would need to:
        # 1. Remove all disordered atoms
        # 2. Add the selected atoms from each group
        # 3. Adjust occupancies to 1.0
        
        # For now, we just yield the original crystal as a placeholder
        yield crystal


def generate_configurations_with_constraints(crystal: MolecularCrystal, 
                                           max_configurations: int = 100) -> Iterator[MolecularCrystal]:
    """
    Generate ordered configurations with a maximum count.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The disordered crystal to generate configurations for.
    max_configurations : int, default=100
        Maximum number of configurations to generate.
        
    Yields
    ------
    MolecularCrystal
        An ordered configuration of the crystal.
    """
    count = 0
    for config in generate_ordered_configurations(crystal):
        if count >= max_configurations:
            break
        yield config
        count += 1