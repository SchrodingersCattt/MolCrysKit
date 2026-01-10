"""
Heuristic rules engine for hydrogenation of molecular crystals.

This module provides functions to determine hydrogenation strategies based on
chemical environments, coordination numbers, and ring structures.
"""

from typing import Dict, Tuple
from ..constants.config import BOND_LENGTHS


def determine_hydrogenation_needs(atom_symbol: str, env_stats: Dict, ring_info: Dict) -> Dict:
    """
    Determine hydrogenation strategy for an atom based on its environment.
    
    Parameters
    ----------
    atom_symbol : str
        Atomic symbol (e.g., 'C', 'N', 'O')
    env_stats : dict
        Local geometry statistics from ChemicalEnvironment.get_local_geometry_stats
    ring_info : dict
        Ring information from ChemicalEnvironment.detect_ring_info
        
    Returns
    -------
    dict
        Contains:
        - 'num_h': number of hydrogens to add
        - 'geometry': geometry type for hydrogen placement
        - 'bond_length': bond length for the new H
    """
    coordination = env_stats['coordination_number']
    is_planar = env_stats['is_planar']
    in_ring = ring_info['in_ring']
    ring_sizes = ring_info['ring_sizes']
    is_ring_planar = ring_info['is_ring_planar']
    
    # Default values
    num_h = 0
    geometry = 'tetrahedral'
    bond_length = 1.0
    
    # Get the appropriate bond length
    if f"{atom_symbol}-H" in BOND_LENGTHS:
        bond_length = BOND_LENGTHS[f"{atom_symbol}-H"]
    
    # Carbon rules
    if atom_symbol == 'C':
        if coordination == 3:
            if is_planar or (in_ring and is_ring_planar):
                # Target sp2 (Total coordination 3) - already satisfied by double bonds
                # For standard hydrogenation, if input is skeleton missing H:
                num_h = 0  # Assuming valence satisfied by double bonds
                geometry = 'trigonal_planar'
        elif coordination == 2:
            if is_planar:  # angle ~120 -> Target sp2 -> Add 1 H (in-plane)
                num_h = 1
                geometry = 'trigonal_planar'  # This will be handled specially for 2 neighbors
            else:  # angle ~109 -> Target sp3 -> Add 2 H (tetrahedral)
                num_h = 2
                geometry = 'tetrahedral'
        elif coordination == 1:
            # triple bond or cumulene, likely sp hybridized
            num_h = 1
            geometry = 'linear'
    
    # Nitrogen rules
    elif atom_symbol == 'N':
        if coordination == 2:
            # Pyridine-like: N in 6-membered ring with planar structure -> Don't add H
            if in_ring and 6 in ring_sizes and is_ring_planar:
                num_h = 0
                geometry = 'planar_aromatic'  # Special case for pyridine N
            # Pyrrole-like: N in 5-membered ring with planar structure -> Add 1 H in plane
            elif in_ring and 5 in ring_sizes and is_ring_planar:
                num_h = 1
                geometry = 'planar_bisector'  # Special case for pyrrole H placement
            # Amine-like: N with 2 neighbors but not in planar ring -> Add 1 H
            else:
                num_h = 1
                geometry = 'tetrahedral'  # Like in amine, pyramidal
        elif coordination == 1:
            # Likely amide or similar, could add 2 H for primary amine
            num_h = 2
            geometry = 'trigonal_pyramidal'
    
    # Oxygen rules
    elif atom_symbol == 'O':
        if coordination == 1:
            # Usually alcohols, phenols, ethers, carbonyls, etc.
            # For alcohol/ether: add 1 H to make water/alcohol
            # For carbonyl: would need to reduce first
            num_h = 1
            geometry = 'bent'  # Or 'tetrahedral' if considering 2 lone pairs
        elif coordination == 0:
            # Free oxygen, could potentially add 2 H to make water
            num_h = 2
            geometry = 'bent'
    
    # Other elements (simplified rules)
    elif atom_symbol == 'S':
        if coordination == 1:
            # Thiol group, add 1 H
            num_h = 1
            geometry = 'bent'
    
    return {
        'num_h': num_h,
        'geometry': geometry,
        'bond_length': bond_length
    }