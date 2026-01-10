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
    coord = env_stats['coordination_number']
    avg_len = env_stats['average_bond_length']
    
    # Defaults
    num_h = 0
    geometry = 'tetrahedral'
    bond_length = BOND_LENGTHS.get(f"{atom_symbol}-H", 1.0)
    
    # Carbon rules
    if atom_symbol == 'C':
        if coord == 3:
            # Case: sp2 (Planar, ~360 sum) vs sp3 (Pyramidal, ~328.5 sum)
            # Previous logic was flawed: it prioritized ring planarity over local geometry
            # New logic: Local geometry takes precedence over global ring properties
            
            angle_sum = env_stats['bond_angle_sum']
            # --- NEW LOGIC: Local Geometry First ---
            
            # 1. Definitely sp3 region (Pyramidal)
            # Ideal tetrahedral is 328.5 degrees. With tolerance to 345 degrees.
            # If less than this value, regardless of ring environment, it must be pyramidal.
            if angle_sum < 345.0:
                num_h = 1
                geometry = 'tetrahedral'
                
            # 2. Definitely sp2 region (Planar)
            # Close to 360 degrees, definitely planar.
            elif angle_sum > 355.0:
                num_h = 0
                geometry = 'trigonal_planar'
                
            # 3. Ambiguous region (Distorted/Intermediate)
            # E.g. 348 degrees. Could be a strained sp3 or a distorted sp2.
            # Only in this case do we consider the ring environment for arbitration.
            else:
                if ring_info['in_ring'] and ring_info['is_ring_planar']:
                    # Ring is planar -> likely aromatic/conjugated system -> sp2
                    num_h = 0
                    geometry = 'trigonal_planar'
                else:
                    # Ring not planar, or not in ring -> default to sp3
                    num_h = 1
                    geometry = 'tetrahedral'

        elif coord == 2:
            # Case: -CH2- (sp3) vs =CH- (sp2 aromatic) vs =C= (sp linear)
            angle = env_stats['bond_angle_single']
            
            # --- SCORING SYSTEM ---
            # Ideal models
            # sp:   Angle 180,   Len ~1.2-1.3 (cumulene)
            # sp2:  Angle 120,   Len ~1.34-1.42 (aromatic)
            # sp3:  Angle 109.5, Len ~1.50-1.54 (aliphatic)
            
            # 1. Angle Penalty (Weighted heavily)
            score_sp   = abs(angle - 180.0)
            score_sp2  = abs(angle - 120.0)
            score_sp3  = abs(angle - 109.5)
            
            # 2. Bond Length Bias (Adjust scores based on length)
            # If length > 1.46 (typical single bond), heavily penalize sp/sp2
            if avg_len > 1.46: 
                score_sp3 -= 15.0  # Strong bonus for sp3
                score_sp  += 20.0  # Penalty for sp
            # If length < 1.38 (typical double/aromatic), penalize sp3
            elif avg_len < 1.38:
                score_sp2 -= 10.0  # Bonus for sp2
                score_sp3 += 20.0  # Penalty for sp3
            
            # 3. Decision
            best_match = min(score_sp, score_sp2, score_sp3)
            
            if best_match == score_sp and score_sp < 15.0: # Must be reasonably close
                num_h = 0
                geometry = 'linear'
            elif best_match == score_sp2:
                num_h = 1
                geometry = 'trigonal_planar'
            else:
                # Default to sp3 if ambiguous or matches sp3 best
                num_h = 2
                geometry = 'tetrahedral'
                
        elif coord == 1:
            # Case: Methyl (-CH3, sp3) vs Alkyne (-C#C-H, sp)
            # Threshold adjusted from 1.35 to 1.28 to be safe.
            # C-N single is ~1.47, C-C single ~1.54.
            # C#C triple is ~1.20. C#N triple is ~1.16.
            # Only strictly short bonds should be linear.
            if avg_len < 1.28 and avg_len > 0.1: 
                num_h = 1
                geometry = 'linear'
            else:
                num_h = 3
                geometry = 'tetrahedral'
    
    # Nitrogen rules
    elif atom_symbol == 'N':
        if coord == 2:
            # Pyridine (sp2, 0H) vs Pyrrole (sp2, 1H) vs Amine (sp3, 1H)
            in_ring = ring_info['in_ring']
            is_planar_ring = ring_info['is_ring_planar']
            ring_sizes = ring_info['ring_sizes']
            
            if in_ring and is_planar_ring:
                if 6 in ring_sizes: # Pyridine-like
                    num_h = 0
                    geometry = 'planar_aromatic'
                elif 5 in ring_sizes: # Pyrrole-like
                    num_h = 1
                    geometry = 'planar_bisector' # Use the corrected geometry
                else:
                    # Generic planar ring N? Likely sp2 conjugated.
                    num_h = 0 
            else:
                # Amine-like (Secondary amine)
                num_h = 1
                geometry = 'tetrahedral' # Pyramidal
                
        elif coord == 1:
            # Primary amine (-NH2) or Amide
            num_h = 2
            geometry = 'tetrahedral'
    
    # Oxygen rules
    elif atom_symbol == 'O':
        if coord == 1:
            if avg_len < 1.4:
                num_h = 0
            else:
                num_h = 1
                geometry = 'bent'
            
    elif atom_symbol == 'S':
        if coord == 1:
            num_h = 1
            geometry = 'bent'
    
    return {
        'num_h': num_h,
        'geometry': geometry,
        'bond_length': bond_length
    }