"""
Chemical environment analyzer for molecular crystals.

This module provides classes and functions to analyze the local environment
of atoms in molecular crystals, including coordination numbers, bond angles,
and ring detection.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx
from abc import ABC, abstractmethod
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import angle_between_vectors


class ChemicalEnvironment:
    """
    Analyzes the local chemical environment of atoms in a molecular crystal.
    
    Provides methods to calculate coordination numbers, bond angles, 
    planarity, and ring membership for atoms in a crystal structure.
    """
    
    def __init__(self, crystal_molecule):
        """
        Initialize with a CrystalMolecule or graph + positions.
        
        Parameters
        ----------
        crystal_molecule : CrystalMolecule or tuple
            Either a CrystalMolecule object or a tuple containing
            (graph, positions) where graph is a NetworkX graph and 
            positions is a list of 3D coordinates.
        """
        if hasattr(crystal_molecule, 'graph') and hasattr(crystal_molecule, 'get_positions'):
            # It's a CrystalMolecule object
            self.graph = crystal_molecule.graph
            self.positions = crystal_molecule.get_positions()
        else:
            # It's a (graph, positions) tuple
            self.graph, self.positions = crystal_molecule
        
        # Precompute cycle basis ONCE per molecule to avoid repeated heavy calculations
        self._precompute_ring_info()
    
    def _precompute_ring_info(self):
        """Precompute ring membership for all atoms to avoid repeated heavy calculations."""
        try:
            cycles = nx.minimum_cycle_basis(self.graph)
            self._atom_rings = {}
            for idx in self.graph.nodes():
                # Find all cycles this atom belongs to
                atom_cycles = [cycle for cycle in cycles if idx in cycle]
                self._atom_rings[idx] = atom_cycles
        except Exception:
            self._atom_rings = {}

    def get_local_geometry_stats(self, atom_index: int) -> Dict:
        """
        Calculate raw geometry statistics. Does NOT make decisions (e.g. is_planar).
        Returns raw angles and lengths for heuristics to decide.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom to analyze
            
        Returns
        -------
        dict
            Dictionary containing:
            - coordination_number: Number of neighbors
            - bond_angle_sum: Sum of angles between all neighbor pairs
            - bond_angle_single: Specifically for coordination=2, the angle between the two neighbors
            - average_bond_length: Average distance to neighbors
        """
        neighbors = list(self.graph.neighbors(atom_index))
        n_neighbors = len(neighbors)
        
        center_pos = self.positions[atom_index]
        
        distances = []
        neighbor_positions = []
        for neighbor_idx in neighbors:
            neighbor_pos = self.positions[neighbor_idx]
            dist = np.linalg.norm(neighbor_pos - center_pos)
            distances.append(dist)
            neighbor_positions.append(neighbor_pos)
        avg_bond_length = sum(distances) / len(distances) if distances else 0.0
        
        # Calculate angle statistics
        bond_angle_sum = 0.0
        bond_angle_avg = 0.0  # Only meaningful for coord=2
        
        if n_neighbors >= 2:
            angles = []
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    vec1 = neighbor_positions[i] - center_pos
                    vec2 = neighbor_positions[j] - center_pos
                    angle = np.degrees(angle_between_vectors(vec1, vec2))
                    angles.append(angle)
            
            bond_angle_sum = sum(angles)
            if n_neighbors == 2 and angles:
                bond_angle_avg = angles[0]  # For coord=2, there is only 1 angle
        
        return {
            'coordination_number': n_neighbors,
            'bond_angle_sum': bond_angle_sum,
            'bond_angle_single': bond_angle_avg,  # Specifically for coord=2
            'average_bond_length': avg_bond_length
        }
    
    def detect_ring_info(self, atom_index: int) -> Dict:
        """
        Detect ring information for an atom using precomputed ring info.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom to analyze
            
        Returns
        -------
        dict
            Dictionary containing:
            - in_ring: Boolean indicating if atom is in a ring
            - ring_sizes: List of sizes of rings this atom belongs to
            - is_ring_planar: Boolean for the smallest ring, whether it's planar
        """
        try:
            # Use cached ring info
            atom_cycles = self._atom_rings.get(atom_index, [])
            
            if not atom_cycles:
                return {
                    'in_ring': False,
                    'ring_sizes': [],
                    'is_ring_planar': False
                }
            
            # Get ring sizes
            ring_sizes = [len(cycle) for cycle in atom_cycles]
            
            # Determine if the smallest ring is planar
            is_ring_planar = False
            if ring_sizes:
                # Find the smallest ring
                min_size = min(ring_sizes)
                smallest_ring_cycles = [cycle for cycle in atom_cycles if len(cycle) == min_size]
                
                if smallest_ring_cycles:
                    # Take the first occurrence of the smallest ring size
                    smallest_ring = smallest_ring_cycles[0]
                    
                    # Check if the atoms in the ring lie on a plane
                    is_ring_planar = self._is_ring_planar(smallest_ring)
            
            return {
                'in_ring': True,
                'ring_sizes': sorted(ring_sizes),
                'is_ring_planar': is_ring_planar
            }
        except Exception:
            # If there's an issue with cycle detection, fall back to basic check
            neighbors = list(self.graph.neighbors(atom_index))
            if len(neighbors) < 2:
                return {
                    'in_ring': False,
                    'ring_sizes': [],
                    'is_ring_planar': False
                }
            
            # Simple check: if neighbors are interconnected, it might be in a ring
            # This is a simplified fallback
            return {
                'in_ring': False,
                'ring_sizes': [],
                'is_ring_planar': False
            }
    
    def _is_ring_planar(self, ring_atom_indices: List[int], tolerance: float = 0.1) -> bool:
        """
        Check if the atoms in a ring lie on a plane using SVD.
        
        Parameters
        ----------
        ring_atom_indices : List[int]
            Indices of atoms forming the ring
        tolerance : float
            Tolerance for considering atoms to be coplanar
            
        Returns
        -------
        bool
            True if the ring is planar, False otherwise
        """
        if len(ring_atom_indices) < 3:
            return False  # Need at least 3 atoms to define a plane
        
        # Get positions of ring atoms
        ring_positions = np.array([self.positions[idx] for idx in ring_atom_indices])
        
        # Calculate centroid of ring atoms
        centroid = np.mean(ring_positions, axis=0)
        
        # Center the positions at the origin
        centered_positions = ring_positions - centroid
        
        # Perform SVD to find principal components
        U, s, Vt = np.linalg.svd(centered_positions)
        
        # The singular values represent the spread along each principal component
        # For a planar ring, the smallest singular value should be close to zero
        # (meaning little variation in the third dimension)
        if len(s) >= 3:
            # The third singular value represents variation in the direction 
            # perpendicular to the plane defined by the first two components
            smallest_sv = s[2]  # Third singular value
            return smallest_sv < tolerance
        else:
            # If fewer than 3 dimensions are found, consider it planar
            return True

    def get_site(self, atom_index: int) -> 'HybridizedSite':
        """
        Factory method to create the appropriate HybridizedSite subclass based on the atom's symbol.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom in the crystal
            
        Returns
        -------
        HybridizedSite
            An instance of the appropriate subclass based on the atom's symbol
        """
        atom_symbol = self.graph.nodes[atom_index]['symbol'] if 'symbol' in self.graph.nodes[atom_index] else self.graph.nodes[atom_index]
        if isinstance(atom_symbol, dict):
            atom_symbol = atom_symbol.get('symbol', 'X')  # Fallback to 'X' if no symbol
        elif hasattr(atom_symbol, 'symbol'):
            atom_symbol = atom_symbol.symbol
        
        if atom_symbol == 'C':
            return CarbonSite(atom_index, atom_symbol, self)
        elif atom_symbol == 'N':
            return NitrogenSite(atom_index, atom_symbol, self)
        else:
            return GenericSite(atom_index, atom_symbol, self)


class HybridizedSite(ABC):
    """
    Abstract base class for representing hybridized sites in a chemical environment.
    """
    
    def __init__(self, atom_index: int, element: str, env: 'ChemicalEnvironment'):
        """
        Initialize with atom index, element, and chemical environment.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom in the crystal
        element : str
            Element symbol
        env : ChemicalEnvironment
            The chemical environment of the crystal
        """
        self.atom_index = atom_index
        self.element = element
        self.env = env
    
    @property
    def geometry_stats(self) -> Dict:
        """
        Lazy access to local geometry statistics.
        """
        return self.env.get_local_geometry_stats(self.atom_index)
    
    @property
    def ring_info(self) -> Dict:
        """
        Lazy access to ring information.
        """
        return self.env.detect_ring_info(self.atom_index)
    
    @abstractmethod
    def get_hydrogenation_strategy(self) -> Dict:
        """
        Abstract method to determine hydrogenation strategy for this site.
        
        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        pass


class CarbonSite(HybridizedSite):
    """
    Carbon-specific hybridized site implementation.
    """
    
    def get_hydrogenation_strategy(self) -> Dict:
        """
        Determine hydrogenation strategy for carbon based on its environment.
        
        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        from ..constants.config import BOND_LENGTHS
        
        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']
        
        # Defaults
        num_h = 0
        geometry = 'tetrahedral'
        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)
        
        if coord == 3:
            # Case: sp2 (Planar, ~360 sum) vs sp3 (Pyramidal, ~328.5 sum)
            # Previous logic was flawed: it prioritized ring planarity over local geometry
            # New logic: Local geometry takes precedence over global ring properties
            
            angle_sum = self.geometry_stats['bond_angle_sum']
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
                if self.ring_info['in_ring'] and self.ring_info['is_ring_planar']:
                    # Ring is planar -> likely aromatic/conjugated system -> sp2
                    num_h = 0
                    geometry = 'trigonal_planar'
                else:
                    # Ring not planar, or not in ring -> default to sp3
                    num_h = 1
                    geometry = 'tetrahedral'

        elif coord == 2:
            # Case: -CH2- (sp3) vs =CH- (sp2 aromatic) vs =C= (sp linear)
            angle = self.geometry_stats['bond_angle_single']
            
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
        
        return {
            'num_h': num_h,
            'geometry': geometry,
            'bond_length': bond_length
        }


class NitrogenSite(HybridizedSite):
    """
    Nitrogen-specific hybridized site implementation.
    """
    
    def get_hydrogenation_strategy(self) -> Dict:
        """
        Determine hydrogenation strategy for nitrogen based on its environment.
        
        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        from ..constants.config import BOND_LENGTHS
        
        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']
        
        # Defaults
        num_h = 0
        geometry = 'tetrahedral'
        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)
        
        if coord == 2:
            # Pyridine (sp2, 0H) vs Pyrrole (sp2, 1H) vs Amine (sp3, 1H)
            in_ring = self.ring_info['in_ring']
            is_planar_ring = self.ring_info['is_ring_planar']
            ring_sizes = self.ring_info['ring_sizes']
            
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
        
        return {
            'num_h': num_h,
            'geometry': geometry,
            'bond_length': bond_length
        }


class GenericSite(HybridizedSite):
    """
    Generic hybridized site implementation for elements other than C and N.
    """
    
    def get_hydrogenation_strategy(self) -> Dict:
        """
        Determine hydrogenation strategy for generic elements based on their environment.
        
        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        from ..constants.config import BOND_LENGTHS
        
        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']
        
        # Defaults
        num_h = 0
        geometry = 'tetrahedral'
        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)
        
        atom_symbol = self.element
        
        # Oxygen rules
        if atom_symbol == 'O':
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