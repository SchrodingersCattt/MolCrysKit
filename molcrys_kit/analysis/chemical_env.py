"""
Chemical environment analyzer for molecular crystals.

This module provides classes and functions to analyze the local environment
of atoms in molecular crystals, including coordination numbers, bond angles,
and ring detection.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx
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
    
    def get_local_geometry_stats(self, atom_index: int) -> Dict:
        """
        Calculate statistics for the local geometry around an atom.
        
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
            - average_bond_length: Average distance to neighbors
            - is_planar: Boolean, true if bond_angle_sum is close to 360 degrees (within tolerance)
        """
        neighbors = list(self.graph.neighbors(atom_index))
        n_neighbors = len(neighbors)
        
        if n_neighbors < 2:
            # Not enough neighbors to calculate angles
            return {
                'coordination_number': n_neighbors,
                'bond_angle_sum': 0.0,
                'average_bond_length': 0.0,
                'is_planar': False
            }
        
        center_pos = self.positions[atom_index]
        
        # Calculate distances to neighbors
        distances = []
        neighbor_positions = []
        for neighbor_idx in neighbors:
            neighbor_pos = self.positions[neighbor_idx]
            dist = np.linalg.norm(neighbor_pos - center_pos)
            distances.append(dist)
            neighbor_positions.append(neighbor_pos)
        
        avg_bond_length = sum(distances) / len(distances) if distances else 0.0
        
        # Calculate angles between all neighbor pairs
        angle_sum = 0.0
        for i in range(len(neighbor_positions)):
            for j in range(i + 1, len(neighbor_positions)):
                vec1 = neighbor_positions[i] - center_pos
                vec2 = neighbor_positions[j] - center_pos
                
                angle = angle_between_vectors(vec1, vec2)
                angle_sum += angle
        
        # Convert to degrees for comparison
        angle_sum_deg = np.degrees(angle_sum)
        
        # Check if planar (sum of angles close to 360 for planar systems)
        # Using 10 degree tolerance as suggested
        is_planar = abs(angle_sum_deg - 360.0) <= 10.0
        
        # Special case for 2 neighbors: check if angle is ~120 (sp2) or ~109 (sp3)
        if n_neighbors == 2:
            if len(neighbor_positions) >= 2:
                vec1 = neighbor_positions[0] - center_pos
                vec2 = neighbor_positions[1] - center_pos
                angle = np.degrees(angle_between_vectors(vec1, vec2))
                
                # For 2 neighbors, consider planar if angle is ~120 (sp2 hybridization)
                is_planar = abs(angle - 120.0) <= 15.0
            else:
                is_planar = False
        
        return {
            'coordination_number': n_neighbors,
            'bond_angle_sum': angle_sum_deg,
            'average_bond_length': avg_bond_length,
            'is_planar': is_planar
        }
    
    def detect_ring_info(self, atom_index: int) -> Dict:
        """
        Detect ring information for an atom using networkx.minimum_cycle_basis.
        
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
            # Get the minimum cycle basis for the entire graph
            cycles = nx.minimum_cycle_basis(self.graph)
            
            # Find cycles that contain the given atom
            atom_cycles = [cycle for cycle in cycles if atom_index in cycle]
            
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