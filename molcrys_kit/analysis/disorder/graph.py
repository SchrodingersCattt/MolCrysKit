"""
Exclusion Graph Construction for Disorder Handling.

This module implements the "Referee" logic that determines which atoms cannot 
coexist in the same physical structure based on raw disorder data.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set
from .info import DisorderInfo
from ...utils.geometry import minimum_image_distance, angle_between_vectors, frac_to_cart


class DisorderGraphBuilder:
    """
    Builds an exclusion graph from DisorderInfo data.
    
    The exclusion graph represents mutual exclusivity between atoms - 
    if there's an edge between two atoms, they cannot coexist in the same structure.
    """
    
    def __init__(self, info: DisorderInfo, lattice: np.ndarray):
        """
        Initialize the graph builder.
        
        Parameters:
        -----------
        info : DisorderInfo
            Raw disorder data from Phase 1
        lattice : np.ndarray
            3x3 matrix representing the lattice vectors
        """
        self.info = info
        self.lattice = lattice
        self.graph = nx.Graph()
        
        # Add nodes to the graph (each node is an atom index)
        for i in range(len(info.labels)):
            self.graph.add_node(i, 
                               label=info.labels[i],
                               symbol=info.symbols[i],
                               frac_coord=info.frac_coords[i],
                               occupancy=info.occupancies[i],
                               disorder_group=info.disorder_groups[i])
    
    def build(self) -> nx.Graph:
        """
        Build the exclusion graph by applying the 3-layer conflict detection logic.
        
        Returns:
        --------
        networkx.Graph
            The exclusion graph where edges represent mutually exclusive atoms
        """
        # Layer 1: Explicit conflicts based on disorder groups
        self._add_explicit_conflicts()
        
        # Layer 2: Geometric conflicts based on atomic distances
        self._add_geometric_conflicts(threshold=0.8)
        
        # Layer 3: Valence/inferred conflicts based on coordination geometry
        self._resolve_valence_conflicts()
        
        return self.graph
    
    def _add_explicit_conflicts(self):
        """
        Add edges between atoms that have different non-zero disorder groups.
        
        Rule: If two atoms have non-zero disorder groups and group_A != group_B,
        they are mutually exclusive.
        """
        n_atoms = len(self.info.labels)
        
        for i in range(n_atoms):
            group_i = self.info.disorder_groups[i]
            
            # Only consider non-zero groups
            if group_i == 0:
                continue
                
            for j in range(i + 1, n_atoms):
                group_j = self.info.disorder_groups[j]
                
                # If both have non-zero groups and they are different, add exclusion
                if group_j != 0 and group_i != group_j:
                    # Only add if not already added with a different conflict type
                    if not self.graph.has_edge(i, j):
                        self.graph.add_edge(i, j, conflict_type="explicit")
                    else:
                        # If already exists, update to explicit if it's not already explicit
                        if self.graph[i][j]['conflict_type'] != 'explicit':
                            self.graph[i][j]['conflict_type'] = 'explicit'
    
    def _add_geometric_conflicts(self, threshold: float = 0.8):
        """
        Add edges between atoms that are too close to coexist (hard sphere collision).
        
        Parameters:
        -----------
        threshold : float
            Distance threshold (in Angstroms) below which atoms are considered colliding
        """
        n_atoms = len(self.info.labels)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Skip if same atom (shouldn't happen but just in case)
                if i == j:
                    continue
                
                # Get fractional coordinates
                frac_i = self.info.frac_coords[i]
                frac_j = self.info.frac_coords[j]
                
                # Calculate minimum image distance
                distance = minimum_image_distance(frac_i, frac_j, self.lattice)
                
                # If distance is below threshold, check if atoms are bonded
                if distance < threshold:
                    # Check if the atoms are bonded - if so, don't add geometric conflict
                    symbol_i = self.info.symbols[i]
                    symbol_j = self.info.symbols[j]
                    
                    if not self._are_bonded(symbol_i, symbol_j, distance):
                        # Only add if not already added with explicit conflict
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(i, j, conflict_type="geometric", distance=distance)
                        else:
                            # If there's already a connection, check if we should update the type
                            # Explicit conflicts take priority over geometric
                            if self.graph[i][j]['conflict_type'] != 'explicit':
                                self.graph[i][j]['conflict_type'] = 'geometric'
                                self.graph[i][j]['distance'] = distance
    
    def _resolve_valence_conflicts(self):
        """
        Detect and resolve valence conflicts (implicit disorder like in 'DAP-4').
        
        This handles cases where atoms are not explicitly grouped but would
        overcrowd a coordination environment (like 8 H atoms around a N atom).
        """
        n_atoms = len(self.info.labels)
        
        # Build connectivity graph for neighbors
        connectivity_graph = nx.Graph()
        for i in range(n_atoms):
            connectivity_graph.add_node(i)
            
        # Add edges for bonded atoms (using a generous threshold)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                frac_i = self.info.frac_coords[i]
                frac_j = self.info.frac_coords[j]
                
                # Calculate distance
                distance = minimum_image_distance(frac_i, frac_j, self.lattice)
                
                # Use element-dependent bonding thresholds
                symbol_i = self.info.symbols[i].strip()
                symbol_j = self.info.symbols[j].strip()
                
                # Determine if bonded based on element types
                is_bonded = self._are_bonded(symbol_i, symbol_j, distance)
                
                if is_bonded:
                    connectivity_graph.add_edge(i, j, distance=distance)
        
        # Identify overcrowded centers
        for center_idx in range(n_atoms):
            neighbors = list(connectivity_graph.neighbors(center_idx))
            
            # Get max coordination number for this element
            center_symbol = self.info.symbols[center_idx]
            max_coordination = self._get_max_coordination(center_symbol)
            
            # Check if overcrowded and has low occupancy neighbors
            if len(neighbors) > max_coordination:
                low_occ_neighbors = [
                    n for n in neighbors 
                    if self.info.occupancies[n] < 1.0
                ]
                
                if len(low_occ_neighbors) > 1:
                    # Try to decompose the overcrowded situation using geometric analysis
                    self._decompose_cliques(low_occ_neighbors, center_idx)
    
    def _are_bonded(self, symbol1: str, symbol2: str, distance: float) -> bool:
        """
        Determine if two atoms are bonded based on element types and distance.
        
        Parameters:
        -----------
        symbol1, symbol2 : str
            Element symbols
        distance : float
            Distance between atoms
            
        Returns:
        --------
        bool
            True if atoms are considered bonded
        """
        # Define bonding thresholds based on element types
        if {'H', 'D'}.intersection({symbol1, symbol2}):
            # H/D with any other element
            if {'C', 'N', 'O', 'S', 'P'}.intersection({symbol1, symbol2}):
                return 0.6 < distance < 1.4
            elif {'H', 'D'}.intersection({symbol1, symbol2}):
                return False  # H-H unlikely to bond
            else:
                return 0.8 < distance < 1.8
        elif {'C', 'N', 'O'}.intersection({symbol1, symbol2}):
            # C, N, O with each other
            return 0.8 < distance < 1.9
        elif {'C', 'N', 'O'}.intersection({symbol1}) and {'C', 'N', 'O'}.intersection({symbol2}):
            # C-N, C-O, N-O
            return 0.8 < distance < 2.0
        else:
            # General threshold for other element pairs
            return 0.5 < distance < 2.2
    
    def _get_max_coordination(self, element_symbol: str) -> int:
        """
        Get the maximum coordination number for an element.
        
        Parameters:
        -----------
        element_symbol : str
            Element symbol
            
        Returns:
        --------
        int
            Maximum expected coordination number
        """
        element = element_symbol.strip().title()  # Capitalize first letter
        
        # Common coordination numbers for main group elements
        coordination_map = {
            'H': 1, 'D': 1,
            'C': 4, 'Si': 4,
            'N': 4, 'P': 4,  # Usually 3+1 for N/P with lone pair
            'O': 2, 'S': 2, 'Se': 2, 'Te': 2,
            'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
            # Common metals
            'Li': 4, 'Na': 6, 'K': 8,
            'Mg': 6, 'Ca': 6, 'Sr': 6, 'Ba': 8,
            'Al': 4, 'Ga': 4,
        }
        
        return coordination_map.get(element, 6)  # Default to 6 if unknown
    
    def _decompose_cliques(self, atom_indices: List[int], center_idx: int):
        """
        Decompose overcrowded atom sets into mutually exclusive subsets.
        
        Parameters:
        -----------
        atom_indices : List[int]
            Indices of atoms that are overcrowding around a center
        center_idx : int
            Index of the center atom
        """
        if len(atom_indices) < 2:
            return  # Need at least 2 atoms to create exclusions
        
        # Calculate positions relative to the center
        center_frac = self.info.frac_coords[center_idx]
        relative_positions = []
        
        for idx in atom_indices:
            atom_frac = self.info.frac_coords[idx]
            # Apply minimum image convention relative to center
            delta = atom_frac - center_frac
            delta = delta - np.round(delta)  # Minimum image
            relative_positions.append(delta)
        
        # Convert to Cartesian coordinates
        cart_positions = []
        for rel_pos in relative_positions:
            cart_pos = frac_to_cart(rel_pos, self.lattice)
            cart_positions.append(cart_pos)
        
        # For now, use a simple approach: group atoms that form valid geometries
        # In the case of the "ammonium" example (8 H around N), we expect 2 sets of 4
        if len(atom_indices) == 8 and self.info.symbols[center_idx] in ['N', 'P']:
            # Try to find two tetrahedral arrangements among 8 atoms
            self._find_tetrahedral_groups(atom_indices, cart_positions)
        else:
            # For other cases, if all have low occupancy, make them all mutually exclusive
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
                        # Only add if not already added with explicit or geometric conflict
                        if not self.graph.has_edge(i_idx, j_idx):
                            self.graph.add_edge(i_idx, j_idx, conflict_type="valence")
                        else:
                            # Update to valence if it's not already explicit or geometric
                            current_type = self.graph[i_idx][j_idx]['conflict_type']
                            if current_type not in ['explicit', 'geometric']:
                                self.graph[i_idx][j_idx]['conflict_type'] = 'valence'
    
    def _find_tetrahedral_groups(self, atom_indices: List[int], cart_positions: List[np.ndarray], center_idx: int):
        """
        Find two tetrahedral groups among 8 atoms around a center (like in DAP-4).
        
        Parameters:
        -----------
        atom_indices : List[int]
            Indices of the 8 surrounding atoms
        cart_positions : List[np.ndarray]
            Cartesian positions of the 8 atoms relative to center
        center_idx : int
            Index of the center atom
        """
        if len(atom_indices) != 8:
            return  # Need exactly 8 atoms for this logic
        
        # This is a simplified implementation - we'll look for groups of atoms
        # that could form valid tetrahedral geometries around the center
        
        # Calculate angles between all atom pairs as seen from the center (which is at origin now)
        n_atoms = len(atom_indices)
        
        # For each combination of 4 atoms, check if they form a valid tetrahedron
        from itertools import combinations
        
        valid_groups = []
        
        for combo in combinations(range(n_atoms), 4):
            combo_indices = [atom_indices[i] for i in combo]
            combo_positions = [cart_positions[i] for i in combo]
            
            # Calculate angles between all pairs of positions
            angles = []
            for i in range(4):
                for j in range(i+1, 4):
                    v1 = combo_positions[i]
                    v2 = combo_positions[j]
                    
                    # Calculate angle between vectors from center to these atoms
                    angle_rad = angle_between_vectors(v1, v2)
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
            
            # Check if angles are consistent with tetrahedral geometry (~109.5°)
            # Allow some tolerance (e.g., 109.5° ± 15°)
            tetrahedral_angles = [a for a in angles if 94.5 <= abs(a - 180) <= 124.5 or abs(a) <= 15.5]
            
            # A valid tetrahedron should have 0 or 2 angles of ~109.5°, others are ~70.5°
            # For simplicity, we'll check if there are at least 2 angles near 109.5°
            if len(tetrahedral_angles) >= 2:
                valid_groups.append(combo_indices)
        
        # If we found 2 valid groups, make all atoms in group A exclusive to all atoms in group B
        if len(valid_groups) >= 2:
            group_a = valid_groups[0]
            group_b = valid_groups[1]
            
            for atom_a in group_a:
                for atom_b in group_b:
                    self.graph.add_edge(atom_a, atom_b, conflict_type="valence_geometry")
        else:
            # If we couldn't identify clear geometric groups, fall back to making all low-occ atoms exclusive
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
                        self.graph.add_edge(i_idx, j_idx, conflict_type="valence")