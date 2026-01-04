"""
Structure Generation for Disorder Handling.

This module implements the DisorderSolver that collapses the exclusion graph
into valid, ordered MolecularCrystal objects by solving the Maximum Independent Set problem.
"""

import numpy as np
import networkx as nx
import random
import re
from typing import List

from ase import Atoms
from ase.geometry import get_distances

from ...structures.crystal import MolecularCrystal
from .info import DisorderInfo
from ...io.cif import identify_molecules


class DisorderSolver:
    """
    Solves the disorder problem by finding independent sets in the exclusion graph.
    """

    def __init__(self, info: DisorderInfo, graph: nx.Graph, lattice: np.ndarray):
        """
        Initialize the solver.

        Parameters:
        -----------
        info : DisorderInfo
            Raw disorder data from Phase 1
        graph : networkx.Graph
            Exclusion graph from Phase 2
        lattice : np.ndarray
            3x3 matrix representing the lattice vectors
        """
        self.info = info
        self.graph = graph
        self.lattice = lattice
        self.atom_groups = []  # Will store groups of atoms by (disorder_group, assembly)

    def _identify_atom_groups(self):
        """
        Group atoms by key: (disorder_group, assembly).
        Then use spatial clustering to identify rigid bodies based on distance.
        Finally, check for internal conflicts in each component and explode if needed.
        Store as self.atom_groups (List of Lists of indices).
        """
        # Dictionary to map group key to list of atom indices
        groups_map = {}
        
        for i in range(len(self.info.labels)):
            # Get disorder group and assembly for this atom
            disorder_group = self.info.disorder_groups[i]
            assembly = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
            
            # [CORRECTION] Treat ALL groups (including 0) uniformly to allow spatial clustering
            # This ensures ordered molecules (Group 0) are treated as rigid bodies, not shattered atoms.
            group_key = (disorder_group, assembly)
            
            if group_key not in groups_map:
                groups_map[group_key] = []
            
            groups_map[group_key].append(i)
        
        # Process each initial group with spatial clustering
        for group_atoms in groups_map.values():
            if len(group_atoms) == 1:
                # Single atom group - add directly
                self.atom_groups.append(group_atoms)
            else:
                # Perform spatial clustering on this group
                # Build a temporary distance graph using ASE's get_distances
                group_coords = self.info.frac_coords[group_atoms]
                cart_coords = np.dot(group_coords, self.lattice)
                
                # Calculate pairwise distances between atoms in the group
                # Using ASE's get_distances which handles PBC
                dist_matrix = get_distances(cart_coords, cart_coords, cell=self.lattice, pbc=True)[1]
                
                # Create a temporary graph for clustering
                temp_graph = nx.Graph()
                temp_graph.add_nodes_from(range(len(group_atoms)))
                
                # Connect atoms if distance < 1.8 Å (typical bond length)
                # This keeps molecules (like ClO4 or Methylammonium) together as rigid bodies
                cutoff = 1.8
                for i in range(len(group_atoms)):
                    for j in range(i + 1, len(group_atoms)):
                        if dist_matrix[i, j] < cutoff:
                            temp_graph.add_edge(i, j)
                
                # Find connected components (potential "Rigid Bodies")
                components = list(nx.connected_components(temp_graph))
                
                # For each component, check if it contains internal conflicts
                for comp in components:
                    comp_list = list(comp)
                    comp_atoms = [group_atoms[i] for i in comp_list]
                    
                    # Check if this component has any internal conflicts in the main graph
                    has_internal_conflict = False
                    
                    # Check all pairs of atoms in this component for conflicts in the main graph
                    if len(comp_atoms) > 1:
                        for idx1 in comp_atoms:
                            for idx2 in comp_atoms:
                                if idx1 != idx2 and self.graph.has_edge(idx1, idx2):
                                    has_internal_conflict = True
                                    break
                            if has_internal_conflict:
                                break
                    
                    if has_internal_conflict:
                        # Explode into single-atom groups if the rigid body is self-conflicting
                        for atom_idx in comp_atoms:
                            self.atom_groups.append([atom_idx])
                    else:
                        # Keep as a rigid body group
                        self.atom_groups.append(comp_atoms)

    def _max_weight_independent_set_by_groups(self, graph=None, weight_attr="occupancy"):
        """
        Find an independent set using Group-Based solving approach (Greedy MWIS).
        This mimics the robust logic of the main branch but applied to molecular groups.
        
        Parameters:
        -----------
        graph : networkx.Graph, optional
            Graph to work with. If None, uses self.graph
        weight_attr : str
            The node attribute to use as weight (default: occupancy)

        Returns:
        --------
        List[int]
            List of node indices forming the independent set
        """
        # Use provided graph or fallback to self.graph
        working_graph = graph.copy() if graph is not None else self.graph.copy()
        
        # Calculate weight for each Group (sum of atom weights)
        # Also factor in degree (number of conflicts) to mimic main branch's heuristic:
        # Score = Weight / (Degree + 1) -> Prefer High Weight, Low Conflict
        group_scores = []
        
        for group in self.atom_groups:
            # Check if group is valid in current graph
            if not all(working_graph.has_node(node) for node in group):
                group_scores.append(-1.0) # Mark as invalid
                continue

            # Total Weight of the group
            weight = sum(working_graph.nodes[node].get(weight_attr, 1.0) for node in group)
            
            # Total external degree (conflicts with nodes OUTSIDE the group)
            degree = 0
            for node in group:
                for neighbor in working_graph.neighbors(node):
                    if neighbor not in group: # Ignore internal edges (though there shouldn't be any)
                        degree += 1
            
            # Heuristic score similar to main branch
            score = weight / (degree + 1.0)
            group_scores.append(score)
        
        # Sort Groups by score (descending)
        sorted_group_indices = sorted(range(len(self.atom_groups)), key=lambda i: group_scores[i], reverse=True)
        
        independent_set = []
        
        # Iterate Groups in descending order of score
        for group_idx in sorted_group_indices:
            if group_scores[group_idx] < 0:
                continue
                
            group = self.atom_groups[group_idx]
            
            # Check if ALL atoms in the Group are currently available in the graph
            all_available = all(working_graph.has_node(node) for node in group)
            
            if not all_available:
                continue

            # Since we removed neighbors immediately upon selection, 
            # if the nodes are present, they are guaranteed to be valid candidates 
            # (unless internal conflicts exist, but we checked that).
            
            # Double check for conflicts with already selected set (Defensive coding)
            no_conflicts = True
            for node in group:
                for selected_node in independent_set:
                    if working_graph.has_edge(node, selected_node):
                        no_conflicts = False
                        break
                if not no_conflicts:
                    break
            
            # If the group is valid, add all its atoms to the independent set
            if all_available and no_conflicts:
                # Add all atoms in this group to the independent set
                independent_set.extend(group)
                
                # Remove all group members AND their neighbors from the graph
                nodes_to_remove = []
                for node in group:
                    nodes_to_remove.append(node)
                    # Add neighbors of this node
                    nodes_to_remove.extend(list(working_graph.neighbors(node)))
                
                # Remove duplicates
                nodes_to_remove = list(set(nodes_to_remove))
                
                # Remove from working graph
                working_graph.remove_nodes_from(nodes_to_remove)
        
        return independent_set

    def solve(
        self, num_structures: int = 1, method: str = "optimal"
    ) -> List[MolecularCrystal]:
        """
        Solve the disorder problem and generate ordered structures using Group-Based approach.

        Parameters:
        -----------
        num_structures : int
            Number of structures to generate (for 'random' method)
        method : str
            'optimal' for single best structure, 'random' for ensemble

        Returns:
        --------
        List[MolecularCrystal]
            List of ordered molecular crystal structures
        """
        # Initialize atom groups (Identify Rigid Bodies)
        self._identify_atom_groups()
        
        # Ensure graph has occupancy weights
        for node in self.graph.nodes():
            if "occupancy" not in self.graph.nodes[node]:
                atom_idx = node 
                if atom_idx < len(self.info.occupancies):
                    self.graph.nodes[node]["occupancy"] = self.info.occupancies[atom_idx]

        if method == "optimal":
            # [FIX] Do NOT apply stoichiometry constraints. 
            # Rely strictly on the Exclusion Graph + MWIS (Max Weight Independent Set).
            # This matches 'main' branch logic but with added benefit of Rigid Body groups.
            
            independent_sets = [self._max_weight_independent_set_by_groups(graph=self.graph, weight_attr="occupancy")]

        elif method == "random":
            independent_sets = []
            seen_structures = set() 

            for _ in range(num_structures):
                # Create temp graph with randomized weights
                temp_graph = self.graph.copy()
                
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = base_weight + random_noise

                try:
                    # Solve using Group-Based MWIS
                    solution = self._max_weight_independent_set_by_groups(graph=temp_graph, weight_attr="randomized_weight")
                except:
                    solution = self._random_independent_set()

                solution_tuple = tuple(sorted(solution))
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)

            # Fill remaining with pure random if needed
            attempts = 0
            max_attempts = num_structures * 10
            while len(independent_sets) < num_structures and attempts < max_attempts:
                temp_graph = self.graph.copy()
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = base_weight + random_noise
                
                solution = self._max_weight_independent_set_by_groups(graph=temp_graph, weight_attr="randomized_weight")
                
                solution_tuple = tuple(sorted(solution))
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)
                attempts += 1
        else:
            raise ValueError(f"Unknown method: {method}. Use 'optimal' or 'random'")

        # Reconstruct crystals
        crystals = []
        for independent_set in independent_sets:
            crystal = self._reconstruct_crystal(independent_set)
            crystals.append(crystal)

        return crystals

    def _random_independent_set(self) -> List[int]:
        """
        Fallback: Generate a random independent set using a randomized greedy algorithm.
        """
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)
        independent_set = []
        for node in nodes:
            connected_to_set = False
            for selected_node in independent_set:
                if self.graph.has_edge(node, selected_node):
                    connected_to_set = True
                    break
            if not connected_to_set:
                independent_set.append(node)
        return independent_set

    def _reconstruct_crystal(self, independent_set: List[int]) -> MolecularCrystal:
        """
        Reconstruct a MolecularCrystal from an independent set of atoms.
        """
        selected_symbols = [self.info.symbols[i] for i in independent_set]
        selected_frac_coords = self.info.frac_coords[independent_set]

        atoms = Atoms(
            symbols=selected_symbols,
            scaled_positions=selected_frac_coords,
            cell=self.lattice,
            pbc=True,
        )
        molecules = identify_molecules(atoms)
        pbc = (True, True, True)
        crystal = MolecularCrystal(self.lattice, molecules, pbc)

        return crystal