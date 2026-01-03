"""
Structure Generation for Disorder Handling.

This module implements the DisorderSolver that collapses the exclusion graph
into valid, ordered MolecularCrystal objects by solving the Maximum Independent Set problem.
"""

import numpy as np
import networkx as nx
import random
from typing import List

from ase import Atoms

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
        Atoms with disorder_group=0 are single-atom groups.
        Store as self.atom_groups (List of Lists of indices).
        """
        # Dictionary to map group key to list of atom indices
        groups_map = {}
        
        for i in range(len(self.info.labels)):
            # Get disorder group and assembly for this atom
            disorder_group = self.info.disorder_groups[i]
            assembly = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
            
            # For atoms with disorder_group=0, make them single-atom groups
            if disorder_group == 0:
                # Create a single-atom group for this atom
                self.atom_groups.append([i])
            else:
                # Group by (disorder_group, assembly)
                group_key = (disorder_group, assembly)
                
                if group_key not in groups_map:
                    groups_map[group_key] = []
                
                groups_map[group_key].append(i)
        
        # Add all the grouped atoms to atom_groups
        for group_atoms in groups_map.values():
            if len(group_atoms) > 0:
                self.atom_groups.append(group_atoms)

    def _max_weight_independent_set_by_groups(self, weight_attr="occupancy"):
        """
        Find an independent set using Group-Based solving approach.
        
        Parameters:
        -----------
        weight_attr : str
            The node attribute to use as weight (default: occupancy)

        Returns:
        --------
        List[int]
            List of node indices forming the independent set
        """
        # Calculate weight for each Group (sum of atom weights)
        group_weights = []
        for group in self.atom_groups:
            total_weight = sum(self.graph.nodes[node].get(weight_attr, 1.0) for node in group)
            group_weights.append(total_weight)
        
        # Create a copy of the graph to work with
        working_graph = self.graph.copy()
        
        # Sort Groups by weight (descending)
        sorted_group_indices = sorted(range(len(self.atom_groups)), key=lambda i: group_weights[i], reverse=True)
        
        independent_set = []
        
        # Iterate Groups in descending order of weight
        for group_idx in sorted_group_indices:
            group = self.atom_groups[group_idx]
            
            # Check if ALL atoms in the Group are currently available in the graph
            all_available = all(working_graph.has_node(node) for node in group)
            
            # Check if none of the atoms in the group have conflicts with already selected nodes
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
        # Initialize atom groups
        self._identify_atom_groups()
        
        if method == "optimal":
            # Use max weight independent set with occupancy as weight
            # First, ensure each node has an occupancy attribute for the weight
            for i, node in enumerate(self.graph.nodes()):
                if "occupancy" not in self.graph.nodes[node]:
                    # Use the occupancy from DisorderInfo
                    self.graph.nodes[node]["occupancy"] = self.info.occupancies[i]

            # Find the optimal independent set using Group-Based approach
            independent_sets = [self._max_weight_independent_set_by_groups("occupancy")]
        elif method == "random":
            # Generate multiple random independent sets using Group-Based approach with randomized weights
            independent_sets = []
            seen_structures = set()  # For deduplication

            for _ in range(num_structures):
                # Create a temporary graph with randomized weights
                temp_graph = self.graph.copy()

                # Add randomized weights to nodes
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    # Add tiny random noise to break ties between chemically identical parts
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                # Solve using the Group-Based MWIS solver
                try:
                    solution = self._max_weight_independent_set_by_groups("randomized_weight")
                except:
                    # Fallback to the old random method if needed
                    solution = self._random_independent_set()

                # Create a hashable representation for deduplication
                solution_tuple = tuple(sorted(solution))

                # Only add if it's unique
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)

            # If we didn't get enough unique structures, fill with random ones
            attempts = 0
            max_attempts = num_structures * 10
            while len(independent_sets) < num_structures and attempts < max_attempts:
                # Generate another solution
                temp_graph = self.graph.copy()
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                try:
                    solution = self._max_weight_independent_set_by_groups("randomized_weight")
                except:
                    solution = self._random_independent_set()

                solution_tuple = tuple(sorted(solution))
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)

                attempts += 1
        else:
            raise ValueError(f"Unknown method: {method}. Use 'optimal' or 'random'")

        # Reconstruct crystals from the independent sets
        crystals = []
        for independent_set in independent_sets:
            crystal = self._reconstruct_crystal(independent_set)
            crystals.append(crystal)

        return crystals

    def _random_independent_set(self) -> List[int]:
        """
        Generate a random independent set using a randomized greedy algorithm.

        Returns:
        --------
        List[int]
            List of atom indices forming an independent set
        """
        # Get all nodes and shuffle them
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)

        # Greedily build an independent set
        independent_set = []
        for node in nodes:
            # Check if this node is connected to any node already in the set
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

        Parameters:
        -----------
        independent_set : List[int]
            List of atom indices to include in the structure

        Returns:
        --------
        MolecularCrystal
            Reconstructed ordered crystal structure
        """
        # Filter the DisorderInfo data for the selected atoms
        selected_symbols = [self.info.symbols[i] for i in independent_set]
        selected_frac_coords = self.info.frac_coords[independent_set]

        atoms = Atoms(
            symbols=selected_symbols,
            scaled_positions=selected_frac_coords,
            cell=self.lattice,
            pbc=True,
        )

        # Rebuild molecular topology using the imported function
        molecules = identify_molecules(atoms)

        # Create MolecularCrystal
        pbc = (True, True, True)
        crystal = MolecularCrystal(self.lattice, molecules, pbc)

        return crystal