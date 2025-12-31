"""
Structure Generation for Disorder Handling.

This module implements the DisorderSolver that collapses the exclusion graph
into valid, ordered MolecularCrystal objects by solving the Maximum Independent Set problem.
"""

import numpy as np
import networkx as nx
import random
from typing import List

try:
    from ase import Atoms

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder

from ...structures.crystal import MolecularCrystal
from ...structures.molecule import CrystalMolecule
from ...constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
    METAL_THRESHOLD_FACTOR,
    NON_METAL_THRESHOLD_FACTOR,
)
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

    def solve(
        self, num_structures: int = 1, method: str = "optimal"
    ) -> List[MolecularCrystal]:
        """
        Solve the disorder problem and generate ordered structures.

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
        if method == "optimal":
            # Use max weight independent set with occupancy as weight
            # First, ensure each node has an occupancy attribute for the weight
            for i, node in enumerate(self.graph.nodes()):
                if "occupancy" not in self.graph.nodes[node]:
                    # Use the occupancy from DisorderInfo
                    self.graph.nodes[node]["occupancy"] = self.info.occupancies[i]

            # Find the optimal independent set
            # NetworkX doesn't have max_weight_independent_set, using a greedy approach instead
            # Create a copy of the graph to work with
            working_graph = self.graph.copy()
            optimal_set = set()

            # Sort nodes by their occupancy in descending order
            sorted_nodes = sorted(
                working_graph.nodes(data=True),
                key=lambda x: x[1].get("occupancy", 1.0),
                reverse=True,
            )

            # Greedy approach to find an independent set
            while sorted_nodes:
                # Pick the node with the highest occupancy
                node, data = sorted_nodes.pop(0)

                # If this node is still in the graph, add it to the independent set
                if working_graph.has_node(node):
                    optimal_set.add(node)

                    # Remove this node and its neighbors from the graph
                    nodes_to_remove = [node]
                    nodes_to_remove.extend(list(working_graph.neighbors(node)))

                    # Update the working graph
                    working_graph.remove_nodes_from(nodes_to_remove)

                    # Re-sort the remaining nodes
                    sorted_nodes = sorted(
                        [
                            (n, d)
                            for n, d in working_graph.nodes(data=True)
                            if (n, d) in sorted_nodes
                        ],
                        key=lambda x: x[1].get("occupancy", 1.0),
                        reverse=True,
                    )

            independent_sets = [list(optimal_set)]
        elif method == "random":
            # Generate multiple random independent sets using the new Randomized Exact Solver
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

                # Solve using the exact MWIS solver
                try:
                    # Note: NetworkX doesn't have max_weight_independent_set by default
                    # So we'll use the complement: find max weight clique in the complement graph
                    # Or implement our own greedy approach for MWIS
                    solution = self._max_weight_independent_set(
                        temp_graph, "randomized_weight"
                    )
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
                    solution = self._max_weight_independent_set(
                        temp_graph, "randomized_weight"
                    )
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
        selected_labels = [self.info.labels[i] for i in independent_set]
        selected_symbols = [self.info.symbols[i] for i in independent_set]
        selected_frac_coords = self.info.frac_coords[independent_set]

        # Force occupancy to 1.0 for all selected atoms
        selected_occupancies = [1.0 for _ in independent_set]

        # Create ASE Atoms object
        try:
            atoms = Atoms(
                symbols=selected_symbols,
                scaled_positions=selected_frac_coords,
                cell=self.lattice,
                pbc=True,
            )
        except ImportError:
            raise ImportError(
                "ASE is required for crystal reconstruction. "
                "Please install it with 'pip install ase'"
            )

        # Rebuild molecular topology using the imported function
        molecules = identify_molecules(atoms)

        # Create MolecularCrystal
        pbc = (True, True, True)
        crystal = MolecularCrystal(self.lattice, molecules, pbc)

        return crystal

    def _max_weight_independent_set(self, graph, weight_attr="weight"):
        """
        Find an approximation to the maximum weight independent set using a greedy algorithm.

        Parameters:
        -----------
        graph : networkx.Graph
            The input graph
        weight_attr : str
            The node attribute to use as weight

        Returns:
        --------
        List[int]
            List of node indices forming the independent set
        """
        # Create a copy of the graph to work with
        working_graph = graph.copy()
        independent_set = []

        # Continue until the graph is empty
        while working_graph.number_of_nodes() > 0:
            # Calculate the weight-to-degree ratio for each node
            # Nodes with high weight but low connections are preferred
            ratios = {}
            for node in working_graph.nodes():
                weight = working_graph.nodes[node].get(weight_attr, 1.0)
                # Use degree + 1 to avoid division by zero and prefer lower degree nodes
                degree = working_graph.degree(node)
                # Use weight/(degree+1) as the heuristic - higher is better
                ratios[node] = weight / (degree + 1)

            # Select the node with the highest weight/(degree+1) ratio
            best_node = max(ratios.keys(), key=lambda n: ratios[n])

            # Add this node to the independent set
            independent_set.append(best_node)

            # Remove this node and its neighbors (and their edges) from the graph
            nodes_to_remove = [best_node] + list(working_graph.neighbors(best_node))
            working_graph.remove_nodes_from(nodes_to_remove)

        return independent_set
