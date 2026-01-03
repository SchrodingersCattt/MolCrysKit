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
                    comp_atoms_set = set(comp_atoms)
                    
                    # Check all pairs of atoms in this component for conflicts in the main graph
                    for idx1 in comp_atoms:
                        for idx2 in comp_atoms:
                            if idx1 != idx2 and self.graph.has_edge(idx1, idx2):
                                has_internal_conflict = True
                                break
                        if has_internal_conflict:
                            break
                    
                    if has_internal_conflict:
                        # Explode into single-atom groups
                        for atom_idx in comp_atoms:
                            self.atom_groups.append([atom_idx])
                    else:
                        # Keep as a rigid body group
                        self.atom_groups.append(comp_atoms)

    def _max_weight_independent_set_by_groups(self, graph=None, weight_attr="occupancy"):
        """
        Find an independent set using Group-Based solving approach.
        
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
        group_weights = []
        for group in self.atom_groups:
            # Use the working graph to get weights (in case nodes were removed)
            total_weight = sum(working_graph.nodes[node].get(weight_attr, 1.0) for node in group if working_graph.has_node(node))
            group_weights.append(total_weight)
        
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

    def _apply_stoichiometry_constraints(self, graph):
        """
        Apply stoichiometry constraints to the graph by removing excess atom groups
        that would violate the expected stoichiometry if all selected.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The exclusion graph to modify
        
        Returns:
        --------
        networkx.Graph
            Modified graph with excess atoms removed
        """
        # Initialize atom groups if not already done
        if not self.atom_groups:
            self._identify_atom_groups()
        
        # Create a mapping from source atom label (e.g., "N1") to its groups
        family_groups = {}
        for group_idx, group in enumerate(self.atom_groups):
            if not group:  # Skip empty groups
                continue
            # Get the source atom label from the first atom in the group
            # All atoms in the same group should have the same base label
            source_label = self.info.labels[group[0]]
            # Extract the base label using regex: element symbol + number (e.g., "C1", "C12", "Fe1")
            match = re.match(r"^([A-Za-z]{1,2}\d+)", source_label)
            if match:
                base_label = match.group(1)
            else:
                base_label = source_label  # Fallback if no match
            
            if base_label not in family_groups:
                family_groups[base_label] = []
            family_groups[base_label].append((group_idx, group))
        
        # Process each family to apply stoichiometry constraints
        modified_graph = graph.copy()
        for base_label, groups_with_idx in family_groups.items():
            if not groups_with_idx:
                continue
                
            # Extract all atom indices in this family
            family_atom_indices = []
            for group_idx, group in groups_with_idx:
                family_atom_indices.extend(group)
            
            # Calculate target count for this family
            family_occupancies = [self.info.occupancies[i] for i in family_atom_indices]
            target_count = int(round(sum(family_occupancies)))
            
            # Identify "Sites": A "Site" is a set of mutually conflicting groups within the family
            # Construct a subgraph of the family's groups
            family_subgraph = nx.Graph()
            for i, (group_idx1, group1) in enumerate(groups_with_idx):
                for j, (group_idx2, group2) in enumerate(groups_with_idx[i+1:], i+1):
                    # Check if any atom in group1 conflicts with any atom in group2
                    conflict = False
                    for atom1 in group1:
                        for atom2 in group2:
                            if modified_graph.has_edge(atom1, atom2):
                                conflict = True
                                break
                        if conflict:
                            break
                    
                    if conflict:
                        family_subgraph.add_edge(group_idx1, group_idx2)
            
            # Find connected components in the subgraph (these represent "Sites")
            if family_subgraph.number_of_nodes() > 0:
                components = list(nx.connected_components(family_subgraph))
                num_sites = len(components)
                
                # Also add isolated nodes as separate sites
                all_group_indices = {group_idx for group_idx, _ in groups_with_idx}
                component_nodes = set()
                for comp in components:
                    component_nodes.update(comp)
                
                isolated_nodes = all_group_indices - component_nodes
                num_sites += len(isolated_nodes)
            else:
                # All groups are isolated
                num_sites = len(groups_with_idx)
            
            # Enforce stoichiometry constraints
            if num_sites > target_count:
                # We need to remove atoms from some sites
                # Randomly select target_count components (Sites) to KEEP
                if target_count <= 0:
                    # Remove all groups in this family
                    groups_to_remove = groups_with_idx
                else:
                    # Create list of all components (connected + isolated)
                    all_components = list(nx.connected_components(family_subgraph))
                    # Add isolated nodes as single-node components
                    isolated_nodes = all_group_indices - set(family_subgraph.nodes())
                    for node in isolated_nodes:
                        all_components.append({node})
                    
                    # Randomly select which components to keep
                    if len(all_components) <= target_count:
                        # We don't need to remove anything
                        continue
                    
                    selected_components = random.sample(all_components, target_count)
                    selected_group_indices = set()
                    for comp in selected_components:
                        selected_group_indices.update(comp)
                    
                    # Groups to remove are those not in selected components
                    groups_to_remove = [(idx, group) for idx, group in groups_with_idx 
                                        if idx not in selected_group_indices]
                
                # Remove the unselected groups from the graph
                for group_idx, group in groups_to_remove:
                    for atom_idx in group:
                        if modified_graph.has_node(atom_idx):
                            modified_graph.remove_node(atom_idx)
        
        return modified_graph

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
            # Apply stoichiometry constraints to the graph
            working_graph = self._apply_stoichiometry_constraints(self.graph)
            
            # Use max weight independent set with occupancy as weight
            # First, ensure each node in the working graph has an occupancy attribute for the weight
            for node in working_graph.nodes():
                if "occupancy" not in working_graph.nodes[node]:
                    # Use the occupancy from DisorderInfo - find the corresponding atom index
                    atom_idx = node  # node should be the atom index
                    if atom_idx < len(self.info.occupancies):
                        working_graph.nodes[node]["occupancy"] = self.info.occupancies[atom_idx]

            # Find the optimal independent set using Group-Based approach
            # Temporarily update atom_groups to reflect the modified graph
            original_atom_groups = self.atom_groups
            # Re-identify groups based on the modified graph (only for the nodes that remain)
            temp_groups = []
            for group in self.atom_groups:
                # Only keep groups that have at least one atom still in the working graph
                remaining_atoms = [atom for atom in group if working_graph.has_node(atom)]
                if remaining_atoms:
                    temp_groups.append(remaining_atoms)
            self.atom_groups = temp_groups
            
            independent_sets = [self._max_weight_independent_set_by_groups(graph=working_graph, weight_attr="occupancy")]
            self.atom_groups = original_atom_groups  # Restore original groups
        elif method == "random":
            # Generate multiple random independent sets using Group-Based approach with randomized weights
            independent_sets = []
            seen_structures = set()  # For deduplication

            for _ in range(num_structures):
                # Apply stoichiometry constraints to the temporary graph
                temp_graph = self._apply_stoichiometry_constraints(self.graph)
                
                # Add randomized weights to nodes
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    # If no occupancy attribute exists, use from DisorderInfo
                    if base_weight == 1.0 and "occupancy" not in temp_graph.nodes[node]:
                        atom_idx = node
                        if atom_idx < len(self.info.occupancies):
                            base_weight = self.info.occupancies[atom_idx]
                            temp_graph.nodes[node]["occupancy"] = base_weight
                    # Add tiny random noise to break ties between chemically identical parts
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                # Solve using the Group-Based MWIS solver
                try:
                    # Temporarily update atom_groups to reflect the modified graph
                    original_atom_groups = self.atom_groups
                    # Re-identify groups based on the modified graph (only for the nodes that remain)
                    temp_groups = []
                    for group in self.atom_groups:
                        # Only keep groups that have at least one atom still in the temp_graph
                        remaining_atoms = [atom for atom in group if temp_graph.has_node(atom)]
                        if remaining_atoms:
                            temp_groups.append(remaining_atoms)
                    self.atom_groups = temp_groups
                    
                    solution = self._max_weight_independent_set_by_groups(graph=temp_graph, weight_attr="randomized_weight")
                    self.atom_groups = original_atom_groups  # Restore original groups
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
                # Apply stoichiometry constraints to the temporary graph
                temp_graph = self._apply_stoichiometry_constraints(self.graph)
                
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    # If no occupancy attribute exists, use from DisorderInfo
                    if base_weight == 1.0 and "occupancy" not in temp_graph.nodes[node]:
                        atom_idx = node
                        if atom_idx < len(self.info.occupancies):
                            base_weight = self.info.occupancies[atom_idx]
                            temp_graph.nodes[node]["occupancy"] = base_weight
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                try:
                    # Temporarily update atom_groups to reflect the modified graph
                    original_atom_groups = self.atom_groups
                    # Re-identify groups based on the modified graph (only for the nodes that remain)
                    temp_groups = []
                    for group in self.atom_groups:
                        # Only keep groups that have at least one atom still in the temp_graph
                        remaining_atoms = [atom for atom in group if temp_graph.has_node(atom)]
                        if remaining_atoms:
                            temp_groups.append(remaining_atoms)
                    self.atom_groups = temp_groups
                    
                    solution = self._max_weight_independent_set_by_groups(graph=temp_graph, weight_attr="randomized_weight")
                    self.atom_groups = original_atom_groups  # Restore original groups
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