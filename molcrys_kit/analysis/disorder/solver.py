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
    from ase.neighborlist import neighbor_list
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


def identify_molecules_in_atoms(
    atoms: Atoms, bond_thresholds=None
) -> List[CrystalMolecule]:
    """
    Identify discrete molecular units in a crystal using graph-based approach.
    This is a copy of the function from io.cif to avoid circular imports.

    This function builds a graph of all atoms in the crystal structure, connecting
    atoms that are likely bonded based on distance criteria. Connected components
    in this graph represent individual molecules.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the crystal structure.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.

    Returns
    -------
    List[CrystalMolecule]
        List of CrystalMolecule objects, each representing a molecular unit.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is required for molecule identification. Please install it with 'pip install ase'"
        )

    # Build a global graph for the entire crystal structure
    crystal_graph = nx.Graph()

    # Add all atoms as nodes
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    for i in range(len(atoms)):
        crystal_graph.add_node(i, symbol=symbols[i], position=positions[i])

    # Use ASE neighbor list for efficient bond detection
    i_list, j_list, d_list = neighbor_list(
        "ijd", atoms, cutoff=3.0
    )  # 3Å should cover most bonds

    # Add edges (bonds) based on distance criteria
    for i, j, distance in zip(i_list, j_list, d_list):
        # Check if custom threshold is provided
        pair_key1 = (symbols[i], symbols[j])
        pair_key2 = (symbols[j], symbols[i])

        if bond_thresholds and (
            pair_key1 in bond_thresholds or pair_key2 in bond_thresholds
        ):
            # Use custom threshold if provided
            threshold = bond_thresholds.get(pair_key1, bond_thresholds.get(pair_key2))
        else:
            # Determine threshold based on atomic radii and element types
            radius_i = (
                get_atomic_radius(symbols[i]) if has_atomic_radius(symbols[i]) else 0.5
            )
            radius_j = (
                get_atomic_radius(symbols[j]) if has_atomic_radius(symbols[j]) else 0.5
            )
            is_metal_i = is_metal_element(symbols[i])
            is_metal_j = is_metal_element(symbols[j])

            # Determine threshold factor based on element types
            if is_metal_i and is_metal_j:  # Metal-Metal
                factor = METAL_THRESHOLD_FACTOR
            elif not is_metal_i and not is_metal_j:  # Non-metal-Non-metal
                factor = NON_METAL_THRESHOLD_FACTOR
            else:  # Metal-Non-metal
                factor = (METAL_THRESHOLD_FACTOR + NON_METAL_THRESHOLD_FACTOR) / 2

            # Bonding threshold as sum of covalent radii multiplied by factor
            threshold = (radius_i + radius_j) * factor

        # Add edge if atoms are close enough to be bonded
        if distance < threshold:
            crystal_graph.add_edge(i, j, distance=distance)

    # Find connected components (molecular units)
    components = list(nx.connected_components(crystal_graph))

    # Create separate CrystalMolecule objects for each molecular unit
    molecules = []
    for component in components:
        # Get indices of atoms in this component
        atom_indices = list(component)

        # Extract atoms for this molecule
        molecule_atoms = atoms[atom_indices]

        # Create CrystalMolecule object
        molecule = CrystalMolecule(molecule_atoms)
        molecules.append(molecule)

    return molecules


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
    
    def solve(self, num_structures: int = 1, method: str = 'optimal') -> List[MolecularCrystal]:
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
        if method == 'optimal':
            # Use max weight independent set with occupancy as weight
            # First, ensure each node has an occupancy attribute for the weight
            for i, node in enumerate(self.graph.nodes()):
                if 'occupancy' not in self.graph.nodes[node]:
                    # Use the occupancy from DisorderInfo
                    self.graph.nodes[node]['occupancy'] = self.info.occupancies[i]
            
            # Find the optimal independent set
            # NetworkX doesn't have max_weight_independent_set, using a greedy approach instead
            # Create a copy of the graph to work with
            working_graph = self.graph.copy()
            optimal_set = set()
            
            # Sort nodes by their occupancy in descending order
            sorted_nodes = sorted(
                working_graph.nodes(data=True), 
                key=lambda x: x[1].get('occupancy', 1.0), 
                reverse=True
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
                        [(n, d) for n, d in working_graph.nodes(data=True) if (n, d) in sorted_nodes],
                        key=lambda x: x[1].get('occupancy', 1.0),
                        reverse=True
                    )
            
            independent_sets = [list(optimal_set)]
        elif method == 'random':
            # Generate multiple random independent sets
            independent_sets = []
            attempts = 0
            max_attempts = num_structures * 10  # Allow more attempts to find unique sets
            
            while len(independent_sets) < num_structures and attempts < max_attempts:
                random_set = self._random_independent_set()
                
                # Only add if it's not already in the list (check for uniqueness)
                is_unique = True
                for existing_set in independent_sets:
                    if set(random_set) == set(existing_set):
                        is_unique = False
                        break
                
                if is_unique:
                    independent_sets.append(random_set)
                
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
                pbc=True
            )
        except ImportError:
            raise ImportError(
                "ASE is required for crystal reconstruction. "
                "Please install it with 'pip install ase'"
            )
        
        # Rebuild molecular topology using our local function to avoid circular import
        molecules = identify_molecules_in_atoms(atoms)
        
        # Create MolecularCrystal
        pbc = (True, True, True)
        crystal = MolecularCrystal(self.lattice, molecules, pbc)
        
        return crystal