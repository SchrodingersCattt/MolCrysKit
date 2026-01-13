"""
Topology-aware stoichiometry analysis for molecular crystals.

This module provides functionality for identifying molecular species (including isomers)
and calculating stoichiometry based on molecular topology.
"""

import networkx as nx
from typing import Dict, Optional
from collections import defaultdict
from ..structures.crystal import MolecularCrystal
from ..constants.config import COMMON_SOLVENTS


class StoichiometryAnalyzer:
    """
    Analyzes the stoichiometry of a molecular crystal based on molecular topology.

    This class identifies distinct molecular species by comparing their
    internal connectivity graphs, enabling distinction between isomers.
    """

    def __init__(self, crystal: MolecularCrystal):
        """
        Initialize the analyzer with a molecular crystal.

        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to analyze.
        """
        self.crystal = crystal
        self.species_map = {}  # Maps species ID to list of molecule indices
        self.species_graphs = {}  # Maps species ID to graph for reference
        self._analyze_species()

    @staticmethod
    def inspect_solvent(formula: str) -> Optional[str]:
        """
        Check if a given formula matches any known solvent.

        Parameters
        ----------
        formula : str
            The chemical formula to check.

        Returns
        -------
        Optional[str]
            The name of the matching solvent if found, None otherwise.
        """
        for solvent_name, solvent_info in COMMON_SOLVENTS.items():
            if formula == solvent_info['formula'] or formula == solvent_info['heavy_formula']:
                return solvent_name
        return None

    def _analyze_species(self):
        """
        Classify all molecules in the crystal based on their topology.
        """
        # Group molecules by chemical formula first
        formula_groups = defaultdict(list)

        for idx, molecule in enumerate(self.crystal.molecules):
            formula = molecule.get_chemical_formula()
            formula_groups[formula].append((idx, molecule))

        # For each formula group, further distinguish isomers using graph isomorphism
        species_counter = {}

        for formula, mol_list in formula_groups.items():
            # Group by topology using graph isomorphism
            topology_groups = []

            for idx, molecule in mol_list:
                mol_graph = molecule.graph
                is_new_topology = True

                # Compare with existing topologies in this formula group
                for topo_idx, (ref_graph, topo_mols) in enumerate(topology_groups):
                    # Check for isomorphism considering node attributes (element types)
                    def node_match(n1, n2):
                        return n1["symbol"] == n2["symbol"]

                    if nx.is_isomorphic(ref_graph, mol_graph, node_match=node_match):
                        topology_groups[topo_idx][1].append((idx, molecule))
                        is_new_topology = False
                        break

                if is_new_topology:
                    topology_groups.append((mol_graph, [(idx, molecule)]))

            # Create species IDs and populate species_map
            for topo_idx, (graph, topo_mols) in enumerate(topology_groups):
                species_id = f"{formula}_{topo_idx + 1}"
                self.species_graphs[species_id] = graph
                self.species_map[species_id] = [mol_idx for mol_idx, _ in topo_mols]

    def get_simplest_unit(self) -> Dict[str, int]:
        """
        Calculate the simplest stoichiometric unit (Z=1) using GCD algorithm.

        Returns
        -------
        Dict[str, int]
            A dictionary mapping species IDs to their counts in the simplest unit.
        """
        if not self.species_map:
            return {}

        # Get counts of each species
        counts = {
            species_id: len(indices) for species_id, indices in self.species_map.items()
        }

        # Find GCD of all counts to get the simplest ratio
        count_values = list(counts.values())
        if not count_values:
            return {}

        # Calculate GCD of all counts
        from math import gcd
        from functools import reduce

        overall_gcd = reduce(gcd, count_values)

        # Divide each count by the GCD to get the simplest unit
        simplest_unit = {}
        for species_id, count in counts.items():
            simplest_unit[species_id] = count // overall_gcd

        return simplest_unit

    def print_species_summary(self):
        """
        Print a summary table of identified species with solvent identification.
        """
        print("Species Summary:")
        print(f"{'ID':<15} {'Count':<8} {'Formula':<15} {'Reference Molecule Index':<25} {'Notes':<20}")
        print("-" * 85)

        notes = ""
        for species_id, indices in self.species_map.items():
            # Extract formula from species ID (before the underscore and number)
            formula_parts = species_id.split("_")
            formula = "_".join(formula_parts[:-1])
            
            # Check if this formula matches any solvent
            possible_solvent = self.inspect_solvent(formula)
            
            count = len(indices)
            example_idx = indices[0] if indices else "N/A"
            if possible_solvent:
                notes += f"[Possible Solvent: {possible_solvent}]"
            
            print(f"{species_id:<15} {count:<8} {formula:<15} {example_idx:<25} {notes:<20}")