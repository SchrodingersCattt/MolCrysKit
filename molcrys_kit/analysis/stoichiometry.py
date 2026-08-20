"""
Topology-aware stoichiometry analysis for molecular crystals.

This module provides functionality for identifying molecular species (including isomers)
and calculating stoichiometry based on molecular topology.
"""

import networkx as nx
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from collections import defaultdict
from ase.geometry import minkowski_reduce
from ..structures.crystal import MolecularCrystal
from ..constants.config import COMMON_SOLVENTS
from ..utils.graph import graph_invariant


@dataclass(frozen=True)
class FormulaUnitMember:
    """One molecule selected for a compact stoichiometric formula unit."""

    species_id: str
    molecule_index: int
    image_shift: tuple[int, int, int]


@dataclass(frozen=True)
class FormulaUnitSelection:
    """Deterministic molecule/image selection for one formula unit."""

    members: tuple[FormulaUnitMember, ...]
    species_counts: tuple[tuple[str, int], ...]

    @property
    def molecule_indices(self) -> tuple[int, ...]:
        """Selected molecule indices in deterministic assembly order."""
        return tuple(member.molecule_index for member in self.members)

    def counts(self) -> Dict[str, int]:
        """Return the simplest-unit species counts as a new dictionary."""
        return dict(self.species_counts)


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

        Uses a two-stage approach for efficiency:
        1. Fast invariant-based grouping (degree sequence + element-degree
           signature) to avoid expensive VF2 graph isomorphism in the
           common case.
        2. Full VF2 isomorphism only when invariants match but the caller
           needs a definitive answer.

        This eliminates the O(N!) worst-case that occurs when VF2 is run
        on large molecular graphs (>50 nodes) produced by erroneous bond
        perception.
        """
        formula_groups = defaultdict(list)

        for idx, molecule in enumerate(self.crystal.molecules):
            formula = molecule.get_chemical_formula()
            formula_groups[formula].append((idx, molecule))

        for formula, mol_list in formula_groups.items():
            topology_groups = []

            for idx, molecule in mol_list:
                mol_graph = molecule.graph
                mol_invariant = graph_invariant(mol_graph)
                is_new_topology = True

                for topo_idx, (ref_graph, ref_inv, topo_mols) in enumerate(topology_groups):
                    # Stage 1: fast invariant comparison (O(N log N))
                    if mol_invariant != ref_inv:
                        continue

                    # Stage 2: full VF2 only when invariants match
                    # For small graphs this is fast; for large graphs with
                    # matching invariants it's still necessary for correctness.
                    def node_match(n1, n2):
                        return n1["symbol"] == n2["symbol"]

                    if nx.is_isomorphic(ref_graph, mol_graph, node_match=node_match):
                        topology_groups[topo_idx] = (ref_graph, ref_inv, topo_mols + [(idx, molecule)])
                        is_new_topology = False
                        break

                if is_new_topology:
                    topology_groups.append((mol_graph, mol_invariant, [(idx, molecule)]))

            for topo_idx, (graph, _inv, topo_mols) in enumerate(topology_groups):
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

    def select_formula_unit(self) -> FormulaUnitSelection:
        """Select a spatially compact realisation of the simplest unit.

        Each molecule of the heaviest species is evaluated as an anchor.
        Remaining molecules are chosen greedily by nearest periodic centroid
        image, then complete selections are ranked by maximum and total
        pairwise centroid distance.  Molecule index and lattice shift provide
        deterministic tie-breaks.  Returned shifts are additional translations
        relative to the molecule positions stored in ``crystal``.
        """
        simplest = self.get_simplest_unit()
        if not simplest:
            return FormulaUnitSelection((), ())

        lattice = np.asarray(self.crystal.lattice, dtype=float)
        periodic = np.asarray(self.crystal.pbc, dtype=bool)
        reduced_lattice, reduction = minkowski_reduce(lattice, pbc=periodic)
        reduced_lattice = np.asarray(reduced_lattice, dtype=float)
        reduction = np.asarray(reduction, dtype=int)
        inv_reduced_lattice = np.linalg.inv(reduced_lattice)
        # In a Minkowski-reduced basis, the nearest image is guaranteed to
        # lie in this Voronoi-relevant {-1, 0, 1} neighbour set.
        neighbor_ranges = [range(-int(pbc), int(pbc) + 1) for pbc in periodic]

        def _species_priority(species_id: str):
            sample = self.crystal.molecules[self.species_map[species_id][0]]
            heavy_atoms = sum(
                symbol != "H" for symbol in sample.get_chemical_symbols()
            )
            return (-heavy_atoms, -len(sample), species_id)

        species_order = sorted(simplest, key=_species_priority)
        anchor_species = species_order[0]
        counts = tuple(
            (species_id, int(simplest[species_id]))
            for species_id in sorted(simplest)
        )

        def _selection_for_anchor(anchor_index: int) -> FormulaUnitSelection:
            anchor = self.crystal.molecules[anchor_index]
            running_centroid = np.asarray(anchor.get_centroid(), dtype=float)
            running_weight = len(anchor)
            selected = [
                FormulaUnitMember(anchor_species, anchor_index, (0, 0, 0))
            ]
            used = {anchor_index}

            def _best_shift(molecule_index: int):
                centroid = np.asarray(
                    self.crystal.molecules[molecule_index].get_centroid(), dtype=float
                )
                delta = centroid - running_centroid
                delta_reduced_frac = delta @ inv_reduced_lattice
                base_reduced_shift = np.zeros(3, dtype=int)
                base_reduced_shift[periodic] = -np.floor(
                    delta_reduced_frac[periodic]
                ).astype(int)
                scored = []
                for offset in itertools.product(*neighbor_ranges):
                    reduced_shift = base_reduced_shift + np.asarray(
                        offset, dtype=int
                    )
                    shift = reduced_shift @ reduction
                    shifted = centroid + shift @ lattice
                    distance = float(np.linalg.norm(shifted - running_centroid))
                    scored.append((distance, tuple(int(v) for v in shift)))
                return min(scored, key=lambda item: (round(item[0], 12), item[1]))

            for species_id in species_order:
                required = int(simplest[species_id])
                if species_id == anchor_species:
                    required -= 1
                for _ in range(required):
                    candidates = []
                    for molecule_index in sorted(self.species_map[species_id]):
                        if molecule_index in used:
                            continue
                        distance, shift = _best_shift(molecule_index)
                        candidates.append((round(distance, 12), molecule_index, shift))
                    if not candidates:
                        raise RuntimeError(
                            f"Not enough molecules to select {simplest[species_id]} "
                            f"member(s) of species {species_id!r}"
                        )
                    _, molecule_index, shift = min(candidates)
                    molecule = self.crystal.molecules[molecule_index]
                    shifted_centroid = (
                        np.asarray(molecule.get_centroid())
                        + np.asarray(shift) @ lattice
                    )
                    new_weight = running_weight + len(molecule)
                    running_centroid = (
                        running_centroid * running_weight
                        + shifted_centroid * len(molecule)
                    ) / new_weight
                    running_weight = new_weight
                    used.add(molecule_index)
                    selected.append(
                        FormulaUnitMember(species_id, molecule_index, shift)
                    )

            return FormulaUnitSelection(tuple(selected), counts)

        def _compactness(selection: FormulaUnitSelection):
            centroids = [
                np.asarray(
                    self.crystal.molecules[member.molecule_index].get_centroid(),
                    dtype=float,
                )
                + np.asarray(member.image_shift) @ lattice
                for member in selection.members
            ]
            distances = [
                float(np.linalg.norm(left - right))
                for left, right in itertools.combinations(centroids, 2)
            ]
            deterministic_key = tuple(
                (member.molecule_index, member.image_shift)
                for member in selection.members
            )
            return (
                round(max(distances, default=0.0), 12),
                round(sum(distances), 12),
                deterministic_key,
            )

        selections = [
            _selection_for_anchor(anchor_index)
            for anchor_index in sorted(self.species_map[anchor_species])
        ]
        return min(selections, key=_compactness)

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


__all__ = [
    "FormulaUnitMember",
    "FormulaUnitSelection",
    "StoichiometryAnalyzer",
]
