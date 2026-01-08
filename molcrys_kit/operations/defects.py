"""
Spatial defect generation for molecular crystals.

This module provides functionality for generating vacancies (defects) by removing
specific molecular clusters based on spatial relationships.
"""

from typing import Dict, List, Optional, Tuple, Union
from ..structures.crystal import MolecularCrystal
from ..analysis.stoichiometry import StoichiometryAnalyzer
from ..utils.geometry import minimum_image_distance


class VacancyGenerator:
    """
    Generates spatial defects (vacancies) in molecular crystals by removing specific molecular clusters.
    """

    def __init__(self, crystal: MolecularCrystal):
        """
        Initialize the vacancy generator with a molecular crystal.

        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to generate vacancies in.
        """
        self.crystal = crystal
        self.analyzer = StoichiometryAnalyzer(crystal)

    def find_removable_cluster_indices(
        self,
        target_spec: Optional[Dict[str, int]] = None,
        seed_index: Optional[int] = None,
    ) -> List[int]:
        """
        Find the indices of molecules that would be removed to form a vacancy cluster.

        Parameters
        ----------
        target_spec : Dict[str, int], optional
            Dictionary mapping species IDs to counts to remove. If None, uses simplest unit.
        seed_index : int, optional
            Index of the molecule to start removing from. If None, picks randomly from rarest species.

        Returns
        -------
        List[int]
            List of molecule indices to remove.
        """
        if target_spec is None:
            target_spec = self.analyzer.get_simplest_unit()

        # Validate that crystal has enough molecules of requested types
        for species_id, count in target_spec.items():
            available_count = len(self.analyzer.species_map[species_id])
            if count > available_count:
                raise ValueError(
                    f"Not enough molecules of type {species_id}. "
                    f"Requested: {count}, Available: {available_count}"
                )

        # Create a copy of the target spec to track what's still needed
        needed_spec = target_spec.copy()

        # Initialize removal list
        removal_list = []

        # Handle seeding
        if seed_index is not None:
            # Add the seed molecule to removal list
            seed_mol = self.crystal.molecules[seed_index]
            seed_formula = seed_mol.get_chemical_formula()

            # Find which species this seed belongs to
            seed_species_id = None
            for species_id, indices in self.analyzer.species_map.items():
                if seed_index in indices:
                    seed_species_id = species_id
                    break

            if seed_species_id is None or seed_species_id not in needed_spec:
                raise ValueError(
                    f"Seed index {seed_index} does not belong to a requested species"
                )

            removal_list.append(seed_index)
            needed_spec[seed_species_id] -= 1
            if needed_spec[seed_species_id] <= 0:
                del needed_spec[seed_species_id]
        else:
            # Pick a random molecule from the rarest requested species
            rarest_species = min(
                (
                    s_id
                    for s_id in needed_spec.keys()
                    if s_id in self.analyzer.species_map
                ),
                key=lambda s_id: len(
                    [
                        idx
                        for idx in self.analyzer.species_map[s_id]
                        if idx not in removal_list
                    ]
                ),
            )

            # Find an available molecule of the rarest species
            available_indices = [
                idx
                for idx in self.analyzer.species_map[rarest_species]
                if idx not in removal_list
            ]
            if not available_indices:
                raise ValueError(
                    f"No available molecules of rarest species {rarest_species}"
                )

            seed_index = available_indices[0]
            removal_list.append(seed_index)
            needed_spec[rarest_species] -= 1
            if needed_spec[rarest_species] <= 0:
                del needed_spec[rarest_species]

        # Cluster expansion loop
        while needed_spec:
            # Find all candidate molecules that match types currently in needed_spec
            candidate_indices = []
            for species_id, count in needed_spec.items():
                species_indices = self.analyzer.species_map[species_id]
                # Only include those not already in removal_list
                for idx in species_indices:
                    if idx not in removal_list:
                        candidate_indices.append(idx)

            if not candidate_indices:
                break  # Should not happen if validation passed

            # Calculate distance from any molecule in removal_list to each candidate
            # Using minimum image convention for periodic boundary conditions
            min_distances = []
            for candidate_idx in candidate_indices:
                candidate_pos = self.crystal.molecules[
                    candidate_idx
                ].get_centroid_frac()

                # Find minimum distance to any molecule already in removal_list
                min_dist = float("inf")
                for removed_idx in removal_list:
                    removed_pos = self.crystal.molecules[
                        removed_idx
                    ].get_centroid_frac()
                    dist = minimum_image_distance(
                        removed_pos, candidate_pos, self.crystal.lattice
                    )
                    if dist < min_dist:
                        min_dist = dist

                min_distances.append((min_dist, candidate_idx))

            # Find the candidate with the absolute minimum distance
            min_dist, closest_candidate_idx = min(min_distances, key=lambda x: x[0])

            # Add this candidate to removal list
            closest_mol = self.crystal.molecules[closest_candidate_idx]
            closest_formula = closest_mol.get_chemical_formula()

            # Find which species this belongs to
            closest_species_id = None
            for species_id, indices in self.analyzer.species_map.items():
                if closest_candidate_idx in indices:
                    closest_species_id = species_id
                    break

            if closest_species_id is not None and closest_species_id in needed_spec:
                removal_list.append(closest_candidate_idx)
                needed_spec[closest_species_id] -= 1
                if needed_spec[closest_species_id] <= 0:
                    del needed_spec[closest_species_id]

        return removal_list

    def generate_vacancy(
        self,
        target_spec: Optional[Dict[str, int]] = None,
        seed_index: Optional[int] = None,
        method: str = "spatial_cluster",
        return_removed_cluster: bool = False,
    ) -> Union[MolecularCrystal, Tuple[MolecularCrystal, MolecularCrystal]]:
        """
        Generate a vacancy by removing a cluster of molecules.

        Parameters
        ----------
        target_spec : Dict[str, int], optional
            Dictionary mapping species IDs to counts to remove. If None, uses simplest unit.
        seed_index : int, optional
            Index of the molecule to start removing from. If None, picks randomly from rarest species.
        method : str, default='spatial_cluster'
            Method to use for selecting molecules to remove. Currently only supports 'spatial_cluster'.
        return_removed_cluster : bool, default=False
            If True, also returns the cluster of removed molecules as a separate crystal.

        Returns
        -------
        MolecularCrystal or Tuple[MolecularCrystal, MolecularCrystal]
            If return_removed_cluster is False, returns a new crystal with the specified molecules removed.
            If return_removed_cluster is True, returns a tuple containing:
            - The new crystal with the specified molecules removed
            - A crystal containing only the removed molecules
        """
        if method != "spatial_cluster":
            raise ValueError(f"Method {method} not supported. Use 'spatial_cluster'.")

        removal_indices = self.find_removable_cluster_indices(target_spec, seed_index)

        # Create new list of molecules excluding those to be removed
        remaining_molecules = [
            mol.copy()
            for idx, mol in enumerate(self.crystal.molecules)
            if idx not in removal_indices
        ]

        # Create new crystal with remaining molecules
        new_crystal = MolecularCrystal(
            lattice=self.crystal.lattice.copy(),
            molecules=remaining_molecules,
            pbc=self.crystal.pbc,
        )

        if return_removed_cluster:
            # Create list of removed molecules
            removed_molecules = [
                mol.copy()
                for idx, mol in enumerate(self.crystal.molecules)
                if idx in removal_indices
            ]

            # Create new crystal with removed molecules
            removed_cluster = MolecularCrystal(
                lattice=self.crystal.lattice.copy(),
                molecules=removed_molecules,
                pbc=self.crystal.pbc,
            )

            return new_crystal, removed_cluster

        return new_crystal


def generate_vacancy(
    crystal: MolecularCrystal,
    species_list: Optional[List[Dict[str, Union[str, int]]]] = None,
    seed_index: Optional[int] = None,
    method: str = "spatial_cluster",
    return_removed_cluster: bool = False,
) -> Union[MolecularCrystal, Tuple[MolecularCrystal, MolecularCrystal]]:
    """
    Public API wrapper to generate a vacancy by removing a cluster of molecules.

    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to generate vacancies in.
    species_list : List[Dict[str, Union[str, int]]], optional
        List of dictionaries mapping species IDs to counts to remove. Each dict has format:
        {"species_id": "identifier", "count": int}. If None, uses simplest unit.
    seed_index : int, optional
        Index of the molecule to start removing from. If None, picks randomly from rarest species.
    method : str, default='spatial_cluster'
        Method to use for selecting molecules to remove. Currently only supports 'spatial_cluster'.
    return_removed_cluster : bool, default=False
        If True, also returns the cluster of removed molecules as a separate crystal.

    Returns
    -------
    MolecularCrystal or Tuple[MolecularCrystal, MolecularCrystal]
        If return_removed_cluster is False, returns a new crystal with the specified molecules removed.
        If return_removed_cluster is True, returns a tuple containing:
        - The new crystal with the specified molecules removed
        - A crystal containing only the removed molecules
    """
    # Convert species_list to the internal target_spec format if provided
    target_spec = None
    if species_list is not None:
        target_spec = {}
        for item in species_list:
            species_id = item["species_id"]
            count = item["count"]
            target_spec[species_id] = count

    generator = VacancyGenerator(crystal)
    return generator.generate_vacancy(
        target_spec=target_spec,
        seed_index=seed_index,
        method=method,
        return_removed_cluster=return_removed_cluster,
    )
