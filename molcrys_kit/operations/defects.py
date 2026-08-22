"""
Spatial defect generation for molecular crystals.

This module provides functionality for generating vacancies (defects) by removing
specific molecular clusters based on spatial relationships.
"""

from __future__ import annotations

import copy
import itertools
import math
import random as _random
from collections.abc import Iterator, Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..analysis.stoichiometry import StoichiometryAnalyzer
from ..utils.geometry import minimum_image_distance
from .implicit_shape import (
    DEFAULT_SHAPE_BATCH_SIZE,
    ImplicitShape,
    evaluate_shape_field,
    merge_stable_topk,
    resolve_shape_center,
)
from .modeling_readiness import (
    require_complete_topology_units,
    warn_if_unresolved_disorder,
)


_CHARGE_TOLERANCE = 1e-8


def _copy_partition_molecule(molecule: CrystalMolecule) -> CrystalMolecule:
    copied = molecule.copy()
    copied.info.pop("atom_indices", None)
    copied.info.pop("bond_records", None)
    copied.info.pop("bond_pairs", None)
    return copied


def _partition_extra_arrays(
    crystal: MolecularCrystal,
    molecule_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    source_indices = crystal._molecule_global_indices()
    result: dict[str, np.ndarray] = {}
    for key, values in crystal.extra_arrays.items():
        array = np.asarray(values)
        blocks = [array[source_indices[index]] for index in molecule_indices]
        result[key] = np.concatenate(blocks, axis=0) if blocks else array[:0].copy()
    return result


def _partition_by_molecule_indices(
    crystal: MolecularCrystal,
    removal_indices: Sequence[int],
    *,
    operation_key: str,
    operation_info: Mapping[str, object],
    return_removed_cluster: bool,
) -> MolecularCrystal | tuple[MolecularCrystal, MolecularCrystal]:
    """Materialize a topology-preserving molecule partition."""
    removal_set = {int(index) for index in removal_indices}
    remaining_indices = [
        index for index in range(len(crystal.molecules)) if index not in removal_set
    ]
    removed_indices = [
        index for index in range(len(crystal.molecules)) if index in removal_set
    ]
    if not remaining_indices:
        raise ValueError("The requested defect would remove every molecule from the structure.")
    if not removed_indices:
        raise ValueError("The requested defect selected no complete molecules or ions.")

    base_metadata = copy.deepcopy(crystal.metadata)
    base_metadata[operation_key] = copy.deepcopy(dict(operation_info))
    remaining = MolecularCrystal(
        lattice=crystal.lattice.copy(),
        molecules=[
            _copy_partition_molecule(crystal.molecules[index])
            for index in remaining_indices
        ],
        pbc=crystal.pbc,
        metadata=base_metadata,
        extra_arrays=_partition_extra_arrays(crystal, remaining_indices),
    )
    if not return_removed_cluster:
        return remaining

    removed_metadata = copy.deepcopy(crystal.metadata)
    removed_metadata[f"{operation_key}_removed"] = copy.deepcopy(dict(operation_info))
    removed = MolecularCrystal(
        lattice=crystal.lattice.copy(),
        molecules=[
            _copy_partition_molecule(crystal.molecules[index])
            for index in removed_indices
        ],
        pbc=crystal.pbc,
        metadata=removed_metadata,
        extra_arrays=_partition_extra_arrays(crystal, removed_indices),
    )
    return remaining, removed


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
        self._readiness = warn_if_unresolved_disorder(
            crystal, operation="VacancyGenerator"
        )
        require_complete_topology_units(
            self._readiness, operation="VacancyGenerator"
        )

    def find_removable_cluster_indices(
        self,
        target_spec: Optional[Dict[str, int]] = None,
        seed_index: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> List[int]:
        """
        Find the indices of molecules that would be removed to form a vacancy cluster.

        Parameters
        ----------
        target_spec : Dict[str, int], optional
            Dictionary mapping species IDs to counts to remove. If None, uses simplest unit.
        seed_index : int, optional
            Index of the molecule to start removing from. If None, a molecule is chosen
            stochastically from the rarest requested species.
        random_seed : int, optional
            Seed for the random number generator used when ``seed_index`` is None.
            Pass an integer for reproducible selection; omit (or pass None) for
            non-deterministic behaviour.

        Returns
        -------
        List[int]
            List of molecule indices to remove.
        """
        if target_spec is None:
            target_spec = self.analyzer.get_simplest_unit()

        # Validate that crystal has enough molecules of requested types
        for species_id, count in target_spec.items():
            if species_id not in self.analyzer.species_map:
                raise ValueError(
                    f"Species '{species_id}' not found in crystal. "
                    f"Available species: {sorted(self.analyzer.species_map)}"
                )
            available_count = len(self.analyzer.species_map[species_id])
            if count > available_count:
                raise ValueError(
                    f"Not enough molecules of type {species_id}. "
                    f"Requested: {count}, Available: {available_count}"
                )

        # Local RNG so we never touch the global random state.
        rng = _random.Random(random_seed)

        # Create a copy of the target spec to track what's still needed
        needed_spec = target_spec.copy()

        # Initialize removal list
        removal_list = []

        # Handle seeding
        if seed_index is not None:
            if seed_index < 0 or seed_index >= len(self.crystal.molecules):
                raise ValueError(f"seed_index {seed_index} is out of range")
            # Add the seed molecule to removal list
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
            # Stochastically pick a molecule from the rarest requested species.
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

            # Find available molecules of the rarest species
            available_indices = [
                idx
                for idx in self.analyzer.species_map[rarest_species]
                if idx not in removal_list
            ]
            if not available_indices:
                raise ValueError(
                    f"No available molecules of rarest species {rarest_species}"
                )

            seed_index = rng.choice(available_indices)
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
        random_seed: Optional[int] = None,
    ) -> Union[MolecularCrystal, Tuple[MolecularCrystal, MolecularCrystal]]:
        """
        Generate a vacancy by removing a cluster of molecules.

        Parameters
        ----------
        target_spec : Dict[str, int], optional
            Dictionary mapping species IDs to counts to remove. If None, uses simplest unit.
        seed_index : int, optional
            Index of the molecule to start removing from. If None, a molecule is chosen
            stochastically from the rarest requested species.
        method : str, default='spatial_cluster'
            Method to use for selecting molecules to remove. Currently only supports 'spatial_cluster'.
        return_removed_cluster : bool, default=False
            If True, also returns the cluster of removed molecules as a separate crystal.
        random_seed : int, optional
            Seed for the random number generator used when ``seed_index`` is None.
            Pass an integer for reproducible results; omit for non-deterministic behaviour.

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

        actual_spec = (
            self.analyzer.get_simplest_unit()
            if target_spec is None
            else dict(target_spec)
        )
        removal_indices = self.find_removable_cluster_indices(
            actual_spec, seed_index, random_seed=random_seed
        )
        info = {
            "selection_mode": "spatial_cluster",
            "target_spec": {key: int(value) for key, value in actual_spec.items()},
            "removed_molecule_indices": [int(index) for index in removal_indices],
            "removed_molecule_count": len(removal_indices),
            "removed_atom_count": sum(
                len(self.crystal.molecules[index]) for index in removal_indices
            ),
            "modeling_readiness": self._readiness.to_dict(),
        }
        return _partition_by_molecule_indices(
            self.crystal,
            removal_indices,
            operation_key="vacancy",
            operation_info=info,
            return_removed_cluster=return_removed_cluster,
        )


class VoidCarver:
    """Remove complete molecules or ions according to an implicit 3-D shape.

    The source lattice and periodic boundary flags are preserved.  Selection
    is globally stoichiometric by topology-derived species; no atoms are cut,
    no bonds are rebuilt, and no surface caps are added.
    """

    def __init__(
        self,
        crystal: MolecularCrystal,
        batch_size: int = DEFAULT_SHAPE_BATCH_SIZE,
    ):
        lattice = np.asarray(crystal.lattice, dtype=float)
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            raise ValueError("crystal lattice must be a finite 3 x 3 matrix.")
        if abs(float(np.linalg.det(lattice))) < 1e-12:
            raise ValueError("void carving requires a non-singular 3-D lattice.")
        if not crystal.molecules:
            raise ValueError("crystal must contain at least one molecule or ion.")
        if any(len(molecule) == 0 for molecule in crystal.molecules):
            raise ValueError("void carving does not support empty topology units.")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        self.crystal = crystal
        self.batch_size = int(batch_size)
        self._lattice = lattice.copy()
        self._inverse_lattice = np.linalg.inv(lattice)
        self._periodic = np.asarray(crystal.pbc, dtype=bool)
        self.analyzer = StoichiometryAnalyzer(crystal)
        self._species_by_index = np.empty(len(crystal.molecules), dtype=object)
        for species_id, indices in self.analyzer.species_map.items():
            self._species_by_index[np.asarray(indices, dtype=int)] = species_id
        self._readiness = warn_if_unresolved_disorder(crystal, operation="VoidCarver")
        require_complete_topology_units(self._readiness, operation="VoidCarver")

    def carve(
        self,
        shape: ImplicitShape,
        *,
        center: Sequence[float] | None = None,
        center_frac: Sequence[float] | None = None,
        hit_mode: str = "centroid",
        target_spec: Mapping[str, int] | None = None,
        target_units: int | None = None,
        boundary_policy: str = "inside",
        periodic_images: bool = True,
        species_charge_map: Mapping[str, float] | None = None,
        return_removed_cluster: bool = False,
    ) -> MolecularCrystal | tuple[MolecularCrystal, MolecularCrystal]:
        """Return the periodic source structure with a shaped void removed."""
        if not isinstance(shape, ImplicitShape):
            raise TypeError("shape must be an ImplicitShape.")
        if hit_mode not in {"centroid", "any_atom", "all_atoms"}:
            raise ValueError("hit_mode must be 'centroid', 'any_atom', or 'all_atoms'.")
        if boundary_policy not in {"inside", "cover"}:
            raise ValueError("boundary_policy must be 'inside' or 'cover'.")
        if not isinstance(periodic_images, (bool, np.bool_)):
            raise TypeError("periodic_images must be a boolean.")
        if target_units is not None:
            if (
                isinstance(target_units, bool)
                or not isinstance(target_units, (int, np.integer))
                or target_units <= 0
            ):
                raise ValueError("target_units must be a positive integer.")
            target_units = int(target_units)

        clean_spec = self._validate_target_spec(target_spec)
        unit_charge = self._validate_species_charges(clean_spec, species_charge_map)
        shape_center, shape_center_frac = resolve_shape_center(
            self._lattice, center, center_frac
        )
        image_translations = self._shape_image_translations(
            shape,
            shape_center,
            hit_mode,
            periodic_images=bool(periodic_images),
        )

        if target_units is not None:
            selected_unit_count = target_units
            desired = {
                species_id: selected_unit_count * ratio
                for species_id, ratio in clean_spec.items()
            }
            raw_inside_counts, selected = self._scan_scores(
                shape,
                shape_center,
                image_translations,
                hit_mode,
                species_ids=clean_spec,
                desired=desired,
                inside_only=False,
            )
        else:
            raw_inside_counts, _ = self._scan_scores(
                shape,
                shape_center,
                image_translations,
                hit_mode,
                species_ids=clean_spec,
            )
            if boundary_policy == "inside":
                selected_unit_count = min(
                    raw_inside_counts[species_id] // ratio
                    for species_id, ratio in clean_spec.items()
                )
                inside_only = True
            else:
                selected_unit_count = max(
                    math.ceil(raw_inside_counts[species_id] / ratio)
                    for species_id, ratio in clean_spec.items()
                )
                inside_only = False
            if selected_unit_count <= 0:
                raise ValueError(
                    "The requested shape does not contain enough species to remove one "
                    "complete stoichiometric unit."
                )
            desired = {
                species_id: selected_unit_count * ratio
                for species_id, ratio in clean_spec.items()
            }
            _, selected = self._scan_scores(
                shape,
                shape_center,
                image_translations,
                hit_mode,
                species_ids=clean_spec,
                desired=desired,
                inside_only=inside_only,
            )

        removal_indices = np.sort(
            np.concatenate([selected[species_id] for species_id in sorted(selected)])
        )
        removed_species_counts = {
            species_id: int(len(selected[species_id]))
            for species_id in sorted(selected)
        }
        removed_atom_count = sum(
            len(self.crystal.molecules[int(index)]) for index in removal_indices
        )
        removed_mass = sum(
            float(self.crystal.molecules[int(index)].get_masses().sum())
            for index in removal_indices
        )
        removed_charge = (
            float(selected_unit_count * unit_charge)
            if species_charge_map is not None
            else None
        )
        info: dict[str, object] = {
            "shape": shape.name,
            "shape_parameters": dict(shape.parameters),
            "bounds_A": shape.bounds.tolist(),
            "center_A": shape_center.tolist(),
            "center_frac": shape_center_frac.tolist(),
            "selection_mode": (
                "fixed_count" if target_units is not None else "fixed_geometry"
            ),
            "hit_mode": hit_mode,
            "boundary_policy": boundary_policy,
            "periodic_images": bool(periodic_images),
            "shape_image_count": int(len(image_translations)),
            "target_spec": {key: int(value) for key, value in clean_spec.items()},
            "target_units": target_units,
            "selected_unit_count": int(selected_unit_count),
            "raw_inside_species_counts": {
                key: int(value) for key, value in raw_inside_counts.items()
            },
            "removed_species_counts": removed_species_counts,
            "removed_molecule_count": int(len(removal_indices)),
            "removed_atom_count": int(removed_atom_count),
            "removed_mass_amu": float(removed_mass),
            "remaining_molecule_count": int(
                len(self.crystal.molecules) - len(removal_indices)
            ),
            "remaining_atom_count": int(
                sum(len(molecule) for molecule in self.crystal.molecules)
                - removed_atom_count
            ),
            "charge_verified": species_charge_map is not None,
            "removed_net_charge_e": removed_charge,
            "batch_size": int(self.batch_size),
            "modeling_readiness": self._readiness.to_dict(),
            "source_formula_moiety": self.crystal.formula_moiety,
        }
        return _partition_by_molecule_indices(
            self.crystal,
            removal_indices,
            operation_key="void",
            operation_info=info,
            return_removed_cluster=return_removed_cluster,
        )

    def _validate_target_spec(
        self,
        target_spec: Mapping[str, int] | None,
    ) -> dict[str, int]:
        if target_spec is None:
            target_spec = self.analyzer.get_simplest_unit()
        if not isinstance(target_spec, Mapping) or not target_spec:
            raise ValueError("target_spec must be a non-empty species/count mapping.")
        clean: dict[str, int] = {}
        for species_id, count in target_spec.items():
            if species_id not in self.analyzer.species_map:
                raise ValueError(
                    f"Species '{species_id}' not found in crystal. "
                    f"Available species: {sorted(self.analyzer.species_map)}"
                )
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count <= 0
            ):
                raise ValueError(f"target_spec count for {species_id!r} must be positive.")
            clean[str(species_id)] = int(count)
        return dict(sorted(clean.items()))

    @staticmethod
    def _validate_species_charges(
        target_spec: Mapping[str, int],
        species_charge_map: Mapping[str, float] | None,
    ) -> float:
        if species_charge_map is None:
            return 0.0
        if not isinstance(species_charge_map, Mapping):
            raise TypeError("species_charge_map must be a mapping.")
        missing = sorted(set(target_spec) - set(species_charge_map))
        if missing:
            raise ValueError(f"species_charge_map is missing target species: {missing}")
        unit_charge = 0.0
        for species_id, ratio in target_spec.items():
            charge = float(species_charge_map[species_id])
            if not np.isfinite(charge):
                raise ValueError("species charges must be finite values.")
            unit_charge += ratio * charge
        if abs(unit_charge) > _CHARGE_TOLERANCE:
            raise ValueError(
                f"target_spec has non-zero net charge {unit_charge:g} e under "
                "species_charge_map."
            )
        return unit_charge

    def _position_limits(self, hit_mode: str) -> tuple[np.ndarray, np.ndarray]:
        if hit_mode == "centroid":
            representatives = np.asarray(
                [molecule.get_positions().mean(axis=0) for molecule in self.crystal.molecules]
            )
            return representatives.min(axis=0), representatives.max(axis=0)
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for molecule in self.crystal.molecules:
            positions = molecule.get_positions()
            lower = np.minimum(lower, positions.min(axis=0))
            upper = np.maximum(upper, positions.max(axis=0))
        return lower, upper

    def _shape_image_translations(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        hit_mode: str,
        *,
        periodic_images: bool,
    ) -> np.ndarray:
        hkl = shape.parameters.get("direction_hkl")
        if hkl is not None:
            direction = np.asarray(hkl, dtype=int)
            if not periodic_images:
                raise ValueError("through_cylinder requires periodic_images=True.")
            if np.any((direction != 0) & ~self._periodic):
                raise ValueError(
                    "through_cylinder direction_hkl uses a non-periodic lattice direction."
                )
        if not periodic_images or not np.any(self._periodic):
            return np.zeros((1, 3), dtype=float)

        position_min, position_max = self._position_limits(hit_mode)
        position_corners = np.asarray(
            list(itertools.product(*np.column_stack((position_min, position_max))))
        )
        shape_corners = np.asarray(list(itertools.product(*shape.bounds)))
        fractional = (
            position_corners[:, None, :] - center - shape_corners[None, :, :]
        ) @ self._inverse_lattice
        lower = np.floor(fractional.min(axis=(0, 1))).astype(int)
        upper = np.ceil(fractional.max(axis=(0, 1))).astype(int)
        ranges = [
            range(int(lower[index]), int(upper[index]) + 1)
            if self._periodic[index]
            else (0,)
            for index in range(3)
        ]
        shifts = np.asarray(list(itertools.product(*ranges)), dtype=int)
        return shifts @ self._lattice

    def _iter_score_batches(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        image_translations: np.ndarray,
        hit_mode: str,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if hit_mode == "centroid":
            for start in range(0, len(self.crystal.molecules), self.batch_size):
                stop = min(start + self.batch_size, len(self.crystal.molecules))
                molecule_ids = np.arange(start, stop, dtype=np.int64)
                points = np.asarray(
                    [
                        self.crystal.molecules[index].get_positions().mean(axis=0)
                        for index in range(start, stop)
                    ]
                )
                scores = np.full(len(points), np.inf)
                for translation in image_translations:
                    scores = np.minimum(
                        scores,
                        evaluate_shape_field(shape, points - center - translation),
                    )
                yield molecule_ids, scores
            return

        group_ids: list[int] = []
        position_blocks: list[np.ndarray] = []
        atom_count = 0
        for molecule_id, molecule in enumerate(self.crystal.molecules):
            positions = molecule.get_positions()
            if len(positions) > self.batch_size:
                if group_ids:
                    yield self._score_atom_group(
                        shape,
                        center,
                        image_translations,
                        hit_mode,
                        group_ids,
                        position_blocks,
                    )
                    group_ids, position_blocks, atom_count = [], [], 0
                yield self._score_large_molecule(
                    shape,
                    center,
                    image_translations,
                    hit_mode,
                    molecule_id,
                    positions,
                )
                continue
            if group_ids and atom_count + len(positions) > self.batch_size:
                yield self._score_atom_group(
                    shape,
                    center,
                    image_translations,
                    hit_mode,
                    group_ids,
                    position_blocks,
                )
                group_ids, position_blocks, atom_count = [], [], 0
            group_ids.append(molecule_id)
            position_blocks.append(positions)
            atom_count += len(positions)
        if group_ids:
            yield self._score_atom_group(
                shape,
                center,
                image_translations,
                hit_mode,
                group_ids,
                position_blocks,
            )

    @staticmethod
    def _reduce_atom_scores(
        values: np.ndarray,
        offsets: np.ndarray,
        hit_mode: str,
    ) -> np.ndarray:
        if hit_mode == "any_atom":
            return np.minimum.reduceat(values, offsets)
        return np.maximum.reduceat(values, offsets)

    def _score_atom_group(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        image_translations: np.ndarray,
        hit_mode: str,
        group_ids: Sequence[int],
        position_blocks: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.vstack(position_blocks)
        offsets = np.cumsum([0, *[len(block) for block in position_blocks[:-1]]])
        scores = np.full(len(group_ids), np.inf)
        for translation in image_translations:
            values = evaluate_shape_field(shape, positions - center - translation)
            image_scores = self._reduce_atom_scores(values, offsets, hit_mode)
            scores = np.minimum(scores, image_scores)
        return np.asarray(group_ids, dtype=np.int64), scores

    def _score_large_molecule(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        image_translations: np.ndarray,
        hit_mode: str,
        molecule_id: int,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        score = np.inf
        for translation in image_translations:
            image_score = np.inf if hit_mode == "any_atom" else -np.inf
            for start in range(0, len(positions), self.batch_size):
                values = evaluate_shape_field(
                    shape,
                    positions[start : start + self.batch_size] - center - translation,
                )
                if hit_mode == "any_atom":
                    image_score = min(image_score, float(values.min()))
                else:
                    image_score = max(image_score, float(values.max()))
            score = min(score, image_score)
        return np.asarray([molecule_id], dtype=np.int64), np.asarray([score])

    def _scan_scores(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        image_translations: np.ndarray,
        hit_mode: str,
        *,
        species_ids: Sequence[str],
        desired: Mapping[str, int] | None = None,
        inside_only: bool = False,
    ) -> tuple[dict[str, int], dict[str, np.ndarray]]:
        target_species = sorted(species_ids)
        raw_inside_counts = {species_id: 0 for species_id in target_species}
        best_ids = {
            species_id: np.empty(0, dtype=np.int64) for species_id in target_species
        }
        best_scores = {
            species_id: np.empty(0, dtype=float) for species_id in target_species
        }
        for molecule_ids, scores in self._iter_score_batches(
            shape, center, image_translations, hit_mode
        ):
            species_values = self._species_by_index[molecule_ids]
            for species_id in target_species:
                species_mask = species_values == species_id
                raw_inside_counts[species_id] += int(
                    np.count_nonzero(species_mask & (scores <= 0.0))
                )
                if desired is None:
                    continue
                candidate_mask = species_mask
                if inside_only:
                    candidate_mask = candidate_mask & (scores <= 0.0)
                candidate_ids = molecule_ids[candidate_mask]
                if len(candidate_ids) == 0:
                    continue
                candidate_scores = scores[candidate_mask]
                best_ids[species_id], best_scores[species_id] = merge_stable_topk(
                    best_ids[species_id],
                    best_scores[species_id],
                    candidate_ids,
                    candidate_scores,
                    int(desired[species_id]),
                )
        if desired is not None:
            for species_id, count in desired.items():
                if len(best_ids[species_id]) < count:
                    qualifier = " inside the shape" if inside_only else ""
                    raise ValueError(
                        f"Need {count} molecule(s) of {species_id}{qualifier}, but only "
                        f"{len(best_ids[species_id])} are available."
                    )
        return raw_inside_counts, best_ids


def generate_vacancy(
    crystal: MolecularCrystal,
    species_list: Optional[List[Dict[str, Union[str, int]]]] = None,
    seed_index: Optional[int] = None,
    method: str = "spatial_cluster",
    return_removed_cluster: bool = False,
    random_seed: Optional[int] = None,
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
        Index of the molecule to start removing from. If None, a molecule is chosen
        stochastically from the rarest requested species.
    method : str, default='spatial_cluster'
        Method to use for selecting molecules to remove. Currently only supports 'spatial_cluster'.
    return_removed_cluster : bool, default=False
        If True, also returns the cluster of removed molecules as a separate crystal.
    random_seed : int, optional
        Seed for the random number generator used when ``seed_index`` is None.
        Pass an integer for reproducible results; omit for non-deterministic behaviour.

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
        random_seed=random_seed,
    )


def carve_void(
    crystal: MolecularCrystal,
    shape: ImplicitShape,
    *,
    center: Sequence[float] | None = None,
    center_frac: Sequence[float] | None = None,
    hit_mode: str = "centroid",
    target_spec: Mapping[str, int] | None = None,
    target_units: int | None = None,
    boundary_policy: str = "inside",
    periodic_images: bool = True,
    species_charge_map: Mapping[str, float] | None = None,
    return_removed_cluster: bool = False,
    batch_size: int = DEFAULT_SHAPE_BATCH_SIZE,
) -> MolecularCrystal | tuple[MolecularCrystal, MolecularCrystal]:
    """Convenience wrapper around :class:`VoidCarver`."""
    return VoidCarver(crystal, batch_size=batch_size).carve(
        shape,
        center=center,
        center_frac=center_frac,
        hit_mode=hit_mode,
        target_spec=target_spec,
        target_units=target_units,
        boundary_policy=boundary_policy,
        periodic_images=periodic_images,
        species_charge_map=species_charge_map,
        return_removed_cluster=return_removed_cluster,
    )


__all__ = [
    "VacancyGenerator",
    "VoidCarver",
    "carve_void",
    "generate_vacancy",
]
