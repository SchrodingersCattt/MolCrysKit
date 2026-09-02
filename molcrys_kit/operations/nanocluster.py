"""Topology-preserving nanocluster carving with implicit 3-D shapes.

Unlike :mod:`molcrys_kit.operations.cluster`, which deliberately cuts and
hydrogen-caps coordination clusters for finite-cluster QM calculations, this
module never cuts a molecule.  It selects either complete
:class:`~molcrys_kit.structures.molecule.CrystalMolecule` objects or complete
translated unit-cell packets from a periodic molecular crystal.

The shape is a vectorized implicit field ``f(x, y, z)`` with ``f <= 0`` inside
and an explicit Cartesian search box.  Candidate representatives are evaluated
in bounded NumPy batches; rejected candidates are never materialized as atom
objects.  Fixed-count carving keeps the globally smallest ``(field, stable_id)``
pairs with a streaming top-k, so memory is bounded by ``batch_size`` plus the
requested output size rather than by the full candidate supercell.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from typing import Optional, Sequence

import numpy as np

from ..analysis.disorder import UnresolvedDisorderWarning
from ..constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_IMAGE_SHIFT,
    KEY_OCCUPANCY,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule, _strip_stale_frac_arrays
from ..constants.config import DEFAULT_NANOCLUSTER_BATCH_SIZE
from .implicit_shape import (
    ImplicitShape,
    NanoShape,
    evaluate_shape_field,
    merge_stable_topk,
    resolve_shape_center,
)


def _warn_if_unresolved_disorder(
    crystal: MolecularCrystal,
) -> dict[str, bool | int]:
    nonunit_occupancies = 0
    active_groups = 0
    active_assemblies = 0
    for molecule in crystal.molecules:
        occupancies = np.asarray(molecule.arrays.get(KEY_OCCUPANCY, []), dtype=float)
        nonunit_occupancies += int(
            np.count_nonzero(~np.isclose(occupancies, 1.0, rtol=0.0, atol=1e-8))
        )
        active_groups += sum(
            str(value).strip() not in {"", ".", "?", "0"}
            for value in molecule.arrays.get(KEY_DISORDER_GROUP, [])
        )
        active_assemblies += sum(
            str(value).strip() not in {"", ".", "?", "0"}
            for value in molecule.arrays.get(KEY_ASSEMBLY, [])
        )
    stats = {
        "all_atom_ordered": not bool(
            nonunit_occupancies or active_groups or active_assemblies
        ),
        "nonunit_occupancy_count": nonunit_occupancies,
        "active_disorder_group_count": active_groups,
        "active_disorder_assembly_count": active_assemblies,
    }
    if not stats["all_atom_ordered"]:
        warnings.warn(
            "NanoClusterCarver is continuing with unresolved disorder; resolve an "
            "ordered replica before production MD.",
            UnresolvedDisorderWarning,
            stacklevel=2,
        )
    return stats


def _require_complete_topology_units(crystal: MolecularCrystal) -> None:
    incomplete_count = sum(
        molecule.info.get("unwrap_completed") is False
        for molecule in crystal.molecules
    )
    if incomplete_count:
        raise ValueError(
            "NanoClusterCarver requires finite, completely unwrapped molecules or "
            f"ions; {incomplete_count} topology unit(s) are incomplete. Periodic 3-D "
            "frameworks/MOFs are not supported and are not automatically cut or capped."
        )


class NanoClusterCarver:
    """Carve finite shapes from a 3-D periodic molecular crystal.

    Candidate representatives are molecule centroids/centers of mass for
    ``topology_unit='molecule'`` and translated crystallographic cell centers
    for ``topology_unit='unit_cell'``.  Only selected units are copied.
    """

    def __init__(
        self,
        crystal: MolecularCrystal,
        batch_size: int = DEFAULT_NANOCLUSTER_BATCH_SIZE,
    ):
        lattice = np.asarray(crystal.lattice, dtype=float)
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            raise ValueError("crystal lattice must be a finite 3 x 3 matrix.")
        if abs(float(np.linalg.det(lattice))) < 1e-12:
            raise ValueError("nanocluster carving requires a non-singular 3-D lattice.")
        if not np.asarray(crystal.pbc, dtype=bool).all():
            raise ValueError("nanocluster carving requires periodicity in all three dimensions.")
        if not crystal.molecules:
            raise ValueError("crystal must contain at least one molecule.")
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
        self._source_molecule_indices = crystal._molecule_global_indices()
        self._input_disorder = _warn_if_unresolved_disorder(crystal)
        _require_complete_topology_units(crystal)

    def carve(
        self,
        shape: ImplicitShape,
        *,
        topology_unit: str = "molecule",
        center: Optional[Sequence[float]] = None,
        center_frac: Optional[Sequence[float]] = None,
        center_kind: str = "centroid",
        target_units: Optional[int] = None,
        vacuum: float = 0.0,
    ) -> MolecularCrystal:
        """Return a finite topology-preserving nanocluster.

        When ``target_units`` is omitted, representatives inside ``shape``
        (``field <= 0``) are selected.  When it is provided, the globally
        smallest field values inside ``shape.bounds`` are selected, with the
        stable candidate id breaking ties.
        """
        if not isinstance(shape, ImplicitShape):
            raise TypeError("shape must be an ImplicitShape.")
        if topology_unit not in {"molecule", "unit_cell"}:
            raise ValueError("topology_unit must be 'molecule' or 'unit_cell'.")
        if center_kind not in {"centroid", "com"}:
            raise ValueError("center_kind must be 'centroid' or 'com'.")
        if target_units is not None:
            if (
                isinstance(target_units, bool)
                or not isinstance(target_units, (int, np.integer))
                or target_units <= 0
            ):
                raise ValueError("target_units must be a positive integer.")
            target_units = int(target_units)
        vacuum_value = float(vacuum)
        if not np.isfinite(vacuum_value) or vacuum_value < 0:
            raise ValueError("vacuum must be a non-negative finite value.")

        shape_center, shape_center_frac = resolve_shape_center(
            self._lattice, center, center_frac
        )

        representatives = self._base_representatives(topology_unit, center_kind)
        if topology_unit == "molecule":
            formulas = {molecule.get_chemical_formula() for molecule in self.crystal.molecules}
            if len(formulas) > 1:
                warnings.warn(
                    "Molecule-level nanocluster carving preserves each molecule but does "
                    "not guarantee charge neutrality or source-cell stoichiometry; use "
                    "topology_unit='unit_cell' when exact composition is required.",
                    UserWarning,
                    stacklevel=2,
                )

        lower, dimensions = self._translation_grid(shape, shape_center, representatives)
        selected_ids, valid_candidate_count, grid_candidate_count = self._select_candidate_ids(
            shape=shape,
            center=shape_center,
            representatives=representatives,
            lower=lower,
            dimensions=dimensions,
            target_units=target_units,
        )
        if selected_ids.size == 0:
            raise ValueError("The requested shape selected no topology units.")

        return self._materialize(
            selected_ids=selected_ids,
            lower=lower,
            dimensions=dimensions,
            topology_unit=topology_unit,
            center_kind=center_kind,
            shape=shape,
            shape_center=shape_center,
            shape_center_frac=shape_center_frac,
            target_units=target_units,
            vacuum=vacuum_value,
            valid_candidate_count=valid_candidate_count,
            grid_candidate_count=grid_candidate_count,
        )

    def _base_representatives(self, topology_unit: str, center_kind: str) -> np.ndarray:
        if topology_unit == "unit_cell":
            return np.array([0.5 * np.sum(self._lattice, axis=0)], dtype=float)
        if center_kind == "centroid":
            return np.array(
                [molecule.get_positions().mean(axis=0) for molecule in self.crystal.molecules],
                dtype=float,
            )
        return np.array(
            [molecule.get_center_of_mass() for molecule in self.crystal.molecules],
            dtype=float,
        )

    def _translation_grid(
        self,
        shape: ImplicitShape,
        center: np.ndarray,
        representatives: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        corners = np.array(list(itertools.product(*shape.bounds)), dtype=float)
        fractional = (
            center[None, None, :] + corners[:, None, :] - representatives[None, :, :]
        ) @ self._inverse_lattice
        fractional_min = fractional.min(axis=(0, 1))
        fractional_max = fractional.max(axis=(0, 1))
        index_limits = np.iinfo(np.int64)
        lower_float = np.floor(fractional_min)
        upper_float = np.ceil(fractional_max)
        safe_min = np.nextafter(float(index_limits.min), 0.0)
        safe_max = np.nextafter(float(index_limits.max), 0.0)
        if (
            not np.isfinite(lower_float).all()
            or not np.isfinite(upper_float).all()
            or np.any(lower_float < safe_min)
            or np.any(upper_float > safe_max)
        ):
            raise ValueError("shape bounds generate an invalid integer translation range.")
        lower = lower_float.astype(np.int64)
        upper = upper_float.astype(np.int64)
        dimensions = tuple(
            int(upper_value) - int(lower_value) + 1
            for lower_value, upper_value in zip(lower, upper)
        )
        if any(value <= 0 for value in dimensions):  # pragma: no cover
            raise ValueError("shape bounds produced an empty translation grid.")
        return lower, dimensions

    @staticmethod
    def _decode_translations(
        translation_ids: np.ndarray,
        lower: np.ndarray,
        dimensions: tuple[int, int, int],
    ) -> np.ndarray:
        n_yz = dimensions[1] * dimensions[2]
        x_index = translation_ids // n_yz
        remainder = translation_ids % n_yz
        y_index = remainder // dimensions[2]
        z_index = remainder % dimensions[2]
        return np.column_stack((x_index, y_index, z_index)).astype(np.int64) + lower

    def _select_candidate_ids(
        self,
        *,
        shape: ImplicitShape,
        center: np.ndarray,
        representatives: np.ndarray,
        lower: np.ndarray,
        dimensions: tuple[int, int, int],
        target_units: Optional[int],
    ) -> tuple[np.ndarray, int, int]:
        translation_count = math.prod(dimensions)
        base_count = len(representatives)
        grid_candidate_count = translation_count * base_count
        if grid_candidate_count > np.iinfo(np.int64).max:
            raise ValueError("shape bounds generate too many candidates for stable indexing.")

        selected_chunks: list[np.ndarray] = []
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=float)
        valid_candidate_count = 0
        bounds_lower = shape.bounds[:, 0]
        bounds_upper = shape.bounds[:, 1]

        for start in range(0, grid_candidate_count, self.batch_size):
            stop = min(start + self.batch_size, grid_candidate_count)
            candidate_ids = np.arange(start, stop, dtype=np.int64)
            base_ids = candidate_ids % base_count
            translation_ids = candidate_ids // base_count
            shifts = self._decode_translations(translation_ids, lower, dimensions)
            positions = representatives[base_ids] + shifts @ self._lattice
            local_positions = positions - center
            in_bounds = np.all(
                (local_positions >= bounds_lower) & (local_positions <= bounds_upper),
                axis=1,
            )
            if not np.any(in_bounds):
                continue
            bounded_ids = candidate_ids[in_bounds]
            bounded_positions = local_positions[in_bounds]
            scores = evaluate_shape_field(shape, bounded_positions)
            valid_candidate_count += len(bounded_ids)

            if target_units is None:
                selected_chunks.append(bounded_ids[scores <= 0.0])
                continue

            best_ids, best_scores = merge_stable_topk(
                best_ids,
                best_scores,
                bounded_ids,
                scores,
                target_units,
            )

        if target_units is None:
            selected_ids = (
                np.concatenate(selected_chunks)
                if selected_chunks
                else np.empty(0, dtype=np.int64)
            )
        else:
            if valid_candidate_count < target_units:
                raise ValueError(
                    f"target_units={target_units} exceeds the {valid_candidate_count} "
                    "candidate units inside bounds; enlarge the shape bounds."
                )
            selected_ids = best_ids

        return np.sort(selected_ids), valid_candidate_count, grid_candidate_count

    def _copy_molecule(
        self,
        molecule: CrystalMolecule,
        translation: np.ndarray,
    ) -> CrystalMolecule:
        copied = molecule.copy()
        copied.positions += translation
        copied.set_pbc(False)
        copied.info.pop("atom_indices", None)
        copied.info.pop("bond_records", None)
        copied.info.pop("bond_pairs", None)
        _strip_stale_frac_arrays(copied)
        if KEY_IMAGE_SHIFT in copied.arrays:
            del copied.arrays[KEY_IMAGE_SHIFT]
        return copied

    def _materialize(
        self,
        *,
        selected_ids: np.ndarray,
        lower: np.ndarray,
        dimensions: tuple[int, int, int],
        topology_unit: str,
        center_kind: str,
        shape: ImplicitShape,
        shape_center: np.ndarray,
        shape_center_frac: np.ndarray,
        target_units: Optional[int],
        vacuum: float,
        valid_candidate_count: int,
        grid_candidate_count: int,
    ) -> MolecularCrystal:
        base_count = len(self.crystal.molecules) if topology_unit == "molecule" else 1
        base_ids = selected_ids % base_count
        translation_ids = selected_ids // base_count
        shifts = self._decode_translations(translation_ids, lower, dimensions)
        translations = shifts @ self._lattice

        molecules: list[CrystalMolecule] = []
        selected_base_molecules: list[int] = []
        if topology_unit == "molecule":
            for base_id, translation in zip(base_ids, translations):
                molecule_index = int(base_id)
                molecules.append(
                    self._copy_molecule(self.crystal.molecules[molecule_index], translation)
                )
                selected_base_molecules.append(molecule_index)
        else:
            for translation in translations:
                for molecule_index, molecule in enumerate(self.crystal.molecules):
                    molecules.append(self._copy_molecule(molecule, translation))
                    selected_base_molecules.append(molecule_index)

        all_positions = np.vstack([molecule.get_positions() for molecule in molecules])
        atom_min = np.min(all_positions, axis=0)
        atom_max = np.max(all_positions, axis=0)
        span = atom_max - atom_min
        cell_lengths = span + 2.0 * vacuum
        cell_lengths[cell_lengths < 1e-8] = 1.0
        output_lattice = np.diag(cell_lengths)
        output_shift = np.full(3, vacuum, dtype=float) - atom_min
        for molecule in molecules:
            molecule.positions += output_shift
            molecule.set_cell(output_lattice)
            molecule.set_pbc(False)

        extra_arrays = self._replicate_extra_arrays(selected_base_molecules)
        selected_atom_count = sum(len(molecule) for molecule in molecules)
        metadata = copy.deepcopy(self.crystal.metadata)
        metadata["nanocluster"] = {
            "shape": shape.name,
            "shape_parameters": dict(shape.parameters),
            "bounds_A": shape.bounds.tolist(),
            "source_center_A": shape_center.tolist(),
            "source_center_frac": shape_center_frac.tolist(),
            "output_shift_A": output_shift.tolist(),
            "selection_mode": "fixed_count" if target_units is not None else "fixed_geometry",
            "topology_unit": topology_unit,
            "center_kind": center_kind if topology_unit == "molecule" else "cell_center",
            "target_units": target_units,
            "selected_unit_count": int(len(selected_ids)),
            "selected_molecule_count": int(len(molecules)),
            "selected_atom_count": int(selected_atom_count),
            "candidate_count": int(valid_candidate_count),
            "grid_candidate_count": int(grid_candidate_count),
            "batch_size": int(self.batch_size),
            "vacuum_A": float(vacuum),
            "input_disorder": dict(self._input_disorder),
        }

        return MolecularCrystal(
            lattice=output_lattice,
            molecules=molecules,
            pbc=(False, False, False),
            metadata=metadata,
            extra_arrays=extra_arrays,
        )

    def _replicate_extra_arrays(
        self,
        selected_base_molecules: Sequence[int],
    ) -> dict[str, np.ndarray]:
        replicated: dict[str, np.ndarray] = {}
        for key, values in self.crystal.extra_arrays.items():
            array = np.asarray(values)
            blocks = [
                array[self._source_molecule_indices[molecule_index]]
                for molecule_index in selected_base_molecules
            ]
            replicated[key] = np.concatenate(blocks, axis=0)
        return replicated


def carve_nanocluster(
    crystal: MolecularCrystal,
    shape: ImplicitShape,
    *,
    topology_unit: str = "molecule",
    center: Optional[Sequence[float]] = None,
    center_frac: Optional[Sequence[float]] = None,
    center_kind: str = "centroid",
    target_units: Optional[int] = None,
    vacuum: float = 0.0,
    batch_size: int = DEFAULT_NANOCLUSTER_BATCH_SIZE,
) -> MolecularCrystal:
    """Convenience wrapper around :class:`NanoClusterCarver`."""
    return NanoClusterCarver(crystal, batch_size=batch_size).carve(
        shape,
        topology_unit=topology_unit,
        center=center,
        center_frac=center_frac,
        center_kind=center_kind,
        target_units=target_units,
        vacuum=vacuum,
    )


__all__ = [
    "DEFAULT_NANOCLUSTER_BATCH_SIZE",
    "NanoShape",
    "NanoClusterCarver",
    "carve_nanocluster",
]
