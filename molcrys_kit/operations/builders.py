"""
Structure builders for molecular crystals.

This module provides functionality to build complex structures from simpler units.
"""

import copy
import dataclasses
import itertools
import numbers
import warnings
from typing import Sequence, Tuple

import numpy as np

from ..analysis.disorder import UnresolvedDisorderWarning
from ..constants.config import KEY_ASSEMBLY, KEY_DISORDER_GROUP, KEY_OCCUPANCY
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import _strip_stale_frac_arrays


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
            "create_supercell is continuing with unresolved disorder; resolve an "
            "ordered replica before production MD.",
            UnresolvedDisorderWarning,
            stacklevel=2,
        )
    return stats


def create_supercell(
    crystal: MolecularCrystal, scaling_factors: Tuple[int, int, int]
) -> MolecularCrystal:
    """
    Create a supercell by replicating the unit cell.

    Parameters
    ----------
    crystal : MolecularCrystal
        The unit cell.
    scaling_factors : Tuple[int, int, int]
        Scaling factors for each lattice vector.

    Returns
    -------
    MolecularCrystal
        Supercell structure.  Delegates to
        :meth:`MolecularCrystal.get_supercell` which internally copies
        molecules while preserving per-atom disorder metadata and the
        original ``sym_op_index``/``asym_id`` source provenance. Repeated
        calls append their build records to ``metadata["supercell_history"]``.
    """
    input_disorder = _warn_if_unresolved_disorder(crystal)
    result = crystal.get_supercell(*scaling_factors)
    result.metadata = copy.deepcopy(crystal.metadata)
    result.metadata["input_disorder"] = input_disorder
    supercell_info = {
        "scaling_factors": [int(value) for value in scaling_factors],
        "source_molecule_count": len(crystal.molecules),
        "source_atom_count": sum(len(molecule) for molecule in crystal.molecules),
    }
    history = copy.deepcopy(result.metadata.get("supercell_history", []))
    if not history and "supercell" in result.metadata:
        history.append(copy.deepcopy(result.metadata["supercell"]))
    history.append(copy.deepcopy(supercell_info))
    result.metadata["supercell_history"] = history
    result.metadata["supercell"] = supercell_info
    return result


def _replica_schema(crystal: MolecularCrystal, replica_index: int) -> tuple:
    """Return the order-independent molecule and extra-array schema."""
    molecule_schema = []
    for molecule in crystal.molecules:
        array_schema = tuple(
            sorted(
                (
                    key,
                    tuple(np.asarray(values).shape[1:]),
                    np.asarray(values).dtype.kind,
                )
                for key, values in molecule.arrays.items()
            )
        )
        molecule_schema.append(
            (tuple(sorted(molecule.get_chemical_symbols())), array_schema)
        )

    atom_count = sum(len(molecule) for molecule in crystal.molecules)
    extra_array_schema = []
    for key, values in crystal.extra_arrays.items():
        array = np.asarray(values)
        if len(array) != atom_count:
            raise ValueError(
                f"Replica {replica_index} extra array {key!r} has length "
                f"{len(array)}; expected {atom_count}."
            )
        extra_array_schema.append((key, tuple(array.shape[1:]), array.dtype.kind))

    return tuple(sorted(molecule_schema)), tuple(sorted(extra_array_schema))


def _serialise_provenance(provenance):
    if provenance is None:
        return None
    if hasattr(provenance, "to_dict"):
        return provenance.to_dict()
    if dataclasses.is_dataclass(provenance):
        return dataclasses.asdict(provenance)
    if isinstance(provenance, dict):
        return copy.deepcopy(provenance)
    return str(provenance)


def assemble_replica_supercell(
    replicas: Sequence[MolecularCrystal],
    scaling_factors: Tuple[int, int, int],
    replica_indices: Sequence[int],
) -> MolecularCrystal:
    """Assemble selected ordered unit-cell replicas into one supercell.

    Cell translations follow ``itertools.product(range(n_a), range(n_b),
    range(n_c))``: ``k`` varies fastest, followed by ``j`` and then ``i``.
    ``replica_indices[position]`` selects the replica copied into the
    translation at that position. Repeated indices are allowed.

    Molecules are kept unwrapped. Output ordering is translation-major, then
    follows the selected replica's molecule and atom order. Stored fractional
    coordinates are removed after translation and ``image_shift`` is reset to
    zero in the contiguous output coordinate frame.

    Parameters
    ----------
    replicas : Sequence[MolecularCrystal]
        Compatible ordered unit-cell replicas.
    scaling_factors : Tuple[int, int, int]
        Positive supercell dimensions ``(n_a, n_b, n_c)``.
    replica_indices : Sequence[int]
        One zero-based replica index per translated unit cell.

    Returns
    -------
    MolecularCrystal
        The assembled supercell with per-cell selection and disorder
        provenance recorded in ``metadata["replica_supercell"]``.

    Raises
    ------
    TypeError
        If a scaling factor or replica index is not an integer.
    ValueError
        If the mapping length or an index is invalid, or selected replicas
        have incompatible lattices, periodicity, molecule schemas, or
        per-atom extra-array schemas.
    """
    replica_list = list(replicas)
    if not replica_list:
        raise ValueError("At least one replica is required.")

    if len(scaling_factors) != 3:
        raise ValueError("scaling_factors must contain exactly three values.")
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral)
        for value in scaling_factors
    ):
        raise TypeError("scaling_factors must contain integers.")
    scale = tuple(int(value) for value in scaling_factors)
    if any(value < 1 for value in scale):
        raise ValueError("scaling_factors must each be >= 1.")

    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral)
        for value in replica_indices
    ):
        raise TypeError("replica_indices must contain integers.")
    selected_indices = [int(value) for value in replica_indices]
    expected_count = int(np.prod(scale))
    if len(selected_indices) != expected_count:
        raise ValueError(
            f"replica_indices has length {len(selected_indices)}; "
            f"expected {expected_count} for scaling_factors={scale}."
        )
    for mapping_position, replica_index in enumerate(selected_indices):
        if not 0 <= replica_index < len(replica_list):
            raise ValueError(
                f"Replica index {replica_index} at mapping position "
                f"{mapping_position} is out of range for {len(replica_list)} replicas."
            )

    reference_index = selected_indices[0]
    reference = replica_list[reference_index]
    if not isinstance(reference, MolecularCrystal):
        raise TypeError(f"Replica {reference_index} is not a MolecularCrystal.")
    reference_lattice = np.asarray(reference.lattice, dtype=float)
    reference_pbc = tuple(bool(value) for value in reference.pbc)
    reference_schema = _replica_schema(reference, reference_index)
    for replica_index in dict.fromkeys(selected_indices):
        replica = replica_list[replica_index]
        if not isinstance(replica, MolecularCrystal):
            raise TypeError(f"Replica {replica_index} is not a MolecularCrystal.")
        if not np.allclose(
            np.asarray(replica.lattice, dtype=float),
            reference_lattice,
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError(
                f"Replica {replica_index} has a lattice incompatible with "
                f"replica {reference_index}."
            )
        if tuple(bool(value) for value in replica.pbc) != reference_pbc:
            raise ValueError(
                f"Replica {replica_index} has periodic boundary conditions "
                f"incompatible with replica {reference_index}."
            )
        if _replica_schema(replica, replica_index) != reference_schema:
            raise ValueError(
                f"Replica {replica_index} has an atom/molecule schema "
                f"incompatible with replica {reference_index}."
            )

    translations = list(
        itertools.product(range(scale[0]), range(scale[1]), range(scale[2]))
    )
    new_lattice = np.asarray(
        [
            reference_lattice[0] * scale[0],
            reference_lattice[1] * scale[1],
            reference_lattice[2] * scale[2],
        ],
        dtype=float,
    )
    from ..constants.config import KEY_IMAGE_SHIFT

    new_molecules = []
    provenance_cells = []
    for translation, replica_index in zip(translations, selected_indices):
        replica = replica_list[replica_index]
        for source_molecule_index, molecule in enumerate(replica.molecules):
            new_atoms = molecule.copy()
            new_atoms.info.pop("atom_indices", None)
            new_atoms.info.pop("bond_records", None)
            new_atoms.info.pop("bond_pairs", None)
            new_atoms.info["source_replica_index"] = replica_index
            new_atoms.info["source_molecule_index"] = source_molecule_index
            new_atoms.info["unit_cell_translation"] = list(translation)
            new_atoms.positions += np.dot(np.asarray(translation), reference_lattice)
            new_atoms.set_array(
                KEY_IMAGE_SHIFT,
                np.zeros((len(new_atoms), 3), dtype=int),
            )
            _strip_stale_frac_arrays(new_atoms)
            new_molecules.append(new_atoms)

        provenance_cells.append(
            {
                "translation": [int(value) for value in translation],
                "replica_index": replica_index,
                "disorder_provenance": _serialise_provenance(
                    replica.disorder_provenance
                ),
            }
        )

    extra_arrays = {}
    for key in reference.extra_arrays:
        cell_arrays = []
        for replica_index in selected_indices:
            replica = replica_list[replica_index]
            molecule_indices = replica._molecule_global_indices()
            ordered_indices = [
                index for indices in molecule_indices for index in indices
            ]
            cell_arrays.append(np.asarray(replica.extra_arrays[key])[ordered_indices])
        extra_arrays[key] = np.concatenate(cell_arrays, axis=0)

    mapping_order = (
        "itertools.product(range(n_a), range(n_b), range(n_c)); k varies fastest"
    )
    supercell_info = {
        "scaling_factors": list(scale),
        "source_molecule_count": len(reference.molecules),
        "source_atom_count": sum(len(molecule) for molecule in reference.molecules),
        "replica_indices": selected_indices,
        "translation_order": mapping_order,
    }
    metadata = copy.deepcopy(reference.metadata)
    history = copy.deepcopy(metadata.get("supercell_history", []))
    if not history and "supercell" in metadata:
        history.append(copy.deepcopy(metadata["supercell"]))
    history.append(copy.deepcopy(supercell_info))
    metadata["supercell_history"] = history
    metadata["supercell"] = copy.deepcopy(supercell_info)
    metadata["replica_supercell"] = {
        "translation_order": mapping_order,
        "cells": copy.deepcopy(provenance_cells),
    }
    disorder_provenance = {
        "kind": "replica_supercell",
        "translation_order": mapping_order,
        "cells": copy.deepcopy(provenance_cells),
    }

    return MolecularCrystal(
        new_lattice,
        new_molecules,
        reference_pbc,
        disorder_provenance=disorder_provenance,
        metadata=metadata,
        extra_arrays=extra_arrays,
    )


__all__ = ["assemble_replica_supercell", "create_supercell"]
