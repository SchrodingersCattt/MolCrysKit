"""Disorder-input status for carving and cluster builders.

``all_atom_ordered`` means the structure went through the disorder solver and
incompatible alternative *sites* were removed (``disorder_provenance`` present).
Refined partial occupancies and retained CIF ``disorder_group`` / ``assembly``
labels on survivors are source metadata, not evidence of unresolved disorder.
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np

from ..analysis.disorder.diagnostics import UnresolvedDisorderWarning
from ..constants.config import KEY_ASSEMBLY, KEY_DISORDER_GROUP, KEY_OCCUPANCY
from ..structures.crystal import MolecularCrystal


class DisorderInputStats(TypedDict):
    all_atom_ordered: bool
    ordered_replica: bool
    nonunit_occupancy_count: int
    retained_disorder_group_label_count: int
    retained_disorder_assembly_label_count: int


def disorder_input_stats(crystal: MolecularCrystal) -> DisorderInputStats:
    nonunit_occupancies = 0
    retained_groups = 0
    retained_assemblies = 0
    for molecule in crystal.molecules:
        occupancies = np.asarray(molecule.arrays.get(KEY_OCCUPANCY, []), dtype=float)
        nonunit_occupancies += int(
            np.count_nonzero(~np.isclose(occupancies, 1.0, rtol=0.0, atol=1e-8))
        )
        retained_groups += sum(
            int(str(value).strip() not in {"", ".", "?", "0"})
            for value in molecule.arrays.get(KEY_DISORDER_GROUP, [])
        )
        retained_assemblies += sum(
            int(str(value).strip() not in {"", ".", "?", "0"})
            for value in molecule.arrays.get(KEY_ASSEMBLY, [])
        )
    ordered_replica = crystal.disorder_provenance is not None
    if ordered_replica:
        all_atom_ordered = True
    else:
        all_atom_ordered = not bool(
            nonunit_occupancies or retained_groups or retained_assemblies
        )
    return DisorderInputStats(
        all_atom_ordered=all_atom_ordered,
        ordered_replica=ordered_replica,
        nonunit_occupancy_count=nonunit_occupancies,
        retained_disorder_group_label_count=retained_groups,
        retained_disorder_assembly_label_count=retained_assemblies,
    )


def warn_if_unresolved_disorder(
    crystal: MolecularCrystal,
    *,
    operation: str,
) -> dict[str, bool | int]:
    """Return sidecar stats and warn when the input is not an ordered replica."""
    stats = disorder_input_stats(crystal)
    payload = {
        "all_atom_ordered": stats["all_atom_ordered"],
        "ordered_replica": stats["ordered_replica"],
        "nonunit_occupancy_count": stats["nonunit_occupancy_count"],
        "active_disorder_group_count": stats["retained_disorder_group_label_count"],
        "active_disorder_assembly_count": stats[
            "retained_disorder_assembly_label_count"
        ],
        "retained_disorder_group_label_count": stats[
            "retained_disorder_group_label_count"
        ],
        "retained_disorder_assembly_label_count": stats[
            "retained_disorder_assembly_label_count"
        ],
    }
    if not stats["all_atom_ordered"]:
        warnings.warn(
            f"{operation} is continuing with unresolved disorder; resolve an ordered "
            "replica before production MD.",
            UnresolvedDisorderWarning,
            stacklevel=2,
        )
    return payload