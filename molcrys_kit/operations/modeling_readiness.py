"""Preflight checks for structures used in large atomistic models."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from ..constants.config import KEY_ASSEMBLY, KEY_DISORDER_GROUP, KEY_OCCUPANCY
from ..structures.crystal import MolecularCrystal


class UnresolvedDisorderWarning(UserWarning):
    """A modeling operation is continuing with unresolved disorder."""


@dataclass(frozen=True)
class ModelingReadinessReport:
    """Deterministic summary of disorder and topology-unit readiness."""

    all_atom_ordered: bool
    nonunit_occupancy_count: int
    active_disorder_group_count: int
    active_disorder_assembly_count: int
    incomplete_unwrap_molecule_count: int

    def to_dict(self) -> dict[str, bool | int]:
        return asdict(self)


def _is_active_marker(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value != 0)
    return str(value).strip() not in {"", ".", "?", "0"}


def inspect_modeling_readiness(crystal: MolecularCrystal) -> ModelingReadinessReport:
    """Inspect unresolved site disorder and incomplete topology units."""
    nonunit_occupancy_count = 0
    active_disorder_group_count = 0
    active_disorder_assembly_count = 0
    incomplete_unwrap_molecule_count = 0

    for molecule in crystal.molecules:
        occupancies = np.asarray(molecule.arrays.get(KEY_OCCUPANCY, []), dtype=float)
        nonunit_occupancy_count += int(
            np.count_nonzero(~np.isclose(occupancies, 1.0, rtol=0.0, atol=1e-8))
        )
        active_disorder_group_count += sum(
            _is_active_marker(value)
            for value in molecule.arrays.get(KEY_DISORDER_GROUP, [])
        )
        active_disorder_assembly_count += sum(
            _is_active_marker(value)
            for value in molecule.arrays.get(KEY_ASSEMBLY, [])
        )
        incomplete_unwrap_molecule_count += int(
            molecule.info.get("unwrap_completed") is False
        )

    all_atom_ordered = not (
        nonunit_occupancy_count
        or active_disorder_group_count
        or active_disorder_assembly_count
    )
    return ModelingReadinessReport(
        all_atom_ordered=all_atom_ordered,
        nonunit_occupancy_count=nonunit_occupancy_count,
        active_disorder_group_count=active_disorder_group_count,
        active_disorder_assembly_count=active_disorder_assembly_count,
        incomplete_unwrap_molecule_count=incomplete_unwrap_molecule_count,
    )


def warn_if_unresolved_disorder(
    crystal: MolecularCrystal,
    *,
    operation: str,
) -> ModelingReadinessReport:
    """Warn, but do not block, when a large-model input is still disordered."""
    report = inspect_modeling_readiness(crystal)
    if not report.all_atom_ordered:
        warnings.warn(
            f"{operation} is continuing with unresolved disorder: "
            f"{report.nonunit_occupancy_count} non-unit occupancies, "
            f"{report.active_disorder_group_count} active disorder-group sites, and "
            f"{report.active_disorder_assembly_count} active disorder-assembly sites. "
            "Resolve an ordered replica first (read_mol_crystal(..., "
            "resolve_disorder=True) or 'mck operate disorder') before production MD.",
            UnresolvedDisorderWarning,
            stacklevel=2,
        )
    return report


def require_complete_topology_units(
    report: ModelingReadinessReport,
    *,
    operation: str,
) -> None:
    """Reject molecule-preserving operations when unwrapping was incomplete."""
    if report.incomplete_unwrap_molecule_count:
        raise ValueError(
            f"{operation} requires finite, completely unwrapped molecules or ions; "
            f"{report.incomplete_unwrap_molecule_count} topology unit(s) are incomplete. "
            "Periodic 3-D frameworks/MOFs are not supported by this topology-preserving "
            "operation and are not automatically cut or capped."
        )


__all__ = [
    "ModelingReadinessReport",
    "UnresolvedDisorderWarning",
    "inspect_modeling_readiness",
]
