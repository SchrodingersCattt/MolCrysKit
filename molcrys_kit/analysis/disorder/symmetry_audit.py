"""Parent-symmetry audit for ordered replicas from disorder resolution.

An ordered replica is typically written as a P1 packing of selected atoms.
This module reports whether that packing still closes under each parent
crystallographic operation: for every kept atom, whether an image exists among
the kept atoms, which disorder component that image belongs to, and whether
occupancy bookkeeping for one asymmetric-unit parent closes to the declared
occupancy sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ...constants.config import DISORDER_CONFIG
from ...structures.symmetry import CrystalSymmetry


@dataclass(frozen=True)
class ParentSymmetryOperationAudit:
    """Audit of one parent operation against one ordered replica."""

    operation_index: int
    xyz: Optional[str]
    residual_angstrom_max: float
    residual_angstrom_mean: float
    mapped_atom_count: int
    unmapped_atom_count: int
    component_map: dict[str, str]
    closes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OccupancyClosureAudit:
    """Occupancy bookkeeping for one asymmetric-unit parent atom."""

    asym_id: int
    symbol: str
    declared_occupancy_sum: float
    kept_occupancy_sum: float
    closes: bool
    kept_groups: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kept_groups"] = list(self.kept_groups)
        return payload


@dataclass(frozen=True)
class ParentSymmetryAudit:
    """Per-replica audit of parent-group closure after disorder resolution."""

    parent_space_group_number: Optional[int]
    parent_space_group_symbol: Optional[str]
    parent_operation_count: int
    mapping_tolerance_angstrom: float
    respects_parent_group: bool
    operations: tuple[ParentSymmetryOperationAudit, ...] = ()
    occupancy_closure: tuple[OccupancyClosureAudit, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_space_group_number": self.parent_space_group_number,
            "parent_space_group_symbol": self.parent_space_group_symbol,
            "parent_operation_count": self.parent_operation_count,
            "mapping_tolerance_angstrom": self.mapping_tolerance_angstrom,
            "respects_parent_group": self.respects_parent_group,
            "operations": [item.to_dict() for item in self.operations],
            "occupancy_closure": [item.to_dict() for item in self.occupancy_closure],
            "notes": list(self.notes),
        }


def _component_key(info, atom_index: int) -> str:
    group = (
        int(info.disorder_groups[atom_index])
        if atom_index < len(info.disorder_groups)
        else 0
    )
    assembly = ""
    if info.assemblies and atom_index < len(info.assemblies):
        assembly = str(info.assemblies[atom_index] or "")
    if assembly:
        return f"{assembly}:{group}"
    return str(group)


def _parent_symmetry(info) -> Optional[CrystalSymmetry]:
    symmetry = getattr(info, "crystal_symmetry", None)
    if symmetry is not None and getattr(symmetry, "operations", None):
        return symmetry
    source = getattr(info, "source_symmetry", None)
    if source is not None and getattr(source, "operations", None):
        return source
    return None


def _pairwise_min_image_distances(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
) -> np.ndarray:
    """Return Cartesian min-image distances with shape ``(len(a), len(b))``."""
    delta = (
        np.asarray(frac_a, dtype=float)[:, None, :]
        - np.asarray(frac_b, dtype=float)[None, :, :]
    )
    pbc_mask = np.asarray(pbc, dtype=bool)
    if pbc_mask.all():
        delta = delta - np.round(delta)
    else:
        delta[..., pbc_mask] -= np.round(delta[..., pbc_mask])
    cart = np.einsum("...j,jk->...k", delta, np.asarray(lattice, dtype=float))
    return np.linalg.norm(cart, axis=-1)


def audit_parent_symmetry(
    info,
    kept_indices: Sequence[int],
    lattice: np.ndarray,
    *,
    tolerance_angstrom: float | None = None,
) -> ParentSymmetryAudit:
    """Return a parent-symmetry audit for one ordered independent set.

    Parameters
    ----------
    info :
        Expanded-cell disorder metadata used by the solver.
    kept_indices :
        Indices into ``info`` that survive in the ordered replica.
    lattice :
        3x3 lattice matrix with row vectors in Angstrom.
    tolerance_angstrom :
        Cartesian mapping tolerance. Defaults to
        ``DISORDER_CONFIG["PARENT_SYMMETRY_AUDIT_TOLERANCE"]``.
    """
    if tolerance_angstrom is None:
        tolerance_angstrom = float(DISORDER_CONFIG["PARENT_SYMMETRY_AUDIT_TOLERANCE"])
    notes: list[str] = []
    symmetry = _parent_symmetry(info)
    if symmetry is None:
        return ParentSymmetryAudit(
            parent_space_group_number=None,
            parent_space_group_symbol=None,
            parent_operation_count=0,
            mapping_tolerance_angstrom=float(tolerance_angstrom),
            respects_parent_group=False,
            notes=("parent symmetry unavailable",),
        )

    operations = tuple(symmetry.operations)
    if len(operations) <= 1:
        notes.append("parent symmetry has only the identity")

    kept = [int(index) for index in kept_indices]
    kept_set = set(kept)
    if not kept:
        return ParentSymmetryAudit(
            parent_space_group_number=symmetry.space_group_number,
            parent_space_group_symbol=symmetry.space_group_symbol,
            parent_operation_count=len(operations),
            mapping_tolerance_angstrom=float(tolerance_angstrom),
            respects_parent_group=False,
            notes=("empty kept set",),
        )

    pbc = getattr(info, "pbc", None) or (True, True, True)
    frac = np.asarray(info.frac_coords, dtype=float)
    symbols = list(info.symbols)
    kept_frac = frac[kept]
    kept_symbols = np.asarray([symbols[index] for index in kept], dtype=object)
    kept_component_keys = [_component_key(info, index) for index in kept]
    lattice_array = np.asarray(lattice, dtype=float)
    same_symbol = kept_symbols[:, None] == kept_symbols[None, :]

    operation_audits: list[ParentSymmetryOperationAudit] = []
    all_close = True
    for operation in operations:
        mapped = operation.apply(kept_frac, wrap=False)
        distances = _pairwise_min_image_distances(mapped, kept_frac, lattice_array, pbc)
        masked = np.where(same_symbol, distances, np.inf)
        best_target_local = np.argmin(masked, axis=1)
        best_distance = masked[np.arange(len(kept)), best_target_local]

        unmapped = int(
            np.sum(~np.isfinite(best_distance) | (best_distance > tolerance_angstrom))
        )
        finite = best_distance[np.isfinite(best_distance)]
        residual_max = float(np.max(finite)) if finite.size else float("inf")
        residual_mean = float(np.mean(finite)) if finite.size else float("inf")

        component_map: dict[str, str] = {}
        mapped_mask = np.isfinite(best_distance) & (best_distance <= tolerance_angstrom)
        for source_local in np.flatnonzero(mapped_mask):
            source_key = kept_component_keys[int(source_local)]
            target_key = kept_component_keys[int(best_target_local[source_local])]
            previous = component_map.get(source_key)
            if previous is None:
                component_map[source_key] = target_key
            elif previous != target_key:
                component_map[source_key] = f"{previous}|{target_key}"

        closes = unmapped == 0 and residual_max <= tolerance_angstrom
        all_close = all_close and closes
        operation_audits.append(
            ParentSymmetryOperationAudit(
                operation_index=int(
                    operation.index if operation.index is not None else 0
                ),
                xyz=operation.xyz,
                residual_angstrom_max=residual_max,
                residual_angstrom_mean=residual_mean,
                mapped_atom_count=len(kept) - unmapped,
                unmapped_atom_count=unmapped,
                component_map=dict(sorted(component_map.items())),
                closes=closes,
            )
        )

    occupancy_rows: list[OccupancyClosureAudit] = []
    asym_ids = getattr(info, "asym_id", None) or []
    if asym_ids:
        parents: dict[int, list[int]] = {}
        for index, asym in enumerate(asym_ids):
            parents.setdefault(int(asym), []).append(index)
        for asym_id, members in sorted(parents.items()):
            disordered = [
                index
                for index in members
                if (
                    float(info.occupancies[index]) < 1.0
                    or int(info.disorder_groups[index]) != 0
                )
            ]
            if not disordered:
                continue
            kept_members = [index for index in disordered if index in kept_set]
            # Exclusive alternatives that were not selected are expected to
            # contribute zero kept occupancy; only audit parents that appear
            # in the replica.
            if not kept_members:
                continue
            # Declared occupancy for one ASU parent is the unique (group, assembly)
            # occupancy sum, not the expanded-image sum.
            declared_sum = 0.0
            declared_component_occs: list[float] = []
            seen = set()
            for index in disordered:
                key = (
                    int(info.disorder_groups[index]),
                    str(info.assemblies[index] if index < len(info.assemblies) else ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                occ = float(info.occupancies[index])
                declared_sum += occ
                declared_component_occs.append(occ)
            kept_sum = 0.0
            kept_groups = []
            seen_kept = set()
            for index in kept_members:
                key = (
                    int(info.disorder_groups[index]),
                    str(info.assemblies[index] if index < len(info.assemblies) else ""),
                )
                if key in seen_kept:
                    continue
                seen_kept.add(key)
                kept_sum += float(info.occupancies[index])
                kept_groups.append(int(info.disorder_groups[index]))
            # Ordered replicas keep one exclusive alternative; closure means the
            # kept alternative's occupancy is one of the declared component
            # occupancies and does not exceed the declared sum.
            closes = kept_sum <= declared_sum + 1e-6 and (
                any(abs(kept_sum - occ) <= 1e-6 for occ in declared_component_occs)
                or abs(kept_sum - declared_sum) <= 1e-6
            )
            occupancy_rows.append(
                OccupancyClosureAudit(
                    asym_id=int(asym_id),
                    symbol=str(info.symbols[disordered[0]]),
                    declared_occupancy_sum=float(declared_sum),
                    kept_occupancy_sum=float(kept_sum),
                    closes=bool(closes),
                    kept_groups=tuple(sorted(kept_groups)),
                )
            )
            if not closes:
                all_close = False
    else:
        notes.append("asym_id unavailable; occupancy closure skipped")

    if len(operations) > 1 and not all(
        audit.closes for audit in operation_audits if audit.operation_index != 0
    ):
        notes.append(
            "replica does not close under every non-identity parent operation; "
            "ordered models often lower to a subgroup"
        )

    respects = bool(all_close and all(item.closes for item in operation_audits))
    return ParentSymmetryAudit(
        parent_space_group_number=symmetry.space_group_number,
        parent_space_group_symbol=symmetry.space_group_symbol,
        parent_operation_count=len(operations),
        mapping_tolerance_angstrom=float(tolerance_angstrom),
        respects_parent_group=respects,
        operations=tuple(operation_audits),
        occupancy_closure=tuple(occupancy_rows),
        notes=tuple(notes),
    )


def replica_respects_parent_symmetry(
    info,
    kept_indices: Sequence[int],
    lattice: np.ndarray,
    *,
    tolerance_angstrom: float | None = None,
) -> bool:
    """Return whether ``kept_indices`` close under every parent operation."""
    audit = audit_parent_symmetry(
        info,
        kept_indices,
        lattice,
        tolerance_angstrom=tolerance_angstrom,
    )
    return bool(audit.respects_parent_group)


__all__ = [
    "OccupancyClosureAudit",
    "ParentSymmetryAudit",
    "ParentSymmetryOperationAudit",
    "audit_parent_symmetry",
    "replica_respects_parent_symmetry",
]
