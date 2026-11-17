"""Parent-symmetry audit for ordered replicas from disorder resolution.

An ordered replica is typically written as a P1 packing of selected atoms.
This module reports whether that packing still closes under each parent
crystallographic operation: for every kept atom, whether an image exists among
the kept atoms, which disorder component that image belongs to, and whether
occupancy bookkeeping for one asymmetric-unit parent closes to the declared
occupancy sum.
"""

from __future__ import annotations

import itertools
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
    respects_parent_group: Optional[bool]
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


class ParentSymmetryNotAssessed(Exception):
    """Exact closest-image search was not completed for this cell."""


def _pairwise_min_image_distances(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    lattice: np.ndarray,
    pbc: Sequence[bool],
    tolerance_angstrom: float,
) -> np.ndarray:
    """Return Cartesian distances with shape ``(len(a), len(b))``.

    Every image that can lie inside ``tolerance_angstrom`` is enumerated.
    The integer box is the smallest cube implied by the metric's least
    eigenvalue, not a fixed 27-image window. An orthogonal metric reduces
    to fractional rounding, which is exact. A degenerate metric, or a proven
    box that exceeds :data:`PARENT_SYMMETRY_EXACT_PAIR_BUDGET`, raises
    :class:`ParentSymmetryNotAssessed` instead of returning a shorter-looking
    but incomplete minimum.
    """
    delta = (
        np.asarray(frac_a, dtype=float)[:, None, :]
        - np.asarray(frac_b, dtype=float)[None, :, :]
    )
    periodic = np.asarray(pbc, dtype=bool)
    rounded = np.array(delta, copy=True)
    rounded[..., periodic] -= np.round(rounded[..., periodic])
    lattice_array = np.asarray(lattice, dtype=float)
    if _independent_rounding_is_exact(lattice_array):
        cartesian = np.einsum("...j,jk->...k", rounded, lattice_array)
        return np.linalg.norm(cartesian, axis=-1)
    half_width = _exact_shift_half_width(lattice_array, float(tolerance_angstrom))
    ranges = [
        range(-half_width, half_width + 1) if axis_periodic else range(1)
        for axis_periodic in periodic
    ]
    n_images = 1
    for axis_range in ranges:
        n_images *= len(axis_range)
    n_pairs = int(rounded.shape[0] * rounded.shape[1])
    budget = int(DISORDER_CONFIG["PARENT_SYMMETRY_EXACT_PAIR_BUDGET"])
    if n_pairs * n_images > budget:
        raise ParentSymmetryNotAssessed(
            "exact closest-image box exceeds the parent-symmetry pair budget; "
            "distances were not assessed"
        )
    best = np.full(rounded.shape[:2], np.inf)
    for shift in itertools.product(*ranges):
        cartesian = np.einsum(
            "...j,jk->...k",
            rounded + np.asarray(shift, dtype=float),
            lattice_array,
        )
        np.minimum(best, np.linalg.norm(cartesian, axis=-1), out=best)
    return best


def _independent_rounding_is_exact(lattice: np.ndarray) -> bool:
    """Fractional rounding is the minimum image when the metric is orthogonal."""
    metric = lattice @ lattice.T
    off_diagonal = np.array(metric, copy=True)
    np.fill_diagonal(off_diagonal, 0.0)
    scale = float(np.max(np.abs(np.diag(metric))))
    return bool(np.max(np.abs(off_diagonal)) <= 1e-12 * max(scale, 1.0))


def _exact_shift_half_width(lattice: np.ndarray, tolerance_angstrom: float) -> int:
    """Largest |integer| that can still produce a vector inside the tolerance.

    For metric :math:`G` with least eigenvalue :math:`\\lambda`, a fractional
    displacement :math:`u` has Cartesian length at least :math:`\\sqrt{\\lambda}\\,
    \\|u\\|`. After rounding, :math:`u = r + k` with :math:`\\|r\\|_\\infty \\le 1/2`
    and :math:`k` integral, so any image shorter than ``tolerance_angstrom``
    has :math:`\\|k\\|_\\infty \\le \\mathrm{tolerance}/\\sqrt{\\lambda} + 1/2`.
    """
    metric = np.asarray(lattice, dtype=float) @ np.asarray(lattice, dtype=float).T
    eigenvalues = np.linalg.eigvalsh(metric)
    least = float(eigenvalues[0])
    scale = float(np.max(np.abs(np.diag(metric))))
    if not np.isfinite(least) or least <= 1e-12 * max(scale, 1.0):
        raise ParentSymmetryNotAssessed(
            "cell metric is degenerate; parent-symmetry distances were not assessed"
        )
    if tolerance_angstrom < 0.0:
        raise ValueError("tolerance_angstrom must be non-negative")
    limit = float(tolerance_angstrom) / np.sqrt(least) + 0.5
    return int(np.ceil(limit - 1e-12))


def _perfect_matching(allowed: np.ndarray) -> list[int] | None:
    """Return one bijection ``source -> target`` or ``None``.

    ``allowed`` is a square boolean matrix. A nearest-neighbour column can
    assign two sources to one target; a symmetry of a finite kept set has to
    hit every kept atom once. The search is Hopcroft–Karp, which stays
    polynomial when many atoms of one element lie inside the tolerance.
    """
    count = int(allowed.shape[0])
    if count == 0:
        return []
    row_counts = allowed.sum(axis=1)
    column_counts = allowed.sum(axis=0)
    if np.any(row_counts == 0) or np.any(column_counts == 0):
        return None
    if np.all(row_counts == 1) and np.all(column_counts == 1):
        return [int(index) for index in allowed.argmax(axis=1)]

    import networkx as nx

    graph = nx.Graph()
    sources = [f"s{index}" for index in range(count)]
    targets = [f"t{index}" for index in range(count)]
    graph.add_nodes_from(sources, bipartite=0)
    graph.add_nodes_from(targets, bipartite=1)
    rows, columns = np.nonzero(allowed)
    graph.add_edges_from(
        (sources[int(row)], targets[int(column)])
        for row, column in zip(rows, columns)
    )
    matching = nx.bipartite.hopcroft_karp_matching(graph, top_nodes=sources)
    if any(node not in matching for node in sources):
        return None
    return [int(matching[node][1:]) for node in sources]


def _operation_is_identity(operation) -> bool:
    rotation = np.asarray(operation.rotation, dtype=float)
    translation = np.mod(np.asarray(operation.translation, dtype=float), 1.0)
    translation[np.isclose(translation, 1.0)] = 0.0
    return bool(
        np.allclose(rotation, np.eye(3), atol=1e-8, rtol=0.0)
        and np.allclose(translation, 0.0, atol=1e-8, rtol=0.0)
    )


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
    identity_flags: list[bool] = []
    all_close = True
    for operation in operations:
        mapped = operation.apply(kept_frac, wrap=False)
        try:
            distances = _pairwise_min_image_distances(
                mapped,
                kept_frac,
                lattice_array,
                pbc,
                float(tolerance_angstrom),
            )
        except ParentSymmetryNotAssessed as exc:
            notes.append(str(exc))
            return ParentSymmetryAudit(
                parent_space_group_number=symmetry.space_group_number,
                parent_space_group_symbol=symmetry.space_group_symbol,
                parent_operation_count=len(operations),
                mapping_tolerance_angstrom=float(tolerance_angstrom),
                respects_parent_group=None,
                notes=tuple(dict.fromkeys(notes)),
            )
        allowed = same_symbol & np.isfinite(distances) & (distances <= tolerance_angstrom)
        assignment = _perfect_matching(allowed)
        component_map: dict[str, str] = {}
        if assignment is None:
            best_distance = np.full(len(kept), np.inf)
            unmapped = len(kept)
            residual_max = float("inf")
            residual_mean = float("inf")
            closes = False
        else:
            best_target_local = np.asarray(assignment, dtype=int)
            best_distance = distances[np.arange(len(kept)), best_target_local]
            unmapped = 0
            residual_max = float(np.max(best_distance)) if len(kept) else 0.0
            residual_mean = float(np.mean(best_distance)) if len(kept) else 0.0
            for source_local, target_local in enumerate(best_target_local):
                source_key = kept_component_keys[int(source_local)]
                target_key = kept_component_keys[int(target_local)]
                previous = component_map.get(source_key)
                if previous is None:
                    component_map[source_key] = target_key
                elif previous != target_key:
                    component_map[source_key] = f"{previous}|{target_key}"
            closes = True
        identity_flags.append(_operation_is_identity(operation))
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
            declared_by_key: dict[tuple[int, str], float] = {}
            for index in disordered:
                key = (
                    int(info.disorder_groups[index]),
                    str(info.assemblies[index] if index < len(info.assemblies) else ""),
                )
                declared_by_key.setdefault(key, float(info.occupancies[index]))
            declared_sum = float(sum(declared_by_key.values()))
            kept_sum = 0.0
            kept_groups = []
            kept_keys: set[tuple[int, str]] = set()
            for index in kept_members:
                key = (
                    int(info.disorder_groups[index]),
                    str(info.assemblies[index] if index < len(info.assemblies) else ""),
                )
                if key in kept_keys:
                    continue
                kept_keys.add(key)
                kept_sum += float(info.occupancies[index])
                kept_groups.append(int(info.disorder_groups[index]))
            # One exclusive component, or every declared component. A numeric
            # sum of several parts must not alias a different component.
            declared_keys = set(declared_by_key)
            closes = kept_keys == declared_keys or (
                len(kept_keys) == 1 and kept_keys <= declared_keys
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
        audit.closes
        for audit, is_identity in zip(operation_audits, identity_flags)
        if not is_identity
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
    """Return whether the audit completed and every parent operation closes.

    An unassessed metric returns False. That is not a report that the
    replica breaks the parent group.
    """
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
