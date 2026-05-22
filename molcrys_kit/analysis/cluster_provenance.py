"""Provenance records for QM cluster models carved from a periodic crystal.

Mirrors the design of `molcrys_kit.analysis.disorder.provenance.DisorderProvenance`:
a frozen dataclass capturing the audit trail of a single derived structure,
plus ``to_dict`` / ``from_dict`` helpers for JSON sidecar serialization.

The sidecar JSON is the canonical record of how a cluster was carved; it
travels with the XYZ output so that downstream Gaussian/ORCA-input scripts
(outside this package) can pick up frozen-atom indices, cap atom indices,
and the kept-atom-to-parent index map without re-running the carver.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ClusterProvenance:
    """Audit trail for a single carved cluster.

    Attributes
    ----------
    mode : str
        ``"bond_shells"`` (chemistry-aware carving) or ``"rcut"`` (radial
        diagnostic carving).
    seed_global_indices : List[int]
        Global atom indices in the parent ``MolecularCrystal.to_ase()`` that
        seeded this cluster (after seed-merge auto-grouping).
    n_shells : Optional[int]
        Number of cut-boundary layers crossed beyond the SBU, when
        ``mode == "bond_shells"``.  ``None`` for radial mode.
    rcut_A : Optional[float]
        Radial cutoff in Angstrom, when ``mode == "rcut"``.  ``None`` for
        bond-shells mode.
    kept_global_indices : List[int]
        Global parent-atom indices retained in the cluster.  Cap H atoms
        do not appear here (they have no parent atom).
    cut_bonds : List[Tuple[int, int]]
        For each dangling bond cut during carving, ``(kept_global_idx,
        dropped_global_idx)``.  The same length as ``cap_local_indices``.
    cap_local_indices : List[int]
        Indices, in the emitted cluster's atom list, of the H atoms that
        cap the dangling bonds.
    frozen_local_indices : List[int]
        Indices, in the emitted cluster's atom list, of atoms to be held
        fixed during QM geometry optimisation.  See ``freeze_shell``
        semantics in :func:`molcrys_kit.operations.cluster.ClusterCarver`.
    freeze_shell : int
        Which freeze convention was used: ``0`` = none, ``1`` = caps +
        the heavy atom they replace, ``2`` = shell-1 plus one more layer
        of heavy atoms inward.
    cap_distance_A : float or None
        Uniform cap-distance override (Angstrom) supplied by the caller,
        or ``None`` when the carver looked up per-element X-H bond
        lengths from the ``cap_bond_lengths_A`` table.
    cap_bond_lengths_A : dict[str, float]
        Element-keyed X-H bond lengths consulted by the carver (typically
        the shared ``molcrys_kit.constants.config.BOND_LENGTHS`` table
        plus any user overrides).  Recorded so the sidecar JSON is
        self-describing even when the global table changes between MCK
        versions.
    cap_distances_used_A : list[float]
        Per-cap distance actually applied, in the same order as
        ``cap_local_indices`` / ``cut_bonds``.
    seed_merge_radius_A : float
        Distance threshold (Angstrom) under which adjacent metal seeds
        were auto-grouped into one cluster.
    parent_label : Optional[str]
        Free-text label of the parent structure (e.g. the source CIF
        path).  Not used by the algorithm; for human bookkeeping.
    convention_reference : str
        Free-text citation block describing the published convention
        that motivated the specific parameter choices for this carve
        (``n_shells``, ``freeze_shell``, ``cap_distance`` override, cap
        bond length overrides).  Empty by default; callers are
        encouraged to fill it in with the DOI(s) appropriate to their
        system so the sidecar JSON is self-documenting.
    """

    mode: str
    seed_global_indices: List[int]
    n_shells: Optional[int]
    rcut_A: Optional[float]
    kept_global_indices: List[int]
    cut_bonds: List[Tuple[int, int]]
    cap_local_indices: List[int]
    frozen_local_indices: List[int]
    freeze_shell: int
    cap_distance_A: Optional[float] = None
    cap_bond_lengths_A: Dict[str, float] = field(default_factory=dict)
    cap_distances_used_A: List[float] = field(default_factory=list)
    seed_merge_radius_A: float = 0.0
    parent_label: Optional[str] = None
    convention_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain-dict (JSON-ready) representation.

        Tuples become lists; the dataclass field names are preserved.
        """
        payload = asdict(self)
        # Tuples in cut_bonds become lists under asdict already; ensure
        # the integer values are plain ints (not numpy) so json.dumps works.
        payload["seed_global_indices"] = [int(i) for i in payload["seed_global_indices"]]
        payload["kept_global_indices"] = [int(i) for i in payload["kept_global_indices"]]
        payload["cap_local_indices"] = [int(i) for i in payload["cap_local_indices"]]
        payload["frozen_local_indices"] = [int(i) for i in payload["frozen_local_indices"]]
        payload["cut_bonds"] = [[int(a), int(b)] for a, b in payload["cut_bonds"]]
        if payload["n_shells"] is not None:
            payload["n_shells"] = int(payload["n_shells"])
        if payload["rcut_A"] is not None:
            payload["rcut_A"] = float(payload["rcut_A"])
        payload["freeze_shell"] = int(payload["freeze_shell"])
        if payload["cap_distance_A"] is not None:
            payload["cap_distance_A"] = float(payload["cap_distance_A"])
        payload["cap_bond_lengths_A"] = {
            str(k): float(v) for k, v in payload.get("cap_bond_lengths_A", {}).items()
        }
        payload["cap_distances_used_A"] = [
            float(v) for v in payload.get("cap_distances_used_A", [])
        ]
        payload["seed_merge_radius_A"] = float(payload["seed_merge_radius_A"])
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ClusterProvenance":
        """Inverse of :meth:`to_dict`.  Tuples are restored from list pairs."""
        cut_bonds: List[Tuple[int, int]] = [
            (int(a), int(b)) for a, b in payload.get("cut_bonds", [])
        ]
        kwargs = dict(payload)
        kwargs["cut_bonds"] = cut_bonds
        kwargs["seed_global_indices"] = [int(i) for i in payload["seed_global_indices"]]
        kwargs["kept_global_indices"] = [int(i) for i in payload["kept_global_indices"]]
        kwargs["cap_local_indices"] = [int(i) for i in payload["cap_local_indices"]]
        kwargs["frozen_local_indices"] = [int(i) for i in payload["frozen_local_indices"]]
        kwargs["cap_bond_lengths_A"] = {
            str(k): float(v)
            for k, v in payload.get("cap_bond_lengths_A", {}).items()
        }
        kwargs["cap_distances_used_A"] = [
            float(v) for v in payload.get("cap_distances_used_A", [])
        ]
        return cls(**kwargs)


__all__ = ["ClusterProvenance"]
