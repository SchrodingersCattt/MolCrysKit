"""C1-C10 invariants every correctly-carved coordination cluster must
satisfy.

A *carve invariant* is a condition that holds, by construction, for any
cluster emitted by :class:`molcrys_kit.operations.cluster.ClusterCarver`
operating on a sensible parent crystal.  The functions in this module
*check* those invariants on the produced artefacts (parent + cluster +
sidecar provenance), and return the list of violations.  An empty list
means the cluster passes.

The check runs purely on parent-crystal data plus the cluster artefacts
so it does not depend on the carver internals: the carver can change
implementation and the checker still tells us whether the emitted
artefact is a valid coordination cluster.

Invariants
----------
* ``C1`` -- every seed atom retains its first-shell non-metal donors that
  were bonded in the parent crystal.
* ``C2`` -- no phantom bonds: every parent-bonded pair of kept atoms is
  also geometrically close in the cluster (except topologically
  required ``loop_cuts``).
* ``C3`` -- the cluster (heavy atoms + cap H) is one connected component.
* ``C4`` -- per-anion-group cap pairing: every cut keeper appears in
  ``cap_keeper_global_indices`` after anion-group lookup (carboxylate /
  sulfonate / phosphonate / hypercoordinate oxo / deprotonated aromatic
  N-heterocycle); cap-index columns are length-consistent.
* ``C5`` -- each cap H sits at the recorded ``cap_distances_used_A`` from
  its keeper atom.
* ``C6`` -- every entry in ``cut_bonds`` is either a metal-boundary cut, a
  requested-and-applied C-C cut, or a loop cut.
* ``C7`` -- bonds between seed atoms must survive the carve.
* ``C8`` -- chemistry-aware cap count: at most one cap H per anion
  group (no -C(OH)2 geminal diols, no dihydro-N-heterocycle tautomers)
  and per non-C keeper (no NH2 from mu-bridging N over-capping); the
  total cap count equals (unique non-C anion groups) + (C-C cuts).
* ``C9`` -- element conservation: the cluster's non-H element counts
  exactly equal the parent counts on ``kept_global_indices``; the
  cluster's H count equals (kept parent H) + ``len(cap_local_indices)``.
* ``C10`` -- linker inventory: every connected non-metal fragment that
  appears in the cluster has a parent-side counterpart (after counting
  cap H atoms back as missing metal coordination), i.e. the carver did
  not invent or destroy a ligand species.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx

from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

from ..constants import (
    DEFAULT_NEIGHBOR_CUTOFF,
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from .interactions import get_bonding_threshold


__all__ = [
    "ClusterInvariantReport",
    "build_parent_bond_graph",
    "check_cluster_invariants",
    "check_cluster_invariants_from_files",
]


@dataclass
class ClusterInvariantReport:
    """Result of running the carve-invariant check on one cluster."""

    name: str
    failures: List[str]

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> str:
        if self.ok:
            return f"[OK] {self.name}"
        lines = [f"[FAIL] {self.name} ({len(self.failures)} violations)"]
        lines.extend(f"  - {msg}" for msg in self.failures)
        return "\n".join(lines)


def _radius(sym: str) -> float:
    return get_atomic_radius(sym) if has_atomic_radius(sym) else 0.5


def _pair_threshold(sym_i: str, sym_j: str) -> float:
    return get_bonding_threshold(
        _radius(sym_i),
        _radius(sym_j),
        is_metal_element(sym_i),
        is_metal_element(sym_j),
    )


def build_parent_bond_graph(atoms: Atoms) -> nx.Graph:
    """Build the parent bond graph (PBC=True) using MCK's threshold rules."""
    syms = atoms.get_chemical_symbols()
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF)
    graph = nx.Graph()
    graph.add_nodes_from((i, {"symbol": s}) for i, s in enumerate(syms))
    for i, j, d in zip(i_list, j_list, d_list):
        if i >= j:
            continue
        if d < _pair_threshold(syms[i], syms[j]):
            graph.add_edge(int(i), int(j), distance=float(d))
    return graph


def _build_cluster_bond_graph(cluster_atoms: Atoms) -> nx.Graph:
    """Cluster bond graph in the non-periodic Cartesian frame."""
    syms = cluster_atoms.get_chemical_symbols()
    i_list, j_list, d_list = neighbor_list(
        "ijd", cluster_atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(syms)))
    for i, j, d in zip(i_list, j_list, d_list):
        if i >= j:
            continue
        if d < _pair_threshold(syms[i], syms[j]):
            graph.add_edge(int(i), int(j))
    return graph


def check_cluster_invariants(
    parent_atoms: Atoms,
    cluster_atoms: Atoms,
    provenance: Dict[str, object],
    parent_graph: nx.Graph | None = None,
    cap_distance_tol: float = 1e-2,
    bond_slack: float = 1.0,
) -> List[str]:
    """Check the C1-C10 carve invariants on one cluster artefact.

    Returns the list of violations (each a human-readable string).  An
    empty list means the cluster passes every invariant.

    Parameters
    ----------
    parent_atoms : Atoms
        Parent periodic crystal as ASE Atoms (``crystal.to_ase()``).
    cluster_atoms : Atoms
        Cluster as ASE Atoms (``cluster.to_ase()`` or read from XYZ).
    provenance : dict
        Sidecar JSON payload (``ClusterProvenance.to_dict()``).
    parent_graph : networkx.Graph, optional
        Precomputed parent bond graph; recomputed if ``None``.
    cap_distance_tol : float, default 1e-2
        Tolerance (Angstrom) on cap-H placement vs the recorded value.
    bond_slack : float, default 1.0
        Multiplier applied to the bond threshold when judging phantom
        bonds; ``1.0`` means "must be below the same threshold the parent
        graph used".  Slightly relaxed slack lets a relaxed cluster pass
        without re-deriving thresholds.
    """
    failures: List[str] = []
    parent_syms = parent_atoms.get_chemical_symbols()
    if parent_graph is None:
        parent_graph = build_parent_bond_graph(parent_atoms)

    seed = list(int(s) for s in provenance.get("seed_global_indices", []))
    kept = list(int(g) for g in provenance.get("kept_global_indices", []))
    kept_set = set(kept)
    seed_set = set(seed)
    loop_cut_keys = {
        tuple(sorted((int(a), int(b))))
        for a, b in provenance.get("loop_cuts", [])
    }

    if not kept:
        failures.append("kept_global_indices is empty")
        return failures

    cluster_syms = cluster_atoms.get_chemical_symbols()
    cluster_positions = cluster_atoms.get_positions()
    if len(cluster_syms) < len(kept):
        failures.append(
            f"cluster has {len(cluster_syms)} atoms but provenance lists {len(kept)} kept parents"
        )
        return failures

    kept_sorted = sorted(kept)
    global_to_local: Dict[int, int] = {g: lo for lo, g in enumerate(kept_sorted)}

    # C1 + C7: seed-side checks.
    for s in seed:
        if s not in kept_set:
            failures.append(f"C1: seed atom {parent_syms[s]}#{s} not in kept set")
            continue
        for nb in parent_graph.neighbors(s):
            nb_sym = parent_syms[nb]
            if nb in seed_set:
                if nb not in kept_set:
                    failures.append(
                        f"C7: seed-seed bond {parent_syms[s]}#{s}-{nb_sym}#{nb} not kept"
                    )
                continue
            if is_metal_element(nb_sym):
                continue  # other metal nodes are out of scope by design
            if nb not in kept_set:
                failures.append(
                    f"C1: seed {parent_syms[s]}#{s} lost first-shell donor {nb_sym}#{nb}"
                )

    # C2: phantom bonds inside the kept set, except topologically
    # nontrivial loop cuts (which the carver explicitly broke and
    # capped on both sides).
    for u, v in parent_graph.edges():
        if u in kept_set and v in kept_set:
            if tuple(sorted((int(u), int(v)))) in loop_cut_keys:
                continue
            su, sv = parent_syms[u], parent_syms[v]
            d = float(np.linalg.norm(
                cluster_positions[global_to_local[u]]
                - cluster_positions[global_to_local[v]]
            ))
            thr = _pair_threshold(su, sv) * bond_slack
            if d >= thr:
                failures.append(
                    f"C2: phantom bond {su}#{u}-{sv}#{v}: parent bonded, "
                    f"cluster distance {d:.3f} A >= threshold {thr:.3f} A"
                )

    # C3: cluster must be one connected component.
    cluster_graph = _build_cluster_bond_graph(cluster_atoms)
    if cluster_graph.number_of_nodes() > 0 and not nx.is_connected(cluster_graph):
        comps = list(nx.connected_components(cluster_graph))
        sizes = sorted((len(c) for c in comps), reverse=True)
        failures.append(
            f"C3: cluster has {len(comps)} disconnected components, sizes={sizes}"
        )

    # C4: per-anion-group cap pairing.  After chemistry-aware dedup
    # the number of caps may be smaller than the number of cuts, and
    # several keepers may share a single cap H via the anion-group
    # rule (e.g. both O's of a carboxylate -> one OH).  The invariant
    # is therefore: every cut keeper's *anion group* (or, for atoms
    # not in any recognised anion group, the keeper atom itself) must
    # appear in ``cap_keeper_global_indices`` after group lookup.
    #
    # Implementation mirrors the carver
    # (``operations.cluster._build_anion_group_map``): we feed
    # :class:`ChemicalEnvironment` the organic skeleton (parent bond
    # graph with metal-non-metal edges stripped) so the carboxylate /
    # sulfonate / hypercoordinate-oxo detectors see the same picture
    # they do on a fully protonated crystal where no metals are
    # present.
    from .chemical_env import ChemicalEnvironment
    chem_graph = nx.Graph()
    for i, s in enumerate(parent_syms):
        chem_graph.add_node(i, symbol=s)
    for u, v in parent_graph.edges():
        if is_metal_element(parent_syms[int(u)]) or is_metal_element(parent_syms[int(v)]):
            continue
        chem_graph.add_edge(int(u), int(v))
    chem_env = ChemicalEnvironment((chem_graph, parent_atoms.get_positions()))
    anion_groups = chem_env.compute_anion_protonation_groups()

    cut_bonds = [tuple(c) for c in provenance.get("cut_bonds", [])]
    cap_locals = list(int(i) for i in provenance.get("cap_local_indices", []))
    cap_keepers = list(int(g) for g in provenance.get("cap_keeper_global_indices", []))
    cap_distances = list(float(d) for d in provenance.get("cap_distances_used_A", []))
    cut_keepers = {int(c[0]) for c in cut_bonds}
    cap_keeper_groups = {
        anion_groups.get(int(g), int(g)) for g in cap_keepers
    }
    missing_keepers = {
        g for g in cut_keepers
        if anion_groups.get(int(g), int(g)) not in cap_keeper_groups
    }
    if missing_keepers:
        failures.append(
            "C4: cut keepers without any cap H "
            "(after anion-group lookup): "
            + ", ".join(f"{parent_syms[g]}#{g}" for g in sorted(missing_keepers))
        )
    if len(cap_locals) != len(cap_keepers):
        failures.append(
            f"C4: cap_local_indices ({len(cap_locals)}) and "
            f"cap_keeper_global_indices ({len(cap_keepers)}) length mismatch"
        )
    if len(cap_locals) != len(cap_distances):
        failures.append(
            f"C4: cap_local_indices ({len(cap_locals)}) and "
            f"cap_distances_used_A ({len(cap_distances)}) length mismatch"
        )

    # C5: cap H geometry vs sidecar.
    pairs = min(len(cap_locals), len(cap_keepers), len(cap_distances))
    for idx in range(pairs):
        cap_local = cap_locals[idx]
        kept_g = int(cap_keepers[idx])
        expected = cap_distances[idx]
        if cap_local >= len(cluster_syms):
            failures.append(f"C5: cap_local {cap_local} out of cluster range")
            continue
        if cluster_syms[cap_local] != "H":
            failures.append(
                f"C5: cap_local {cap_local} symbol is {cluster_syms[cap_local]}, not H"
            )
            continue
        if kept_g not in global_to_local:
            failures.append(f"C5: keeper #{kept_g} for cap {cap_local} not in kept set")
            continue
        d = float(np.linalg.norm(
            cluster_positions[cap_local]
            - cluster_positions[global_to_local[kept_g]]
        ))
        if abs(d - expected) > cap_distance_tol:
            failures.append(
                f"C5: cap H@{cap_local} sits {d:.3f} A from keeper #{kept_g}, "
                f"expected {expected:.3f}"
            )

    # C6: unauthorized cuts.  Each cut must be either a metal-boundary
    # cut, a requested-and-applied C-C cut, or a loop-cut (recorded
    # order-independently in ``loop_cuts``).
    metal_boundary = {tuple(c) for c in provenance.get("metal_boundary_cuts", [])}
    cc_applied = {tuple(c) for c in provenance.get("cut_cc_bonds_applied", [])}
    for cut in cut_bonds:
        if cut in metal_boundary or cut in cc_applied:
            continue
        kept_g, dropped_g = cut
        if tuple(sorted((int(kept_g), int(dropped_g)))) in loop_cut_keys:
            continue
        failures.append(
            f"C6: cut {parent_syms[kept_g]}#{kept_g}-{parent_syms[dropped_g]}#{dropped_g} "
            "is neither a metal-boundary cut, a requested C-C cut, nor a loop cut"
        )

    # C8: chemistry-aware cap count.  The rule (shared with the
    # carver and ``add_hydrogens``) is one cap H per *anion group*
    # (carboxylate, sulfonate, phosphonate, hypercoordinate oxo,
    # deprotonated aromatic N-heterocycle ring) plus one cap H per
    # C-C cut.  ``anion_groups`` is the same dict computed in the C4
    # block above so the checker uses one source of truth.
    non_c_cut_keepers = [int(g) for g in cut_keepers if parent_syms[int(g)] != "C"]
    unique_non_c_groups = {
        anion_groups.get(int(g), int(g)) for g in non_c_cut_keepers
    }
    c_keeper_cut_count = sum(
        1 for c in cut_bonds if parent_syms[int(c[0])] == "C"
    )
    expected_caps = len(unique_non_c_groups) + c_keeper_cut_count
    if len(cap_locals) != expected_caps:
        failures.append(
            f"C8: expected {expected_caps} cap H atoms "
            f"({len(unique_non_c_groups)} unique non-C anion groups + "
            f"{c_keeper_cut_count} C-keeper cuts), got {len(cap_locals)}"
        )
    # Per-anion-group: at most one cap H (excluding C-keeper cuts).
    caps_per_group: Dict[int, int] = {}
    for g in cap_keepers:
        if parent_syms[int(g)] == "C":
            continue
        gid = anion_groups.get(int(g), int(g))
        caps_per_group[gid] = caps_per_group.get(gid, 0) + 1
    for gid, n in caps_per_group.items():
        if n > 1:
            offenders = [
                int(g) for g in cap_keepers
                if anion_groups.get(int(g), int(g)) == gid
                and parent_syms[int(g)] != "C"
            ]
            failures.append(
                f"C8: anion group {gid} ({offenders}) carries {n} cap H atoms; "
                "chemistry rule allows at most one cap per anion group "
                "(no -C(OH)2 geminal diols, no dihydro triazoles)"
            )
    # Additional sanity: no kept N/O/S atom should have more H neighbors
    # in the cluster than (parent H neighbors + 1).
    parent_h_count: Dict[int, int] = {}
    for g in kept:
        if parent_syms[int(g)] in ("N", "O", "S", "P"):
            n_h = sum(
                1 for nb in parent_graph.neighbors(int(g))
                if parent_syms[int(nb)] == "H"
            )
            parent_h_count[int(g)] = n_h
    cluster_h_count: Dict[int, int] = {g: 0 for g in parent_h_count}
    for local_i, sym in enumerate(cluster_syms):
        if sym != "H":
            continue
        # find neighbours of this H in the cluster
        h_pos = cluster_positions[local_i]
        for g, lo in global_to_local.items():
            if g not in cluster_h_count:
                continue
            d = float(np.linalg.norm(cluster_positions[lo] - h_pos))
            if d < 1.3:
                cluster_h_count[g] = cluster_h_count.get(g, 0) + 1
    non_c_capped_atoms = {
        int(g) for g in cap_keepers if parent_syms[int(g)] != "C"
    }
    for g, n_cluster in cluster_h_count.items():
        n_parent = parent_h_count.get(g, 0)
        allowed = n_parent + (1 if int(g) in non_c_capped_atoms else 0)
        if n_cluster > allowed:
            failures.append(
                f"C8: kept {parent_syms[g]}#{g} has {n_cluster} H neighbours "
                f"in cluster, expected at most {allowed} "
                f"(parent={n_parent}, capped={int(g) in non_c_capped_atoms})"
            )

    # C9: element conservation.
    parent_kept_counts: Counter = Counter(parent_syms[g] for g in kept)
    cluster_counts: Counter = Counter(cluster_syms)
    for el, n_parent in parent_kept_counts.items():
        n_cluster = cluster_counts.get(el, 0)
        if el == "H":
            expected_h = n_parent + len(cap_locals)
            if n_cluster != expected_h:
                failures.append(
                    f"C9: cluster has {n_cluster} H atoms, expected "
                    f"{expected_h} (parent kept H={n_parent} + caps={len(cap_locals)})"
                )
        else:
            if n_cluster != n_parent:
                failures.append(
                    f"C9: cluster has {n_cluster} {el} atoms, parent kept has {n_parent}"
                )
    # Catch any element that appears in the cluster but not in kept (other than H).
    for el, n_cluster in cluster_counts.items():
        if el == "H":
            continue
        if parent_kept_counts.get(el, 0) == 0:
            failures.append(
                f"C9: cluster has {n_cluster} {el} atoms not present in parent kept set"
            )

    # C10: linker inventory.  Count cluster fragments after deleting the
    # seed metal atoms; each fragment's formula (excluding cap H) must
    # appear in the parent's non-metal-fragment inventory.
    non_seed_cluster_atoms = [
        i for i, s in enumerate(cluster_syms)
        if i not in set(cap_locals) and (
            not is_metal_element(s)
        )
    ]
    if non_seed_cluster_atoms:
        cluster_atoms_obj = Atoms(
            symbols=[cluster_syms[i] for i in non_seed_cluster_atoms],
            positions=cluster_positions[non_seed_cluster_atoms],
            pbc=False,
        )
        cluster_frag_graph = _build_cluster_bond_graph(cluster_atoms_obj)
        # Compute parent linker inventory.
        parent_non_metal = [
            i for i, s in enumerate(parent_syms) if not is_metal_element(s)
        ]
        parent_frag_graph = parent_graph.subgraph(parent_non_metal).copy()
        parent_fragments: Counter = Counter()
        for component in nx.connected_components(parent_frag_graph):
            formula = "".join(
                f"{el}{c}"
                for el, c in sorted(
                    Counter(parent_syms[g] for g in component).items()
                )
            )
            parent_fragments[formula] += 1
        # Cluster fragments (excluding cap H).  Subtract cap H atoms by
        # not including them above; the remaining H atoms are real
        # parent H.  Each kept linker that lost some Zn coordination
        # would show one fewer Zn bond and one extra H from the cap --
        # but we already excluded caps.  So a cluster fragment's
        # element count should match a parent fragment exactly.
        cluster_fragments: Counter = Counter()
        for component in nx.connected_components(cluster_frag_graph):
            formula = "".join(
                f"{el}{c}"
                for el, c in sorted(
                    Counter(cluster_atoms_obj.get_chemical_symbols()[i] for i in component).items()
                )
            )
            cluster_fragments[formula] += 1
        for formula, n_cluster in cluster_fragments.items():
            n_parent = parent_fragments.get(formula, 0)
            if n_parent == 0:
                failures.append(
                    f"C10: cluster fragment {formula} (count {n_cluster}) "
                    "has no parent counterpart; ligand chemistry was distorted"
                )

    return failures


def check_cluster_invariants_from_files(
    parent_cif: Path | str,
    xyz_path: Path | str,
    sidecar_path: Path | str | None = None,
) -> ClusterInvariantReport:
    """File-driven entry point: read parent CIF + cluster XYZ + sidecar JSON."""
    from ..io.cif import read_mol_crystal

    xyz_path = Path(xyz_path)
    if sidecar_path is None:
        sidecar_path = xyz_path.with_suffix(xyz_path.suffix + ".cluster.json")
    sidecar_path = Path(sidecar_path)

    crystal = read_mol_crystal(str(parent_cif))
    parent_atoms = crystal.to_ase()
    cluster_atoms = ase_read(str(xyz_path))
    with open(sidecar_path) as fh:
        provenance = json.load(fh)

    failures = check_cluster_invariants(parent_atoms, cluster_atoms, provenance)
    return ClusterInvariantReport(name=str(xyz_path), failures=failures)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Check C1-C10 carve invariants on carved QM clusters."
    )
    parser.add_argument("--parent-cif", required=True, type=Path)
    parser.add_argument("xyz", nargs="+", type=Path)
    args = parser.parse_args()

    all_ok = True
    for xyz in args.xyz:
        result = check_cluster_invariants_from_files(args.parent_cif, xyz)
        print(result.report())
        if not result.ok:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
