"""
Repro / diagnostic script for the DAI-4 NH4+ hydrogen-count regression.

Run with:
    python scripts/repro_implicit_nh4.py /path/to/DAI-4.cif

Expected output BEFORE the fix:
    NH4+ H-counts: Counter({4: 4, 3: 4})  <- 4 sites correct, 4 sites wrong

Expected output AFTER the fix:
    NH4+ H-counts: Counter({4: 8})         <- all 8 sites correct

The script also prints intermediate diagnostics that were used to
narrow down the root cause (edge types around N1, atom-group sizes,
motif-centre detection).
"""

import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(__file__ + "/../../"))  # repo root on any machine

from ase.geometry import get_distances

from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.analysis.disorder.solver import DisorderSolver
from pymatgen.io.cif import CifParser


def _nh4_h_count(mc):
    """Return per-ammonium H-count list from a resolved MolecularCrystal."""
    atoms = mc.to_ase()
    sy = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    cell = atoms.cell[:]
    N_idx = np.where(sy == "N")[0]
    H_idx = np.where(sy == "H")[0]
    C_idx = np.where(sy == "C")[0]

    if not len(N_idx) or not len(H_idx):
        return []

    _, d_NH = get_distances(pos[N_idx], pos[H_idx], cell=cell, pbc=True)
    _, d_NC = get_distances(pos[N_idx], pos[C_idx], cell=cell, pbc=True) if len(C_idx) else (None, np.full((len(N_idx), 1), 99.0))

    counts = []
    for k, ni in enumerate(N_idx):
        if (d_NC[k] < 1.7).sum() > 0:
            continue  # N bonded to C — not ammonium
        counts.append(int((d_NH[k] < 1.3).sum()))
    return counts


def main(cif_path: str) -> None:
    print(f"\n=== Repro: {cif_path} ===\n")

    info = scan_cif_disorder(cif_path)
    lat = CifParser(cif_path).parse_structures()[0].lattice.matrix
    n_atoms = len(info.labels)

    # ── Step 1: Build exclusion graph ──────────────────────────────────────
    builder = DisorderGraphBuilder(info, lat)
    graph = builder.build()

    # Characterise edges around the *first* N1 copy
    n1_indices = [i for i in range(n_atoms) if info.labels[i] == "N1"]
    if n1_indices:
        n1 = n1_indices[0]
        H_near_n1 = [j for j in range(n_atoms) if builder.dist_matrix[n1, j] < 1.3
                     and info.symbols[j] in ("H", "D")]
        edge_types = Counter(
            graph[i][j].get("conflict_type", "?")
            for i in H_near_n1 for j in H_near_n1
            if i < j and graph.has_edge(i, j)
        )
        cross = [(info.labels[i], i, info.labels[j], j,
                  graph[i][j].get("conflict_type", "?"))
                 for i in H_near_n1 for j in H_near_n1
                 if i < j and graph.has_edge(i, j) and info.labels[i] != info.labels[j]]
        print(f"Edges among H near N1#{n1}:")
        print(f"  by type : {dict(edge_types)}")
        print(f"  cross-cluster count: {len(cross)}")
        if cross:
            print(f"  cross-cluster sample: {cross[:3]}")
    else:
        print("  (no N1 label found — skipping edge diagnostics)")

    # ── Step 2: Atom-group sizes for N1/N4 ─────────────────────────────────
    solver = DisorderSolver(info, graph, lat)
    solver._identify_atom_groups()
    a2g = {a: gi for gi, g in enumerate(solver.atom_groups) for a in g}

    for lbl in ("N1", "N4"):
        idx_list = [i for i in range(n_atoms) if info.labels[i] == lbl][:2]
        for ni in idx_list:
            gi = a2g[ni]
            gsz = len(solver.atom_groups[gi])
            member_labels = [info.labels[a] for a in solver.atom_groups[gi]]
            print(f"  {lbl}#{ni} → group #{gi} (size={gsz})  members: {member_labels}")

    # ── Step 3: Motif-centre detection ─────────────────────────────────────
    centers = solver._find_motif_centers(a2g)
    N_centers = [c for c in centers if info.symbols[c] == "N"]
    N1_centers = [c for c in N_centers if info.labels[c].startswith("N1")]
    N4_centers = [c for c in N_centers if info.labels[c].startswith("N4")]
    print(f"\nMotif N-centres found: {len(N_centers)} total")
    print(f"  N1-type: {len(N1_centers)}  (labels: {[info.labels[c] for c in N1_centers[:4]]})")
    print(f"  N4-type: {len(N4_centers)}  (labels: {[info.labels[c] for c in N4_centers[:4]]})")

    # ── Step 4: Resolve and count H per NH4+ ───────────────────────────────
    solver2 = DisorderSolver(info, graph, lat)
    results = solver2.solve(num_structures=1, method="optimal")
    mc = results[0]
    h_counts = _nh4_h_count(mc)
    counter = Counter(h_counts)
    print(f"\nNH4+ H-counts: {counter}  (total ammonium N: {len(h_counts)})")

    if all(h == 4 for h in h_counts):
        print("PASS — all ammonium sites have exactly 4 H atoms.")
    else:
        n_wrong = sum(1 for h in h_counts if h != 4)
        print(f"FAIL — {n_wrong}/{len(h_counts)} ammonium sites have wrong H count.")
        sys.exit(1)


if __name__ == "__main__":
    cif = sys.argv[1] if len(sys.argv) > 1 else (
        "/aisi-nas/guomingyu/personal/prop_pred_abx/data/pems/confs/DAI-4.cif"
    )
    main(cif)
