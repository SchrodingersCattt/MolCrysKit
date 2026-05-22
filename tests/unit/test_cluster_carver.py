"""Unit tests for :mod:`molcrys_kit.operations.cluster`.

Fixtures are constructed programmatically (no CIF dependency) so that the
tests stay tractable in CI and the geometry is fully under our control.
The two reference fixtures are:

* ``_zn_methylbenzoate_crystal``  -- one Zn + ``[OOC-C6H4-CH3]^-`` in a
  large orthorhombic cell.  Exercises ring detection (the benzene ring
  must never be cut) and the cap-placement vector.
* ``_zn_dimer_formate_crystal``  -- two Zn atoms 3.0 A apart with one
  bridging formate each side.  Exercises the multi-metal seed auto-grouping.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from typing import List

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.cluster_provenance import ClusterProvenance
from molcrys_kit.io import write_xyz_with_freeze
from molcrys_kit.operations import (
    ClusterCarver,
    LigandTopologyOverflowError,
    carve_cluster,
)
from molcrys_kit.operations.cluster import _is_cuttable_cc
from molcrys_kit.structures.cluster import CrystalCluster
from molcrys_kit.structures.crystal import MolecularCrystal


# ---------------------------------------------------------------------------
# Programmatic fixtures
# ---------------------------------------------------------------------------


def _benzene_ring_xy(center: np.ndarray) -> List[np.ndarray]:
    """Return the 6 benzene C positions, aromatic radius 1.40 A."""
    out: List[np.ndarray] = []
    for k in range(6):
        theta = math.radians(60 * k)
        out.append(center + np.array([1.40 * math.cos(theta),
                                      1.40 * math.sin(theta), 0.0]))
    return out


def _ring_h_for(c_pos: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Ring C-H pointing radially outward, 1.09 A from carbon."""
    direction = c_pos - center
    direction /= np.linalg.norm(direction)
    return c_pos + 1.09 * direction


def _zn_methylbenzoate_crystal() -> MolecularCrystal:
    """Build Zn + ``[OOC-C6H4-CH3]^-`` in a 20 A cubic cell."""
    cell = 20.0 * np.eye(3)
    ring_center = np.array([0.0, 0.0, 0.0])
    ring_cs = _benzene_ring_xy(ring_center)
    # C0 = (1.40, 0, 0)  -> bears methyl
    # C3 = (-1.40, 0, 0) -> bears carboxylate

    positions: List[np.ndarray] = []
    symbols: List[str] = []

    # Ring carbons C0..C5
    for c in ring_cs:
        positions.append(c)
        symbols.append("C")
    # Ring hydrogens on C1, C2, C4, C5 (NOT on C0/C3, the substituent positions)
    ring_h_indices_for = [1, 2, 4, 5]
    for k in ring_h_indices_for:
        positions.append(_ring_h_for(ring_cs[k], ring_center))
        symbols.append("H")

    # Methyl C attached to C0 at +x, sp3-bond length 1.51 A.
    c_methyl = ring_cs[0] + np.array([1.51, 0.0, 0.0])
    positions.append(c_methyl)
    symbols.append("C")
    # Three methyl H at standard tetrahedral angles, 1.09 A from methyl C.
    h_offsets = [
        np.array([0.36, 1.03, 0.0]),
        np.array([0.36, -0.51, 0.89]),
        np.array([0.36, -0.51, -0.89]),
    ]
    for off in h_offsets:
        positions.append(c_methyl + off)
        symbols.append("H")

    # Carboxylate C attached to C3 at -x, 1.51 A.
    c_carb = ring_cs[3] + np.array([-1.51, 0.0, 0.0])
    positions.append(c_carb)
    symbols.append("C")
    # Two O atoms at ~1.25 A from C(carb), symmetric in y, lying in xy plane.
    # Place O1 at C(carb) + (-0.68, +1.05, 0); O2 at C(carb) + (-0.68, -1.05, 0).
    o1 = c_carb + np.array([-0.68, 1.05, 0.0])
    o2 = c_carb + np.array([-0.68, -1.05, 0.0])
    positions.append(o1); symbols.append("O")
    positions.append(o2); symbols.append("O")

    # Zn coordinated to both O atoms.  Place along -x roughly bisecting them.
    zn = np.array([c_carb[0] - 2.40, 0.0, 0.0])  # ~2.0-2.1 A from each O
    positions.append(zn); symbols.append("Zn")

    # Translate everything into the cell interior so wrapping is trivial.
    pos_arr = np.array(positions, dtype=float)
    pos_arr += np.array([cell[0, 0] / 2.0, cell[1, 1] / 2.0, cell[2, 2] / 2.0])

    atoms = Atoms(symbols=symbols, positions=pos_arr, cell=cell, pbc=True)
    return MolecularCrystal.from_ase(atoms)


def _zn_dimer_formate_crystal() -> MolecularCrystal:
    """Two Zn atoms 3.0 A apart with bridging formate-style ligands.

    Used to exercise multi-metal seed auto-grouping and per-element cap
    distances on Zn-O metal-boundary cuts.  Geometry follows real
    bridging-formate paddle-wheel proportions (Zn-O 2.00 A, C-O 1.25 A,
    O-C-O 120 deg) so the auto-detected bond graph contains the desired
    Zn-O and C-O bonds but NOT Zn-C.
    """
    cell = 20.0 * np.eye(3)
    positions: List[np.ndarray] = []
    symbols: List[str] = []

    zn_a = np.array([0.0, 0.0, 0.0])
    zn_b = np.array([3.0, 0.0, 0.0])
    positions.append(zn_a); symbols.append("Zn")
    positions.append(zn_b); symbols.append("Zn")

    # Two bridging formates above (+z) and below (-z) the Zn-Zn axis.
    # O atoms 2.00 A from each Zn; C lifted further out so it stays well
    # beyond the Zn-C bonding threshold (~2.39 A).
    for sign in (+1.0, -1.0):
        positions.append(np.array([0.42, 0.0, sign * 1.95])); symbols.append("O")
        positions.append(np.array([2.58, 0.0, sign * 1.95])); symbols.append("O")
        c = np.array([1.50, 0.0, sign * 2.58])
        positions.append(c); symbols.append("C")
        positions.append(c + np.array([0.0, 0.0, sign * 1.09])); symbols.append("H")

    pos_arr = np.array(positions, dtype=float)
    pos_arr += np.array([cell[0, 0] / 2.0, cell[1, 1] / 2.0, cell[2, 2] / 2.0])
    atoms = Atoms(symbols=symbols, positions=pos_arr, cell=cell, pbc=True)
    return MolecularCrystal.from_ase(atoms)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_keeps_complete_ligand_topology_without_cc_cuts():
    crystal = _zn_methylbenzoate_crystal()
    clusters = carve_cluster(crystal, seed="Zn", mode="bond_shells")
    assert len(clusters) == 1
    cluster = clusters[0]

    symbols = cluster.get_chemical_symbols()
    counts = {s: symbols.count(s) for s in set(symbols)}
    # Default bond_shells is topology-preserving: the whole ligand remains
    # intact and no single C-C bond is cut unless the caller explicitly asks.
    assert counts["Zn"] == 1
    assert counts["O"] == 2
    assert counts["C"] == 8
    assert counts["H"] == 7
    assert cluster.provenance.mode == "bond_shells"
    assert cluster.provenance.cut_bonds == []
    assert cluster.provenance.cut_cc_bonds_applied == []
    assert cluster.provenance.max_atoms == 500
    assert cluster.cap_local_indices == []


def test_explicit_cut_cc_bonds_respected():
    crystal = _zn_methylbenzoate_crystal()
    clusters = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", cut_cc_bonds=[(0, 10)]
    )
    cluster = clusters[0]

    symbols = cluster.get_chemical_symbols()
    counts = {s: symbols.count(s) for s in set(symbols)}
    # Expected: Zn=1, O=2, C=1(carb) + 6(ring) = 7, ring-H=4, cap-H=1 -> H=5.
    assert counts["Zn"] == 1
    assert counts["O"] == 2
    assert counts["C"] == 7
    assert counts["H"] == 5

    # The single cap H must sit ~1.09 A from a kept ring carbon, along
    # the direction toward the methyl carbon at (1.51, 0, 0) offset.
    cap_idx = cluster.cap_local_indices[0]
    cap_pos = cluster.get_positions()[cap_idx]
    # The kept ring carbon attached to the cut bond is the same C0 we
    # placed at angle 0 of the benzene ring.  Its kept-side neighbour
    # in the cluster should be exactly 1.09 A away from cap_pos.
    distances = np.linalg.norm(cluster.get_positions() - cap_pos, axis=1)
    distances[cap_idx] = np.inf
    nearest_distance = float(distances.min())
    assert nearest_distance == pytest.approx(1.09, abs=1e-3)
    assert cluster.provenance.cut_cc_bonds_requested == [(0, 10)]
    assert cluster.provenance.cut_cc_bonds_applied == [(0, 10)]


def test_bond_shells_does_not_break_aromatic_ring():
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", cut_cc_bonds=[(0, 10)]
    )[0]

    # Explicit truncation at C0-Cmethyl must not permit cuts inside the
    # aromatic ring.
    cut_pairs = cluster.provenance.cut_bonds
    parent_atoms = crystal.to_ase()
    parent_positions = parent_atoms.get_positions()
    for kept_g, dropped_g in cut_pairs:
        # Neither atom of a cut should match a ring C-C pair length (1.40 A).
        dist = np.linalg.norm(parent_positions[kept_g] - parent_positions[dropped_g])
        assert dist > 1.42 or parent_atoms.get_chemical_symbols()[kept_g] != "C" \
               or parent_atoms.get_chemical_symbols()[dropped_g] != "C"


def test_freeze_shell_layers_count():
    crystal = _zn_methylbenzoate_crystal()
    kwargs = dict(seed="Zn", mode="bond_shells", cut_cc_bonds=[(0, 10)])
    c0 = carve_cluster(crystal, freeze_shell=0, **kwargs)[0]
    c1 = carve_cluster(crystal, freeze_shell=1, **kwargs)[0]
    c2 = carve_cluster(crystal, freeze_shell=2, **kwargs)[0]

    assert c0.frozen_local_indices == []
    assert len(c1.frozen_local_indices) >= 2  # cap + keeper
    assert len(c2.frozen_local_indices) > len(c1.frozen_local_indices)
    # Caps must always be inside the frozen set when freeze_shell >= 1.
    for cap in c1.cap_local_indices:
        assert cap in c1.frozen_local_indices


def test_rcut_warns_on_non_cc_cut():
    crystal = _zn_methylbenzoate_crystal()
    # rcut=1.5 is below the Zn-O distance (~2.0 A) so BFS will try to
    # leave the Zn through a Zn-O bond and immediately fail the radial
    # test, producing a non-C-C cut.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        carve_cluster(crystal, seed="Zn", mode="rcut", rcut=1.5)
    messages = [str(w.message) for w in caught]
    assert any("not a C-C" in m for m in messages), (
        "Expected a non-C-C cut warning when rcut slices through Zn-O.\n"
        + "\n".join(messages)
    )


def test_stop_at_non_seed_metals_caps_zn_dimer():
    """With seed_merge_radius small enough to split the Zn dimer, BFS from
    one Zn must terminate at the other Zn (an implicit metal boundary)."""
    crystal = _zn_dimer_formate_crystal()
    # Tight merge radius -> two separate seed groups (one Zn each).
    clusters = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        seed_merge_radius=2.0,
    )
    assert len(clusters) == 2
    for cluster in clusters:
        # Each cluster should have at least one cut whose dropped atom
        # is the *other* Zn (a non-seed metal).
        parent_symbols = crystal.to_ase().get_chemical_symbols()
        dropped_metals = [
            dropped for (_kept, dropped) in cluster.provenance.cut_bonds
            if parent_symbols[dropped] == "Zn"
        ]
        assert len(dropped_metals) >= 1, (
            f"Expected at least one metal-boundary cut, got {cluster.provenance.cut_bonds}"
        )
    # Disabling the metal-boundary rule should make BFS reach the other
    # Zn through the bridging formate framework, yielding bigger clusters
    # with no metal-boundary cuts.
    clusters_no_stop = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        seed_merge_radius=2.0,
        stop_at_non_seed_metals=False,
    )
    for cluster in clusters_no_stop:
        parent_symbols = crystal.to_ase().get_chemical_symbols()
        for kept, dropped in cluster.provenance.cut_bonds:
            assert parent_symbols[dropped] != "Zn"


def test_default_no_cc_cuts_on_dma_like_topology():
    """Metal-boundary cuts should not be reported as user C-C truncation."""
    crystal = _zn_dimer_formate_crystal()
    clusters = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        seed_merge_radius=2.0,
    )
    assert len(clusters) == 2
    for cluster in clusters:
        assert cluster.provenance.cut_cc_bonds_requested == []
        assert cluster.provenance.cut_cc_bonds_applied == []
        assert cluster.provenance.cut_bonds == cluster.provenance.metal_boundary_cuts


def test_max_atoms_raises_with_candidates():
    crystal = _zn_methylbenzoate_crystal()
    carver = ClusterCarver(crystal)
    with pytest.raises(LigandTopologyOverflowError) as caught:
        carver.carve_bond_shells("Zn", max_atoms=3)

    err = caught.value
    assert err.actual_atom_count > err.max_atoms
    assert err.candidates

    symbols = crystal.to_ase().get_chemical_symbols()
    candidate_nodes = {idx for edge in err.candidates for idx in edge}
    rings_of = carver._ring_membership_near(sorted(candidate_nodes))
    for a, b in err.candidates:
        assert _is_cuttable_cc(symbols, carver._graph, a, b, rings_of)


@pytest.mark.parametrize(
    ("bad_bond", "message"),
    [
        ((15, 17), "expected C-C"),
        ((0, 1), "small ring"),
        ((0, 14), "no bond"),
    ],
)
def test_invalid_cut_cc_bonds_rejected(bad_bond, message):
    crystal = _zn_methylbenzoate_crystal()
    with pytest.raises(ValueError, match=message):
        carve_cluster(
            crystal,
            seed="Zn",
            mode="bond_shells",
            cut_cc_bonds=[bad_bond],
        )


def test_seed_merge_groups_paddle_wheel_metals():
    crystal = _zn_dimer_formate_crystal()
    # Default seed_merge_radius=0.0 means "no auto-grouping": the two Zn
    # at 3.0 A apart end up in separate clusters.
    clusters_split = carve_cluster(crystal, seed="Zn", mode="bond_shells")
    assert len(clusters_split) == 2

    # Explicit radius of 3.5 A (> the 3.0 A Zn-Zn separation) merges them
    # into a single seed group -> exactly one cluster.
    clusters = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", seed_merge_radius=3.5
    )
    assert len(clusters) == 1


def test_provenance_json_round_trip(tmp_path):
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", cut_cc_bonds=[(0, 10)]
    )[0]
    xyz_path = tmp_path / "cluster.xyz"
    sidecar = write_xyz_with_freeze(cluster, str(xyz_path))

    with open(sidecar) as fh:
        payload = json.load(fh)

    restored = ClusterProvenance.from_dict(payload)
    assert restored == cluster.provenance


def test_xyz_emits_freeze_and_cap_flags(tmp_path):
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        cut_cc_bonds=[(0, 10)],
        freeze_shell=1,
    )[0]
    xyz_path = tmp_path / "cluster.xyz"
    write_xyz_with_freeze(cluster, str(xyz_path))
    text = xyz_path.read_text().splitlines()
    # Comment line carries our header bits.
    assert text[1].startswith(f"natoms={len(cluster)}")
    assert "mode=bond_shells" in text[1]
    # Each per-atom line ends in one of {F, C, -}.
    body = text[2:]
    flags = {line.split()[-1] for line in body}
    assert flags <= {"F", "C", "-"}
    assert "C" in flags  # at least one cap atom
    assert "F" in flags  # freeze_shell=1 ensures at least cap + keeper frozen


def test_cap_distance_uses_bond_lengths_table_per_element():
    """Without an explicit cap_distance override, caps are placed at the
    element-specific X-H length from molcrys_kit.constants.config.BOND_LENGTHS:
    C-H = 1.09, N-H = 1.01, O-H = 0.96.

    On the Zn-dimer formate fixture with stop_at_non_seed_metals=True and
    a tight seed_merge_radius, every cut is a Zn-O metal boundary, so
    every cap must sit at 0.96 A from the kept O.
    """
    crystal = _zn_dimer_formate_crystal()
    clusters = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        seed_merge_radius=2.0,
    )
    parent_symbols = crystal.to_ase().get_chemical_symbols()
    for cluster in clusters:
        cluster_symbols = cluster.get_chemical_symbols()
        cluster_positions = cluster.get_positions()
        # Every cut here must drop the *other* Zn (a metal-boundary cut)
        # with the kept-side atom being O.
        for (kept_g, _dropped_g), cap_local in zip(
            cluster.provenance.cut_bonds, cluster.provenance.cap_local_indices
        ):
            assert parent_symbols[kept_g] == "O", (
                f"Expected O on the kept side of a Zn-O metal-boundary cut, "
                f"got {parent_symbols[kept_g]}."
            )
            cap_pos = cluster_positions[cap_local]
            # Find the kept O atom in the cluster (it's the one closest to
            # the cap H, since the cap is 0.96 A out along the bond vector).
            o_indices = [i for i, s in enumerate(cluster_symbols) if s == "O"]
            distances = [
                float(np.linalg.norm(cluster_positions[i] - cap_pos))
                for i in o_indices
            ]
            assert min(distances) == pytest.approx(0.96, abs=1e-3), (
                f"Cap H should sit at O-H = 0.96 A, got {min(distances):.4f}."
            )
        # Provenance must reflect the per-element lookup.
        assert cluster.provenance.cap_distance_A is None
        assert cluster.provenance.cap_bond_lengths_A["O"] == pytest.approx(0.96)
        assert cluster.provenance.cap_bond_lengths_A["C"] == pytest.approx(1.09)
        assert all(
            d == pytest.approx(0.96, abs=1e-6)
            for d in cluster.provenance.cap_distances_used_A
        )


def test_cap_distance_explicit_override_is_uniform():
    """Passing a positive ``cap_distance`` forces a uniform cap length
    regardless of element (the original v0 behaviour, preserved for
    backwards compatibility)."""
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        cut_cc_bonds=[(0, 10)],
        cap_distance=1.20,
    )[0]
    assert cluster.provenance.cap_distance_A == pytest.approx(1.20)
    assert all(
        d == pytest.approx(1.20)
        for d in cluster.provenance.cap_distances_used_A
    )


def test_cap_bond_lengths_user_override_changes_a_specific_element():
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal,
        seed="Zn",
        mode="bond_shells",
        cut_cc_bonds=[(0, 10)],
        cap_bond_lengths={"C-H": 1.05},
    )[0]
    # The cap is on C (cut at C0-Cmethyl), so 1.05 A must show up.
    assert cluster.provenance.cap_bond_lengths_A["C"] == pytest.approx(1.05)
    assert all(
        d == pytest.approx(1.05)
        for d in cluster.provenance.cap_distances_used_A
    )


def test_crystal_cluster_is_non_periodic():
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(crystal, seed="Zn", mode="bond_shells")[0]
    assert isinstance(cluster, CrystalCluster)
    assert not cluster.get_pbc().any()
    assert np.allclose(cluster.get_cell(), 0.0)


# ---------------------------------------------------------------------------
# Regression: bonds spanning a periodic face must NOT leave phantom bonds
# ---------------------------------------------------------------------------


def _two_zn_bridge_crystal_wrapped() -> MolecularCrystal:
    """Two Zn atoms bridged by one in-cell O *and* one image-of-C in a small
    a-axis cell, so the resulting bond graph contains a topologically
    non-trivial loop that crosses the +a face once.

    Geometry (a = 6.0, b = c = 20.0, all coordinates in Angstrom)::

        Zn1 (1.0, 10.0, 10.0)   <-- O at (2.5, 10.0, 11.0) (intra-cell bridge,
                                     1.80 A from each Zn)
        Zn2 (4.0, 10.0, 10.0)
        C   (5.5, 10.0, 11.0)   <-- bonded to Zn1 via -a image at (-0.5, ...)
                                    (Zn1-C = 1.80 A) and to Zn2 directly
                                    (1.80 A).

    The Zn-O-Zn-C-Zn loop is closed but the C->Zn1 leg crosses +a, so the
    cycle's edge-vector sum equals +a (a topologically non-trivial loop in
    the periodic bond graph).  Multi-source BFS that initialises both Zn
    seeds at offset = 0 will assign C an offset reachable via one Zn only,
    leaving the other Zn-C edge geometrically broken in the cluster -- a
    phantom bond.  This regression test asserts that the carver places no
    such phantom bonds in the final cluster geometry.
    """
    cell = np.diag([6.0, 20.0, 20.0])
    positions = np.array([
        [1.0, 10.0, 10.0],   # Zn1
        [4.0, 10.0, 10.0],   # Zn2
        [2.5, 10.0, 11.0],   # O (intra-cell bridge)
        [5.5, 10.0, 11.0],   # C (bridges Zn2 direct, Zn1 via -a image)
    ], dtype=float)
    symbols = ["Zn", "Zn", "O", "C"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return MolecularCrystal.from_ase(atoms)


def test_no_phantom_bonds_when_loop_crosses_periodic_face():
    """Cluster geometry must remain self-consistent even when the parent
    bond graph contains a topologically non-trivial loop (a cycle whose
    edge vectors sum to a non-zero cell vector).  Such loops arise in any
    framework whose seed group spans a periodic boundary, e.g. metal
    trimers in azolate-carboxylate MOFs.

    Concretely: for every parent bond between two kept atoms, the bond
    must be either geometrically realised in the cluster (Cartesian
    distance below the bond threshold) *or* explicitly broken as a
    ``loop_cut`` and capped with H on both sides.  The forbidden state
    is a parent-graph edge whose two endpoints land several Angstrom
    apart in the cluster *without* having been recorded as a loop cut
    -- that would feed an unphysically stretched bond to a downstream
    QM input.
    """
    from molcrys_kit.analysis.interactions import get_bonding_threshold
    from molcrys_kit.constants import get_atomic_radius, is_metal_element
    from molcrys_kit.operations import ClusterCarver

    crystal = _two_zn_bridge_crystal_wrapped()
    # seed_merge_radius = 3.5 A merges the two Zn (3.0 A apart) into a
    # single seed group, triggering the multi-source-BFS code path.
    carver = ClusterCarver(crystal, seed_merge_radius=3.5)
    clusters = carver.carve_bond_shells("Zn")
    assert len(clusters) == 1
    cluster = clusters[0]
    kept = cluster.provenance.kept_global_indices
    assert len(kept) == 4, (
        "All four atoms (Zn1, Zn2, O, C) should be inside the seed-group "
        f"cluster under topology-preserving carving; got {kept}."
    )

    parent_symbols = crystal.to_ase().get_chemical_symbols()
    parent_subgraph = carver._graph.subgraph(kept)
    cluster_positions = cluster.get_positions()
    global_to_local = {g: lo for lo, g in enumerate(kept)}
    loop_cut_keys = {
        tuple(sorted(pair)) for pair in cluster.provenance.loop_cuts
    }
    assert loop_cut_keys, (
        "A periodic-loop fixture must produce at least one loop_cut so the "
        "carver can break the topologically-incompatible edge."
    )

    def bt(s1: str, s2: str) -> float:
        r1 = get_atomic_radius(s1)
        r2 = get_atomic_radius(s2)
        return get_bonding_threshold(r1, r2, is_metal_element(s1), is_metal_element(s2))

    for u, v in parent_subgraph.edges():
        if tuple(sorted((int(u), int(v)))) in loop_cut_keys:
            continue
        d = float(np.linalg.norm(
            cluster_positions[global_to_local[v]]
            - cluster_positions[global_to_local[u]]
        ))
        thr = bt(parent_symbols[u], parent_symbols[v])
        assert d < thr, (
            f"Phantom bond detected: {parent_symbols[u]}#{u}--"
            f"{parent_symbols[v]}#{v} has cluster distance {d:.3f} A "
            f">= bond threshold {thr:.3f} A.  The carver claims these "
            f"atoms are bonded but their cluster positions are too far "
            f"apart -- this is the multi-source-BFS image-offset bug."
        )

    # Each loop cut produces one cap-H per non-metal endpoint (and
    # leaves any metal endpoint as an open coordination site -- Zn-H
    # hydrides would be chemically wrong).
    expected_caps_from_loops = 0
    for u, v in loop_cut_keys:
        if not is_metal_element(parent_symbols[int(u)]):
            expected_caps_from_loops += 1
        if not is_metal_element(parent_symbols[int(v)]):
            expected_caps_from_loops += 1
    assert (
        len(cluster.provenance.cap_local_indices) >= expected_caps_from_loops
    ), (
        f"Expected at least {expected_caps_from_loops} loop-cut caps; "
        f"got {len(cluster.provenance.cap_local_indices)}."
    )


# ---------------------------------------------------------------------------
# Regression: hard correctness checker (cluster_check) must pass on a
# representative periodic-loop fixture so that the C1-C7 acceptance
# criteria are guaranteed by the carver, not just by visual inspection.
# ---------------------------------------------------------------------------


def test_cluster_check_passes_on_periodic_loop_fixture():
    """The cluster checker (analysis.cluster_check) must report zero
    failures for the seeded-group carve of the periodic-loop fixture.

    This wires the acceptance-criteria checker (C1 seed coordination
    intact, C2 no phantom bonds, C3 connectivity, C4 cap/cut pairing,
    C5 cap geometry, C6 no unauthorized cuts, C7 seed-seed bonds
    preserved) into the unit-test suite as a hard regression so that
    any future carver change that violates a chemical invariant fails
    in CI rather than silently producing broken QM input.
    """
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _two_zn_bridge_crystal_wrapped()
    carver = ClusterCarver(crystal, seed_merge_radius=3.5)
    cluster = carver.carve_bond_shells("Zn")[0]

    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=cluster.to_ase(),
        provenance=cluster.provenance.to_dict(),
    )
    assert not failures, "\n".join(failures)


def test_cluster_check_catches_dropped_donor():
    """If we forge a provenance that drops one seed's first-shell
    donor, the checker must flag it as a C1 failure.  This guards the
    checker itself against silent regressions ("checker says OK when
    actually broken")."""
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _two_zn_bridge_crystal_wrapped()
    carver = ClusterCarver(crystal, seed_merge_radius=3.5)
    cluster = carver.carve_bond_shells("Zn")[0]

    bad = cluster.provenance.to_dict()
    # Drop the bridging O (parent index 2) from kept_global_indices.
    bad["kept_global_indices"] = [
        g for g in bad["kept_global_indices"] if int(g) != 2
    ]
    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=cluster.to_ase(),
        provenance=bad,
    )
    assert any(f.startswith("C1:") for f in failures), (
        "Expected a C1 violation when a seed loses its first-shell donor; "
        f"got: {failures}"
    )


# ---------------------------------------------------------------------------
# Chemistry conservation (C8 / C9 / C10): no NH2 from over-capping, atom
# counts are preserved, every cluster ligand fragment exists in the parent.
# ---------------------------------------------------------------------------


def _mu_n_two_zn_fixture() -> MolecularCrystal:
    """Two seed Zn atoms 3.0 A apart, sharing a single bridging N that
    bonds to BOTH Zn -- a topology where the naive "one cap H per cut"
    rule double-protonates the N into an NH2 group.

    Geometry::

        Zn1 (1.0, 10.0, 10.0)  --+
                                  |-- N (2.5, 10.0, 11.0)  (mu2 bridge)
        Zn2 (4.0, 10.0, 10.0)  --+

    Each Zn is also bonded to one terminal O for valence sanity.  When
    the seed group is {Zn1} only (and Zn2 is therefore an external
    metal hit by the metal-boundary cut), the bridging N is "kept" but
    sees two cut Zn-N edges: one to Zn1 (loop-cut at metal boundary?
    no -- Zn1 is in seed), one to Zn2 (metal_boundary).  Wait: with
    seed={Zn1}, Zn2 is external -> one metal_boundary cut at Zn2-N.
    So this fixture really exercises the *DMA-style* failure mode: a
    single N has two non-seed Zn cuts and would receive 2 cap H.
    """
    cell = np.diag([20.0, 20.0, 20.0])
    positions = np.array([
        [1.0, 10.0, 10.0],   # Zn1 (seed)
        [4.0, 10.0, 10.0],   # Zn2 (external -- not a seed)
        [2.5, 10.0, 11.0],   # N  (mu2-bridge, bonded to both Zn)
        [-0.5, 10.0, 10.0],  # O1 (terminal on Zn1)
        [5.5, 10.0, 10.0],   # O2 (terminal on Zn2)
    ], dtype=float)
    symbols = ["Zn", "Zn", "N", "O", "O"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return MolecularCrystal.from_ase(atoms)


def test_mu_bridging_donor_receives_exactly_one_cap_h():
    """A non-C keeper atom bonded to multiple metals outside the seed
    group must receive exactly ONE cap H regardless of how many Zn-N
    bonds were cut at that atom.  This is the rule that prevents the
    "Zn3 trimer + mu-triazolate-N => NH2" pathology observed on the
    DMA MOFs in the piezo-mof project.

    Concretely: seed = {Zn1}, Zn2 is external.  Both Zn1 and Zn2
    are bonded to the same N.  The cluster kept = {Zn1, N, O1}; the
    bridging Zn2-N bond is a metal_boundary cut and the Zn1-N bond is
    intact.  Only ONE cap H should land on the N (in the chemistry
    we are modelling, the bridging N becomes neutral N-H, not NH2).
    """
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _mu_n_two_zn_fixture()
    carver = ClusterCarver(crystal, seed_merge_radius=0.5)
    clusters = carver.carve_bond_shells([0])
    assert len(clusters) == 1
    cluster = clusters[0]
    cluster_atoms = cluster.to_ase()
    syms = cluster_atoms.get_chemical_symbols()
    pos = cluster_atoms.get_positions()
    # Locate the N atom (parent #2 -> local index in kept_sorted).
    kept = sorted(cluster.provenance.kept_global_indices)
    n_local = kept.index(2)
    # Count H within 1.3 A.
    n_h_neighbours = sum(
        1 for j, s in enumerate(syms)
        if s == "H" and j != n_local
        and float(np.linalg.norm(pos[j] - pos[n_local])) < 1.3
    )
    assert n_h_neighbours == 1, (
        f"Bridging N#2 should receive exactly one cap H (=NH), "
        f"got {n_h_neighbours} H neighbours -- this is the over-capping "
        "regression that produces NH2 groups in the DMA MOFs."
    )
    # Full checker must pass too: C8 (cap count), C9 (element conservation),
    # C10 (no foreign linker fragments).
    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=cluster_atoms,
        provenance=cluster.provenance.to_dict(),
    )
    assert not failures, "\n".join(failures)


def test_cluster_check_catches_nh2_overcapping():
    """If we forge a provenance that pretends two cap H were emitted
    for the same N keeper, the checker (C8) must flag it.  This guards
    the chemistry-formula check itself."""
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _mu_n_two_zn_fixture()
    carver = ClusterCarver(crystal, seed_merge_radius=0.5)
    cluster = carver.carve_bond_shells([0])[0]

    cluster_atoms = cluster.to_ase()
    syms = cluster_atoms.get_chemical_symbols()
    pos = cluster_atoms.get_positions()
    kept = sorted(cluster.provenance.kept_global_indices)
    n_local = kept.index(2)
    # Inject a second H next to the bridging N (1.01 A along +z).
    new_pos = np.vstack([pos, pos[n_local] + np.array([0.0, 0.0, 1.01])])
    new_syms = list(syms) + ["H"]
    forged_atoms = Atoms(symbols=new_syms, positions=new_pos, pbc=False)

    bad = cluster.provenance.to_dict()
    bad["cap_local_indices"] = list(bad["cap_local_indices"]) + [len(syms)]
    bad["cap_keeper_global_indices"] = list(
        bad["cap_keeper_global_indices"]
    ) + [2]
    bad["cap_distances_used_A"] = list(bad["cap_distances_used_A"]) + [1.01]

    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=forged_atoms,
        provenance=bad,
    )
    assert any(f.startswith("C8:") for f in failures), (
        "Expected C8 (cap-count) failure on forged NH2 over-capping; "
        f"got: {failures}"
    )


def test_cluster_check_catches_destroyed_ligand_fragment():
    """If we hide one of the bridging O atoms from the kept set, the
    cluster's "linker inventory" (C10) should flag the resulting
    foreign-formula fragment."""
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _two_zn_bridge_crystal_wrapped()
    carver = ClusterCarver(crystal, seed_merge_radius=3.5)
    cluster = carver.carve_bond_shells("Zn")[0]

    # Carry out a manual chemistry-violating mutation: replace one C
    # with an N in the cluster atoms (so the cluster claims a fragment
    # that does not exist in the parent's non-metal inventory).
    cluster_atoms = cluster.to_ase()
    syms = list(cluster_atoms.get_chemical_symbols())
    pos = cluster_atoms.get_positions()
    # Find the parent-C in the cluster (local index of parent #3).
    kept = sorted(cluster.provenance.kept_global_indices)
    c_local = kept.index(3)
    syms[c_local] = "N"
    forged_atoms = Atoms(symbols=syms, positions=pos, pbc=False)
    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=forged_atoms,
        provenance=cluster.provenance.to_dict(),
    )
    assert any(
        f.startswith("C9:") or f.startswith("C10:") for f in failures
    ), (
        "Expected an element-conservation (C9) or linker-inventory "
        f"(C10) failure for the forged cluster; got: {failures}"
    )


def _zn2_mu_oxalate_fixture() -> MolecularCrystal:
    """Two Zn atoms bridged by one oxalate dianion ``OOC-COO^{2-}``.

    Each Zn is chelated by the two O atoms of one carboxylate group.
    Seed = {Zn0}.  When Zn1 is dropped, both O of the *outer*
    carboxylate become keepers and the chemistry rule says only ONE
    should be protonated (=> -COOH, not -C(OH)2 geminal diol).

    Geometry (sp2 carboxylate, C-O = 1.25 A at 120 deg, C-C = 1.40 A,
    Zn-O ~ 2.0 A along the carboxylate bisector):

        Zn0 --- (O0,O1)=C0 --- C1=(O2,O3) --- Zn1
    """
    a = 30.0
    cell = np.eye(3) * a
    cx, cy, cz = a / 2.0, a / 2.0, a / 2.0

    def sp2(c, axis_sign):
        co = 1.25
        return [
            (c[0] - axis_sign * 0.5 * co, c[1] + 0.866 * co, c[2]),
            (c[0] - axis_sign * 0.5 * co, c[1] - 0.866 * co, c[2]),
        ]

    c0 = np.array([cx - 0.7, cy, cz])
    c1 = np.array([cx + 0.7, cy, cz])
    o0a, o0b = sp2(c0, +1)
    o1a, o1b = sp2(c1, -1)
    # Place each Zn so that Zn-O = 2.10 A (within MCK's 2.16 A Zn-O
    # bond cutoff) and Zn-C = 2.425 A (above MCK's 2.28 A Zn-C bond
    # cutoff so no spurious metal-carbon edge is created in the
    # parent bond graph).  Geometry: O sits 1.083 A off the C-C axis
    # at C-O = 1.25 A; the Zn is placed along the bisector at
    # 1.80 A from the O-midpoint so sqrt(1.80^2 + 1.083^2) = 2.10.
    zn_offset = 0.625 + 1.80  # = 2.425 from the C
    zn0 = np.array([c0[0] - zn_offset, cy, cz])
    zn1 = np.array([c1[0] + zn_offset, cy, cz])

    positions = np.array([zn0, zn1, c0, o0a, o0b, c1, o1a, o1b])
    symbols = ["Zn", "Zn", "C", "O", "O", "C", "O", "O"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return MolecularCrystal.from_ase(atoms)


def test_carboxylate_receives_single_cap_h_not_geminal_diol():
    """Both O of one carboxylate are independent keepers when their Zn
    is dropped, but the anion-group dedup must place only ONE cap H
    on the pair (= -COOH, not =-C(OH)2 geminal diol).

    This is the regression that produced ``C(OH)2`` end groups on
    every dicarboxylate in the DMA MOFs and that the user explicitly
    flagged ("端位不可能是偕二醇吧").
    """
    from molcrys_kit.analysis.cluster_check import check_cluster

    crystal = _zn2_mu_oxalate_fixture()
    carver = ClusterCarver(crystal, seed_merge_radius=0.5)
    clusters = carver.carve_bond_shells([0])
    assert len(clusters) == 1
    cluster = clusters[0]
    cluster_atoms = cluster.to_ase()
    syms = cluster_atoms.get_chemical_symbols()
    pos = cluster_atoms.get_positions()

    # Identify the outer C (the one whose Zn was dropped): it's the
    # only C whose both O neighbours are unprotonated *or* exactly
    # one is protonated.  Walk the cluster graph and count O-H per C.
    o_h_count_per_c = {}
    n_h = sum(1 for s in syms if s == "H")
    h_indices = [i for i, s in enumerate(syms) if s == "H"]
    c_indices = [i for i, s in enumerate(syms) if s == "C"]
    o_indices = [i for i, s in enumerate(syms) if s == "O"]
    for ci in c_indices:
        bonded_o = [
            oi for oi in o_indices
            if float(np.linalg.norm(pos[oi] - pos[ci])) < 1.6
        ]
        oh_on_c = 0
        for oi in bonded_o:
            for hi in h_indices:
                if float(np.linalg.norm(pos[hi] - pos[oi])) < 1.3:
                    oh_on_c += 1
                    break
        o_h_count_per_c[ci] = oh_on_c

    geminal = [ci for ci, n in o_h_count_per_c.items() if n >= 2]
    assert not geminal, (
        f"Geminal diol regression: C atoms {geminal} carry two O-H "
        "(should be -COOH not =-C(OH)2)."
    )
    # And exactly one cap H must exist for the cut carboxylate.
    assert n_h == 1, f"Expected 1 cap H for the cut -COO group, got {n_h}"

    failures = check_cluster(
        parent_atoms=crystal.to_ase(),
        cluster_atoms=cluster_atoms,
        provenance=cluster.provenance.to_dict(),
    )
    assert not failures, "\n".join(failures)


def test_chem_env_computes_anion_protonation_groups_for_oxalate():
    """Direct unit test on
    :meth:`ChemicalEnvironment.compute_anion_protonation_groups`:
    both O of each carboxylate must share one group id with the
    central C, distinct from the other carboxylate.
    """
    import networkx as nx
    from molcrys_kit.analysis.chemical_env import ChemicalEnvironment
    from molcrys_kit.constants import is_metal_element

    crystal = _zn2_mu_oxalate_fixture()
    atoms = crystal.to_ase()
    syms = atoms.get_chemical_symbols()
    # Reuse the carver's PBC bond graph and strip metals, like
    # operations.cluster._build_anion_group_map does.
    carver = ClusterCarver(crystal, seed_merge_radius=0.5)
    g = nx.Graph()
    for i, s in enumerate(syms):
        g.add_node(i, symbol=s)
    for u, v in carver._graph.edges():
        if is_metal_element(syms[int(u)]) or is_metal_element(syms[int(v)]):
            continue
        g.add_edge(int(u), int(v))

    groups = ChemicalEnvironment(
        (g, atoms.get_positions())
    ).compute_anion_protonation_groups()

    # The two carboxylates are C2/O3/O4 and C5/O6/O7.
    assert groups[3] == groups[4], "Both O of first carboxylate share group"
    assert groups[6] == groups[7], "Both O of second carboxylate share group"
    assert groups[2] == groups[3], "Central C joins first carboxylate group"
    assert groups[5] == groups[6], "Central C joins second carboxylate group"
    assert groups[3] != groups[6], "The two carboxylates are SEPARATE groups"
    # Metals keep their default identity group.
    assert groups[0] == 0 and groups[1] == 1
