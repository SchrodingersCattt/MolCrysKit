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
from molcrys_kit.operations import ClusterCarver, carve_cluster
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


def test_bond_shells_n0_keeps_only_sbu_and_caps_at_carboxylate_c():
    crystal = _zn_methylbenzoate_crystal()
    clusters = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=0)
    assert len(clusters) == 1
    cluster = clusters[0]

    # n_shells=0 should keep: Zn + 2 O + 1 C(carb) + 1 cap H = 5 atoms.
    elements = sorted(cluster.get_chemical_symbols())
    assert elements == sorted(["Zn", "O", "O", "C", "H"])
    assert cluster.provenance.mode == "bond_shells"
    assert cluster.provenance.n_shells == 0
    assert len(cluster.cap_local_indices) == 1


def test_bond_shells_n1_keeps_ring_and_caps_at_methyl():
    crystal = _zn_methylbenzoate_crystal()
    clusters = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1)
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


def test_bond_shells_does_not_break_aromatic_ring():
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1)[0]

    # In bond_shells mode at n_shells>=1, all 6 ring carbons must be
    # present (count above already enforces 7 C total = 6 ring + 1 carb).
    # As a stronger statement, no cut-bond pair can lie inside the ring.
    cut_pairs = cluster.provenance.cut_bonds
    parent_atoms = crystal.to_ase()
    parent_positions = parent_atoms.get_positions()
    for kept_g, dropped_g in cut_pairs:
        # Carb-ring-C bond was preserved at n_shells=1, so neither atom of
        # a cut should match a ring C-C pair length (1.40 A).
        dist = np.linalg.norm(parent_positions[kept_g] - parent_positions[dropped_g])
        assert dist > 1.42 or parent_atoms.get_chemical_symbols()[kept_g] != "C" \
               or parent_atoms.get_chemical_symbols()[dropped_g] != "C"


def test_freeze_shell_layers_count():
    crystal = _zn_methylbenzoate_crystal()
    c0 = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1, freeze_shell=0)[0]
    c1 = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1, freeze_shell=1)[0]
    c2 = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1, freeze_shell=2)[0]

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
        n_shells=1,
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
        n_shells=1,
        seed_merge_radius=2.0,
        stop_at_non_seed_metals=False,
    )
    for cluster in clusters_no_stop:
        parent_symbols = crystal.to_ase().get_chemical_symbols()
        for kept, dropped in cluster.provenance.cut_bonds:
            assert parent_symbols[dropped] != "Zn"


def test_seed_merge_groups_paddle_wheel_metals():
    crystal = _zn_dimer_formate_crystal()
    # Default seed_merge_radius=0.0 means "no auto-grouping": the two Zn
    # at 3.0 A apart end up in separate clusters.
    clusters_split = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", n_shells=0
    )
    assert len(clusters_split) == 2

    # Explicit radius of 3.5 A (> the 3.0 A Zn-Zn separation) merges them
    # into a single seed group -> exactly one cluster.
    clusters = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", n_shells=0, seed_merge_radius=3.5
    )
    assert len(clusters) == 1


def test_provenance_json_round_trip(tmp_path):
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1)[0]
    xyz_path = tmp_path / "cluster.xyz"
    sidecar = write_xyz_with_freeze(cluster, str(xyz_path))

    with open(sidecar) as fh:
        payload = json.load(fh)

    restored = ClusterProvenance.from_dict(payload)
    assert restored == cluster.provenance


def test_xyz_emits_freeze_and_cap_flags(tmp_path):
    crystal = _zn_methylbenzoate_crystal()
    cluster = carve_cluster(
        crystal, seed="Zn", mode="bond_shells", n_shells=1, freeze_shell=1
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
        n_shells=0,
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
        n_shells=1,
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
        n_shells=1,
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
    cluster = carve_cluster(crystal, seed="Zn", mode="bond_shells", n_shells=1)[0]
    assert isinstance(cluster, CrystalCluster)
    assert not cluster.get_pbc().any()
    assert np.allclose(cluster.get_cell(), 0.0)
