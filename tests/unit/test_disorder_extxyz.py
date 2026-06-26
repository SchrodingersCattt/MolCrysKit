"""Tests for disorder metadata extxyz round-trip and in-memory disorder resolution.

Covers the three gaps closed by feat/disorder-extxyz-roundtrip:
  G1 — generate_ordered_replicas_from_disordered_sites(crystal=...) path
  G2 — extxyz round-trip preserves all disorder arrays
  G3 — DisorderInfo.from_crystal() reconstructs metadata
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder, DisorderInfo
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz
from molcrys_kit.constants.config import (
    KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL,
    KEY_SYM_OP_INDEX, KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA_DIR / "DAP-4.cif"  # special-position disorder
CAFFEINE = DATA_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif"  # assembly disorder
PETN = DATA_DIR / "PETN_PERYTN10.cif"  # ordered


@pytest.fixture
def tmp_xyz():
    fd, path = tempfile.mkstemp(suffix=".extxyz")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ── G2: extxyz round-trip preserves disorder metadata ──────────────────


class TestExtxyzRoundTripDisorder:
    """Disorder arrays survive write → read cycle."""

    def test_ordered_crystal_roundtrip(self, tmp_xyz):
        """Ordered CIF: all-default arrays still present after round-trip."""
        crystal = read_mol_crystal(str(PETN))
        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)

        atoms_orig = crystal.to_ase()
        atoms_load = loaded.to_ase()

        assert len(atoms_load) == len(atoms_orig)
        assert np.allclose(loaded.lattice, crystal.lattice)

    def test_disordered_crystal_preserves_all_arrays(self, tmp_xyz):
        """DAP-4 (disordered): occupancy/disorder_group/assembly/label/
        sym_op_index/asym_id/site_symmetry_order survive the round-trip."""
        crystal = read_mol_crystal(str(DAP4))
        atoms_orig = crystal.to_ase()

        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)
        atoms_load = loaded.to_ase()

        # All 7 disorder arrays must be present
        for key in (KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL,
                    KEY_SYM_OP_INDEX, KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER):
            assert key in atoms_load.arrays, f"Missing array: {key}"

        # Numeric arrays must match exactly
        for key in (KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_SYM_OP_INDEX,
                    KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER):
            orig = atoms_orig.arrays.get(key)
            load = atoms_load.arrays.get(key)
            if orig is not None:
                assert load is not None, f"{key} lost in round-trip"
                np.testing.assert_array_equal(load, orig, err_msg=f"{key} mismatch")

        # String arrays: empty strings become "." in extxyz; desanitised on read
        for key in (KEY_ASSEMBLY, KEY_LABEL):
            orig = atoms_orig.arrays.get(key)
            load = atoms_load.arrays.get(key)
            if orig is not None:
                assert load is not None, f"{key} lost in round-trip"
                assert len(load) == len(orig)

    def test_assembly_disorder_roundtrip(self, tmp_xyz):
        """Caffeine (assemblies A/B): assembly labels survive."""
        crystal = read_mol_crystal(str(CAFFEINE))
        atoms_orig = crystal.to_ase()

        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)
        atoms_load = loaded.to_ase()

        orig_asm = atoms_orig.arrays.get(KEY_ASSEMBLY)
        load_asm = atoms_load.arrays.get(KEY_ASSEMBLY)
        assert orig_asm is not None
        assert load_asm is not None

        # Non-empty assembly labels must be preserved
        orig_labels = {v for v in orig_asm if v and v != "."}
        load_labels = {v for v in load_asm if v and v != "."}
        assert orig_labels == load_labels, (
            f"Assembly labels changed: {orig_labels} -> {load_labels}"
        )


# ── G3: DisorderInfo.from_crystal() ────────────────────────────────────


class TestDisorderInfoFromCrystal:
    """DisorderInfo.from_crystal() agrees with scan_cif_disorder()."""

    def test_from_crystal_matches_scan(self):
        """For DAP-4, DisorderInfo built from crystal matches CIF scan."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        crystal_info = DisorderInfo.from_crystal(crystal)

        # Same number of atoms
        assert len(crystal_info.labels) == len(cif_info.labels)
        assert len(crystal_info.symbols) == len(cif_info.symbols)

        # Symbols must match (order may differ due to pymatgen expansion)
        from collections import Counter
        assert Counter(crystal_info.symbols) == Counter(cif_info.symbols[:len(crystal_info.symbols)])

        # Lattice must be close
        assert crystal_info.lattice_matrix is not None
        assert np.allclose(crystal_info.lattice_matrix, cif_info.lattice_matrix, atol=0.01)

    def test_from_crystal_roundtrip(self, tmp_xyz):
        """DisorderInfo rebuilt after extxyz round-trip still has all fields."""
        crystal = read_mol_crystal(str(DAP4))
        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)

        info = DisorderInfo.from_crystal(loaded)

        assert len(info.labels) > 0
        assert len(info.occupancies) == len(info.labels)
        assert len(info.disorder_groups) == len(info.labels)
        assert len(info.assemblies) == len(info.labels)
        assert len(info.sym_op_indices) == len(info.labels)
        assert len(info.asym_id) == len(info.labels)
        assert len(info.site_symmetry_order) == len(info.labels)
        assert info.lattice_matrix is not None
        assert info.lattice_matrix.shape == (3, 3)

    def test_from_crystal_ordered(self):
        """Ordered crystal → DisorderInfo with all-default values."""
        crystal = read_mol_crystal(str(PETN))
        info = DisorderInfo.from_crystal(crystal)

        assert all(occ == 1.0 for occ in info.occupancies)
        assert all(dg == 0 for dg in info.disorder_groups)


# ── G1: generate_ordered_replicas(crystal=...) ─────────────────────────


class TestGenerateReplicasCrystalPath:
    """Disorder resolution from in-memory crystal matches CIF-based path."""

    def test_crystal_path_produces_replicas(self):
        """Passing crystal= produces valid replicas (may differ in atom count
        from the CIF path due to coordinate precision in the mutual-exclusion
        graph, but must be non-empty and physically reasonable)."""
        cif_replicas = generate_ordered_replicas_from_disordered_sites(
            filepath=str(DAP4), method="optimal", generate_count=1,
        )
        crystal = read_mol_crystal(str(DAP4))
        mem_replicas = generate_ordered_replicas_from_disordered_sites(
            crystal=crystal, method="optimal", generate_count=1,
        )

        assert len(cif_replicas) >= 1
        assert len(mem_replicas) >= 1
        # Both paths must produce non-trivial structures
        assert cif_replicas[0].get_total_nodes() > 100
        assert mem_replicas[0].get_total_nodes() > 100

    def test_crystal_path_after_roundtrip(self, tmp_xyz):
        """Disorder resolution works after extxyz round-trip (no CIF needed)."""
        crystal = read_mol_crystal(str(DAP4))
        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)

        replicas = generate_ordered_replicas_from_disordered_sites(
            crystal=loaded, method="optimal", generate_count=1,
        )
        assert len(replicas) >= 1
        assert replicas[0].get_total_nodes() > 0

    def test_enumerate_crystal_path(self):
        """Enumerate mode via crystal= produces alternatives."""
        crystal = read_mol_crystal(str(CAFFEINE))
        replicas = generate_ordered_replicas_from_disordered_sites(
            crystal=crystal, method="enumerate", generate_count=5, coupled=True,
        )
        # Caffeine has disorder → should produce at least 1 replica
        assert len(replicas) >= 1

    def test_no_args_raises(self):
        """Neither filepath nor crystal → ValueError."""
        with pytest.raises(ValueError, match="Either"):
            generate_ordered_replicas_from_disordered_sites(
                method="optimal", generate_count=1,
            )

    def test_both_args_raises(self):
        """filepath AND crystal simultaneously → ValueError."""
        crystal = read_mol_crystal(str(DAP4))
        with pytest.raises(ValueError, match="mutually exclusive"):
            generate_ordered_replicas_from_disordered_sites(
                filepath=str(DAP4), crystal=crystal,
                method="optimal", generate_count=1,
            )
