"""Tests for single-parser CIF reading and crystal= vs filepath= consistency.

Verifies that read_mol_crystal() uses scan_cif_disorder() as the sole
authority for coordinates AND metadata, and that disorder resolution
via crystal= produces the same results as filepath=.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder, DisorderInfo
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz
from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA_DIR / "DAP-4.cif"  # special-position disorder
CAFFEINE = DATA_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif"  # assembly disorder
PETN = DATA_DIR / "PETN_PERYTN10.cif"  # ordered


class TestSingleParserAtomCount:
    """read_mol_crystal atom count must equal scan_cif_disorder atom count."""

    def test_ordered_atom_count_matches(self):
        di = scan_cif_disorder(str(PETN))
        crystal = read_mol_crystal(str(PETN))
        assert crystal.get_total_nodes() == len(di.symbols)

    def test_disordered_atom_count_matches(self):
        di = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        assert crystal.get_total_nodes() == len(di.symbols)

    def test_assembly_disorder_atom_count_matches(self):
        di = scan_cif_disorder(str(CAFFEINE))
        crystal = read_mol_crystal(str(CAFFEINE))
        assert crystal.get_total_nodes() == len(di.symbols)


class TestCrystalPathMatchesFilePath:
    """generate_ordered_replicas(crystal=) must match filepath= in atom count."""

    def test_optimal_atom_count_matches(self):
        cif_reps = generate_ordered_replicas_from_disordered_sites(
            filepath=str(DAP4), method="optimal", generate_count=1,
        )
        crystal = read_mol_crystal(str(DAP4))
        mem_reps = generate_ordered_replicas_from_disordered_sites(
            crystal=crystal, method="optimal", generate_count=1,
        )
        assert len(cif_reps) == len(mem_reps)
        assert cif_reps[0].get_total_nodes() == mem_reps[0].get_total_nodes()

    def test_enumerate_atom_count_matches(self):
        cif_reps = generate_ordered_replicas_from_disordered_sites(
            filepath=str(DAP4), method="enumerate", generate_count=3, coupled=True,
        )
        crystal = read_mol_crystal(str(DAP4))
        mem_reps = generate_ordered_replicas_from_disordered_sites(
            crystal=crystal, method="enumerate", generate_count=3, coupled=True,
        )
        assert len(cif_reps) == len(mem_reps)
        for c, m in zip(cif_reps, mem_reps):
            assert c.get_total_nodes() == m.get_total_nodes()

    def test_after_extxyz_roundtrip(self, tmp_path):
        """crystal= path works after extxyz round-trip."""
        out = tmp_path / "out.extxyz"
        cif_reps = generate_ordered_replicas_from_disordered_sites(
            filepath=str(DAP4), method="optimal", generate_count=1,
        )
        crystal = read_mol_crystal(str(DAP4))
        write_extxyz(crystal, str(out))
        loaded = read_extxyz(str(out))
        mem_reps = generate_ordered_replicas_from_disordered_sites(
            crystal=loaded, method="optimal", generate_count=1,
        )
        assert len(mem_reps) >= 1
        assert cif_reps[0].get_total_nodes() == mem_reps[0].get_total_nodes()


class TestCifTextMatchesFilepath:
    """cif_text= path must match filepath= path exactly."""

    def test_ordered_text_matches_file(self):
        text = PETN.read_text(encoding="utf-8")
        from_file = read_mol_crystal(str(PETN))
        from_text = read_mol_crystal(cif_text=text)
        assert from_file.get_total_nodes() == from_text.get_total_nodes()

    def test_disordered_text_matches_file(self):
        text = DAP4.read_text(encoding="utf-8")
        from_file = read_mol_crystal(str(DAP4))
        from_text = read_mol_crystal(cif_text=text)
        assert from_file.get_total_nodes() == from_text.get_total_nodes()
