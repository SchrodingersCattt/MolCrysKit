"""Tests for in-memory CIF text parsing (no file I/O).

Verifies that read_mol_crystal(cif_text=...) and scan_cif_disorder(cif_text=...)
produce identical results to the filepath-based path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA_DIR / "DAP-4.cif"
PETN = DATA_DIR / "PETN_PERYTN10.cif"
CAFFEINE = DATA_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif"


@pytest.fixture
def dap4_text() -> str:
    return DAP4.read_text(encoding="utf-8")


@pytest.fixture
def petn_text() -> str:
    return PETN.read_text(encoding="utf-8")


@pytest.fixture
def caffeine_text() -> str:
    return CAFFEINE.read_text(encoding="utf-8")


class TestScanCifDisorderText:
    """scan_cif_disorder(cif_text=...) matches filepath path."""

    def test_disordered_matches(self, dap4_text):
        from_file = scan_cif_disorder(str(DAP4))
        from_text = scan_cif_disorder(cif_text=dap4_text)

        assert len(from_text.labels) == len(from_file.labels)
        assert from_text.labels == from_file.labels
        assert from_text.occupancies == from_file.occupancies
        assert from_text.disorder_groups == from_file.disorder_groups
        assert np.allclose(from_text.frac_coords, from_file.frac_coords)
        assert from_text.lattice_matrix is not None
        assert np.allclose(from_text.lattice_matrix, from_file.lattice_matrix)

    def test_ordered_matches(self, petn_text):
        from_file = scan_cif_disorder(str(PETN))
        from_text = scan_cif_disorder(cif_text=petn_text)

        assert len(from_text.labels) == len(from_file.labels)
        assert not from_text.has_disorder

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            scan_cif_disorder()


class TestReadMolCrystalText:
    """read_mol_crystal(cif_text=...) matches filepath path."""

    def test_ordered_crystal_matches(self, petn_text):
        from_file = read_mol_crystal(str(PETN))
        from_text = read_mol_crystal(cif_text=petn_text)

        assert from_text.get_total_nodes() == from_file.get_total_nodes()
        assert len(from_text.molecules) == len(from_file.molecules)
        assert np.allclose(from_text.lattice, from_file.lattice)

    def test_disordered_crystal_matches(self, dap4_text):
        from_file = read_mol_crystal(str(DAP4))
        from_text = read_mol_crystal(cif_text=dap4_text)

        assert from_text.get_total_nodes() == from_file.get_total_nodes()
        assert len(from_text.molecules) == len(from_file.molecules)

    def test_assembly_disorder_matches(self, caffeine_text):
        from_file = read_mol_crystal(str(CAFFEINE))
        from_text = read_mol_crystal(cif_text=caffeine_text)

        assert from_text.get_total_nodes() == from_file.get_total_nodes()

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            read_mol_crystal()
