"""Tests for CIF fractional coordinate preservation through extxyz round-trip.

Verifies that frac_x/frac_y/frac_z stored by read_mol_crystal survive the
extxyz write→read cycle and that DisorderInfo.from_crystal() uses the exact
stored values instead of recomputing from Cartesian positions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder, DisorderInfo
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz
from molcrys_kit.constants.config import KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA_DIR / "DAP-4.cif"  # disordered
CAFFEINE = DATA_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif"


@pytest.fixture
def tmp_xyz():
    fd, path = tempfile.mkstemp(suffix=".extxyz")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestFracCoordsStored:
    """read_mol_crystal stores frac_x/y/z per-atom arrays."""

    def test_disordered_has_frac_coords(self):
        crystal = read_mol_crystal(str(DAP4))
        atoms = crystal.to_ase()
        assert KEY_FRAC_X in atoms.arrays
        assert KEY_FRAC_Y in atoms.arrays
        assert KEY_FRAC_Z in atoms.arrays

    def test_frac_coords_match_cif_scan(self):
        """Stored frac coords must match scan_cif_disorder exactly."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        atoms = crystal.to_ase()
        n = len(atoms)
        stored = np.column_stack([
            atoms.arrays[KEY_FRAC_X],
            atoms.arrays[KEY_FRAC_Y],
            atoms.arrays[KEY_FRAC_Z],
        ])
        # CIF scan may have more atoms than pymatgen expansion; compare up to n
        np.testing.assert_allclose(
            stored, cif_info.frac_coords[:n],
            atol=1e-10,
            err_msg="Stored frac coords differ from CIF scan values",
        )


class TestFracCoordsRoundTrip:
    """frac_x/y/z survive extxyz write→read cycle."""

    def test_roundtrip_preserves_frac_coords(self, tmp_xyz):
        crystal = read_mol_crystal(str(DAP4))
        atoms_orig = crystal.to_ase()

        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)
        atoms_load = loaded.to_ase()

        for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
            assert key in atoms_load.arrays, f"Missing {key} after round-trip"
            np.testing.assert_allclose(
                atoms_load.arrays[key],
                atoms_orig.arrays[key],
                atol=1e-10,
                err_msg=f"{key} changed after round-trip",
            )


class TestDisorderInfoUsesStoredFracCoords:
    """DisorderInfo.from_crystal() uses exact frac coords, not recomputed."""

    def test_from_crystal_frac_coords_exact(self):
        """from_crystal frac_coords must match CIF scan exactly."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        crystal_info = DisorderInfo.from_crystal(crystal)

        n = len(crystal_info.frac_coords)
        np.testing.assert_allclose(
            crystal_info.frac_coords,
            cif_info.frac_coords[:n],
            atol=1e-10,
            err_msg="from_crystal frac_coords should be exact CIF values",
        )

    def test_from_crystal_after_roundtrip(self, tmp_xyz):
        """Exact frac_coords survive extxyz round-trip into from_crystal."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        write_extxyz(crystal, tmp_xyz)
        loaded = read_extxyz(tmp_xyz)

        info = DisorderInfo.from_crystal(loaded)
        n = len(info.frac_coords)
        np.testing.assert_allclose(
            info.frac_coords,
            cif_info.frac_coords[:n],
            atol=1e-10,
            err_msg="frac_coords lost precision after extxyz round-trip",
        )
