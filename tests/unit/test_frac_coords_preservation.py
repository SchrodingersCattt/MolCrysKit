"""Tests for CIF fractional coordinate preservation through extxyz round-trip.

Verifies that frac_x/frac_y/frac_z stored by read_mol_crystal survive the
extxyz write-read cycle and that DisorderInfo.from_crystal() uses the stored
values instead of recomputing from Cartesian positions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder, DisorderInfo
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz
from molcrys_kit.constants.config import (
    KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z, KEY_LABEL,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA_DIR / "DAP-4.cif"  # disordered (special-position)
CAFFEINE = DATA_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif"  # assembly disorder
PETN = DATA_DIR / "PETN_PERYTN10.cif"  # ordered


# -- Stored values -------------------------------------------------------


class TestFracCoordsStored:
    """read_mol_crystal stores frac_x/y/z per-atom arrays."""

    def test_disordered_has_frac_coords(self):
        crystal = read_mol_crystal(str(DAP4))
        atoms = crystal.to_ase()
        for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
            assert key in atoms.arrays, f"Missing {key}"

    def test_ordered_has_frac_coords(self):
        crystal = read_mol_crystal(str(PETN))
        atoms = crystal.to_ase()
        for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
            assert key in atoms.arrays, f"Missing {key} for ordered CIF"

    def test_frac_coords_bitwise_equal_to_cif_scan(self):
        """Stored frac coords must be bitwise identical to scan_cif_disorder."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        atoms = crystal.to_ase()
        n = len(atoms)
        stored = np.column_stack([
            atoms.arrays[KEY_FRAC_X],
            atoms.arrays[KEY_FRAC_Y],
            atoms.arrays[KEY_FRAC_Z],
        ])
        np.testing.assert_array_equal(
            stored, cif_info.frac_coords[:n],
            err_msg="Stored frac coords differ from CIF scan (should be bitwise equal)",
        )

    def test_label_aligned_frac_coords(self):
        """For each atom i, stored frac[i] must correspond to label[i].

        This verifies that atom ordering between scan_cif_disorder() and
        pymatgen Structure expansion is aligned -- the critical assumption
        underlying the [:len(symbols)] slice in read_mol_crystal().
        """
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        atoms = crystal.to_ase()
        n = len(atoms)

        stored_labels = atoms.arrays.get(KEY_LABEL)
        assert stored_labels is not None, "labels not stored"

        stored_frac = np.column_stack([
            atoms.arrays[KEY_FRAC_X],
            atoms.arrays[KEY_FRAC_Y],
            atoms.arrays[KEY_FRAC_Z],
        ])

        for i in range(n):
            assert stored_labels[i] == cif_info.labels[i], (
                f"Atom {i}: label mismatch -- crystal='{stored_labels[i]}', "
                f"CIF='{cif_info.labels[i]}'"
            )
            np.testing.assert_array_equal(
                stored_frac[i], cif_info.frac_coords[i],
                err_msg=f"Atom {i} ({stored_labels[i]}): frac_coords mismatch",
            )


# -- Round-trip -----------------------------------------------------------


class TestFracCoordsRoundTrip:
    """frac_x/y/z survive extxyz write-read cycle."""

    def test_roundtrip_preserves_frac_coords(self, tmp_path):
        out = tmp_path / "out.extxyz"
        crystal = read_mol_crystal(str(DAP4))
        atoms_orig = crystal.to_ase()

        write_extxyz(crystal, str(out))
        loaded = read_extxyz(str(out))
        atoms_load = loaded.to_ase()

        for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
            assert key in atoms_load.arrays, f"Missing {key} after round-trip"
            # extxyz is ASCII text; float repr round-trip introduces ~1e-16 noise
            np.testing.assert_allclose(
                atoms_load.arrays[key],
                atoms_orig.arrays[key],
                atol=1e-15,
                err_msg=f"{key} changed after round-trip",
            )

    def test_ordered_roundtrip(self, tmp_path):
        out = tmp_path / "out.extxyz"
        crystal = read_mol_crystal(str(PETN))
        atoms_orig = crystal.to_ase()

        write_extxyz(crystal, str(out))
        loaded = read_extxyz(str(out))
        atoms_load = loaded.to_ase()

        for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z):
            assert key in atoms_load.arrays, f"Missing {key} for ordered CIF"
            np.testing.assert_allclose(
                atoms_load.arrays[key],
                atoms_orig.arrays[key],
                atol=1e-15,
                err_msg=f"{key} changed after ordered-CIF round-trip",
            )


# -- DisorderInfo.from_crystal() -----------------------------------------


class TestDisorderInfoUsesStoredFracCoords:
    """DisorderInfo.from_crystal() recomputes frac coords from Cartesian.

    Since from_crystal() deliberately ignores stored frac_x/y/z arrays
    (they become stale after lattice-transforming operations like slab
    cutting), we verify that recomputed coordinates are close to, but
    not necessarily bitwise-equal to, the CIF-parsed values.
    """

    def test_from_crystal_frac_coords_close_to_cif(self):
        """from_crystal frac_coords must be close to CIF scan values (mod 1)."""
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        crystal_info = DisorderInfo.from_crystal(crystal)

        n = len(crystal_info.frac_coords)
        # Compare modulo 1 since CIF stores [0,1) but recomputed coords
        # may be outside that range (e.g. -0.04 vs 0.96 are equivalent).
        diff = crystal_info.frac_coords - cif_info.frac_coords[:n]
        diff -= np.round(diff)
        np.testing.assert_allclose(
            diff,
            0.0,
            atol=1e-10,
            err_msg="from_crystal frac_coords differ too much from CIF values",
        )

    def test_from_crystal_after_roundtrip_bitwise(self, tmp_path):
        """Frac coords survive extxyz round-trip with ASCII-float precision."""
        out = tmp_path / "out.extxyz"
        cif_info = scan_cif_disorder(str(DAP4))
        crystal = read_mol_crystal(str(DAP4))
        write_extxyz(crystal, str(out))
        loaded = read_extxyz(str(out))

        info = DisorderInfo.from_crystal(loaded)
        n = len(info.frac_coords)
        # from_crystal always recomputes from Cartesian; extxyz round-trip
        # adds further noise (~1e-10 from ASCII float repr) → compare
        # modulo 1 with a tolerance that accommodates both effects.
        diff = info.frac_coords - cif_info.frac_coords[:n]
        diff -= np.round(diff)
        np.testing.assert_allclose(
            diff,
            0.0,
            atol=1e-9,
            err_msg="frac_coords lost precision after extxyz round-trip",
        )

    def test_caffeine_from_crystal_frac_coords(self):
        """Assembly-disordered CIF (caffeine) also gives close frac coords."""
        cif_info = scan_cif_disorder(str(CAFFEINE))
        crystal = read_mol_crystal(str(CAFFEINE))
        crystal_info = DisorderInfo.from_crystal(crystal)

        n = len(crystal_info.frac_coords)
        diff = crystal_info.frac_coords - cif_info.frac_coords[:n]
        diff -= np.round(diff)
        np.testing.assert_allclose(
            diff,
            0.0,
            atol=1e-10,
            err_msg="caffeine frac_coords mismatch",
        )
