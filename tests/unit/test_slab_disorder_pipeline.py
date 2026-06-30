"""
Regression tests for the slab ↔ disorder interaction.

Verifies that:
1. Disorder metadata (occupancy, disorder_group, assembly, label) survives
   slab cutting via get_unwrapped_molecules() (Phase A fix).
2. Stale frac_x/y/z arrays are stripped after slab construction (Phase B1).
3. DisorderInfo.from_crystal() recomputes correct frac coords for slabs
   (Phase B2).
4. The disorder pipeline respects slab PBC (True, True, False) (Phase C).
5. Unique sym_op_index/asym_id per slab layer (Phase D).
6. End-to-end: disorder-first→slab pipeline produces correct results.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder, DisorderInfo
from molcrys_kit.operations.surface import (
    TopologicalSlabGenerator,
    generate_topological_slab,
)
from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.constants.config import (
    KEY_OCCUPANCY,
    KEY_DISORDER_GROUP,
    KEY_ASSEMBLY,
    KEY_LABEL,
    KEY_SYM_OP_INDEX,
    KEY_ASYM_ID,
    KEY_FRAC_X,
    KEY_FRAC_Y,
    KEY_FRAC_Z,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
DAN2 = DATA_DIR / "DAN-2.cif"

# Use a simpler disordered CIF for faster tests
DAP4 = DATA_DIR / "DAP-4.cif"


# ---------------------------------------------------------------------------
# Phase A: Disorder metadata survives get_unwrapped_molecules()
# ---------------------------------------------------------------------------


class TestUnwrappedMoleculesPreserveArrays:
    """get_unwrapped_molecules() must preserve per-atom metadata arrays."""

    def test_disorder_arrays_survive_unwrapping(self):
        """Occupancy, disorder_group, assembly, label must survive."""
        crystal = read_mol_crystal(str(DAP4))

        # Before unwrapping: molecules carry disorder metadata
        for mol in crystal.molecules:
            assert KEY_OCCUPANCY in mol.arrays
            assert KEY_DISORDER_GROUP in mol.arrays

        # After unwrapping
        unwrapped = crystal.get_unwrapped_molecules()
        assert len(unwrapped) > 0

        for mol in unwrapped:
            assert KEY_OCCUPANCY in mol.arrays, "occupancy lost after unwrapping"
            assert KEY_DISORDER_GROUP in mol.arrays, "disorder_group lost after unwrapping"
            assert KEY_ASSEMBLY in mol.arrays, "assembly lost after unwrapping"
            assert KEY_LABEL in mol.arrays, "label lost after unwrapping"

    def test_occupancy_values_preserved(self):
        """Partial occupancy values must not be reset to 1.0."""
        crystal = read_mol_crystal(str(DAP4))

        # Collect original occupancies per molecule
        orig_occs = {}
        for i, mol in enumerate(crystal.molecules):
            orig_occs[i] = mol.arrays[KEY_OCCUPANCY].copy()

        unwrapped = crystal.get_unwrapped_molecules()

        for i, mol in enumerate(unwrapped):
            if i < len(orig_occs):
                np.testing.assert_array_equal(
                    mol.arrays[KEY_OCCUPANCY],
                    orig_occs[i],
                    err_msg=f"Molecule {i}: occupancy values changed after unwrapping",
                )

    def test_stale_frac_coords_stripped_after_unwrapping(self):
        """frac_x/y/z must be stripped after unwrapping."""
        crystal = read_mol_crystal(str(DAP4))

        # Before: frac arrays exist
        assert KEY_FRAC_X in crystal.molecules[0].arrays

        unwrapped = crystal.get_unwrapped_molecules()

        for mol in unwrapped:
            assert KEY_FRAC_X not in mol.arrays, "frac_x not stripped"
            assert KEY_FRAC_Y not in mol.arrays, "frac_y not stripped"
            assert KEY_FRAC_Z not in mol.arrays, "frac_z not stripped"


# ---------------------------------------------------------------------------
# Phase B: Stale frac arrays stripped in slab, correct recomputation
# ---------------------------------------------------------------------------


class TestSlabStripsFracArrays:
    """Slab construction must strip stale frac_x/y/z arrays."""

    @pytest.fixture
    def acetaminophen_slab(self):
        """A simple ordered slab for testing array stripping."""
        cif = DATA_DIR / "Acetaminophen_HXACAN.cif"
        crystal = read_mol_crystal(str(cif))
        return generate_topological_slab(crystal, (1, 0, 0), layers=2, vacuum=10.0)

    def test_no_frac_arrays_in_slab(self, acetaminophen_slab):
        slab = acetaminophen_slab
        for mol in slab.molecules:
            assert KEY_FRAC_X not in mol.arrays, "frac_x survived slab cutting"
            assert KEY_FRAC_Y not in mol.arrays, "frac_y survived slab cutting"
            assert KEY_FRAC_Z not in mol.arrays, "frac_z survived slab cutting"


class TestDisorderInfoFromCrystalRecomputes:
    """DisorderInfo.from_crystal() always recomputes frac coords."""

    def test_recomputed_frac_coords_are_valid(self):
        """Recomputed frac coords should match positions / lattice."""
        crystal = read_mol_crystal(str(DAP4))
        info = DisorderInfo.from_crystal(crystal)

        # Verify round-trip: frac @ lattice ≈ Cartesian
        atoms = crystal.to_ase()
        cart_from_frac = info.frac_coords @ info.lattice_matrix
        np.testing.assert_allclose(
            cart_from_frac,
            atoms.get_positions(),
            atol=1e-8,
            err_msg="Recomputed frac coords don't reconstruct positions",
        )


# ---------------------------------------------------------------------------
# Phase C: PBC-aware disorder resolution
# ---------------------------------------------------------------------------


class TestPbcPropagation:
    """PBC is propagated through the disorder pipeline."""

    def test_disorder_info_from_crystal_captures_pbc(self):
        """from_crystal() must capture the crystal's PBC."""
        crystal = read_mol_crystal(str(DAP4))
        info = DisorderInfo.from_crystal(crystal)
        assert info.pbc == (True, True, True)

    def test_slab_pbc_propagated(self):
        """A slab crystal should have pbc=(True,True,False) in DisorderInfo."""
        cif = DATA_DIR / "Acetaminophen_HXACAN.cif"
        crystal = read_mol_crystal(str(cif))
        slab = generate_topological_slab(crystal, (1, 0, 0), layers=2, vacuum=10.0)
        assert slab.pbc == (True, True, False)

        info = DisorderInfo.from_crystal(slab)
        assert info.pbc == (True, True, False), (
            "DisorderInfo should capture slab's non-periodic Z"
        )


# ---------------------------------------------------------------------------
# Phase D: Unique sym_op_index / asym_id per slab layer
# ---------------------------------------------------------------------------


class TestUniqueLayerMetadata:
    """Multi-layer slabs must have unique sym_op_index/asym_id per layer."""

    def test_sym_op_index_unique_across_layers(self):
        """No two layers should share the same sym_op_index values."""
        crystal = read_mol_crystal(str(DAP4))
        slab = generate_topological_slab(crystal, (1, 0, 0), layers=3, vacuum=10.0)

        # Partition slab molecules into layers.
        # With 3 layers, molecules 0..n-1 are layer 0, n..2n-1 are layer 1, etc.
        n_mols_per_layer = len(crystal.get_unwrapped_molecules())

        layer_soi_sets = []
        for layer_idx in range(3):
            layer_sois = set()
            start = layer_idx * n_mols_per_layer
            end = start + n_mols_per_layer
            for mol in slab.molecules[start:end]:
                if KEY_SYM_OP_INDEX in mol.arrays:
                    layer_sois.update(mol.arrays[KEY_SYM_OP_INDEX].tolist())
            layer_soi_sets.append(layer_sois)

        # If sym_op_index arrays exist, verify cross-layer disjointness
        if all(s for s in layer_soi_sets):
            for i in range(len(layer_soi_sets)):
                for j in range(i + 1, len(layer_soi_sets)):
                    overlap = layer_soi_sets[i] & layer_soi_sets[j]
                    assert not overlap, (
                        f"Layer {i} and layer {j} share sym_op_index values: {overlap}"
                    )


# ---------------------------------------------------------------------------
# End-to-end: disorder-first → slab pipeline
# ---------------------------------------------------------------------------


class TestDisorderThenSlabPipeline:
    """The correct pipeline: resolve disorder first, then cut slab."""

    def test_dan2_disorder_then_slab(self):
        """DAN-2: resolve disorder → cut slab produces correct element totals."""
        # Step 1: Resolve disorder
        replicas = generate_ordered_replicas_from_disordered_sites(str(DAN2))
        if isinstance(replicas[0], tuple):
            ordered = replicas[0][0]
        else:
            ordered = replicas[0]

        # Verify disorder resolution produced correct counts
        elem_counter = Counter()
        for mol in ordered.molecules:
            elem_counter.update(mol.get_chemical_symbols())
        assert elem_counter["K"] == 1
        assert elem_counter["C"] == 6
        assert elem_counter["H"] == 14
        assert elem_counter["N"] == 5
        assert elem_counter["O"] == 9
        assert sum(elem_counter.values()) == 35

        # Step 2: Cut slab
        slab = generate_topological_slab(ordered, (1, 0, 0), layers=2, vacuum=10.0)
        assert slab.pbc == (True, True, False)
        assert len(slab.molecules) > 0

        # Slab should have 2× the atoms of the ordered crystal
        n_slab_atoms = sum(len(mol) for mol in slab.molecules)
        n_ordered_atoms = sum(len(mol) for mol in ordered.molecules)
        assert n_slab_atoms == 2 * n_ordered_atoms, (
            f"2-layer slab should have 2× atoms: {n_slab_atoms} != 2×{n_ordered_atoms}"
        )

    def test_acetaminophen_disorder_then_slab_roundtrip(self):
        """Ordered CIF → slab → verify molecule integrity."""
        cif = DATA_DIR / "Acetaminophen_HXACAN.cif"
        crystal = read_mol_crystal(str(cif))
        slab = generate_topological_slab(crystal, (0, 1, 0), layers=2, vacuum=10.0)

        # Every molecule in the slab should have reasonable atom counts
        for mol in slab.molecules:
            assert len(mol) > 0
            # Acetaminophen is C8H9NO2 (20 atoms)
            assert len(mol) <= 30, f"Unexpectedly large molecule: {len(mol)} atoms"


# ---------------------------------------------------------------------------
# Phase 1 tests: CIF round-trip preserves disorder provenance
# ---------------------------------------------------------------------------


class TestCifRoundTripDisorderProvenance:
    """sym_op_index, asym_id, site_symmetry_order survive CIF write→read."""

    @pytest.fixture
    def dap4_crystal(self):
        return read_mol_crystal(str(DAP4))

    def test_cif_roundtrip_preserves_sym_op_index(self, tmp_path, dap4_crystal):
        """sym_op_index survives slab → CIF → re-read."""
        slab = generate_topological_slab(dap4_crystal, (1, 0, 0), layers=2, vacuum=10.0)

        # Write to CIF and re-read
        from molcrys_kit.io.output import write_cif
        cif_path = tmp_path / "slab.cif"
        write_cif(slab, str(cif_path))
        re_read = read_mol_crystal(str(cif_path))

        # Collect sym_op_index from original slab and re-read
        slab_soi = []
        for mol in slab.molecules:
            if KEY_SYM_OP_INDEX in mol.arrays:
                slab_soi.extend(mol.arrays[KEY_SYM_OP_INDEX].tolist())
        re_read_soi = []
        for mol in re_read.molecules:
            if KEY_SYM_OP_INDEX in mol.arrays:
                re_read_soi.extend(mol.arrays[KEY_SYM_OP_INDEX].tolist())

        assert len(slab_soi) == len(re_read_soi), (
            f"Atom count mismatch: {len(slab_soi)} vs {len(re_read_soi)}"
        )
        np.testing.assert_array_equal(
            slab_soi, re_read_soi,
            err_msg="sym_op_index lost/changed after CIF round-trip",
        )

    def test_cif_roundtrip_preserves_asym_id(self, tmp_path, dap4_crystal):
        """asym_id survives slab → CIF → re-read."""
        slab = generate_topological_slab(dap4_crystal, (1, 0, 0), layers=2, vacuum=10.0)

        from molcrys_kit.io.output import write_cif
        cif_path = tmp_path / "slab.cif"
        write_cif(slab, str(cif_path))
        re_read = read_mol_crystal(str(cif_path))

        slab_aid = []
        for mol in slab.molecules:
            if KEY_ASYM_ID in mol.arrays:
                slab_aid.extend(mol.arrays[KEY_ASYM_ID].tolist())
        re_read_aid = []
        for mol in re_read.molecules:
            if KEY_ASYM_ID in mol.arrays:
                re_read_aid.extend(mol.arrays[KEY_ASYM_ID].tolist())

        assert len(slab_aid) == len(re_read_aid)
        np.testing.assert_array_equal(
            slab_aid, re_read_aid,
            err_msg="asym_id lost/changed after CIF round-trip",
        )

    def test_cif_roundtrip_disorder_metadata_complete(self, tmp_path, dap4_crystal):
        """All 7 disorder arrays survive CIF round-trip."""
        slab = generate_topological_slab(dap4_crystal, (1, 0, 0), layers=2, vacuum=10.0)

        from molcrys_kit.io.output import write_cif
        cif_path = tmp_path / "slab.cif"
        write_cif(slab, str(cif_path))

        # Re-read via scan_cif_disorder (CLI path)
        info = scan_cif_disorder(str(cif_path))
        n = len(info.sym_op_indices)
        assert n > 0
        # All sym_op_indices should be non-negative
        assert all(soi >= 0 for soi in info.sym_op_indices)
        # All asym_ids should be present (not all -1)
        assert any(aid != -1 for aid in info.asym_id)


# ---------------------------------------------------------------------------
# Phase 2 tests: Supercell disorder metadata
# ---------------------------------------------------------------------------


class TestSupercellDisorderMetadata:
    """Supercell preserves disorder arrays and offsets sym_op_index/asym_id."""

    @pytest.fixture
    def dap4_crystal(self):
        return read_mol_crystal(str(DAP4))

    def test_get_supercell_preserves_arrays(self, dap4_crystal):
        """get_supercell must preserve occupancy, disorder_group, assembly, label."""
        sc = dap4_crystal.get_supercell(2, 1, 1)

        for mol in sc.molecules:
            assert KEY_OCCUPANCY in mol.arrays
            assert KEY_DISORDER_GROUP in mol.arrays
            assert KEY_ASSEMBLY in mol.arrays
            assert KEY_LABEL in mol.arrays

    def test_get_supercell_strips_frac_coords(self, dap4_crystal):
        """get_supercell must strip stale frac_x/y/z."""
        sc = dap4_crystal.get_supercell(2, 1, 1)

        for mol in sc.molecules:
            assert KEY_FRAC_X not in mol.arrays, "frac_x survived supercell"
            assert KEY_FRAC_Y not in mol.arrays, "frac_y survived supercell"
            assert KEY_FRAC_Z not in mol.arrays, "frac_z survived supercell"

    def test_get_supercell_unique_sym_op_index(self, dap4_crystal):
        """sym_op_index must be unique across repeated cells in supercell."""
        sc = dap4_crystal.get_supercell(2, 2, 1)

        # Molecules 0..n-1 = cell (0,0), n..2n-1 = cell (0,1),
        # 2n..3n-1 = cell (1,0), 3n..4n-1 = cell (1,1)
        n_per_cell = len(dap4_crystal.molecules)
        cell_soi_sets = []
        for cell_idx in range(4):
            cell_sois = set()
            start = cell_idx * n_per_cell
            end = start + n_per_cell
            for mol in sc.molecules[start:end]:
                if KEY_SYM_OP_INDEX in mol.arrays:
                    cell_sois.update(mol.arrays[KEY_SYM_OP_INDEX].tolist())
            cell_soi_sets.append(cell_sois)

        # All cells must have disjoint sym_op_index sets
        if all(s for s in cell_soi_sets):
            for i in range(len(cell_soi_sets)):
                for j in range(i + 1, len(cell_soi_sets)):
                    overlap = cell_soi_sets[i] & cell_soi_sets[j]
                    assert not overlap, (
                        f"Cell {i} and cell {j} share sym_op_index: {overlap}"
                    )

    def test_create_supercell_equals_get_supercell(self, dap4_crystal):
        """create_supercell delegates to get_supercell — results identical."""
        from molcrys_kit.operations.builders import create_supercell
        sc1 = dap4_crystal.get_supercell(2, 1, 1)
        sc2 = create_supercell(dap4_crystal, (2, 1, 1))

        # Same number of molecules
        assert len(sc1.molecules) == len(sc2.molecules)
        # Same number of atoms
        n1 = sum(len(m) for m in sc1.molecules)
        n2 = sum(len(m) for m in sc2.molecules)
        assert n1 == n2

    def test_create_supercell_fractional_sanitation(self, dap4_crystal):
        """create_supercell output should be clean for disorder resolution."""
        from molcrys_kit.operations.builders import create_supercell
        sc = create_supercell(dap4_crystal, (2, 1, 1))

        # from_crystal should work on the supercell
        info = DisorderInfo.from_crystal(sc)
        assert info.pbc == dap4_crystal.pbc
        assert len(info.sym_op_indices) == sum(len(m) for m in sc.molecules)
