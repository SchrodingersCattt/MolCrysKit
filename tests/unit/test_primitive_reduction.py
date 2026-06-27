"""Regression tests for primitive-cell reduction in slab generation."""
from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from molcrys_kit.operations.surface import (
    TopologicalSlabGenerator,
    _try_reduce_to_primitive,
    generate_topological_slab,
)


def _make_fcc_crystal():
    """Build a minimal F-centered cubic MolecularCrystal (NaCl-type).

    Returns the conventional cell with 8 atoms (4 Na + 4 Cl) in Fm-3m.
    The primitive cell should have 2 atoms (1 Na + 1 Cl).
    """
    from ase import Atoms
    from molcrys_kit.structures import MolecularCrystal

    a = 5.64  # NaCl lattice parameter, Angstrom
    # Conventional cell: 4 NaCl formula units
    positions = [
        # Na at face-centered positions
        [0.0, 0.0, 0.0],
        [a / 2, a / 2, 0.0],
        [a / 2, 0.0, a / 2],
        [0.0, a / 2, a / 2],
        # Cl at edge-centered positions
        [a / 2, 0.0, 0.0],
        [0.0, a / 2, 0.0],
        [0.0, 0.0, a / 2],
        [a / 2, a / 2, a / 2],
    ]
    symbols = ["Na"] * 4 + ["Cl"] * 4
    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=[a, a, a], pbc=True)
    return MolecularCrystal.from_ase(atoms)


class TestTryReduceToPrimitive:
    """Tests for _try_reduce_to_primitive()."""

    def test_reduces_fcc_conventional_cell(self):
        crystal = _make_fcc_crystal()
        n_before = crystal.get_total_nodes()
        assert n_before == 8

        reduced = _try_reduce_to_primitive(crystal)
        n_after = reduced.get_total_nodes()

        # F-centered: primitive is 1/4 volume → 8/4 = 2 atoms
        assert n_after == 2
        assert n_after < n_before

    def test_noop_for_primitive_cell(self):
        """Already-primitive structures are returned unchanged."""
        from ase import Atoms
        from molcrys_kit.structures import MolecularCrystal

        # Simple P1 cell with 2 atoms
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]],
                       cell=[5, 5, 5], pbc=True)
        crystal = MolecularCrystal.from_ase(atoms)
        n_before = crystal.get_total_nodes()

        reduced = _try_reduce_to_primitive(crystal)
        assert reduced.get_total_nodes() == n_before

    def test_warns_on_failed_conversion(self):
        """Emit a warning (not silent failure) when conversion fails."""
        from molcrys_kit.structures import MolecularCrystal

        crystal = _make_fcc_crystal()
        # Monkeypatch to_ase to raise
        with patch.object(type(crystal), "to_ase", side_effect=RuntimeError("boom")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _try_reduce_to_primitive(crystal)

        # Should return original, not crash
        assert result is crystal
        # Should have emitted a warning
        assert any("Primitive reduction skipped" in str(x.message) for x in w)

    def test_pymatgen_import_error_returns_original(self):
        """Graceful fallback when pymatgen is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pymatgen" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        crystal = _make_fcc_crystal()
        with patch("builtins.__import__", side_effect=mock_import):
            result = _try_reduce_to_primitive(crystal)
        assert result is crystal


class TestSlabPrimitiveReduction:
    """Integration tests: slab generation with primitive reduction."""

    def test_slab_node_count_reduced(self):
        crystal = _make_fcc_crystal()
        # Without reduction
        slab_full = generate_topological_slab(
            crystal, miller_indices=(1, 0, 0),
            layers=2, vacuum=5.0, reduce_to_primitive=False,
        )
        # With reduction (default)
        slab_prim = generate_topological_slab(
            crystal, miller_indices=(1, 0, 0),
            layers=2, vacuum=5.0, reduce_to_primitive=True,
        )
        assert slab_prim.get_total_nodes() < slab_full.get_total_nodes()

    def test_reduce_false_backward_compat(self):
        """reduce_to_primitive=False gives the old (larger) result."""
        crystal = _make_fcc_crystal()
        slab = generate_topological_slab(
            crystal, miller_indices=(1, 0, 0),
            layers=2, vacuum=5.0, reduce_to_primitive=False,
        )
        # Conventional cell: 8 atoms × 2 layers = 16
        assert slab.get_total_nodes() >= 16

    def test_generator_init_reduces_by_default(self):
        """TopologicalSlabGenerator.__init__ reduces by default."""
        crystal = _make_fcc_crystal()
        gen = TopologicalSlabGenerator(crystal)
        assert gen.crystal.get_total_nodes() < crystal.get_total_nodes()

    def test_generator_init_no_reduce(self):
        """TopologicalSlabGenerator.__init__(reduce_to_primitive=False)."""
        crystal = _make_fcc_crystal()
        gen = TopologicalSlabGenerator(crystal, reduce_to_primitive=False)
        assert gen.crystal.get_total_nodes() == crystal.get_total_nodes()
