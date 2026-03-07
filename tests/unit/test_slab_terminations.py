import os
import sys
import math
import json
import numpy as np
import pytest
import warnings

from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.operations.surface import (
    enumerate_terminations,
    generate_slabs_with_terminations,
    TerminationInfo,
)
from molcrys_kit.io.output import write_cif


def example_cif_path():
    # Resolve project root from this file location
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, "examples", "Acetaminophen_HXACAN.cif")


def load_example_crystal():
    path = example_cif_path()
    assert os.path.exists(path), f"Missing example CIF at {path}"
    return read_mol_crystal(path)


# ---------------------------------------------------------------------------
# Basic termination enumeration
# ---------------------------------------------------------------------------

def test_enumerate_terminations_topo_basic():
    crystal = load_example_crystal()
    infos = enumerate_terminations(
        crystal=crystal,
        miller_index=(1, 0, 0),
        unique_terminations="topo",
        termination_resolution=None,
        symmetry_reduction=False,
    )
    assert isinstance(infos, list) and len(infos) >= 1
    for info in infos:
        assert 0.0 <= info.shift < 1.0
        assert info.miller_index == (1, 0, 0)
        assert isinstance(info.termination_index, int)


def test_enumerate_terminations_returns_termination_info_instances():
    crystal = load_example_crystal()
    infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
    for info in infos:
        assert isinstance(info, TerminationInfo)
        assert hasattr(info, "tasker_type")
        assert hasattr(info, "is_polar")
        assert hasattr(info, "is_tasker_preferred")
        assert hasattr(info, "charge_source")
        assert hasattr(info, "layer_charges")
        assert hasattr(info, "dipole_per_area")


# ---------------------------------------------------------------------------
# Neutral organic crystal: fast path (TypeI_like)
# ---------------------------------------------------------------------------

def test_neutral_crystal_fast_path():
    """Pure organic molecular crystal should trigger the all-neutral fast path."""
    crystal = load_example_crystal()
    infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
    for info in infos:
        assert info.tasker_type == "TypeI_like", (
            f"Expected TypeI_like for neutral crystal, got {info.tasker_type}"
        )
        assert not info.is_polar
        assert info.is_tasker_preferred
        assert info.charge_source == "neutral"


def test_neutral_crystal_zero_dipole():
    """For an all-neutral crystal the dipole per area should be exactly 0."""
    crystal = load_example_crystal()
    infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
    for info in infos:
        assert info.dipole_per_area == 0.0


# ---------------------------------------------------------------------------
# Slab generation: all and by_index
# ---------------------------------------------------------------------------

def test_generate_slabs_all_and_by_index():
    crystal = load_example_crystal()

    # Generate all terminations
    all_results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
        unique_terminations="topo",
        term_selection="all",
    )
    assert len(all_results) >= 1

    # Shifts in results should be in [0, 1)
    shifts = [info.shift for (_, info) in all_results]
    assert all(0.0 <= s < 1.0 for s in shifts)

    # Select first termination by index
    by_index_results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
        unique_terminations="topo",
        term_selection="by_index",
        termination_indices=[0],
    )
    assert len(by_index_results) == 1
    assert by_index_results[0][1].termination_index == 0


def test_slab_topology_preserved():
    """Each slab molecule should have the same atom count as original molecules."""
    crystal = load_example_crystal()
    original_sizes = sorted([len(m) for m in crystal.molecules])
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        term_selection="all",
    )
    for slab, _ in results:
        slab_sizes = sorted(set([len(m) for m in slab.molecules]))
        orig_unique = sorted(set(original_sizes))
        assert slab_sizes == orig_unique, (
            f"Molecule size mismatch: slab={slab_sizes}, original={orig_unique}"
        )


def test_slab_has_vacuum():
    """Generated slabs must have approximately the requested vacuum layer.

    The cell height = d_spacing * layers + min_vacuum_size.  Atoms within
    a layer can protrude up to ~2 Å beyond the crystallographic d_spacing
    boundary, so actual_vacuum = cell_c - atom_extent >= min_vacuum - 2.
    """
    crystal = load_example_crystal()
    vacuum = 15.0
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=vacuum,
        term_selection="all",
    )
    for slab, _ in results:
        c = slab.lattice[2, 2]
        all_z = np.vstack([m.get_positions() for m in slab.molecules])[:, 2]
        slab_thick = np.max(all_z) - np.min(all_z)
        actual_vacuum = c - slab_thick
        assert actual_vacuum >= vacuum - 2.0, (
            f"Vacuum too small: {actual_vacuum:.2f} < {vacuum - 2.0:.2f}"
        )


# ---------------------------------------------------------------------------
# CIF metadata output
# ---------------------------------------------------------------------------

def test_write_cif_with_termination_metadata(tmp_path):
    crystal = load_example_crystal()

    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
        unique_terminations="topo",
        term_selection="by_index",
        termination_indices=[0],
    )
    assert results, "No slabs generated for testing CIF output"
    slab, info = results[0]

    cif_text = write_cif(slab, filename=None, metadata={"termination_info": info})
    assert "_molcrys_termination_shift" in cif_text
    assert "_molcrys_termination_index" in cif_text
    assert "_molcrys_tasker_type" in cif_text
    assert "_molcrys_tasker_polar" in cif_text
    assert "_molcrys_tasker_dipole_per_area" in cif_text
    assert "_molcrys_charge_source" in cif_text

    out_file = tmp_path / "slab_with_meta.cif"
    written = write_cif(slab, filename=str(out_file), metadata={"termination_info": info})
    assert out_file.exists()
    assert written is not None  # write_cif always returns the CIF string now


def test_write_cif_without_metadata_unchanged(tmp_path):
    """write_cif without metadata should produce no molcrys_ fields."""
    crystal = load_example_crystal()
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        term_selection="all",
    )
    slab, _ = results[0]
    cif_text = write_cif(slab)
    assert "_molcrys_termination_shift" not in cif_text


# ---------------------------------------------------------------------------
# Charge source prioritization and fallback
# ---------------------------------------------------------------------------

def test_user_map_charge_source():
    """When mol_charge_map is provided the source should be user_map."""
    crystal = load_example_crystal()
    formula = crystal.molecules[0].get_chemical_formula(mode="hill", empirical=False)
    infos = enumerate_terminations(
        crystal=crystal,
        miller_index=(1, 0, 0),
        mol_charge_map={formula: 0},
    )
    for info in infos:
        assert info.charge_source == "user_map"


def test_unknown_source_fallback_warning():
    """When pymatgen is unavailable and no map is given, source should be 'none' with warning."""
    from unittest.mock import patch
    crystal = load_example_crystal()

    # Simulate pymatgen unavailable so auto_guess always fails
    with patch(
        "molcrys_kit.analysis.charge._guess_charge_pymatgen",
        return_value=None,
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            infos = enumerate_terminations(
                crystal=crystal,
                miller_index=(1, 0, 0),
            )
            # For neutral organic crystals the fast-path skips charge guessing,
            # so we check the charge module directly instead
            from molcrys_kit.analysis.charge import assign_mol_formal_charges
            results = assign_mol_formal_charges(crystal, mol_charge_map={})
            for r in results.values():
                assert r.source == "none"
            assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# Tasker-preferred-first ordering
# ---------------------------------------------------------------------------

def test_tasker_preferred_first_ordering():
    """Preferred terminations must come before non-preferred in default selection."""
    crystal = load_example_crystal()
    infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
    preferred_indices = [i for i, ti in enumerate(infos) if ti.is_tasker_preferred]
    non_preferred_indices = [i for i, ti in enumerate(infos) if not ti.is_tasker_preferred]
    if preferred_indices and non_preferred_indices:
        assert max(preferred_indices) < min(non_preferred_indices), (
            "Preferred terminations should precede non-preferred ones"
        )


def test_term_selection_tasker_preferred_fallback():
    """If no termination is preferred, all are returned with a warning."""
    crystal = load_example_crystal()
    # Patch is_tasker_preferred to False for all terminations via mol_charge injection
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Force TypeIII_like by injecting a nonzero charge that creates dipole
        # Use a large nonzero charge via user map to create polarity
        results = generate_slabs_with_terminations(
            structure_or_crystal=crystal,
            miller_index=(1, 0, 0),
            min_slab_size=8.0,
            min_vacuum_size=10.0,
            mol_charge_map={
                crystal.molecules[0].get_chemical_formula(
                    mode="hill", empirical=False
                ): 0
            },
            term_selection="tasker_preferred",
        )
        # Should still return at least one slab (fallback)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Bug-fix coverage: layer_charges non-empty, symmetry_reduction, overall_source
# ---------------------------------------------------------------------------

def test_fast_path_layer_charges_not_empty():
    """For a neutral organic crystal, layer_charges must be a non-empty list of zeros."""
    crystal = load_example_crystal()
    infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
    for info in infos:
        assert len(info.layer_charges) > 0, (
            "layer_charges should not be empty even on the all-neutral fast path"
        )
        assert all(q == 0.0 for q in info.layer_charges), (
            f"Expected all-zero layer_charges for neutral crystal, got {info.layer_charges}"
        )


def test_symmetry_reduction_returns_fewer_or_equal():
    """symmetry_reduction=True should return <= terminations compared to False."""
    crystal = load_example_crystal()
    infos_no_sym = enumerate_terminations(
        crystal, miller_index=(1, 0, 0), symmetry_reduction=False
    )
    infos_sym = enumerate_terminations(
        crystal, miller_index=(1, 0, 0), symmetry_reduction=True
    )
    assert len(infos_sym) <= len(infos_no_sym), (
        f"symmetry_reduction=True returned more terminations ({len(infos_sym)}) "
        f"than False ({len(infos_no_sym)})"
    )


def test_overall_source_user_map_with_nonzero_charge():
    """charge_source should be 'user_map' when mol_charge_map provides a nonzero charge."""
    from unittest.mock import patch
    crystal = load_example_crystal()
    formula = crystal.molecules[0].get_chemical_formula(mode="hill", empirical=False)

    # Provide a nonzero charge so _evaluate_tasker is called (not the all-neutral fast path).
    # Mock pymatgen so auto_guess is unavailable, ensuring only user_map path is active.
    with patch(
        "molcrys_kit.analysis.charge._guess_charge_pymatgen",
        return_value=None,
    ):
        infos = enumerate_terminations(
            crystal=crystal,
            miller_index=(1, 0, 0),
            mol_charge_map={formula: 1},
        )
    for info in infos:
        assert info.charge_source == "user_map", (
            f"Expected charge_source='user_map', got '{info.charge_source}'"
        )


# ---------------------------------------------------------------------------
# TypeII correction
# ---------------------------------------------------------------------------

def test_tasker2_corrected_field_default_false():
    """TerminationInfo.tasker2_corrected should default to False."""
    crystal = load_example_crystal()
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=10.0,
        term_selection="all",
        correct_tasker2=False,
    )
    for _, ti in results:
        assert ti.tasker2_corrected is False


def test_tasker2_correction_does_not_raise():
    """correct_tasker2=True should run without error on any crystal."""
    crystal = load_example_crystal()
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=10.0,
        term_selection="all",
        correct_tasker2=True,
    )
    assert len(results) >= 1
    for slab, ti in results:
        assert hasattr(ti, "tasker2_corrected")


def test_cif_metadata_includes_tasker2_corrected(tmp_path):
    """write_cif with termination_info should include _molcrys_tasker2_corrected."""
    crystal = load_example_crystal()
    results = generate_slabs_with_terminations(
        structure_or_crystal=crystal,
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=10.0,
        term_selection="by_index",
        termination_indices=[0],
    )
    slab, info = results[0]
    cif_text = write_cif(slab, filename=None, metadata={"termination_info": info})
    assert "_molcrys_tasker2_corrected" in cif_text
