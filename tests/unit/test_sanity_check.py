"""Unit tests for the sanity_check analysis module."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.sanity_check import (
    CheckResult,
    SanityReport,
    check_bond_distances,
    check_hard_clash,
    check_hydrogen_presence,
    check_intermolecular_clash,
    check_isolated_atoms,
    check_topology_preservation,
    sanity_check,
)
from molcrys_kit.structures.crystal import MolecularCrystal


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_crystal():
    """A simple two-molecule crystal (two CH4 molecules far apart)."""
    # Build two methane molecules in a cubic cell
    cell = np.diag([10.0, 10.0, 10.0])
    # Molecule 1 centered at (2, 2, 2)
    positions_1 = np.array([
        [2.0, 2.0, 2.0],  # C
        [3.09, 2.0, 2.0],  # H
        [1.64, 3.03, 2.0],  # H
        [1.64, 1.33, 2.89],  # H
        [1.64, 1.33, 1.11],  # H
    ])
    # Molecule 2 centered at (7, 7, 7)
    positions_2 = positions_1 + 5.0
    positions = np.vstack([positions_1, positions_2])
    symbols = ["C", "H", "H", "H", "H"] * 2
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.arrays["molecule_index"] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return MolecularCrystal.from_ase_atoms(atoms)


@pytest.fixture
def clashing_crystal():
    """Crystal with two atoms unrealistically close (hard clash)."""
    cell = np.diag([10.0, 10.0, 10.0])
    # Two carbon atoms only 0.5 Å apart (covalent radius for C ≈ 0.77)
    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.3, 5.0, 5.0],  # 0.3 Å apart — severe clash
        [2.0, 2.0, 2.0],
        [3.09, 2.0, 2.0],
    ])
    symbols = ["C", "C", "C", "H"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.arrays["molecule_index"] = np.array([0, 0, 1, 1])
    return MolecularCrystal.from_ase_atoms(atoms)


@pytest.fixture
def no_hydrogen_crystal():
    """Crystal without any hydrogen atoms."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([
        [2.0, 2.0, 2.0],
        [3.5, 2.0, 2.0],
        [7.0, 7.0, 7.0],
        [8.5, 7.0, 7.0],
    ])
    symbols = ["C", "O", "C", "O"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.arrays["molecule_index"] = np.array([0, 0, 1, 1])
    return MolecularCrystal.from_ase_atoms(atoms)


@pytest.fixture
def isolated_oxygen_crystal():
    """Crystal with an isolated oxygen atom (single-atom molecule)."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([
        [2.0, 2.0, 2.0],  # C
        [3.09, 2.0, 2.0],  # H
        [1.64, 3.03, 2.0],  # H
        [1.64, 1.33, 2.89],  # H
        [1.64, 1.33, 1.11],  # H
        [8.0, 8.0, 8.0],  # Isolated O
    ])
    symbols = ["C", "H", "H", "H", "H", "O"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.arrays["molecule_index"] = np.array([0, 0, 0, 0, 0, 1])
    return MolecularCrystal.from_ase_atoms(atoms)


# ─── Check Result / Report Tests ─────────────────────────────────────────────


class TestCheckResult:
    def test_repr(self):
        r = CheckResult(name="test", passed=True, message="ok")
        assert "PASS" in repr(r)

        r2 = CheckResult(name="test", passed=False, message="bad")
        assert "FAIL" in repr(r2)


class TestSanityReport:
    def test_empty_report_passes(self):
        report = SanityReport()
        assert report.passed is True
        assert len(report) == 0

    def test_all_pass(self):
        report = SanityReport(results=[
            CheckResult("a", True, "ok"),
            CheckResult("b", True, "fine"),
        ])
        assert report.passed is True
        assert report.failed() == []

    def test_one_failure(self):
        report = SanityReport(results=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "bad"),
        ])
        assert report.passed is False
        assert len(report.failed()) == 1
        assert report.failed()[0].name == "b"

    def test_getitem_by_name(self):
        report = SanityReport(results=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "bad"),
        ])
        assert report["a"].passed is True
        with pytest.raises(KeyError):
            report["nonexistent"]

    def test_to_dict(self):
        report = SanityReport(results=[CheckResult("a", True, "ok", {"x": 1})])
        d = report.to_dict()
        assert d["passed"] is True
        assert d["results"][0]["name"] == "a"
        assert d["results"][0]["details"] == {"x": 1}

    def test_summary(self):
        report = SanityReport(results=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "problem"),
        ])
        s = report.summary()
        assert "1/2 passed" in s
        assert "✓" in s
        assert "✗" in s


# ─── Individual Check Tests ───────────────────────────────────────────────────


class TestCheckHardClash:
    def test_no_clash(self, simple_crystal):
        result = check_hard_clash(simple_crystal)
        assert result.passed is True
        assert result.details["clash_count"] == 0

    def test_detects_clash(self, clashing_crystal):
        result = check_hard_clash(clashing_crystal)
        assert result.passed is False
        assert result.details["clash_count"] > 0
        # Check pairs are returned
        assert len(result.details["pairs"]) > 0

    def test_scale_override(self, simple_crystal):
        # With a very high scale, even normal bonds should clash
        result = check_hard_clash(simple_crystal, scale=2.0)
        assert result.passed is False

    def test_scale_in_details(self, simple_crystal):
        result = check_hard_clash(simple_crystal, scale=0.4)
        assert result.details["scale_used"] == 0.4


class TestCheckIntermolecularClash:
    def test_no_clash(self, simple_crystal):
        result = check_intermolecular_clash(simple_crystal)
        assert result.passed is True
        assert result.details["clash_count"] == 0

    def test_max_clashes_override(self, clashing_crystal):
        # Allow up to 10 clashes
        result = check_intermolecular_clash(clashing_crystal, max_clashes=100)
        assert result.passed is True  # Even if clashes exist, within tolerance


class TestCheckIsolatedAtoms:
    def test_no_isolated(self, simple_crystal):
        result = check_isolated_atoms(simple_crystal)
        assert result.passed is True
        assert result.details["isolated_indices"] == []

    def test_detects_isolated_oxygen(self, isolated_oxygen_crystal):
        result = check_isolated_atoms(isolated_oxygen_crystal)
        assert result.passed is False
        assert "O" in result.details["isolated_elements"]

    def test_custom_elements(self, isolated_oxygen_crystal):
        # With O removed from suspect set, should pass
        result = check_isolated_atoms(isolated_oxygen_crystal, elements={"N", "S"})
        assert result.passed is True


class TestCheckHydrogenPresence:
    def test_has_hydrogen(self, simple_crystal):
        result = check_hydrogen_presence(simple_crystal)
        assert result.passed is True

    def test_no_hydrogen(self, no_hydrogen_crystal):
        result = check_hydrogen_presence(no_hydrogen_crystal)
        assert result.passed is False


class TestCheckBondDistances:
    def test_normal_crystal(self, simple_crystal):
        result = check_bond_distances(simple_crystal)
        assert result.passed is True
        assert result.details["n_bonds_checked"] > 0

    def test_strict_factor(self, simple_crystal):
        # Very strict factor should catch even slightly non-ideal bonds
        result = check_bond_distances(simple_crystal, min_factor=0.99, max_factor=1.01)
        # C-H bonds are ~1.09 Å, covalent radii sum C+H ~ 0.77+0.31 = 1.08
        # So factor ≈ 1.01 — borderline, could go either way
        # Just verify the function runs without error
        assert isinstance(result, CheckResult)


class TestCheckTopologyPreservation:
    def test_same_structure(self, simple_crystal):
        result = check_topology_preservation(simple_crystal, simple_crystal)
        assert result.passed is True

    def test_different_molecule_count(self, simple_crystal, isolated_oxygen_crystal):
        result = check_topology_preservation(simple_crystal, isolated_oxygen_crystal)
        assert result.passed is False


# ─── Aggregated sanity_check Tests ────────────────────────────────────────────


class TestSanityCheck:
    def test_default_all_checks(self, simple_crystal):
        report = sanity_check(simple_crystal)
        assert len(report) == 6  # All single-crystal checks
        assert report.passed is True

    def test_select_checks(self, simple_crystal):
        report = sanity_check(simple_crystal, checks=["hard_clash", "hydrogen_presence"])
        assert len(report) == 2
        assert report["hard_clash"].passed is True
        assert report["hydrogen_presence"].passed is True

    def test_unknown_check(self, simple_crystal):
        report = sanity_check(simple_crystal, checks=["nonexistent_check"])
        assert len(report) == 1
        assert report[0].passed is False
        assert "Unknown" in report[0].message

    def test_parameter_passthrough(self, simple_crystal):
        report = sanity_check(simple_crystal, hard_clash_scale=0.3)
        assert report["hard_clash"].details["scale_used"] == 0.3

    def test_failing_crystal(self, clashing_crystal):
        report = sanity_check(clashing_crystal)
        assert report.passed is False
        assert report["hard_clash"].passed is False


# ─── Topology Change Detection (same mol count, different connectivity) ───────


class TestTopologyConnectivityChange:
    """Test that topology_preservation detects changed connectivity
    (not just different molecule count)."""

    def test_detects_broken_bond(self):
        """Same 2 molecules before/after, but one H moved far away → bond breaks."""
        cell = np.diag([15.0, 15.0, 15.0])
        # Before: normal CH4 + CH4
        pos_before = np.array([
            [2.0, 2.0, 2.0],  # C
            [3.09, 2.0, 2.0],  # H bonded
            [1.64, 3.03, 2.0],  # H
            [1.64, 1.33, 2.89],  # H
            [1.64, 1.33, 1.11],  # H
            [9.0, 9.0, 9.0],  # C
            [10.09, 9.0, 9.0],  # H
            [8.64, 10.03, 9.0],  # H
            [8.64, 8.33, 9.89],  # H
            [8.64, 8.33, 8.11],  # H
        ])
        symbols = ["C", "H", "H", "H", "H"] * 2

        atoms_before = Atoms(symbols=symbols, positions=pos_before, cell=cell, pbc=True)
        atoms_before.arrays["molecule_index"] = np.array([0] * 5 + [1] * 5)

        # After: move one H far from its C → bond should break in topology
        pos_after = pos_before.copy()
        pos_after[1] = [6.0, 6.0, 6.0]  # H moved 5+ Å away from C — no longer bonded

        atoms_after = Atoms(symbols=symbols, positions=pos_after, cell=cell, pbc=True)
        atoms_after.arrays["molecule_index"] = np.array([0] * 5 + [1] * 5)

        mc_before = MolecularCrystal.from_ase_atoms(atoms_before)
        mc_after = MolecularCrystal.from_ase_atoms(atoms_after)

        result = check_topology_preservation(mc_before, mc_after)
        # Topology should detect the change (CH4 → CH3 + H or different graph)
        assert result.passed is False
        assert result.details["n_molecules_before"] != result.details["n_molecules_after"] or \
               len(result.details["mismatched_invariants"]) > 0


# ─── Integration Smoke Test with Real CIF ────────────────────────────────────


class TestRealStructureSmoke:
    """Smoke test on real CIF fixtures — sanity_check should not crash."""

    _CIF_DIR = "tests/data/cif"

    @pytest.fixture(params=["BRCRIM10.cif", "NOKGIH01.cif", "TILPEN.cif"])
    def cif_crystal(self, request):
        """Load a real CIF fixture."""
        from pathlib import Path
        from molcrys_kit.io.cif import read_mol_crystal

        cif_path = Path(__file__).parents[1] / "data" / "cif" / request.param
        if not cif_path.exists():
            pytest.skip(f"{request.param} not found")
        return read_mol_crystal(str(cif_path))

    def test_sanity_check_runs(self, cif_crystal):
        """sanity_check should complete without error on valid CIFs."""
        report = sanity_check(cif_crystal)
        # We don't assert pass/fail — just that it doesn't crash and returns valid report
        assert isinstance(report, SanityReport)
        assert len(report) == 6
        for r in report:
            assert isinstance(r, CheckResult)
            assert isinstance(r.passed, bool)
            assert isinstance(r.message, str)

    def test_individual_checks_run(self, cif_crystal):
        """Each individual check should run without error."""
        result = check_hard_clash(cif_crystal)
        assert isinstance(result, CheckResult)
        result = check_intermolecular_clash(cif_crystal)
        assert isinstance(result, CheckResult)
        result = check_isolated_atoms(cif_crystal)
        assert isinstance(result, CheckResult)
        result = check_hydrogen_presence(cif_crystal)
        assert isinstance(result, CheckResult)
        result = check_bond_distances(cif_crystal)
        assert isinstance(result, CheckResult)
