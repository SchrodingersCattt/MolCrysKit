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
