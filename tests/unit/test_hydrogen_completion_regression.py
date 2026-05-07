"""
Regression tests for add_hydrogens against real CIF structures from examples/.

Each test:
1. Reads a real CIF from the examples/ directory.
2. Calls add_hydrogens with target_elements appropriate for that structure.
3. Asserts that every molecule in the hydrogenated crystal has the chemical
   formula matching the CIF's _chemical_formula_moiety field (Hill notation).

These tests guard against regressions in:
  - Aromatic ring detection (5- and 6-membered rings, fused bicyclics).
  - sp2 / sp3 discrimination for C, N, O.
  - Ring detection robustness (numpy int64 graph node IDs).

Issue reference: https://github.com/SchrodingersCattt/MolCrysKit/issues/18
"""

import os
import pytest
import warnings

# Locate examples/ relative to this file regardless of working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXAMPLES = os.path.join(_REPO_ROOT, "examples")


def _cif(name):
    return os.path.join(_EXAMPLES, name)


def _hydrogenate(cif_path, target_elements, **kwargs):
    """Read CIF, add hydrogens, return list of (formula, n_atoms) per molecule."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.operations import add_hydrogens

        crystal = read_mol_crystal(cif_path)
        hcrystal = add_hydrogens(crystal, target_elements=target_elements, **kwargs)

    return [(m.get_chemical_formula(), len(m)) for m in hcrystal.molecules]


# ---------------------------------------------------------------------------
# Issue #18 regression: nicergoline (GICVUJ03)
# CIF has no H atoms.  Expected: 2 × C24H26BrN3O3 = 57 atoms each, 114 total.
# The molecule contains a fused indole ring system:
#   - 6-membered benzene ring (aromatic CH × 4)
#   - 5-membered pyrrole ring (aromatic N-H × 1, aromatic CH × 2)
#   - bridgehead sp3 carbon
#   - pyridine-like nitrogen (from the bromonicotinate moiety)
# This was the original failing case from the bug report.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mol_idx", [0, 1])
def test_gicvuj03_formula_per_molecule(mol_idx):
    """Each of the two nicergoline molecules must have formula C24H26BrN3O3."""
    path = _cif("GICVUJ03.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=["C", "N", "O"])
    formula, n_atoms = results[mol_idx]
    assert formula == "C24H26BrN3O3", (
        f"Molecule {mol_idx}: expected C24H26BrN3O3, got {formula}"
    )
    assert n_atoms == 57, (
        f"Molecule {mol_idx}: expected 57 atoms (24C+26H+1Br+3N+3O), got {n_atoms}"
    )


def test_gicvuj03_total_nodes():
    """Total atom count across the unit cell must be 114 (2 × 57)."""
    path = _cif("GICVUJ03.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.operations import add_hydrogens

        crystal = read_mol_crystal(path)
        hcrystal = add_hydrogens(crystal, target_elements=["C", "N", "O"])

    total = hcrystal.get_total_nodes()
    assert total == 114, (
        f"Expected 114 total atoms (2 × C24H26BrN3O3), got {total}"
    )


# ---------------------------------------------------------------------------
# ISATIN (C8H5NO2): oxindole / indole-2,3-dione
# Contains fused 5-ring (non-aromatic diketone) + 6-ring (aromatic benzene).
# The NH of the 5-membered amide ring must receive exactly 1 H.
# ---------------------------------------------------------------------------

def test_isatin_formula():
    """Isatin: each molecule must be C8H5NO2 (5 H = 4 aromatic CH + 1 NH)."""
    path = _cif("ISATIN.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=["C", "N", "O"])
    for i, (formula, _) in enumerate(results):
        assert formula == "C8H5NO2", (
            f"Molecule {i}: expected C8H5NO2, got {formula}"
        )


# ---------------------------------------------------------------------------
# Acetaminophen / paracetamol (HXACAN, C8H9NO2)
# CIF already contains all H atoms (full refinement).
# add_hydrogens should not add any spurious extra H.
# ---------------------------------------------------------------------------

def test_acetaminophen_no_extra_h():
    """Acetaminophen CIF already has H; add_hydrogens must leave count unchanged."""
    path = _cif("Acetaminophen_HXACAN.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=["C", "N", "O"])
    for i, (formula, _) in enumerate(results):
        assert formula == "C8H9NO2", (
            f"Molecule {i}: expected C8H9NO2, got {formula}"
        )


# ---------------------------------------------------------------------------
# DACMOR (C21H23NO5): morphine alkaloid
# CIF has no H.  Contains aromatic ring + sp3 and sp2 carbons.
# ---------------------------------------------------------------------------

def test_dacmor_formula():
    """DACMOR: each molecule must be C21H23NO5."""
    path = _cif("DACMOR.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=["C", "N", "O"])
    for i, (formula, _) in enumerate(results):
        assert formula == "C21H23NO5", (
            f"Molecule {i}: expected C21H23NO5, got {formula}"
        )


# ---------------------------------------------------------------------------
# PETN (C5H8N4O12): pentaerythritol tetranitrate — all-sp3 carbon backbone
# CIF has no H.  Tests that the fix does not regress sp3 detection.
# All 8 H must come from the 4 sp3 CH2 groups (2 H each).
# ---------------------------------------------------------------------------

def test_petn_formula():
    """PETN (all-sp3): each molecule must be C5H8N4O12 (H on C only)."""
    path = _cif("PETN_PERYTN10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=["C"])
    for i, (formula, _) in enumerate(results):
        assert formula == "C5H8N4O12", (
            f"Molecule {i}: expected C5H8N4O12, got {formula}"
        )


# ---------------------------------------------------------------------------
# BRCRIM10 / bromocriptine methylsulfonate isopropanol solvate
# Issue #18 reopened case.  PR1 covers the neutral/anion terminal-O failures:
#   - methanesulfonate should be CH3O3S, not CH4O3S
#   - isopropanol should be C3H8O, not C3H7O
# The protonated bromocriptine cation is handled in the follow-up
# charge-aware PR.
# ---------------------------------------------------------------------------

def test_brcrim10_mesylate_formula():
    """BRCRIM10: methanesulfonate fragments must be CH3O3S."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None)
    mesylates = [(formula, n_atoms) for formula, n_atoms in results if formula == "CH3O3S"]
    assert len(mesylates) == 2, (
        f"Expected 2 CH3O3S mesylate fragments, got {results}"
    )
    assert all(n_atoms == 8 for _, n_atoms in mesylates)


def test_brcrim10_isopropanol_formula():
    """BRCRIM10: isopropanol fragments must be C3H8O."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None)
    isopropanols = [
        (formula, n_atoms) for formula, n_atoms in results if formula == "C3H8O"
    ]
    assert len(isopropanols) == 2, (
        f"Expected 2 C3H8O isopropanol fragments, got {results}"
    )
    assert all(n_atoms == 12 for _, n_atoms in isopropanols)


def test_brcrim10_cation_formula():
    """BRCRIM10: bromocriptine cations must be C32H41BrN5O5."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None)
    cations = [
        (formula, n_atoms) for formula, n_atoms in results
        if formula == "C32H41BrN5O5"
    ]
    assert len(cations) == 2, (
        f"Expected 2 C32H41BrN5O5 cation fragments, got {results}"
    )
    assert all(n_atoms == 84 for _, n_atoms in cations)


def test_brcrim10_total_nodes():
    """BRCRIM10: total atom count must match all corrected fragments."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from molcrys_kit.io import read_mol_crystal
        from molcrys_kit.operations import add_hydrogens

        crystal = read_mol_crystal(path)
        hcrystal = add_hydrogens(crystal)

    assert hcrystal.get_total_nodes() == 208


def test_brcrim10_formula_moiety_path():
    """BRCRIM10 should still converge to the moiety formulas via the moiety path."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    from molcrys_kit.io import read_mol_crystal
    from molcrys_kit.operations import add_hydrogens

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal = read_mol_crystal(path)
        hcrystal = add_hydrogens(crystal)

    results = [(m.get_chemical_formula(), len(m)) for m in hcrystal.molecules]
    assert sum(1 for formula, _ in results if formula == "C32H41BrN5O5") == 2
    assert sum(1 for formula, _ in results if formula == "CH3O3S") == 2
    assert sum(1 for formula, _ in results if formula == "C3H8O") == 2


def test_brcrim10_use_formula_moiety_false_keeps_heuristic_path():
    """Opting out of formula moiety keeps the heuristic-only BRCRIM10 result."""
    path = _cif("BRCRIM10.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None, use_formula_moiety=False)

    assert sum(1 for formula, _ in results if formula == "C32H41BrN5O5") == 2
    assert sum(1 for formula, _ in results if formula == "CH3O3S") == 2
    assert sum(1 for formula, _ in results if formula == "C3H8O") == 2


def test_map_methylammonium_perchlorate_formula():
    """MAP: moiety should enforce CH6N+ and ClO4- without perchlorate O-H."""
    path = _cif("MAP.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None)

    assert ("CH6N", 8) in results
    assert ("ClO4", 5) in results


def test_suxrua_ammonium_perchlorate_formula():
    """SUXRUA has unknown moiety; heuristics should infer NH4+ and ClO4-."""
    path = _cif("SUXRUA.cif")
    if not os.path.exists(path):
        pytest.skip(f"CIF not found: {path}")

    results = _hydrogenate(path, target_elements=None)

    assert ("H4N", 5) in results
    assert ("ClO4", 5) in results


def test_isolated_water_coord0_default():
    """A single isolated O without moiety defaults to neutral water."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ase import Atoms
        from molcrys_kit.operations import add_hydrogens
        from molcrys_kit.structures.crystal import MolecularCrystal

        crystal = MolecularCrystal(
            lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            molecules=[Atoms(symbols=["O"], positions=[[5.0, 5.0, 5.0]])],
            pbc=(True, True, True),
        )
        hcrystal = add_hydrogens(crystal, target_elements=None)

    assert len(hcrystal.molecules) == 1
    assert hcrystal.molecules[0].get_chemical_formula() == "H2O"
    assert len(hcrystal.molecules[0]) == 3
