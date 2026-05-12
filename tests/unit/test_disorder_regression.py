"""
Regression tests for the disorder-resolution pipeline.

Each CIF in `tests/data/cif/` that the team has manually validated gets a
parametrised entry below.  Two checks are enforced per case:

1. The total atom count after resolution matches the expected
   ground-truth value (`expected_atoms`).
2. The molecular graphs contain no graph-level defects beyond the
   case's `expected_defects` count (over-coordinated atoms + orphan
   atoms of non-`ISOLATED_OK` elements); usually 0.

Cases that the *current* implementation cannot satisfy are marked as
`xfail` with a short reason.  When a fix lands, simply remove the
`xfail` marker and verify the suite turns green again.

The intent is to make this file the single source of truth for "how the
disorder solver is supposed to behave on real-world CIFs", and to keep
the print-only `scripts/regression_quick.py` from being the only
guardrail.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import pytest

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)


CIF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cif")
)


MAX_COORD = {
    "H": 1,
    "C": 4,
    "N": 4,
    "O": 3,
    "S": 6,
    "Cl": 4,
    "Cd": 8,
    "P": 4,
    "Zn": 6,
}

# Elements that legitimately appear as isolated single-atom species
# (counter-ions / metal cations).  Zero-degree atoms of these elements
# are *not* counted as defects.
ISOLATED_OK = frozenset({
    # halide counter-ions
    "F", "Cl", "Br", "I",
    # alkali / alkaline-earth cations
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
    # divalent / monovalent transition-metal cations that often
    # crystallise as bare ions when the CIF refines no ligand
    "Cd", "Zn", "Ag", "Cu", "Co", "Ni", "Fe", "Mn", "Hg",
})


@dataclass(frozen=True)
class CifCase:
    """A single regression target.

    Attributes
    ----------
    expected_defects:
        Expected number of "graph-level defects" in the resolved
        structure: atoms that are over-coordinated (degree above its
        max-coord limit) plus orphan atoms (degree 0) of elements that
        are *not* in `ISOLATED_OK`.  Defaults to 0.  CIFs whose ground
        truth contains a non-zero number of legitimate orphans (most
        commonly solvent waters with no refined hydrogens) declare that
        count explicitly; see ZIF-8 below.
    """

    name: str
    cif: str
    expected_atoms: int | None
    expected_defects: int = 0
    xfail_reason: str | None = None
    timeout: int = 120


CASES: list[CifCase] = [
    # --- locked baseline (currently green on main) ---
    CifCase("NatComm-1", "NatComm-1.cif", 60),
    CifCase("ammonium-sp-explicit-hm4", "ammonium_sp_explicit_hm4.cif", 176),
    CifCase("DAP-4", "DAP-4.cif", 336),
    CifCase("DAC-4", "DAC-4.cif", 39),
    CifCase("anhydrousCaffeine", "anhydrousCaffeine_CGD_2007_7_1406.cif", 480),
    CifCase("anhydrousCaffeine2", "anhydrousCaffeine2_CGD_2007_7_1406.cif", 144),
    CifCase("ZIF-4", "ZIF-4.cif", 368),
    CifCase("TILPEN", "TILPEN.cif", 84),
    CifCase("1-HTP", "1-HTP.cif", 102),
    CifCase("MAF-4", "MAF-4.cif", 369),
    CifCase("DAN-2", "DAN-2.cif", 35),
    CifCase("PAP-H4", "PAP-H4.cif", 656),
    # 4-fluorophenethylammonium bromide.  The cation is encoded as
    # PART -1 on a mirror plane; SP-completion keeps both half-images,
    # snaps overlapping atoms, and removes/refits ghost hydrogens.
    CifCase("phenethylammonium-sp-p21m", "phenethylammonium_sp_p21m.cif", 88),
    CifCase("DAI-X1", "DAI-X1.cif", 112),
    # ZIF-8: 40 orphan O atoms are solvent waters whose H atoms are not
    # refined in the CIF (labels O1S/O2S/O3S, where -S is SHELXL's
    # solvent suffix).  This is a CIF-side limitation, not a solver bug.
    CifCase("ZIF-8", "ZIF-8.cif", 316, expected_defects=40),
    CifCase("PAP-M5", "PAP-M5.cif", 296),
    CifCase("DAP-O4", "DAP-O4.cif", 344),
    # PAP-4: highly-disordered NH4+ at a special position (24 orientations).
    # Fixed by the chemistry-aware motif merge that ignores soft (geometric /
    # implicit_sp / valence_geometry) conflicts when reconstructing isolated
    # X(H)_n centres.  Full PAP-4 solve takes ~60 s on this machine.
    CifCase("PAP-4", "PAP-4.cif", 304, timeout=180),
    # DAI-4: two chemically equivalent NH4+ sites encoded with different CIF
    # styles (N1 implicit SHELX riding-H vs N4 explicit disorder_assembly
    # tags).  Regression guard for the "motif merge picks same-asym_id H
    # twice" bug where N1 sites were resolved as NH3 instead of NH4+.
    CifCase("DAI-4", "DAI-4.cif", 336),
    # DAP-7: hydrazinium/diaminopropane salt with 2 N per cation.  Checks
    # that the motif merge still handles multi-N cations correctly.  The
    # H1C protons sit on a 0.5-occupancy disorder pair (one proton per
    # hydrazinium); both N1-N1 cations must survive with a single H1C
    # each, giving 6×ClO4 + 2×C6H14N2 + 2×H5N2 = 88 atoms.
    CifCase("DAP-7", "DAP-7.cif", 88),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str):
    """Return (crystal, n_atoms, defects, formulas).

    `defects` counts both over-coordinated atoms (degree > MAX_COORD) and
    orphan atoms (degree 0 for elements that don't normally exist as
    isolated species).
    """
    [crystal] = generate_ordered_replicas_from_disordered_sites(
        path, generate_count=1, method="optimal"
    )
    n_atoms = crystal.get_total_nodes()

    defects = 0
    formulas: list[str] = []
    for mol in crystal.molecules:
        symbols = mol.get_chemical_symbols()
        species = Counter(symbols)
        formulas.append("".join(f"{e}{species[e]}" for e in sorted(species)))
        graph = mol.graph
        for node in graph.nodes:
            elem = symbols[node] if node < len(symbols) else "?"
            limit = MAX_COORD.get(elem, 8)
            degree = graph.degree(node)
            if degree > limit:
                defects += 1
            elif degree == 0 and elem not in ISOLATED_OK:
                defects += 1
    return crystal, n_atoms, defects, Counter(formulas)


def _case_id(case: CifCase) -> str:
    return case.name


# ---------------------------------------------------------------------------
# Parametrised regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cif_data_dir() -> str:
    assert os.path.isdir(CIF_DATA_DIR), f"CIF data directory not found: {CIF_DATA_DIR}"
    return CIF_DATA_DIR


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_disorder_regression(case: CifCase, cif_data_dir: str):
    """Atom-count + over-coordination regression for a single CIF."""
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)

    cif_path = os.path.join(cif_data_dir, case.cif)
    assert os.path.exists(cif_path), f"missing CIF fixture: {case.cif}"

    _, n_atoms, defects, formulas = _resolve(cif_path)

    if case.expected_atoms is not None:
        assert n_atoms == case.expected_atoms, (
            f"{case.name}: expected {case.expected_atoms} atoms, "
            f"got {n_atoms} (formulas={dict(formulas)})"
        )
    assert defects == case.expected_defects, (
        f"{case.name}: expected {case.expected_defects} defective "
        f"atoms (over-coord or orphan), got {defects}"
    )


# ---------------------------------------------------------------------------
# Targeted formula assertions (catch silent topology drift)
# ---------------------------------------------------------------------------


def test_natcomm1_topology(cif_data_dir: str):
    """NatComm-1 should resolve into 2 organic cations + 1 Cd-thiocyanate cluster.

    The earlier expectation of "6 separate SCN units" was wrong: SCN groups
    bridge Cd centres, so the Cd-thiocyanate framework is one connected
    cluster (Cd2(SCN)6 -> formula C6Cd2N6S6 in this representation).
    """
    cif = os.path.join(cif_data_dir, "NatComm-1.cif")
    assert os.path.exists(cif), "NatComm-1.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 60
    assert formulas.get("C5H14N1", 0) == 2, f"organic cation count wrong: {formulas}"
    assert formulas.get("C6Cd2N6S6", 0) == 1, (
        f"expected one bridged Cd2(SCN)6 cluster: {formulas}"
    )


def test_ammonium_sp_explicit_hm4_topology(cif_data_dir: str):
    cif = os.path.join(cif_data_dir, "ammonium_sp_explicit_hm4.cif")
    assert os.path.exists(cif), "ammonium_sp_explicit_hm4.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 176
    assert formulas.get("Cl1O4", 0) == 12
    assert formulas.get("H4N1", 0) == 4
    assert formulas.get("C6H16N2", 0) == 4


def test_dap4_topology(cif_data_dir: str):
    cif = os.path.join(cif_data_dir, "DAP-4.cif")
    assert os.path.exists(cif), "DAP-4.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 336
    assert formulas.get("C6H14N2", 0) == 8
    assert formulas.get("Cl1O4", 0) == 24
    # Each ammonium must have exactly 4 H (not 3 as in the pre-fix regression).
    assert formulas.get("H4N1", 0) == 8, (
        f"Expected 8 × NH4+ (H4N1), got: {dict(formulas)}"
    )


def test_dai4_topology(cif_data_dir: str):
    """DAI-4: both implicit-tagged (N1) and explicit-tagged (N4) NH4+ sites
    must resolve to H4N1.

    This is the primary regression guard for the motif-merge bug where
    the greedy loop picked 3× H1D (same asym_id) for N1, producing NH3
    instead of NH4+.
    """
    cif = os.path.join(cif_data_dir, "DAI-4.cif")
    assert os.path.exists(cif), "DAI-4.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 336, f"Expected 336 atoms, got {n_atoms}"
    assert formulas.get("C6H14N2", 0) == 8, (
        f"Expected 8 × C6H14N2 cation, got: {dict(formulas)}"
    )
    assert formulas.get("I1O4", 0) == 24, (
        f"Expected 24 × IO4- anion, got: {dict(formulas)}"
    )
    # This is the key assertion: all 8 ammonium sites (4 N1 + 4 N4) must
    # have 4 H each.  Before the fix, N1 sites were resolved as H3N1 (NH3).
    assert formulas.get("H4N1", 0) == 8, (
        f"Expected 8 × NH4+ (H4N1), got: {dict(formulas)} — "
        "likely the same-asym_id greedy regression"
    )


def test_dap7_topology(cif_data_dir: str):
    """DAP-7: diaminopropane/hydrazinium salt.

    Checks the chemically correct stoichiometry: 6 perchlorate anions, 2
    diaminopropanediium cations (C6H14N2), and 2 mono-protonated
    hydrazinium cations (H5N2 each, with one of the two 0.5-occupancy
    H1C protons surviving per cation).
    """
    cif = os.path.join(cif_data_dir, "DAP-7.cif")
    assert os.path.exists(cif), "DAP-7.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 88, f"Expected 88 atoms, got {n_atoms}"
    assert formulas.get("Cl1O4", 0) == 6, (
        f"Expected 6 × ClO4- anion, got: {dict(formulas)}"
    )
    assert formulas.get("C6H14N2", 0) == 2, (
        f"Expected 2 × C6H14N2 dication, got: {dict(formulas)}"
    )
    # Hydrazinium cation: monoprotonated N2H5+ (5 H across the 2 N atoms).
    assert formulas.get("H5N2", 0) == 2, (
        f"Expected 2 × H5N2+ hydrazinium cation, got: {dict(formulas)}"
    )


def test_dap7_no_unphysical_proton_split(cif_data_dir: str):
    """DAP-7: every enumerated/random replica must give 2 × H5N2+.

    The two hydrazinium cations are independent decision components, each
    with two mutually-exclusive H1C alternatives.  A regression in the
    implicit-SP conflict edges or the rigid-body splitter could merge
    both cations into a single component (producing N2H6 + N2H4) or
    drop the conflict edge altogether (producing N2H6 × 2).  Walk every
    enumerated alternative and every random sample and refuse anything
    that is not exactly H5N2 × 2.
    """
    cif = os.path.join(cif_data_dir, "DAP-7.cif")
    assert os.path.exists(cif), "DAP-7.cif fixture not found"

    forbidden = {"H4N2", "N2H4", "H6N2", "N2H6"}
    expected_hydrazinium = 2

    for method, count in (("enumerate", 8), ("random", 16)):
        crystals = generate_ordered_replicas_from_disordered_sites(
            cif,
            generate_count=count,
            method=method,
            random_seed=0,
        )
        assert crystals, f"{method}: no replicas returned"
        for replica_idx, crystal in enumerate(crystals):
            moiety = Counter()
            for mol in crystal.molecules:
                ce = Counter(mol.get_chemical_symbols())
                key = "".join(
                    f"{el}{ce[el]}" if ce[el] != 1 else el
                    for el in sorted(ce)
                )
                moiety[key] += 1
            unphysical = forbidden & set(moiety)
            assert not unphysical, (
                f"{method} replica {replica_idx}: forbidden hydrazinium "
                f"moiety {unphysical} present in {dict(moiety)}"
            )
            assert moiety.get("H5N2", 0) == expected_hydrazinium, (
                f"{method} replica {replica_idx}: expected "
                f"{expected_hydrazinium}× H5N2+, got {dict(moiety)}"
            )
