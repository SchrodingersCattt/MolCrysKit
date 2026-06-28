"""
Regression tests for the disorder-resolution pipeline.

Each CIF in `tests/data/cif/` that the team has manually validated gets a
parametrised entry below.  Per case the suite enforces:

1. Total atom count after resolution equals ``expected_atoms``.
2. Molecular graphs contain no graph-level defects beyond
   ``expected_defects`` (over-coordinated atoms + orphans of elements
   that are not in ``ISOLATED_OK``); usually 0.
3. When ``expected_element_totals`` is set, the per-element atom count
   summed over all molecules matches the expected dict.  This is
   intentionally invariant under metal/ligand fragmentation choices
   (e.g. ``K1 N3 O9`` vs ``K1 + 3*N1O3``) so that a future split-metal
   refactor will not need to update the assertion.
4. When ``expected_nh4_count`` is set, the number of isolated H4N1
   fragments matches the expected count, and each NH4 fragment passes a
   tetrahedral-geometry sanity check (N-H bond lengths and H-N-H
   angles within physically meaningful ranges).
5. Closest interatomic contact across the whole crystal is at least
   ``min_interatomic_distance`` (default 0.65 Å); any closer pair would
   indicate that two disorder alternatives were kept simultaneously.

The cross-mode tests enforce the current three-mode contract:
``random`` and ``enumerate`` replica #0 must match optimal exactly;
every ``enumerate`` replica must remain optimal-equivalent; later
``random`` replicas are allowed to sample lower-occupancy populations
but must remain chemically valid (no close contacts, no broken NH4
motifs, and no severe atom-count collapse).

Cases that the *current* implementation cannot satisfy at all are
marked with ``xfail_reason`` on ``CifCase`` and become full xfails.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

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
    expected_element_totals:
        Per-element atom-count totals summed across all molecules.
        Compared with the optimal-mode result element-by-element.
        Intentionally fragmentation-invariant so a future "split bare
        metals into their own molecules" change (DAN-2: K1 N3 O9 -> K1
        + 3*N1 O3) does not require updating the expectation.
    expected_nh4_count:
        Number of isolated H4N1 (ammonium) fragments expected after
        resolution.  When non-zero, every such fragment is also passed
        through ``_assert_nh4_tetrahedral`` for geometry validation.
    min_interatomic_distance:
        Minimum allowed nearest-neighbour distance, in Angstroms,
        across the whole resolved crystal (with PBC).  A closer pair
        almost always means two disorder alternatives were resolved
        on top of each other.
    """

    name: str
    cif: str
    expected_atoms: int | None
    expected_defects: int = 0
    xfail_reason: str | None = None
    timeout: int = 120
    expected_element_totals: Mapping[str, int] | None = None
    expected_nh4_count: int = 0
    min_interatomic_distance: float = 0.65


CASES: list[CifCase] = [
    # --- locked baseline (currently green on main) ---
    CifCase(
        "NatComm-1", "NatComm-1.cif", 60,
        expected_element_totals={"C": 16, "Cd": 2, "H": 28, "N": 8, "S": 6},
    ),
    CifCase(
        "ammonium-sp-explicit-hm4", "ammonium_sp_explicit_hm4.cif", 176,
        expected_element_totals={"C": 24, "Cl": 12, "H": 80, "N": 12, "O": 48},
        expected_nh4_count=4,
    ),
    CifCase(
        "DAP-4", "DAP-4.cif", 336,
        expected_element_totals={"C": 48, "Cl": 24, "H": 144, "N": 24, "O": 96},
        expected_nh4_count=8,
    ),
    CifCase(
        "DAC-4", "DAC-4.cif", 39,
        expected_element_totals={"C": 6, "Cl": 3, "H": 18, "N": 3, "O": 9},
        expected_nh4_count=1,
    ),
    CifCase(
        "anhydrousCaffeine", "anhydrousCaffeine_CGD_2007_7_1406.cif", 480,
        expected_element_totals={"C": 160, "H": 200, "N": 80, "O": 40},
    ),
    CifCase(
        "anhydrousCaffeine2", "anhydrousCaffeine2_CGD_2007_7_1406.cif", 144,
        expected_element_totals={"C": 48, "H": 60, "N": 24, "O": 12},
    ),
    CifCase(
        "ZIF-4", "ZIF-4.cif", 368,
        expected_element_totals={"C": 120, "H": 152, "N": 72, "O": 8, "Zn": 16},
    ),
    CifCase(
        "TILPEN", "TILPEN.cif", 84,
        expected_element_totals={"C": 24, "Fe": 2, "H": 40, "N": 16, "O": 2},
    ),
    CifCase(
        "1-HTP", "1-HTP.cif", 102,
        expected_element_totals={"C": 30, "Fe": 2, "H": 48, "N": 16, "O": 6},
    ),
    # MAF-4: porous Zn-imidazolate framework whose CIF formula_sum
    # ``C8 H14 N4 O2 Zn`` * Z=12 omits the disordered water guests; the
    # solver retains 31 H2O, giving 14 extra H + 7 extra O over the
    # framework-only formula.
    CifCase(
        "MAF-4", "MAF-4.cif", 369,
        expected_element_totals={"C": 96, "H": 182, "N": 48, "O": 31, "Zn": 12},
    ),
    CifCase(
        "DAN-2", "DAN-2.cif", 35,
        expected_element_totals={"C": 6, "H": 14, "K": 1, "N": 5, "O": 9},
    ),
    CifCase(
        "PAP-H4", "PAP-H4.cif", 656,
        expected_element_totals={"C": 80, "Cl": 48, "H": 288, "N": 48, "O": 192},
        expected_nh4_count=16,
    ),
    # 4-fluorophenethylammonium bromide.  The cation is encoded as
    # PART -1 on a mirror plane; SP-completion keeps both half-images,
    # snaps overlapping atoms, and removes/refits ghost hydrogens.  The
    # CIF formula_sum ``C6 H11 Br F N`` is wrong (phenethylammonium has
    # 8 C, not 6); the solver returns the chemically correct C8 cation.
    CifCase(
        "phenethylammonium-sp-p21m", "phenethylammonium_sp_p21m.cif", 88,
        expected_element_totals={"Br": 4, "C": 32, "F": 4, "H": 44, "N": 4},
    ),
    CifCase(
        "DAI-X1", "DAI-X1.cif", 112,
        expected_element_totals={"C": 12, "H": 52, "I": 6, "N": 4, "Na": 2, "O": 36},
    ),
    # ZIF-8: 40 orphan O atoms are solvent waters whose H atoms are not
    # refined in the CIF (labels O1S/O2S/O3S, where -S is SHELXL's
    # solvent suffix).  This is a CIF-side limitation, not a solver bug.
    CifCase(
        "ZIF-8", "ZIF-8.cif", 316, expected_defects=40,
        expected_element_totals={"C": 96, "H": 120, "N": 48, "O": 40, "Zn": 12},
    ),
    CifCase(
        "PAP-M5", "PAP-M5.cif", 296,
        expected_element_totals={"Ag": 8, "C": 40, "Cl": 24, "H": 112, "N": 16, "O": 96},
    ),
    CifCase(
        "DAP-O4", "DAP-O4.cif", 344,
        expected_element_totals={"C": 48, "Cl": 24, "H": 144, "N": 24, "O": 104},
        expected_nh4_count=8,
    ),
    # PAP-4: highly-disordered NH4+ at a special position (24 orientations).
    # Fixed by the chemistry-aware motif merge that ignores soft (geometric /
    # implicit_sp / valence_geometry) conflicts when reconstructing isolated
    # X(H)_n centres.  Full PAP-4 solve takes ~60 s on this machine.
    CifCase(
        "PAP-4", "PAP-4.cif", 304, timeout=180,
        expected_element_totals={"C": 32, "Cl": 24, "H": 128, "N": 24, "O": 96},
        expected_nh4_count=8,
    ),
    # DAI-4: two chemically equivalent NH4+ sites encoded with different CIF
    # styles (N1 implicit SHELX riding-H vs N4 explicit disorder_assembly
    # tags).  Regression guard for the "motif merge picks same-asym_id H
    # twice" bug where N1 sites were resolved as NH3 instead of NH4+.
    CifCase(
        "DAI-4", "DAI-4.cif", 336,
        expected_element_totals={"C": 48, "H": 144, "I": 24, "N": 24, "O": 96},
        expected_nh4_count=8,
    ),
    # DAP-7: hydrazinium/diaminopropane salt with 2 N per cation.  Checks
    # that the motif merge still handles multi-N cations correctly.  The
    # H1C protons sit on a 0.5-occupancy disorder pair (one proton per
    # hydrazinium); both N1-N1 cations must survive with a single H1C
    # each, giving 6×ClO4 + 2×C6H14N2 + 2×H5N2 = 88 atoms.
    CifCase(
        "DAP-7", "DAP-7.cif", 88,
        expected_element_totals={"C": 12, "Cl": 6, "H": 38, "N": 8, "O": 24},
    ),
]


# The solver should not carry any expected cross-mode failures.  Keep the
# mapping as an explicit tripwire: adding a new entry requires documenting the
# reason in the corresponding case comment and should be temporary.
KNOWN_INCONSISTENT_MODES: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# NH4 / minimum-distance helpers
# ---------------------------------------------------------------------------


def _min_interatomic_distance(crystal) -> float:
    """Return the smallest pairwise atom-atom distance in the crystal.

    Uses ASE's ``neighbor_list`` with the crystal cell + PBC so that
    minimum-image contacts across periodic boundaries are picked up.
    Returns ``inf`` when the cutoff catches no neighbours (extremely
    sparse cell).
    """
    atoms = crystal.to_ase()
    distances = neighbor_list("d", atoms, cutoff=2.0)
    if len(distances) == 0:
        return float("inf")
    return float(np.min(distances))


def _assert_nh4_tetrahedral(mol, lattice, pbc, *, label: str) -> None:
    """Validate that an H4N1 fragment has reasonable tetrahedral geometry.

    The bond-length window (0.85-1.20 Å) is a generous covalent N-H
    range that accepts both the typical 1.00 Å neutron-refined value
    and the shorter X-ray riding-H placement.  The angle window
    (85-135°) accepts the ideal 109.5° tetrahedron with substantial
    crystal-field distortion while still rejecting linear or grossly
    pyramidal placements.
    """
    atoms = mol.to_ase()
    atoms.set_cell(lattice)
    atoms.set_pbc(pbc)
    symbols = atoms.get_chemical_symbols()
    n_indices = [i for i, s in enumerate(symbols) if s == "N"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    assert len(n_indices) == 1, f"{label}: expected 1 N, got {len(n_indices)} ({symbols})"
    assert len(h_indices) == 4, f"{label}: expected 4 H, got {len(h_indices)} ({symbols})"

    n = n_indices[0]
    nh_distances = [atoms.get_distance(n, h, mic=True) for h in h_indices]
    bad_bonds = [d for d in nh_distances if not (0.85 <= d <= 1.20)]
    assert not bad_bonds, (
        f"{label}: N-H distances out of [0.85, 1.20] Å: "
        f"{[f'{d:.3f}' for d in nh_distances]}"
    )

    angles = []
    for i in range(4):
        for j in range(i + 1, 4):
            angles.append(atoms.get_angle(h_indices[i], n, h_indices[j], mic=True))
    bad_angles = [a for a in angles if not (85.0 <= a <= 135.0)]
    assert not bad_angles, (
        f"{label}: H-N-H angles out of [85, 135]° (ideal 109.5°): "
        f"{[f'{a:.1f}' for a in angles]}"
    )


def _iter_nh4_fragments(crystal):
    """Yield molecules whose composition is exactly H4N1."""
    nh4_counter = Counter({"H": 4, "N": 1})
    for mol in crystal.molecules:
        if Counter(mol.get_chemical_symbols()) == nh4_counter:
            yield mol


def _element_totals(crystal) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for mol in crystal.molecules:
        totals.update(mol.get_chemical_symbols())
    return dict(totals)


@lru_cache(maxsize=None)
def _solve_cached(
    path: str,
    method: str,
    generate_count: int,
    random_seed: int | None,
    coupled: bool = True,
):
    return generate_ordered_replicas_from_disordered_sites(
        path,
        generate_count=generate_count,
        method=method,
        random_seed=random_seed,
        coupled=coupled,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str):
    """Return (crystal, n_atoms, defects, formulas).

    `defects` counts both over-coordinated atoms (degree > MAX_COORD) and
    orphan atoms (degree 0 for elements that don't normally exist as
    isolated species).
    """
    crystal = _solve_cached(path, "optimal", 1, None)[0]
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
    """Optimal-mode regression: atom count, over-coordination, element
    totals, NH4 chemistry, and minimum interatomic contact."""
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)

    cif_path = os.path.join(cif_data_dir, case.cif)
    assert os.path.exists(cif_path), f"missing CIF fixture: {case.cif}"

    crystal, n_atoms, defects, formulas = _resolve(cif_path)

    if case.expected_atoms is not None:
        assert n_atoms == case.expected_atoms, (
            f"{case.name}: expected {case.expected_atoms} atoms, "
            f"got {n_atoms} (formulas={dict(formulas)})"
        )
    assert defects == case.expected_defects, (
        f"{case.name}: expected {case.expected_defects} defective "
        f"atoms (over-coord or orphan), got {defects}"
    )

    if case.expected_element_totals is not None:
        actual_totals = _element_totals(crystal)
        assert actual_totals == dict(case.expected_element_totals), (
            f"{case.name}: per-element totals mismatch.\n"
            f"  expected: {dict(case.expected_element_totals)}\n"
            f"  actual:   {actual_totals}\n"
            f"  formulas: {dict(formulas)}"
        )

    if case.expected_nh4_count:
        nh4_fragments = list(_iter_nh4_fragments(crystal))
        assert len(nh4_fragments) == case.expected_nh4_count, (
            f"{case.name}: expected {case.expected_nh4_count} H4N1 fragments, "
            f"got {len(nh4_fragments)} (formulas={dict(formulas)})"
        )
        for idx, mol in enumerate(nh4_fragments):
            _assert_nh4_tetrahedral(
                mol,
                lattice=crystal.lattice,
                pbc=crystal.pbc,
                label=f"{case.name} NH4#{idx}",
            )

    d_min = _min_interatomic_distance(crystal)
    assert d_min >= case.min_interatomic_distance, (
        f"{case.name}: closest interatomic contact {d_min:.3f} Å is below "
        f"the {case.min_interatomic_distance:.2f} Å threshold; two disorder "
        f"alternatives may have been resolved on top of each other"
    )


# ---------------------------------------------------------------------------
# Cross-mode consistency (random / enumerate vs optimal)
# ---------------------------------------------------------------------------


_MODE_PARAMS = [
    pytest.param("random", 3, id="random"),
    pytest.param("enumerate", 4, id="enumerate"),
]


@pytest.mark.parametrize("method,n_replicas", _MODE_PARAMS)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_disorder_replica_zero_matches_optimal(
    case: CifCase, method: str, n_replicas: int, cif_data_dir: str
):
    """The first random/enumerate replica is the deterministic MWIS reference."""
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)

    assert (case.name, method) not in KNOWN_INCONSISTENT_MODES

    cif_path = os.path.join(cif_data_dir, case.cif)
    assert os.path.exists(cif_path), f"missing CIF fixture: {case.cif}"

    crystals = _solve_cached(cif_path, method, n_replicas, 42)
    assert crystals, f"{case.name}: {method} returned no replicas"

    _assert_matches_optimal(case, crystals[0], f"{case.name} {method}#0")


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_disorder_enumerate_all_replicas_match_optimal(
    case: CifCase, cif_data_dir: str
):
    """Enumerate is deterministic: every returned replica must be optimal-equivalent."""
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)
    assert (case.name, "enumerate") not in KNOWN_INCONSISTENT_MODES

    cif_path = os.path.join(cif_data_dir, case.cif)
    assert os.path.exists(cif_path), f"missing CIF fixture: {case.cif}"

    crystals = _solve_cached(cif_path, "enumerate", 4, 42)
    assert crystals, f"{case.name}: enumerate returned no replicas"

    for idx, crystal in enumerate(crystals):
        _assert_matches_optimal(case, crystal, f"{case.name} enumerate#{idx}")


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_disorder_random_replica_chemistry_validity(
    case: CifCase, cif_data_dir: str
):
    """Random replicas after #0 may sample lower-occupancy populations, but
    must remain chemically valid: no broken NH4 motifs, no close contacts, and
    no severe atom-count collapse."""
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)
    assert (case.name, "random") not in KNOWN_INCONSISTENT_MODES

    cif_path = os.path.join(cif_data_dir, case.cif)
    assert os.path.exists(cif_path), f"missing CIF fixture: {case.cif}"

    crystals = _solve_cached(cif_path, "random", 3, 42)
    assert len(crystals) >= 2, f"{case.name}: random returned too few replicas"

    for idx, crystal in enumerate(crystals[1:], start=1):
        n_atoms = crystal.get_total_nodes()
        if case.expected_atoms is not None:
            min_atoms = int(case.expected_atoms * 0.85)
            assert n_atoms >= min_atoms, (
                f"{case.name} random#{idx}: atoms {n_atoms} below "
                f"the 85% chemistry-validity floor ({min_atoms})"
            )

        if case.expected_nh4_count:
            nh4_fragments = list(_iter_nh4_fragments(crystal))
            assert len(nh4_fragments) == case.expected_nh4_count, (
                f"{case.name} random#{idx}: NH4 count "
                f"{len(nh4_fragments)} != {case.expected_nh4_count}"
            )
            for nh4_idx, mol in enumerate(nh4_fragments):
                _assert_nh4_tetrahedral(
                    mol,
                    lattice=crystal.lattice,
                    pbc=crystal.pbc,
                    label=f"{case.name} random#{idx} NH4#{nh4_idx}",
                )

        d_min = _min_interatomic_distance(crystal)
        assert d_min >= case.min_interatomic_distance, (
            f"{case.name} random#{idx}: closest contact {d_min:.3f} Å "
            f"below the {case.min_interatomic_distance:.2f} Å threshold"
        )


def _assert_matches_optimal(case: CifCase, crystal, context: str) -> None:
    """Strict optimal-equivalence assertion used by replica #0 and enumerate."""
    expected_totals = (
        dict(case.expected_element_totals)
        if case.expected_element_totals is not None
        else None
    )

    n_atoms = crystal.get_total_nodes()
    if case.expected_atoms is not None:
        assert n_atoms == case.expected_atoms, (
            f"{context}: atoms {n_atoms} != optimal {case.expected_atoms}"
        )

    if expected_totals is not None:
        actual_totals = _element_totals(crystal)
        assert actual_totals == expected_totals, (
            f"{context}: totals {actual_totals} != {expected_totals}"
        )

    if case.expected_nh4_count:
        nh4_fragments = list(_iter_nh4_fragments(crystal))
        assert len(nh4_fragments) == case.expected_nh4_count, (
            f"{context}: NH4 count {len(nh4_fragments)} != "
            f"{case.expected_nh4_count}"
        )
        for nh4_idx, mol in enumerate(nh4_fragments):
            _assert_nh4_tetrahedral(
                mol,
                lattice=crystal.lattice,
                pbc=crystal.pbc,
                label=f"{context} NH4#{nh4_idx}",
            )

    d_min = _min_interatomic_distance(crystal)
    assert d_min >= case.min_interatomic_distance, (
        f"{context}: closest contact {d_min:.3f} Å below "
        f"the {case.min_interatomic_distance:.2f} Å threshold"
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


def test_dapo4_topology(cif_data_dir: str):
    """DAP-O4: Fm-3c (IT 226), 8 diaminopropane dications + 24 perchlorate.

    Regression guard for the graph-invariant pre-check: before the fix,
    VF2 isomorphism on the large (>50-node) perchlorate molecular graphs
    (from disorder-connected bond perception) ran in O(N!) time and hung
    indefinitely.
    """
    cif = os.path.join(cif_data_dir, "DAP-O4.cif")
    assert os.path.exists(cif), "DAP-O4.cif fixture not found"

    _, n_atoms, _, formulas = _resolve(cif)
    assert n_atoms == 344
    # The dication C6H14N2O (diaminopropanol) carries its own hydroxyl
    # O; this is the cation moiety from the CIF, not a perchlorate O.
    assert formulas.get("C6H14N2O1", 0) == 8, (
        f"Expected 8 × C6H14N2O1 dication, got: {dict(formulas)}"
    )
    assert formulas.get("Cl1O4", 0) == 24, (
        f"Expected 24 × ClO4- anion, got: {dict(formulas)}"
    )
    assert formulas.get("H4N1", 0) == 8, (
        f"Expected 8 × NH4+ (H4N1), got: {dict(formulas)}"
    )
