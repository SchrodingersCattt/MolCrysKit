"""Tests for packing-shell polyhedra analysis."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.packing_shell import (
    DEFAULT_MOLECULAR_SEARCH_CUTOFF,
    angular_rmsd_vs_ideals,
    detect_coordination_number,
    find_polyhedra,
    hull_encloses_center,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.polyhedra import convex_hull_payload, ideal_polyhedra_for_cn

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"


def test_ideal_polyhedra_catalog_exposes_cn8_cube():
    cn8 = ideal_polyhedra_for_cn(8)

    assert "cube" in cn8
    assert cn8["cube"].shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(cn8["cube"], axis=1), np.ones(8))


def test_convex_hull_payload_serializes_faces_and_edges():
    coords = ideal_polyhedra_for_cn(8)["cube"]
    payload = convex_hull_payload(coords)

    assert len(payload["vertices"]) == 8
    assert len(payload["simplices"]) > 0
    assert len(payload["edges"]) > 0


def test_hull_encloses_center_for_symmetric_shell():
    coords = ideal_polyhedra_for_cn(8)["cube"]

    assert hull_encloses_center(coords, np.zeros(3)) is True


def test_coordination_number_expands_until_centered_inside_hull():
    coords = np.array(
        [
            [-4.04, 1.06, 2.40],
            [4.08, 1.07, 2.43],
            [0.00, -4.61, 2.52],
            [0.00, 4.80, -2.62],
            [-3.97, 1.06, -7.76],
            [4.07, 1.06, -7.79],
            [0.00, -4.57, -7.71],
            [-4.04, -7.53, -2.60],
            [4.04, -7.53, -2.60],
            [0.00, 4.76, 7.63],
            [-4.10, 7.74, 2.46],
            [4.10, 7.74, 2.46],
        ],
        dtype=float,
    )
    distances = np.linalg.norm(coords, axis=1)

    result = detect_coordination_number(
        distances,
        coords=coords,
        center=[0.0, 0.0, 0.0],
        enforce_enclosure=True,
    )

    assert result["primary_gap_cn"] == 4
    assert result["coordination_number"] == 12
    assert result["enclosed"] is True
    assert result["enclosure_expanded"] is True


def test_angular_rmsd_matches_ideal_cube():
    coords = ideal_polyhedra_for_cn(8)["cube"]
    result = angular_rmsd_vs_ideals(coords)

    assert result["coordination_number"] == 8
    assert result["best_match"]["name"] == "cube"
    assert result["best_match"]["angular_rmsd"] < 1e-10


# --- Cutoff mode and bypass-enclosure tests ---


def test_cutoff_mode_overrides_gap_selection():
    """A user-provided radial cutoff fixes the shell exactly, ignoring gaps."""
    distances = [2.0, 2.05, 2.1, 2.15, 4.0, 4.1, 4.2, 4.3]
    coords = np.array(
        [
            [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
            [4.0, 0.0, 0.0], [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0], [0.0, -4.0, 0.0],
        ],
        dtype=float,
    )
    result = detect_coordination_number(
        distances,
        coords=coords,
        center=np.zeros(3),
        cutoff=3.0,
    )
    assert result["mode"] == "cutoff"
    assert result["coordination_number"] == 4
    assert result["cutoff"] == 3.0
    # gap heuristic would have agreed here, but the field is reported separately
    assert result["primary_gap_cn"] == 4
    assert result["enclosure_expanded"] is False


def test_cutoff_mode_does_not_expand_to_enclose_center():
    """Even when enforce_enclosure=True, cutoff mode never crosses r_c."""
    coords = np.array(
        [
            [-4.04, 1.06, 2.40], [4.08, 1.07, 2.43],
            [0.00, -4.61, 2.52], [0.00, 4.80, -2.62],
            [-3.97, 1.06, -7.76], [4.07, 1.06, -7.79],
            [0.00, -4.57, -7.71], [-4.04, -7.53, -2.60],
            [4.04, -7.53, -2.60], [0.00, 4.76, 7.63],
            [-4.10, 7.74, 2.46], [4.10, 7.74, 2.46],
        ],
        dtype=float,
    )
    distances = np.linalg.norm(coords, axis=1)

    result = detect_coordination_number(
        distances,
        coords=coords,
        center=[0.0, 0.0, 0.0],
        enforce_enclosure=True,
        cutoff=6.0,
    )
    assert result["mode"] == "cutoff"
    assert result["coordination_number"] == int(np.sum(distances <= 6.0))
    assert result["enclosure_expanded"] is False


def test_cutoff_mode_returns_zero_when_nothing_in_range():
    distances = [3.0, 3.5, 4.0]
    result = detect_coordination_number(distances, cutoff=2.5)
    assert result["coordination_number"] == 0
    assert result["mode"] == "cutoff"
    assert result["cutoff"] == 2.5


def test_bypass_enclosure_returns_gap_cn_only():
    """enforce_enclosure=False stops at the gap CN and never expands."""
    coords = np.array(
        [
            [-4.04, 1.06, 2.40], [4.08, 1.07, 2.43],
            [0.00, -4.61, 2.52], [0.00, 4.80, -2.62],
            [-3.97, 1.06, -7.76], [4.07, 1.06, -7.79],
            [0.00, -4.57, -7.71], [-4.04, -7.53, -2.60],
            [4.04, -7.53, -2.60], [0.00, 4.76, 7.63],
            [-4.10, 7.74, 2.46], [4.10, 7.74, 2.46],
        ],
        dtype=float,
    )
    distances = np.linalg.norm(coords, axis=1)

    result = detect_coordination_number(
        distances,
        coords=coords,
        center=[0.0, 0.0, 0.0],
        enforce_enclosure=False,
    )
    assert result["mode"] == "gap"
    assert result["coordination_number"] == result["primary_gap_cn"]
    assert result["enclosure_expanded"] is False


def test_default_mode_label_is_gap_plus_enclosure():
    distances = [2.0, 2.05, 2.1, 2.15, 4.0, 4.1, 4.2, 4.3]
    result = detect_coordination_number(distances)
    assert result["mode"] == "gap+enclosure"
    assert result["cutoff"] is None


# --- find_polyhedra integration tests ---


def _rocksalt_atoms(a: float = 5.0):
    """Build a tiny rock-salt-style A--B test crystal (Pb at origin, I octahedron)."""
    cell = np.eye(3) * a
    positions = np.array(
        [
            [0.0, 0.0, 0.0],            # Pb
            [a / 2, 0.0, 0.0],          # I
            [0.0, a / 2, 0.0],          # I
            [0.0, 0.0, a / 2],          # I
            [a / 2, a / 2, a / 2],      # Pb
            [0.0, a / 2, a / 2],        # I
            [a / 2, 0.0, a / 2],        # I
            [a / 2, a / 2, 0.0],        # I
        ],
        dtype=float,
    )
    symbols = ["Pb", "I", "I", "I", "Pb", "I", "I", "I"]
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def test_find_polyhedra_rocksalt_octahedron_with_cutoff():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(atoms, central="Pb", ligand="I", cutoff=3.0)
    assert len(polys) == 2
    for record in polys:
        assert record["center_symbol"] == "Pb"
        assert record["coordination_number"] == 6
        assert record["mode"] == "cutoff"
        assert all(np.isclose(d, 2.5, atol=1e-6) for d in record["shell_distances"])
        assert len(record["shell_indices"]) == 6


def test_find_polyhedra_rejects_non_ligand_atoms():
    """Decorate the Pb-I cell with a stray Br outside the cutoff: it must NOT enter."""
    atoms = _rocksalt_atoms(a=5.0)
    extra = Atoms(symbols=["Br"], positions=[[0.1, 0.1, 0.1]], cell=atoms.cell, pbc=True)
    combined = atoms + extra
    polys = find_polyhedra(combined, central="Pb", ligand="I", cutoff=3.0)
    for record in polys:
        for shell_idx in record["shell_indices"]:
            assert combined.get_chemical_symbols()[shell_idx] == "I"


def test_find_polyhedra_search_cutoff_can_be_larger_than_cutoff():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(
        atoms, central="Pb", ligand="I", cutoff=3.0, search_cutoff=6.0
    )
    for record in polys:
        assert record["coordination_number"] == 6
        assert record["search_cutoff"] == 6.0


def test_find_polyhedra_enforce_enclosure_can_be_bypassed():
    atoms = _rocksalt_atoms(a=5.0)
    enforced = find_polyhedra(atoms, central="Pb", ligand="I", search_cutoff=6.0)
    bypassed = find_polyhedra(
        atoms,
        central="Pb",
        ligand="I",
        search_cutoff=6.0,
        enforce_enclosure=False,
    )
    for record in enforced:
        assert record["mode"] == "gap+enclosure"
    for record in bypassed:
        assert record["mode"] == "gap"


def test_find_polyhedra_score_shape_returns_octahedron():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(
        atoms, central="Pb", ligand="I", cutoff=3.0, score_shape=True
    )
    for record in polys:
        match = record["best_match"]
        assert match is not None
        assert match["name"] == "octahedron"
        assert match["angular_rmsd"] < 1e-6


def test_find_polyhedra_central_indices_filter():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(
        atoms, central="Pb", ligand="I", cutoff=3.0, central_indices=[0]
    )
    assert len(polys) == 1
    assert polys[0]["center_index"] == 0


def test_find_polyhedra_returns_empty_for_missing_central_element():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(atoms, central="Cs", ligand="I", cutoff=3.0)
    assert polys == []


def test_find_polyhedra_invalid_search_cutoff_raises():
    atoms = _rocksalt_atoms(a=5.0)
    with pytest.raises(ValueError):
        find_polyhedra(atoms, central="Pb", ligand="I", search_cutoff=0.0)


def test_find_polyhedra_atom_level_record_carries_level_field():
    atoms = _rocksalt_atoms(a=5.0)
    polys = find_polyhedra(atoms, central="Pb", ligand="I", cutoff=3.0)
    for record in polys:
        assert record["level"] == "atom"


def test_find_polyhedra_invalid_level_raises():
    atoms = _rocksalt_atoms(a=5.0)
    with pytest.raises(ValueError, match="level must be"):
        find_polyhedra(atoms, central="Pb", ligand="I", level="banana")


# --- find_polyhedra(level="molecule") tests ---


def _rocksalt_molecular_crystal(a: float = 5.0) -> MolecularCrystal:
    """4-N + 4-Cl rocksalt-style MolecularCrystal whose centroids form a
    perfect octahedral A--B coordination at distance ``a/2``.
    """
    cell = np.eye(3) * a
    a_positions = [
        [0.0, 0.0, 0.0],
        [a / 2, a / 2, 0.0],
        [a / 2, 0.0, a / 2],
        [0.0, a / 2, a / 2],
    ]
    b_positions = [
        [a / 2, 0.0, 0.0],
        [0.0, a / 2, 0.0],
        [0.0, 0.0, a / 2],
        [a / 2, a / 2, a / 2],
    ]
    molecules = []
    for pos in a_positions:
        molecules.append(Atoms(symbols=["N"], positions=[pos], cell=cell, pbc=True))
    for pos in b_positions:
        molecules.append(Atoms(symbols=["Cl"], positions=[pos], cell=cell, pbc=True))
    return MolecularCrystal(lattice=cell, molecules=molecules)


def _toy_perchlorate_crystal(a: float = 6.0) -> MolecularCrystal:
    """Tiny molecular crystal with one full NH4 (5 atoms) and one full ClO4
    (5 atoms) placed so their centroids sit on a CsCl-like (0,0,0) /
    (a/2,a/2,a/2) sublattice. Exercises moiety-string parsing ('N H4',
    'Cl O4') against multi-atom molecules.
    """
    cell = np.eye(3) * a

    nh4_centre = np.array([0.0, 0.0, 0.0])
    nh4_local = np.array([
        [0.0, 0.0, 0.0],
        [0.6, 0.6, 0.6],
        [-0.6, -0.6, 0.6],
        [0.6, -0.6, -0.6],
        [-0.6, 0.6, -0.6],
    ])
    nh4 = Atoms(
        symbols=["N", "H", "H", "H", "H"],
        positions=nh4_centre + nh4_local,
        cell=cell,
        pbc=True,
    )
    # Centroid of nh4_local is exactly (0,0,0), so centroid coincides with N.

    clo4_centre = np.array([a / 2, a / 2, a / 2])
    clo4_local = np.array([
        [0.0, 0.0, 0.0],
        [0.7, 0.7, 0.7],
        [-0.7, -0.7, 0.7],
        [0.7, -0.7, -0.7],
        [-0.7, 0.7, -0.7],
    ])
    clo4 = Atoms(
        symbols=["Cl", "O", "O", "O", "O"],
        positions=clo4_centre + clo4_local,
        cell=cell,
        pbc=True,
    )

    return MolecularCrystal(lattice=cell, molecules=[nh4, clo4])


def test_find_polyhedra_molecule_level_rocksalt_octahedron_with_cutoff():
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "N", "Cl", level="molecule", cutoff=3.0)
    assert len(polys) == 4
    for record in polys:
        assert record["level"] == "molecule"
        assert record["center_formula"] == "N"
        assert record["shell_formula"] == "Cl"
        assert record["coordination_number"] == 6
        assert record["mode"] == "cutoff"
        assert len(record["shell_molecule_indices"]) == 6
        assert all(np.isclose(d, 2.5, atol=1e-6) for d in record["shell_distances"])
        assert record["center_kind"] == "centroid"


def test_find_polyhedra_molecule_level_rocksalt_score_shape():
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "N", "Cl", level="molecule",
                           cutoff=3.0, score_shape=True)
    for record in polys:
        match = record["best_match"]
        assert match is not None
        assert match["name"] == "octahedron"
        assert match["angular_rmsd"] < 1e-6


def test_find_polyhedra_molecule_level_central_indices_filter():
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "N", "Cl", level="molecule",
                           cutoff=3.0, central_indices=[0])
    assert len(polys) == 1
    assert polys[0]["center_molecule_index"] == 0


def test_find_polyhedra_molecule_level_homo_pair_excludes_self_image():
    """An A--A search must drop the (t=0, same molecule) self image but must
    keep every periodic image of the same molecule."""
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "N", "N", level="molecule", cutoff=4.0)
    for record in polys:
        ci = record["center_molecule_index"]
        for lig_idx, offset in zip(record["shell_molecule_indices"],
                                   record["shell_offsets"]):
            if lig_idx == ci:
                assert not np.allclose(offset, [0.0, 0.0, 0.0])
        # Closest neighbours are 12 face-diagonal images at √2·a/2 = 3.536 Å
        assert record["coordination_number"] >= 6


def test_find_polyhedra_molecule_level_center_kind_choices():
    crys = _rocksalt_molecular_crystal(a=5.0)
    for kind in ("centroid", "com", "heavy_centroid"):
        polys = find_polyhedra(crys, "N", "Cl", level="molecule",
                               cutoff=3.0, center_kind=kind)
        assert len(polys) == 4
        for record in polys:
            assert record["coordination_number"] == 6
            assert record["center_kind"] == kind


def test_find_polyhedra_molecule_level_returns_empty_for_unknown_central_moiety():
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "Cs", "Cl", level="molecule", cutoff=3.0)
    assert polys == []


def test_find_polyhedra_molecule_level_default_search_cutoff_used():
    crys = _rocksalt_molecular_crystal(a=5.0)
    polys = find_polyhedra(crys, "N", "Cl", level="molecule")
    for record in polys:
        assert record["search_cutoff"] == DEFAULT_MOLECULAR_SEARCH_CUTOFF


def test_find_polyhedra_molecule_level_rejects_non_molecular_crystal():
    atoms = _rocksalt_atoms(a=5.0)
    with pytest.raises(TypeError, match="MolecularCrystal-like"):
        find_polyhedra(atoms, "Pb", "I", level="molecule", cutoff=3.0)


def test_find_polyhedra_molecule_level_invalid_center_kind_raises():
    crys = _rocksalt_molecular_crystal(a=5.0)
    with pytest.raises(ValueError, match="center_kind"):
        find_polyhedra(crys, "N", "Cl", level="molecule",
                       cutoff=3.0, center_kind="bogus")


def test_find_polyhedra_molecule_level_empty_moiety_raises():
    crys = _rocksalt_molecular_crystal(a=5.0)
    with pytest.raises(ValueError, match="non-empty"):
        find_polyhedra(crys, "", "Cl", level="molecule", cutoff=3.0)


def test_find_polyhedra_molecule_level_multifragment_moiety_raises():
    crys = _rocksalt_molecular_crystal(a=5.0)
    with pytest.raises(ValueError, match="exactly one fragment"):
        find_polyhedra(crys, "N H4, C2 H10 N2", "Cl",
                       level="molecule", cutoff=3.0)


def test_find_polyhedra_molecule_level_invalid_search_cutoff_raises():
    crys = _rocksalt_molecular_crystal(a=5.0)
    with pytest.raises(ValueError, match="search_cutoff must be positive"):
        find_polyhedra(crys, "N", "Cl", level="molecule", search_cutoff=0.0)


def test_find_polyhedra_molecule_level_moiety_string_matches_full_fragments():
    """The 'N H4' moiety string must match a real NH4 molecule with 5 atoms
    via heavy-atom signature, and likewise 'Cl O4' must pick up the ClO4."""
    crys = _toy_perchlorate_crystal(a=6.0)
    polys = find_polyhedra(crys, "N H4", "Cl O4", level="molecule", cutoff=8.0)
    assert len(polys) == 1
    record = polys[0]
    assert record["center_formula"] == "H4N"
    assert record["shell_formula"] == "ClO4"
    # CsCl-like geometry: 8 nearest images at sqrt(3)*a/2 = 5.196 Å
    assert record["coordination_number"] == 8
    assert np.isclose(np.mean(record["shell_distances"]),
                      6.0 * np.sqrt(3) / 2, atol=1e-6)


def test_find_polyhedra_molecule_level_dap4_nh4_octahedral_nearest_shell():
    """Sanity check on the real DAP-4 perovskite CIF.  NH4+ occupies the
    A2 site of the Pa-3 perovskite framework; the *nearest* ClO4 shell
    (cutoff=5 Å keeps only first-shell distances ~3.6 Å) is the NaCl-like
    octahedral cage of CN=6.  All 8 NH4 sites must resolve cleanly.

    Before the diagonal-image fix this returned a fragmented CN
    histogram (e.g. ``{6:4, 5:2, 4:1, 3:1}``) because the lattice
    translation enumerator silently dropped face- and body-diagonal
    images (``|t| in (max(lengths), sqrt(3)·max(lengths))``).
    """
    from molcrys_kit.io.cif import read_mol_crystal
    crys = read_mol_crystal(str(_CIF_DIR / "DAP-4.cif"))
    polys = find_polyhedra(crys, "N H4", "Cl O4", level="molecule",
                           cutoff=5.0, score_shape=True)
    assert len(polys) == 8
    for record in polys:
        assert record["coordination_number"] == 6
        assert record["best_match"]["name"] == "octahedron"
        assert 3.4 <= float(np.mean(record["shell_distances"])) <= 3.8


def test_find_polyhedra_molecule_level_dap4_nh4_cuboctahedron_extended_shell():
    """At the perovskite A--X12 cuboctahedron cutoff (~8 Å), NH4 sees
    all 12 ClO4 anions on the surrounding cuboctahedron vertices: 6
    near-corner ClO4 at ~3.6 Å plus 6 farther ClO4 at ~7.9 Å.
    """
    from molcrys_kit.io.cif import read_mol_crystal
    crys = read_mol_crystal(str(_CIF_DIR / "DAP-4.cif"))
    polys = find_polyhedra(crys, "N H4", "Cl O4", level="molecule",
                           cutoff=8.0, score_shape=True)
    assert len(polys) == 8
    for record in polys:
        assert record["coordination_number"] == 12
        assert record["best_match"]["name"] == "cuboctahedron"


def test_find_polyhedra_molecule_level_dap4_clo4_square_planar():
    """Reverse query at cutoff=8 Å: every ClO4 anion is surrounded by 4
    NH4+ cations in a square-planar arrangement (the four corners of the
    A2 cage face that the ClO4 sits on). The cutoff=5 Å view truncates
    the cage to the two nearest NH4 (dimer).
    """
    from molcrys_kit.io.cif import read_mol_crystal
    crys = read_mol_crystal(str(_CIF_DIR / "DAP-4.cif"))
    polys = find_polyhedra(crys, "Cl O4", "N H4", level="molecule",
                           cutoff=8.0, score_shape=True)
    assert len(polys) == 24
    for record in polys:
        assert record["coordination_number"] == 4
        assert record["best_match"]["name"] == "square_planar"


def test_find_polyhedra_molecule_level_diagonal_image_neighbour():
    """Regression: for a cubic cell of side `a`, the body-diagonal image
    is at |t| = a*sqrt(3); a face-diagonal image is at |t| = a*sqrt(2).
    Either is BIGGER than max(|a|, |b|, |c|) = a. An earlier draft used
    `search_cutoff + max(lengths)` as the translation-enumeration bound
    and silently dropped these images, returning CN=0 for centroid pairs
    whose true min-image distance was below `cutoff` but required a
    diagonal translation. The bound must instead be derived from the
    actual centroid extent.
    """
    a = 10.0
    cell = np.eye(3) * a
    # B sits near the (a, a, 0) face-diagonal corner of the home cell;
    # its (-1, -1, 0) image is 0.71 Å from A.
    a_mol = Atoms(symbols=["N"], positions=[[0.0, 0.0, 0.0]],
                  cell=cell, pbc=True)
    b_mol = Atoms(symbols=["Cl"], positions=[[a - 0.5, a - 0.5, 0.0]],
                  cell=cell, pbc=True)
    crys = MolecularCrystal(lattice=cell, molecules=[a_mol, b_mol])
    polys = find_polyhedra(crys, "N", "Cl", level="molecule", cutoff=3.0)
    assert len(polys) == 1
    record = polys[0]
    assert record["coordination_number"] == 1
    assert np.isclose(record["shell_distances"][0], np.sqrt(0.5), atol=1e-6)
    # The required image is the (-1, -1, 0) one
    assert np.allclose(record["shell_offsets"][0], [-a, -a, 0.0], atol=1e-6)


def test_find_polyhedra_molecule_level_dap4_dap_cation_cn12():
    """The DAP^2+ cation occupies the A1 site of the perovskite framework
    and is surrounded by 12 ClO4 anions in the cuboctahedral A1-X12 shell.
    """
    from molcrys_kit.io.cif import read_mol_crystal
    crys = read_mol_crystal(str(_CIF_DIR / "DAP-4.cif"))
    polys = find_polyhedra(crys, "C6 N2", "Cl O4", level="molecule", cutoff=8.0)
    assert len(polys) == 8  # eight DAP cations per Pa-3 cell
    cn_histogram = Counter(r["coordination_number"] for r in polys)
    assert cn_histogram[12] == 8, (
        f"expected every DAP cation to be CN=12, got {dict(cn_histogram)}"
    )
