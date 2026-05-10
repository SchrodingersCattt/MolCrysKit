"""Tests for packing-shell polyhedra analysis."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.packing_shell import (
    angular_rmsd_vs_ideals,
    detect_coordination_number,
    find_polyhedra,
    hull_encloses_center,
)
from molcrys_kit.structures.polyhedra import convex_hull_payload, ideal_polyhedra_for_cn


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
