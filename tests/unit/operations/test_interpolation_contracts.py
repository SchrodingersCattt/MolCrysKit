"""Behavior contracts for molcrys_kit.operations.interpolation and _path_core.

These tests characterize current semantics (including known rigid-endpoint limits
for variable-cell paths). They use synthetic structures only.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations._path_core import (
    InterpolationMethod,
    coerce_interpolation_method,
    copy_crystal_with_molecule_positions,
    path_lambda_values,
)
from molcrys_kit.operations.interpolation import (
    best_atom_mapping,
    interpolate_crystal,
    interpolate_crystal_vc,
    match_molecules,
    match_molecules_vc,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.utils.geometry import get_rotation_matrix


def _water(positions=None) -> CrystalMolecule:
    if positions is None:
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
            dtype=float,
        )
    return CrystalMolecule(Atoms("OHH", positions=positions, pbc=False), check_pbc=False)


def _rigid_water_pair(translation=None, angle_rad=np.pi / 2):
    lattice = np.eye(3) * 12.0
    mol_a = _water()
    com = mol_a.get_center_of_mass()
    rotation = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), angle_rad)
    if translation is None:
        translation = np.array([1.5, 0.8, 0.0])
    pos_b = (mol_a.get_positions() - com) @ rotation.T + com + translation
    crystal_a = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))
    crystal_b = MolecularCrystal(lattice, [_water(pos_b)], pbc=(True, True, True))
    return crystal_a, crystal_b


# ── best_atom_mapping ─────────────────────────────────────────────


def test_best_atom_mapping_identity_order():
    mol = _water()
    order = best_atom_mapping(mol, mol)
    np.testing.assert_array_equal(order, [0, 1, 2])


def test_best_atom_mapping_picks_lowest_rmsd_among_h_permutations():
    mol_a = _water()
    permuted = mol_a.get_positions()[[0, 2, 1]]
    mol_b = CrystalMolecule(Atoms("OHH", positions=permuted, pbc=False), check_pbc=False)
    order = best_atom_mapping(mol_a, mol_b)
    mapped = mol_b.get_positions()[order]
    np.testing.assert_allclose(mapped, mol_a.get_positions(), atol=1e-10)
    assert list(order) in ([0, 1, 2], [0, 2, 1])


def test_best_atom_mapping_formula_mismatch_raises():
    mol_a = _water()
    mol_b = CrystalMolecule(Atoms("HHH", positions=np.zeros((3, 3)), pbc=False), check_pbc=False)
    with pytest.raises(ValueError, match="different formulas"):
        best_atom_mapping(mol_a, mol_b)


def test_best_atom_mapping_count_mismatch_raises():
    mol_a = _water()
    mol_b = CrystalMolecule(Atoms("OH", positions=np.zeros((2, 3)), pbc=False), check_pbc=False)
    with pytest.raises(ValueError, match="different atom counts"):
        best_atom_mapping(mol_a, mol_b)


def test_best_atom_mapping_empty_returns_empty():
    empty = CrystalMolecule(Atoms(pbc=False), check_pbc=False)
    order = best_atom_mapping(empty, empty)
    assert order.shape == (0,)


def test_best_atom_mapping_identity_fallback_without_graph_isomorphism():
    """Non-isomorphic graphs fall back to index order when symbols match."""
    mol_a = CrystalMolecule(
        Atoms("CC", positions=[[0, 0, 0], [1.5, 0, 0]], pbc=False), check_pbc=False
    )
  # Different topology (disconnected vs same) but same formula count — graph may fail
    mol_b = CrystalMolecule(
        Atoms("CC", positions=[[0, 0, 0], [0, 2.0, 0]], pbc=False), check_pbc=False
    )
    order = best_atom_mapping(mol_a, mol_b)
    np.testing.assert_array_equal(order, [0, 1])


def _mapping_shape_rmsd(mol_a, mol_b, order_b):
    pos_a = np.asarray(mol_a.get_positions(), dtype=float)
    pos_b = np.asarray(mol_b.get_positions(), dtype=float)[order_b]
    com_a = pos_a.mean(axis=0)
    com_b = pos_b.mean(axis=0)
    from molcrys_kit.utils.geometry import kabsch_align

    _, rmsd = kabsch_align(pos_a - com_a, pos_b - com_b)
    return float(rmsd)


def test_best_atom_mapping_respects_max_isomorphisms_cap():
    """Stopping at the first isomorphism can miss lower-RMSD permutations."""
    mol_a = _water()
    mol_b = CrystalMolecule(
        Atoms("OHH", positions=mol_a.get_positions()[[0, 2, 1]], pbc=False), check_pbc=False
    )
    full = best_atom_mapping(mol_a, mol_b, max_isomorphisms=256)
    capped = best_atom_mapping(mol_a, mol_b, max_isomorphisms=1)
    assert _mapping_shape_rmsd(mol_a, mol_b, full) < 1e-9
    assert _mapping_shape_rmsd(mol_a, mol_b, capped) > 0.1


def test_best_atom_mapping_order_b_means_target_row_for_each_source_index():
    mol_a = _water()
    shuffled = mol_a.get_positions()[[0, 2, 1]]
    mol_b = CrystalMolecule(Atoms("OHH", positions=shuffled, pbc=False), check_pbc=False)
    order_b = best_atom_mapping(mol_a, mol_b)
    for i in range(3):
        assert mol_a.get_chemical_symbols()[i] == mol_b.get_chemical_symbols()[order_b[i]]


# ── match_molecules ────────────────────────────────────────────────


def test_match_molecules_requires_identical_lattice():
    lat_a = np.eye(3) * 10.0
    lat_b = np.eye(3) * 11.0
    ca = MolecularCrystal(lat_a, [_water()], pbc=(True, True, True))
    cb = MolecularCrystal(lat_b, [_water()], pbc=(True, True, True))
    with pytest.raises(ValueError, match="identical lattices"):
        match_molecules(ca, cb)


def test_match_molecules_count_mismatch_raises():
    lattice = np.eye(3) * 15.0
    ca = MolecularCrystal(lattice, [_water()], pbc=(True, True, True))
    cb = MolecularCrystal(lattice, [_water(), _water()], pbc=(True, True, True))
    with pytest.raises(ValueError, match="same number of molecules"):
        match_molecules(ca, cb)


def test_match_molecules_picks_closest_com_among_equivalents():
    lattice = np.eye(3) * 25.0
    w0 = _water()
    w1 = _water(_water().get_positions() + np.array([8.0, 0.0, 0.0]))
    w2 = _water(_water().get_positions() + np.array([1.0, 0.0, 0.0]))
    ca = MolecularCrystal(lattice, [w0, w1], pbc=(True, True, True))
    cb = MolecularCrystal(lattice, [w2, w1], pbc=(True, True, True))
    matches = match_molecules(ca, cb)
    assert {m.idx_a for m in matches} == {0, 1}
    m0 = next(m for m in matches if m.idx_a == 0)
    assert m0.idx_b == 0


def test_match_molecules_rotation_is_proper_det_one():
    crystal_a, crystal_b = _rigid_water_pair()
    match = match_molecules(crystal_a, crystal_b)[0]
    det = float(np.linalg.det(match.rotation_matrix))
    np.testing.assert_allclose(det, 1.0, atol=1e-8)


# ── interpolate_crystal (fixed cell) ───────────────────────────────


@pytest.mark.parametrize("method", list(InterpolationMethod))
def test_interpolate_crystal_all_methods_share_frame_count(method):
    crystal_a, crystal_b = _rigid_water_pair()
    frames = interpolate_crystal(crystal_a, crystal_b, method=method, n_images=7)
    assert len(frames) == 7
    np.testing.assert_allclose(
        frames[0].molecules[0].get_positions(),
        crystal_a.molecules[0].get_positions(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        frames[-1].molecules[0].get_positions(),
        crystal_b.molecules[0].get_positions(),
        atol=1e-8,
    )


def test_interpolate_crystal_include_endpoints_false_returns_interior_only():
    crystal_a, crystal_b = _rigid_water_pair()
    lambdas = path_lambda_values(5, include_endpoints=False)
    frames = interpolate_crystal(
        crystal_a, crystal_b, n_images=5, include_endpoints=False
    )
    assert len(frames) == len(lambdas) == 5
    assert not np.allclose(
        frames[0].molecules[0].get_positions(),
        crystal_a.molecules[0].get_positions(),
        atol=1e-8,
    )


def test_interpolate_crystal_single_frame_returns_lambda_zero():
    crystal_a, crystal_b = _rigid_water_pair()
    frames = interpolate_crystal(crystal_a, crystal_b, n_images=1)
    assert len(frames) == 1
    np.testing.assert_allclose(
        frames[0].molecules[0].get_positions(),
        crystal_a.molecules[0].get_positions(),
        atol=1e-10,
    )


def test_interpolate_crystal_does_not_mutate_inputs():
    crystal_a, crystal_b = _rigid_water_pair()
    pos_a_before = crystal_a.molecules[0].get_positions().copy()
    lat_before = crystal_a.lattice.copy()
    _ = interpolate_crystal(crystal_a, crystal_b, n_images=3)
    np.testing.assert_allclose(crystal_a.molecules[0].get_positions(), pos_a_before)
    np.testing.assert_allclose(crystal_a.lattice, lat_before)


def test_interpolate_crystal_unknown_molecule_index_raises():
    crystal_a, crystal_b = _rigid_water_pair()
    with pytest.raises(ValueError, match="Unknown molecule indices"):
        interpolate_crystal(crystal_a, crystal_b, molecule_indices=[99])


def test_coerce_interpolation_method_aliases():
    assert coerce_interpolation_method("com_alignment") == InterpolationMethod.COM_SO3
    assert coerce_interpolation_method("screw") == InterpolationMethod.SE3_SCREW
    with pytest.raises(ValueError, match="Unknown"):
        coerce_interpolation_method("not_a_method")


def test_proper_rotation_path_does_not_imply_chirality_invariance_under_bond_stretch():
    """Chirality is not certified: a deformed endpoint can invert handedness vs rigid fit."""
    chiral = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.5, 1.2, 0.0],
            [0.2, 0.3, 1.1],
        ],
        dtype=float,
    )
    mol_a = CrystalMolecule(Atoms("CCCC", positions=chiral, pbc=False), check_pbc=False)
    reflected = chiral.copy()
    reflected[:, 0] *= -1.0
    mol_b = CrystalMolecule(Atoms("CCCC", positions=reflected, pbc=False), check_pbc=False)
    lattice = np.eye(3) * 20.0
    ca = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))
    cb = MolecularCrystal(lattice, [mol_b], pbc=(True, True, True))
    frames = interpolate_crystal(ca, cb, n_images=3)
    det_end = np.linalg.det(
        match_molecules(ca, cb)[0].rotation_matrix
    )
    np.testing.assert_allclose(det_end, 1.0, atol=1e-8)
    assert not np.allclose(
        frames[-1].molecules[0].get_positions(),
        mol_b.get_positions(),
        atol=0.05,
    )


# ── variable-cell ──────────────────────────────────────────────────


def _vc_pair_with_internal_stretch():
    """Same rigid pose class but mol_b has a stretched bond (non-rigid internal delta)."""
    lat_a = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    lat_b = np.array([[5.5, 0.0, 0.0], [1.1, 6.5, 0.0], [0.0, 0.0, 7.5]])
    pos_a = np.array([[2.0, 3.0, 3.0], [2.5, 3.8, 3.0], [2.0, 3.0, 3.9]], dtype=float)
    mol_a = CrystalMolecule(Atoms("OHH", positions=pos_a, pbc=False), check_pbc=False)
    pos_b = pos_a.copy()
    pos_b[1] += np.array([0.35, 0.0, 0.0])
    mol_b = CrystalMolecule(Atoms("OHH", positions=pos_b, pbc=False), check_pbc=False)
    ca = MolecularCrystal(lat_a, [mol_a], pbc=(True, True, True))
    cb = MolecularCrystal(lat_b, [mol_b], pbc=(True, True, True))
    return ca, cb, mol_b


def test_vc_last_frame_lattice_matches_b():
    ca, cb, _ = _vc_pair_with_internal_stretch()
    frames = interpolate_crystal_vc(ca, cb, n_images=5, include_endpoints=True)
    np.testing.assert_allclose(frames[-1].lattice, cb.lattice, atol=1e-8)
    np.testing.assert_allclose(frames[0].lattice, ca.lattice, atol=1e-8)


def test_vc_rigid_internal_coords_last_frame_not_exact_deformed_b():
    """include_endpoints reproduces lattices and start coords; rigid model need not match B internals."""
    ca, cb, mol_b = _vc_pair_with_internal_stretch()
    frames = interpolate_crystal_vc(ca, cb, n_images=5, include_endpoints=True)
    np.testing.assert_allclose(
        frames[0].molecules[0].get_positions(),
        ca.molecules[0].get_positions(),
        atol=1e-8,
    )
    last = frames[-1].molecules[0].get_positions()
    target = mol_b.get_positions()
    assert not np.allclose(last, target, atol=1e-3)
    stretch_b = np.linalg.norm(target[1] - target[0])
    stretch_last = np.linalg.norm(last[1] - last[0])
    assert stretch_last < stretch_b - 0.1


def test_match_molecules_vc_nonorthogonal_mean_lattice():
    ca, cb, _ = _vc_pair_with_internal_stretch()
    matches = match_molecules_vc(ca, cb)
    assert len(matches) == 1
    assert matches[0].fit_rmsd > 1e-6


def test_interpolate_crystal_vc_preserves_pbc_and_molecule_count():
    ca, cb, _ = _vc_pair_with_internal_stretch()
    n_mol = len(ca.molecules)
    frames = interpolate_crystal_vc(ca, cb, n_images=4)
    for frame in frames:
        assert len(frame.molecules) == n_mol
        assert tuple(frame.pbc) == tuple(ca.pbc)


def test_interpolate_crystal_vc_static_molecule_fractional_tracking():
    lattice_a = np.eye(3) * 18.0
    lattice_b = np.eye(3) * 20.0
    w_move = _water(_water().get_positions() + np.array([2.0, 0.0, 0.0]))
    w_static = _water(_water().get_positions() + np.array([9.0, 0.0, 0.0]))
    ca = MolecularCrystal(lattice_a, [w_move, w_static], pbc=(True, True, True))
    wb = _water(w_move.get_positions() + np.array([0.5, 0.3, 0.0]))
    cb = MolecularCrystal(lattice_b, [wb, w_static], pbc=(True, True, True))
    frames = interpolate_crystal_vc(ca, cb, molecule_indices=[0], n_images=3)
    static_mid = frames[1].molecules[1].get_positions()
    static_a = ca.molecules[1].get_positions()
    assert not np.allclose(static_mid, static_a, atol=0.01)


def test_copy_crystal_with_molecule_positions_shallow_metadata():
    crystal_a, crystal_b = _rigid_water_pair()
    crystal_a.metadata = {"tag": "a"}
    new_pos = crystal_b.molecules[0].get_positions()
    frame = copy_crystal_with_molecule_positions(crystal_a, {0: new_pos})
    assert frame.metadata == crystal_a.metadata
    frame.metadata["tag"] = "mutated"
    assert crystal_a.metadata["tag"] == "a"


def test_match_molecules_unmatched_formula_raises():
    lattice = np.eye(3) * 12.0
    ca = MolecularCrystal(lattice, [_water()], pbc=(True, True, True))
    cb = MolecularCrystal(
        lattice,
        [CrystalMolecule(Atoms("HH", positions=np.zeros((2, 3)), pbc=False), check_pbc=False)],
        pbc=(True, True, True),
    )
    with pytest.raises(ValueError, match="No unmatched molecule"):
        match_molecules(ca, cb)