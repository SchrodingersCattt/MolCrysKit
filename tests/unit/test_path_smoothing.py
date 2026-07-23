"""Tests for geometric IDPP path smoothing."""

import numpy as np

from ase import Atoms as AseAtoms

from molcrys_kit.operations.path_smoothing import (
    _idpp_energy_and_gradient,
    _pair_indices,
    smooth_interpolation_idpp,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.utils.geometry import minimum_image_distance


def _crystal_from_positions(positions, lattice):
    atoms = AseAtoms("HHH", positions=np.asarray(positions, dtype=float), cell=lattice, pbc=True)
    mol = CrystalMolecule(atoms=atoms, check_pbc=False)
    return MolecularCrystal(np.asarray(lattice, dtype=float), [mol], (True, True, True))


def _crystal_from_scaled_positions(scaled_positions, lattice):
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(scaled_positions, dtype=float) @ lattice
    return _crystal_from_positions(positions, lattice)


def _min_distance(crystal):
    atoms = crystal.to_ase()
    frac = atoms.get_scaled_positions()
    lattice = np.asarray(crystal.lattice, dtype=float)
    dmin = float("inf")
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = minimum_image_distance(frac[i], frac[j], lattice)
            dmin = min(dmin, d)
    return dmin


def test_fixed_cell_clash_improves():
    lattice = np.eye(3) * 10.0
    start = _crystal_from_positions([[1, 1, 1], [5, 5, 5], [8, 8, 8]], lattice)
    middle = _crystal_from_positions([[1, 1, 1], [1.05, 1, 1], [8, 8, 8]], lattice)
    end = _crystal_from_positions([[1, 1, 1], [8, 8, 8], [5, 5, 5]], lattice)

    before = _min_distance(middle)
    smoothed = smooth_interpolation_idpp([start, middle, end], max_steps=300, step_size=0.005)
    after = _min_distance(smoothed[1])

    assert before < 0.1
    assert after > before


def test_variable_cell_clash_improves_and_cells_unchanged():
    lat0 = np.eye(3) * 10.0
    lat1 = np.eye(3) * 11.0
    lat2 = np.eye(3) * 12.0
    start = _crystal_from_positions([[1, 1, 1], [5, 5, 5], [8, 8, 8]], lat0)
    middle = _crystal_from_positions([[1, 1, 1], [1.05, 1, 1], [8, 8, 8]], lat1)
    end = _crystal_from_positions([[1, 1, 1], [8, 8, 8], [5, 5, 5]], lat2)

    middle_lattice_before = np.asarray(middle.lattice).copy()
    before = _min_distance(middle)
    smoothed = smooth_interpolation_idpp([start, middle, end], max_steps=300, step_size=0.005)
    after = _min_distance(smoothed[1])

    np.testing.assert_allclose(smoothed[1].lattice, middle_lattice_before)
    assert after > before


def test_non_orthogonal_cell_clash_improves():
    lat0 = np.array([[10.0, 1.0, 0.2], [0.0, 11.0, 0.3], [0.0, 0.0, 12.0]])
    lat1 = np.array([[10.5, 1.1, 0.2], [0.0, 11.5, 0.4], [0.0, 0.0, 12.5]])
    lat2 = np.array([[11.0, 1.2, 0.3], [0.0, 12.0, 0.5], [0.0, 0.0, 13.0]])
    start = _crystal_from_scaled_positions([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], lat0)
    middle = _crystal_from_scaled_positions([[0.1, 0.1, 0.1], [0.105, 0.1, 0.1], [0.8, 0.8, 0.8]], lat1)
    end = _crystal_from_scaled_positions([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5]], lat2)

    before = _min_distance(middle)
    smoothed = smooth_interpolation_idpp([start, middle, end])
    after = _min_distance(smoothed[1])

    assert before < 0.1
    assert after > before


def test_default_parameters_improve_clash():
    lattice = np.eye(3) * 10.0
    start = _crystal_from_positions([[1, 1, 1], [5, 5, 5], [8, 8, 8]], lattice)
    middle = _crystal_from_positions([[1, 1, 1], [1.05, 1, 1], [8, 8, 8]], lattice)
    end = _crystal_from_positions([[1, 1, 1], [8, 8, 8], [5, 5, 5]], lattice)

    before = _min_distance(middle)
    smoothed = smooth_interpolation_idpp([start, middle, end])
    after = _min_distance(smoothed[1])

    assert after > before


def test_idpp_gradient_matches_finite_difference_non_orthogonal():
    lattice = np.array([[5.0, 0.5, 0.2], [0.0, 6.0, 0.3], [0.0, 0.0, 7.0]])
    frac = np.array([[0.12, 0.23, 0.34], [0.41, 0.52, 0.63], [0.74, 0.25, 0.36]])
    pairs = _pair_indices(3)
    target = np.array([2.8, 3.4, 4.1])
    weights = 1.0 / target**4
    _, grad = _idpp_energy_and_gradient(frac, lattice, pairs, target, weights)

    eps = 1e-6
    numerical = np.zeros_like(frac)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            plus = frac.copy()
            minus = frac.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            e_plus, _ = _idpp_energy_and_gradient(plus, lattice, pairs, target, weights)
            e_minus, _ = _idpp_energy_and_gradient(minus, lattice, pairs, target, weights)
            numerical[i, j] = (e_plus - e_minus) / (2 * eps)

    np.testing.assert_allclose(grad, numerical, rtol=1e-5, atol=1e-7)


def test_endpoints_unchanged():
    lattice = np.eye(3) * 10.0
    start = _crystal_from_positions([[1, 1, 1], [5, 5, 5], [8, 8, 8]], lattice)
    middle = _crystal_from_positions([[1, 1, 1], [1.05, 1, 1], [8, 8, 8]], lattice)
    end = _crystal_from_positions([[1, 1, 1], [8, 8, 8], [5, 5, 5]], lattice)

    start_pos = start.to_ase().get_positions().copy()
    end_pos = end.to_ase().get_positions().copy()
    smoothed = smooth_interpolation_idpp([start, middle, end])

    np.testing.assert_allclose(smoothed[0].lattice, start.lattice)
    np.testing.assert_allclose(smoothed[-1].lattice, end.lattice)
    np.testing.assert_allclose(smoothed[0].to_ase().get_positions(), start_pos)
    np.testing.assert_allclose(smoothed[-1].to_ase().get_positions(), end_pos)


def test_image_count_preserved():
    lattice = np.eye(3) * 10.0
    images = [
        _crystal_from_positions([[1, 1, 1], [5, 5, 5], [8, 8, 8]], lattice),
        _crystal_from_positions([[1, 1, 1], [1.05, 1, 1], [8, 8, 8]], lattice),
        _crystal_from_positions([[1, 1, 1], [8, 8, 8], [5, 5, 5]], lattice),
    ]
    smoothed = smooth_interpolation_idpp(images)
    assert len(smoothed) == len(images)
