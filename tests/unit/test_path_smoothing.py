"""Tests for geometric IDPP path smoothing."""

import numpy as np

from ase import Atoms as AseAtoms

from molcrys_kit.operations.path_smoothing import smooth_interpolation_idpp
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.utils.geometry import minimum_image_distance


def _crystal_from_positions(positions, lattice):
    atoms = AseAtoms("HHH", positions=np.asarray(positions, dtype=float), cell=lattice, pbc=True)
    mol = CrystalMolecule(atoms=atoms, check_pbc=False)
    return MolecularCrystal(np.asarray(lattice, dtype=float), [mol], (True, True, True))


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
