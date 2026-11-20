"""Independent numerical and interface checks for IDPP refinement."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian
from ase.mep.neb import IDPP

from molcrys_kit.operations import refine_idpp_path
from molcrys_kit.operations.idpp import _objective


@pytest.mark.parametrize(
    "cell,pbc",
    [
        (np.diag([9.0, 10.0, 11.0]), (False, False, False)),
        (np.array([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 10.0]]), (True, True, True)),
        (np.diag([10.0, 0.0, 0.0]), (True, False, False)),
    ],
)
def test_objective_and_gradient_agree_with_ase_reference(cell, pbc):
    atoms = Atoms(
        "CHOH",
        positions=[[0.3, 0.4, 0.5], [1.7, 0.7, 0.9], [2.2, 1.8, 1.3], [3.2, 2.1, 0.2]],
        cell=cell,
        pbc=pbc,
    )
    target = atoms.get_all_distances(mic=True) * 1.07
    first, second = np.triu_indices(len(atoms), 1)
    value, grad = _objective(
        atoms.positions, atoms.cell, atoms.pbc, first, second, target[first, second], 1e-8
    )
    atoms.calc = IDPP(target, mic=True)
    assert value == pytest.approx(atoms.get_potential_energy(), abs=1e-11)
    np.testing.assert_allclose(grad, -atoms.get_forces(), atol=1e-10, rtol=1e-10)


def _images():
    a = Atoms("NHOCl", positions=[[0, 0, 0], [1.0, 0.5, 0], [3.0, 0, 0], [4.7, 0, 0]])
    middle = Atoms("NHOCl", positions=[[0.1, 0, 0], [1.1, 1.4, 0], [3.1, 0, 0], [4.9, 0, 0]])
    b = Atoms("NHOCl", positions=[[0.2, 0, 0], [2.0, 0.5, 0], [3.2, 0, 0], [5.0, 0, 0]])
    return [a, middle, b]


def test_prescribed_moving_fragment_and_free_atom_preserve_inputs():
    images = _images()
    before = [a.positions.copy() for a in images]
    out = refine_idpp_path(images, prescribed_atom_indices=(0, 2, 3))
    for actual, expected in zip(images, before):
        np.testing.assert_array_equal(actual.positions, expected)
    np.testing.assert_array_equal(out.images[0].positions, before[0])
    np.testing.assert_array_equal(out.images[-1].positions, before[-1])
    np.testing.assert_array_equal(out.images[1].positions[[0, 2, 3]], before[1][[0, 2, 3]])
    assert not np.allclose(out.images[1].positions[1], before[1][1])
    assert out.diagnostics[0]["final_objective"] < out.diagnostics[0]["initial_objective"]
    assert out.diagnostics[0]["prescribed_atom_count"] == 3


def test_ase_fixed_atoms_and_all_prescribed():
    images = _images()
    images[1].set_constraint(FixAtoms(indices=[0, 2]))
    out = refine_idpp_path(images, prescribed_atom_indices=(1, 3))
    np.testing.assert_array_equal(out.images[1].positions, images[1].positions)
    assert out.converged
    assert out.diagnostics[0]["iterations"] == 0
    assert out.diagnostics[0]["message"] == "all atoms prescribed"


def test_periodic_rewrapping_is_equivalent_in_skew_cell():
    cell = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    images = _images()
    for frame in images:
        frame.set_cell(cell)
        frame.set_pbc(True)
    shifted = [frame.copy() for frame in images]
    for frame in shifted:
        frame.positions[1] += cell[0] - cell[1]
    a = refine_idpp_path(images, prescribed_atom_indices=(0, 2, 3))
    b = refine_idpp_path(shifted, prescribed_atom_indices=(0, 2, 3))
    from ase.geometry import find_mic

    delta, _ = find_mic(a.images[1].positions - b.images[1].positions, cell, pbc=True)
    np.testing.assert_allclose(delta, 0, atol=1e-8)


def test_nonconvergence_is_explicit():
    out = refine_idpp_path(_images(), max_iterations=1, gradient_tolerance=1e-12)
    assert not out.converged
    assert not out.diagnostics[0]["converged"]


def test_reject_overlap_and_unsupported_constraint():
    images = _images()
    images[1].positions[1] = images[1].positions[0]
    with pytest.raises(ValueError, match="overlapping"):
        refine_idpp_path(images)
    images = _images()
    images[1].set_constraint(FixCartesian(0, mask=(True, False, False)))
    with pytest.raises(ValueError, match="FixAtoms"):
        refine_idpp_path(images)


@pytest.mark.parametrize("indices", [(-1,), (0, 0), (True,), (8,), (1.2,)])
def test_reject_invalid_prescribed_indices(indices):
    with pytest.raises(ValueError, match="prescribed atom"):
        refine_idpp_path(_images(), prescribed_atom_indices=indices)


def test_reject_inconsistent_atom_ids_and_path_fractions():
    images = _images()
    for frame in images:
        frame.set_array("atom_id", np.arange(4))
    images[1].arrays["atom_id"][[0, 1]] = images[1].arrays["atom_id"][[1, 0]]
    with pytest.raises(ValueError, match="atom_id"):
        refine_idpp_path(images)
    with pytest.raises(ValueError, match="fractions"):
        refine_idpp_path(_images(), fractions=[0.0, 0.9, 0.8])


@pytest.mark.parametrize("pbc", [(True, True, False), (True, True, True)])
def test_skew_cell_minimum_image_matches_explicit_lattice_search(pbc):
    from itertools import product
    from molcrys_kit.operations.idpp import _distances

    cell = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    displacement = np.array([0.49, 0.49, 0]) @ cell
    positions = np.array([[0.0, 0.0, 0.0], displacement])
    _, distances = _distances(positions, cell, pbc, np.array([0]), np.array([1]))
    expected = min(
        np.linalg.norm(displacement + np.array([i, j, 0]) @ cell)
        for i, j in product(range(-2, 3), repeat=2)
    )
    assert distances[0] == pytest.approx(expected, abs=1e-12)


def test_skew_periodic_gradient_matches_finite_difference():
    cell = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    positions = np.array([[0.0, 0.0, 0.0], [9.31, 0.49, 0.0], [2.1, 0.2, 0.7]])
    first, second = np.triu_indices(len(positions), 1)
    targets = np.array([0.7, 2.0, 2.3])
    _, gradient = _objective(positions, cell, (True, True, False), first, second, targets, 1e-8)
    numerical = np.zeros_like(positions)
    step = 1e-6
    for i in range(len(positions)):
        for j in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i, j] += step
            minus[i, j] -= step
            numerical[i, j] = (
                _objective(plus, cell, (True, True, False), first, second, targets, 1e-8)[0]
                - _objective(minus, cell, (True, True, False), first, second, targets, 1e-8)[0]
            ) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-6)


def test_variable_cells_preserve_cells_and_requested_progress():
    images = [
        Atoms("CH", positions=[[1, 1, 1], [3, 1, 1]], cell=np.diag([8.0, 9, 10]), pbc=True),
        Atoms("CH", positions=[[1, 1, 1], [4.5, 1, 1]], cell=np.diag([9.0, 9, 10]), pbc=True),
        Atoms("CH", positions=[[1, 1, 1], [4, 1, 1]], cell=np.diag([10.0, 9, 10]), pbc=True),
    ]
    result = refine_idpp_path(
        images, fractions=[0.0, 0.3, 1.0], prescribed_atom_indices=(0,), gradient_tolerance=1e-5
    )
    assert result.converged
    assert result.images[1].get_distance(0, 1, mic=True) == pytest.approx(2.3, abs=1e-4)
    for source, frame in zip(images, result.images):
        np.testing.assert_array_equal(source.cell.array, frame.cell.array)
    np.testing.assert_array_equal(result.images[1].positions[0], images[1].positions[0])
