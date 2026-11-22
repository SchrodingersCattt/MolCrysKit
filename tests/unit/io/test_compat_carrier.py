"""Parity checks for the private core-install structure carrier."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms as AseAtoms
from ase.geometry import get_distances as ase_get_distances
from ase.neighborlist import neighbor_list as ase_neighbor_list

from molcrys_kit.compat.atoms import Atoms, get_distances, neighbor_list
from molcrys_kit.compat.symmetry import SymmOp

_SYMBOLS = ["H", "C", "O", "N"]
_POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 1.4, 0.3], [0.2, 1.0, 2.5]],
    dtype=float,
)
_CELL = np.diag([10.0, 11.0, 12.0])


def _pair(cell=None, pbc=True, positions=_POSITIONS, symbols=_SYMBOLS):
    kwargs = {
        "symbols": symbols,
        "positions": np.asarray(positions, dtype=float),
        "cell": _CELL if cell is None else cell,
        "pbc": pbc,
    }
    return Atoms(**kwargs), AseAtoms(**kwargs)


def _assert_same_atoms(carrier: Atoms, reference: AseAtoms) -> None:
    assert carrier.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(carrier.get_positions(), reference.positions)


def test_boolean_mask_selects_true_positions_in_order():
    carrier, reference = _pair()
    mask = [True, False, True, False]
    _assert_same_atoms(carrier[mask], reference[mask])
    _assert_same_atoms(carrier[np.array(mask)], reference[np.array(mask)])


def test_integer_array_slice_and_scalar_indexing_match_ase():
    carrier, reference = _pair()
    order = np.array([2, 0, 2])
    _assert_same_atoms(carrier[order], reference[order])
    _assert_same_atoms(carrier[1:4:2], reference[1:4:2])
    chosen = carrier[-1]
    assert chosen.get_chemical_symbols() == [reference[-1].symbol]
    np.testing.assert_allclose(chosen.get_positions()[0], reference[-1].position)


def test_invalid_indexes_match_ase_errors():
    carrier, reference = _pair()
    with pytest.raises(IndexError, match="Length of mask"):
        carrier[[True, False]]
    with pytest.raises(IndexError, match="Length of mask"):
        reference[[True, False]]
    with pytest.raises(IndexError, match="integer"):
        carrier[np.array([0.5, 1.2])]
    with pytest.raises(IndexError, match="Index out of range"):
        carrier[5]
    with pytest.raises(IndexError, match="Index out of range"):
        reference[5]
    with pytest.raises(IndexError, match="atom index must be integers"):
        carrier[1.5]
    with pytest.raises(IndexError, match="atom index must be integers"):
        carrier[np.float64(1.5)]


def test_scaled_positions_match_ase_cell_completion():
    positions = np.array([[1.5, -2.0, 0.25], [0.0, 0.2, 1.0]])
    cases = [
        (np.zeros((3, 3)), False, True),
        (np.zeros((3, 3)), True, True),
        (np.zeros((3, 3)), True, False),
        (np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]), True, False),
        (np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), True, True),
        (np.diag([3.0, 4.0, 5.0]), True, True),
    ]
    for cell, pbc, wrap in cases:
        carrier, reference = _pair(
            cell=cell, pbc=pbc, positions=positions, symbols=["H", "He"]
        )
        np.testing.assert_allclose(
            carrier.get_scaled_positions(wrap=wrap),
            reference.get_scaled_positions(wrap=wrap),
            atol=1e-12,
        )


def test_singular_nonzero_cell_is_not_replaced_by_zeros():
    cell = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    carrier, reference = _pair(cell=cell, positions=[[0.2, 0.3, 0.4]], symbols=["H"])
    with pytest.raises(np.linalg.LinAlgError):
        carrier.get_scaled_positions()
    with pytest.raises(np.linalg.LinAlgError):
        reference.get_scaled_positions()


def test_symmetry_parser_matches_affine_operations_and_rejects_eval():
    identity = SymmOp.from_xyz_str("X, Y, Z")
    np.testing.assert_allclose(identity.rotation_matrix, np.eye(3))
    np.testing.assert_allclose(identity.translation_vector, np.zeros(3))

    screw = SymmOp.from_xyz_str("-x,-y,z+1/2")
    np.testing.assert_allclose(screw.rotation_matrix, np.diag([-1.0, -1.0, 1.0]))
    np.testing.assert_allclose(screw.translation_vector, [0.0, 0.0, 0.5])

    hexagonal = SymmOp.from_xyz_str("x-y,x,z+1/3")
    np.testing.assert_allclose(
        hexagonal.rotation_matrix, [[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(hexagonal.translation_vector, [0.0, 0.0, 1.0 / 3.0])

    wrapped = SymmOp.from_xyz_str("x+1,y,z")
    np.testing.assert_allclose(wrapped.translation_vector, np.zeros(3))

    grouped = SymmOp.from_xyz_str("(x+1/2),-y,z")
    np.testing.assert_allclose(grouped.rotation_matrix, np.diag([1.0, -1.0, 1.0]))
    np.testing.assert_allclose(grouped.translation_vector, [0.5, 0.0, 0.0])
    negated = SymmOp.from_xyz_str("-(x+y),x,z+1/2")
    np.testing.assert_allclose(
        negated.rotation_matrix, [[-1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(negated.translation_vector, [0.0, 0.0, 0.5])

    for expression in ("2**1000000000,y,z", "x*y,y,z", "x**2,y,z", "2*x,y,z", "x,x,z"):
        with pytest.raises(ValueError, match="invalid symmetry operation"):
            SymmOp.from_xyz_str(expression)


def test_vendored_space_group_operations_are_crystallographic():
    table = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "molcrys_kit/compat/data/symm_ops.json"
        ).read_text(encoding="utf-8")
    )
    seen = 0
    for entry in table:
        for operation in entry["symops"]:
            parsed = SymmOp.from_xyz_str(operation)
            determinant = round(float(np.linalg.det(parsed.rotation_matrix)))
            assert determinant in (-1, 1)
            assert np.all(np.isin(np.rint(parsed.rotation_matrix), [-1, 0, 1]))
            seen += 1
    assert seen > 1000


def _exhaustive_minimum(delta, cell, pbc):
    """Shortest image inside the interplanar bound around the raw vector.

    Any lattice image shorter than ``delta`` has fractional component
    ``|f_k| < ||delta|| / h_k`` along each periodic axis, where ``h_k`` is
    the spacing between lattice planes. The box below contains that image.
    """
    delta = np.asarray(delta, dtype=float)
    cell = np.asarray(cell, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    ranges = []
    raw = float(np.linalg.norm(delta))
    for axis in range(3):
        if not periodic[axis]:
            ranges.append([0])
            continue
        other = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        height = abs(float(np.dot(cell[axis], other))) / (
            float(np.linalg.norm(other)) + 1e-15
        )
        count = int(np.ceil(raw / max(height, 1e-8))) + 1
        ranges.append(range(-count, count + 1))
    best_vector = delta.copy()
    best_length = raw
    for shift in itertools.product(*ranges):
        vector = delta + np.asarray(shift, dtype=float) @ cell
        length = float(np.linalg.norm(vector))
        if length < best_length:
            best_vector = vector
            best_length = length
    return best_vector


def test_get_distance_uses_minimum_image_on_the_carrier():
    carrier, _reference = _pair(
        positions=np.array([[0.0, 0.0, 0.0], [20.5, 0.0, 0.0]]),
        symbols=["C", "C"],
    )
    vector = carrier.get_distance(0, 1, mic=True, vector=True)
    np.testing.assert_allclose(vector, [0.5, 0.0, 0.0])
    assert carrier.get_distance(0, 1, mic=True) == pytest.approx(0.5)
    np.testing.assert_allclose(
        carrier.get_distance(0, 1, mic=False, vector=True), [20.5, 0.0, 0.0]
    )


def test_neighbor_list_finds_contact_several_cells_away():
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.0, 0.0, 0.0], [20.5, 0.0, 0.0]])
    atoms = Atoms(symbols=["C", "C"], positions=positions, cell=cell, pbc=True)
    indices_i, indices_j, distances, shifts = neighbor_list("ijdS", atoms, 1.0)
    found = {
        (int(left), int(right)): (
            float(distance),
            tuple(int(value) for value in shift),
        )
        for left, right, distance, shift in zip(indices_i, indices_j, distances, shifts)
    }
    assert found[(0, 1)][0] == pytest.approx(0.5)
    assert found[(0, 1)][1] == (-2, 0, 0)
    assert found[(1, 0)][1] == (2, 0, 0)
    reference = AseAtoms(symbols=["C", "C"], positions=positions, cell=cell, pbc=True)
    ref_i, ref_j, ref_d, ref_s = ase_neighbor_list("ijdS", reference, 1.0)
    assert sorted(zip(ref_i.tolist(), ref_j.tolist(), ref_s.tolist())) == sorted(
        zip(indices_i.tolist(), indices_j.tolist(), shifts.tolist())
    )
    np.testing.assert_allclose(sorted(distances), sorted(ref_d))


def test_neighbor_images_follow_individual_and_collective_lattice_translations():
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.0, 0.0], [0.2, -0.4, 5.0]], dtype=float)
    base = np.array([[0.2, 0.1, 0.0], [1.2, 2.4, 0.3], [3.5, 0.2, 4.0]], dtype=float)
    cutoff = 3.0

    def pairs(positions):
        atoms = Atoms(symbols=["C", "C", "C"], positions=positions, cell=cell, pbc=True)
        indices_i, indices_j, _distances, vectors, shifts = neighbor_list(
            "ijdDS", atoms, cutoff
        )
        found = {}
        for left, right, vector, shift in zip(indices_i, indices_j, vectors, shifts):
            shift = np.asarray(shift, dtype=int)
            reconstructed = positions[right] + shift @ cell - positions[left]
            np.testing.assert_allclose(reconstructed, vector, atol=1e-8)
            found[(int(left), int(right), tuple(int(value) for value in shift))] = (
                np.asarray(vector, dtype=float)
            )
        return found

    original = pairs(base)
    assert original
    translation = np.array([2, -1, 3])
    moved = np.array(base, copy=True)
    moved[1] = moved[1] + translation @ cell
    shifted = pairs(moved)
    for (left, right, shift), vector in original.items():
        updated = np.asarray(shift, dtype=int)
        if right == 1:
            updated = updated - translation
        if left == 1:
            updated = updated + translation
        key = (left, right, tuple(int(value) for value in updated))
        assert key in shifted
        np.testing.assert_allclose(shifted[key], vector, atol=1e-8)

    collective = base + np.array([-1, 2, 1], dtype=float) @ cell
    again = pairs(collective)
    assert set(again) == set(original)
    for key, vector in original.items():
        np.testing.assert_allclose(again[key], vector, atol=1e-8)


def test_get_distances_matches_exhaustive_search_on_skew_cells():
    cells = [
        np.diag([10.0, 11.0, 12.0]),
        np.array([[1.0, 0.0, 0.0], [0.6, 0.2, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[4.0, 0.0, 0.0], [1.5, 3.0, 0.0], [0.2, -0.4, 5.0]]),
    ]
    masks = [
        np.array([True, True, True]),
        np.array([True, True, False]),
        np.array([True, False, True]),
    ]
    points = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [2.2, -0.4, 0.7], [20.5, 1.0, -3.0]]
    )
    for cell in cells:
        for mask in masks:
            vectors, distances = get_distances(
                points[:2], points[2:], cell=cell, pbc=mask
            )
            for row, origin in enumerate(points[:2]):
                for column, target in enumerate(points[2:]):
                    expected = _exhaustive_minimum(target - origin, cell, mask)
                    np.testing.assert_allclose(
                        vectors[row, column], expected, atol=1e-8
                    )
                    assert distances[row, column] == pytest.approx(
                        float(np.linalg.norm(expected))
                    )


def test_get_distances_matches_ase_for_full_periodicity_and_is_not_longer():
    generator = np.random.default_rng(0)
    for _trial in range(12):
        factor = np.linalg.qr(generator.normal(size=(3, 3)))[0]
        cell = factor * generator.uniform(2.0, 6.0, size=3)
        cell = cell + generator.normal(scale=0.2, size=(3, 3))
        if abs(float(np.linalg.det(cell))) < 0.3:
            continue
        points = generator.normal(scale=8.0, size=(4, 3))
        full_vectors, full_distances = get_distances(
            points[:2], points[2:], cell=cell, pbc=True
        )
        ase_vectors, ase_distances = ase_get_distances(
            points[:2], points[2:], cell=cell, pbc=True
        )
        np.testing.assert_allclose(full_distances, ase_distances, atol=1e-6)
        np.testing.assert_allclose(full_vectors, ase_vectors, atol=1e-6)
        partial = generator.random(3) > 0.25
        if not np.any(partial):
            partial[0] = True
        _vectors, distances = get_distances(
            points[:2], points[2:], cell=cell, pbc=partial
        )
        _ase_vectors, ase_partial = ase_get_distances(
            points[:2], points[2:], cell=cell, pbc=partial
        )
        assert np.all(distances <= ase_partial + 1e-6)
