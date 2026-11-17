"""Parity checks for the private core-install structure carrier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms as AseAtoms

from molcrys_kit.compat.atoms import Atoms
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
