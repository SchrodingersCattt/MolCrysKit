"""Generic crystallographic affine-operation and coset tests."""

import numpy as np
import pytest

from molcrys_kit.constants.symmetry_path import (
    AFFINE_EQUIVALENCE_TOLERANCE,
    AFFINE_ORTHOGONALITY_TOLERANCE,
)
from molcrys_kit.structures.symmetry import (
    FractionalAffineOperation,
    LatticeBasisChange,
    domain_representatives,
    left_cosets,
    validate_subgroup,
)


def op(rotation, translation=(0, 0, 0)):
    return FractionalAffineOperation(rotation, translation)


def test_fractional_affine_uses_row_coordinate_convention():
    operation = op(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [0, 0.5, 0],
    )
    result = operation.apply([[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(result, [[-0.1, 0.7, -0.3]])


def test_affine_compose_inverse_and_translation_equivalence():
    operation = op(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [0, 0.5, 0],
    )
    identity = operation.compose(operation.inverse())
    assert identity.equivalent_to(op(np.eye(3)))
    assert operation.equivalent_to(
        op(operation.rotation, operation.translation + [1, -2, 3]),
        tolerance=AFFINE_EQUIVALENCE_TOLERANCE,
    )


def test_operation_validates_metric_in_triclinic_cell():
    lattice = np.array([[5.0, 0.0, 0.0], [1.2, 6.0, 0.0], [0.3, 0.4, 7.0]])
    identity = op(np.eye(3))
    identity.validate_metric(lattice)
    np.testing.assert_allclose(
        identity.cartesian_linear(lattice),
        np.eye(3),
        atol=AFFINE_ORTHOGONALITY_TOLERANCE,
    )


def test_non_metric_operation_rejected_for_given_lattice():
    lattice = np.diag([5.0, 6.0, 7.0])
    swap_xy = op([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="metric"):
        swap_xy.validate_metric(lattice)


def test_unimodular_basis_round_trip_and_operation_conjugation():
    change = LatticeBasisChange([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    fractional = np.array([[0.2, 0.3, 0.4]])
    np.testing.assert_allclose(
        change.new_to_old_fractional(change.old_to_new_fractional(fractional)),
        fractional,
    )
    inversion = op(-np.eye(3), [0.25, 0.5, 0.75])
    transformed = change.transform_operation(inversion)
    old_result = inversion.apply(fractional)
    new_result = transformed.apply(change.old_to_new_fractional(fractional))
    np.testing.assert_allclose(change.new_to_old_fractional(new_result), old_result)


def test_non_unimodular_basis_change_is_explicitly_rejected():
    with pytest.raises(ValueError, match="unimodular"):
        LatticeBasisChange(np.diag([2, 1, 1]))


def test_general_left_cosets_do_not_depend_on_operation_indices():
    identity = op(np.eye(3))
    screw = op([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0.5, 0])
    inversion = op(-np.eye(3))
    mirror = op([[1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, -0.5, 0])
    group = (mirror, identity, inversion, screw)  # deliberately noncanonical order
    subgroup = (screw, identity)

    validate_subgroup(group, subgroup)
    cosets = left_cosets(group, subgroup)
    assert len(cosets) == 2
    assert all(len(coset) == 2 for coset in cosets)
    assert any(any(item.equivalent_to(identity) for item in coset) for coset in cosets)
    representatives = domain_representatives(group, subgroup)
    assert len(representatives) == 2
    assert any(representative.is_improper for representative in representatives)


def test_invalid_rotation_matrix_is_rejected():
    with pytest.raises(ValueError, match="integer-valued"):
        op([[0.5, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="determinant"):
        op([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
