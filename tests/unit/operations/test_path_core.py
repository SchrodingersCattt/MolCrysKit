"""Contracts for private path-generation primitives."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations._path_core import (
    InterpolationMethod,
    interpolate_rigid_positions,
    materialize_atoms_frame,
    minimum_image_displacement,
    path_lambda_values,
)
from molcrys_kit.utils.geometry import get_rotation_matrix


def test_lambda_values_use_frame_count_semantics():
    np.testing.assert_allclose(path_lambda_values(3, True), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(path_lambda_values(3, False), [0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="integer"):
        path_lambda_values(2.5, True)


def test_minimum_image_respects_partial_pbc():
    vector, shift = minimum_image_displacement(
        np.array([-8.0, -8.0, 0.0]),
        np.eye(3) * 10.0,
        (False, True, False),
    )
    np.testing.assert_allclose(vector, [-8.0, 2.0, 0.0])
    assert shift == (0, 1, 0)


@pytest.mark.parametrize("method", list(InterpolationMethod))
def test_rigid_interpolation_preserves_exact_endpoints(method):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    center = np.mean(positions, axis=0)
    rotation = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    translation = np.array([2.0, -1.0, 0.5])
    expected = (positions - center) @ rotation.T + center + translation

    start = interpolate_rigid_positions(
        positions,
        center=center,
        rotation=rotation,
        translation=translation,
        lam=0.0,
        method=method,
    )
    end = interpolate_rigid_positions(
        positions,
        center=center,
        rotation=rotation,
        translation=translation,
        lam=1.0,
        method=method,
    )
    np.testing.assert_allclose(start, positions, atol=1.0e-12)
    np.testing.assert_allclose(end, expected, atol=1.0e-12)


def test_materialized_frame_preserves_payloads_without_aliasing():
    reference = Atoms("HH", positions=np.zeros((2, 3)))
    reference.info = {"nested": {"labels": ["source"]}}
    reference.set_array("atom_id", np.array([4, 5]))
    reference.set_array("frac_x", np.zeros(2))

    frame = materialize_atoms_frame(
        reference,
        np.ones((2, 3)),
        info_updates={"path_lambda": 0.5},
    )

    np.testing.assert_allclose(frame.positions, np.ones((2, 3)))
    np.testing.assert_array_equal(frame.arrays["atom_id"], [4, 5])
    assert "frac_x" not in frame.arrays
    assert frame.info["path_lambda"] == 0.5
    frame.info["nested"]["labels"].append("frame")
    assert reference.info["nested"]["labels"] == ["source"]
