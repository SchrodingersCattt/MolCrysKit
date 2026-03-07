"""
Unit tests for molcrys_kit.utils.geometry.
"""

import numpy as np
import pytest

from molcrys_kit.utils.geometry import (
    frac_to_cart,
    cart_to_frac,
    normalize_vector,
    distance_between_points,
    angle_between_vectors,
    dihedral_angle,
    minimum_image_distance,
    minimum_image_vector,
    volume_of_cell,
    calculate_center_of_mass,
    rotate_vector,
    get_rotation_matrix,
    get_missing_vectors,
    reduce_surface_lattice,
    calculate_dihedral_and_adjustment,
)


class TestFracCart:
    """Fractional/cartesian coordinate conversion."""

    def test_frac_to_cart_center(self, cubic_lattice_5):
        cart = frac_to_cart(np.array([0.5, 0.5, 0.5]), cubic_lattice_5)
        np.testing.assert_allclose(cart, [2.5, 2.5, 2.5])

    def test_cart_to_frac_center(self, cubic_lattice_5):
        frac = cart_to_frac(np.array([2.5, 2.5, 2.5]), cubic_lattice_5)
        np.testing.assert_allclose(frac, [0.5, 0.5, 0.5])

    def test_roundtrip(self, cubic_lattice_5):
        frac = np.array([0.1, 0.2, 0.3])
        cart = frac_to_cart(frac, cubic_lattice_5)
        back = cart_to_frac(cart, cubic_lattice_5)
        np.testing.assert_allclose(back, frac)

    def test_rectangular_lattice(self):
        lattice = np.diag([10.0, 20.0, 30.0])
        frac = np.array([0.1, 0.2, 0.3])
        cart = frac_to_cart(frac, lattice)
        np.testing.assert_allclose(cart, [1.0, 4.0, 9.0])


class TestNormalizeVector:
    """Vector normalization."""

    def test_unit_axis(self):
        n = normalize_vector(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0])

    def test_zero_vector(self):
        n = normalize_vector(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(n, [0.0, 0.0, 0.0])

    def test_diagonal_unit_magnitude(self):
        n = normalize_vector(np.array([1.0, 1.0, 1.0]))
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10


class TestDistanceAngle:
    """Distance and angle helpers."""

    def test_distance_axial(self):
        d = distance_between_points(np.array([0, 0, 0]), np.array([3, 0, 0]))
        assert abs(d - 3.0) < 1e-10

    def test_distance_diagonal(self):
        d = distance_between_points(np.array([0, 0, 0]), np.array([1.0, 1.0, 1.0]))
        assert abs(d - np.sqrt(3.0)) < 1e-10

    def test_angle_parallel(self):
        a = angle_between_vectors(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))
        assert abs(a) < 1e-10

    def test_angle_perpendicular(self):
        a = angle_between_vectors(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        assert abs(a - np.pi / 2) < 1e-10

    def test_angle_antiparallel(self):
        a = angle_between_vectors(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))
        assert abs(a - np.pi) < 1e-10

    def test_dihedral_finite(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 1.0])
        d = dihedral_angle(p1, p2, p3, p4)
        assert np.isfinite(d)

    def test_dihedral_trans(self):
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        d = dihedral_angle(p1, p2, p3, p4)
        assert np.isfinite(d)
        assert abs(d) <= np.pi + 1e-10


class TestMinimumImage:
    """Minimum image distance."""

    def test_close_points(self, cubic_lattice_10):
        f1, f2 = np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2])
        d = minimum_image_distance(f1, f2, cubic_lattice_10)
        direct = np.linalg.norm(frac_to_cart(f1 - f2, cubic_lattice_10))
        assert abs(d - direct) < 1e-10

    def test_wrap_opposite_edges(self, cubic_lattice_10):
        f1 = np.array([0.1, 0.1, 0.1])
        f2 = np.array([0.9, 0.9, 0.9])
        d = minimum_image_distance(f1, f2, cubic_lattice_10)
        expected = np.linalg.norm(np.array([0.2, 0.2, 0.2]) * 10.0)
        assert abs(d - expected) < 1e-10


class TestMinimumImageVector:
    """Minimum image vector function."""

    def test_single_vector(self, cubic_lattice_10):
        delta = np.array([0.8, 0.0, 0.0])
        v = minimum_image_vector(delta, cubic_lattice_10)
        assert abs(v[0] - (-2.0)) < 1e-10

    def test_batch_vectors(self, cubic_lattice_10):
        deltas = np.array([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
        vecs = minimum_image_vector(deltas, cubic_lattice_10)
        assert vecs.shape == (2, 3)
        assert abs(vecs[0, 0] - (-2.0)) < 1e-10
        np.testing.assert_allclose(vecs[1], [0, 0, 0], atol=1e-10)


class TestVolumeCenterOfMass:
    """Volume and center-of-mass."""

    def test_volume_cubic(self, cubic_lattice_5):
        v = volume_of_cell(cubic_lattice_5)
        assert abs(v - 125.0) < 1e-10

    def test_volume_rectangular(self):
        v = volume_of_cell(np.diag([3.0, 4.0, 5.0]))
        assert abs(v - 60.0) < 1e-10

    def test_com_equal_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        com = calculate_center_of_mass(coords, ["H", "H"])
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0])

    def test_com_different_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        com = calculate_center_of_mass(coords, ["C", "H"])
        np.testing.assert_allclose(com, [2.0 / 13.0, 0.0, 0.0], atol=0.01)

    def test_com_empty(self):
        com = calculate_center_of_mass(np.array([]), [])
        np.testing.assert_allclose(com, [0, 0, 0])


class TestRotateVector:
    """Vector rotation."""

    def test_rotate_z_90(self):
        v = np.array([1.0, 0.0, 0.0])
        r = rotate_vector(v, np.array([0, 0, 1]), 90.0)
        np.testing.assert_allclose(r, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotate_zero_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        r = rotate_vector(v, np.array([0, 1, 0]), 0.0)
        np.testing.assert_allclose(r, v, atol=1e-10)

    def test_rotate_360_returns_original(self):
        v = np.array([1.0, 2.0, 3.0])
        r = rotate_vector(v, np.array([0, 0, 1]), 360.0)
        np.testing.assert_allclose(r, v, atol=1e-10)


class TestRotationMatrix:
    """get_rotation_matrix utility."""

    def test_identity(self):
        R = get_rotation_matrix(np.array([0, 0, 1.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_z(self):
        R = get_rotation_matrix(np.array([0, 0, 1.0]), np.pi / 2)
        v = R @ np.array([1.0, 0, 0])
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-10)

    def test_preserves_magnitude(self):
        R = get_rotation_matrix(np.array([1.0, 1.0, 1.0]), np.pi / 3)
        v = np.array([3.0, 4.0, 5.0])
        rotated = R @ v
        assert abs(np.linalg.norm(rotated) - np.linalg.norm(v)) < 1e-10

    def test_orthogonal(self):
        R = get_rotation_matrix(np.array([1.0, 0, 0]), np.pi / 4)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestGetMissingVectors:
    """Missing coordination vectors."""

    def test_linear_no_neighbors(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(center, [], "linear", 1.0)
        assert len(vecs) == 2

    def test_linear_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(center, [np.array([1, 0, 0])], "linear", 1.0)
        assert len(vecs) == 1

    def test_linear_saturated(self):
        center = np.zeros(3)
        vecs = get_missing_vectors(
            center,
            [np.array([1, 0, 0]), np.array([-1, 0, 0])],
            "linear",
            1.0,
        )
        assert len(vecs) == 0

    def test_tetrahedral_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(center, [np.array([1, 0, 0])], "tetrahedral", 1.0)
        assert len(vecs) == 3

    def test_tetrahedral_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "tetrahedral", 1.0)
        assert len(vecs) == 4

    def test_tetrahedral_two_neighbors(self):
        center = np.zeros(3)
        neighbors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
        vecs = get_missing_vectors(center, neighbors, "tetrahedral", 1.0)
        assert len(vecs) == 2

    def test_tetrahedral_three_neighbors(self):
        s = 1 / np.sqrt(3)
        center = np.zeros(3)
        neighbors = [
            np.array([s, s, s]),
            np.array([s, -s, -s]),
            np.array([-s, s, -s]),
        ]
        vecs = get_missing_vectors(center, neighbors, "tetrahedral", 1.0)
        assert len(vecs) == 1

    def test_trigonal_planar_two_neighbors(self):
        center = np.array([0.0, 0.0, 0.0])
        neighbors = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]
        vecs = get_missing_vectors(center, neighbors, "trigonal_planar", 1.0)
        assert len(vecs) == 1

    def test_trigonal_planar_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "trigonal_planar", 1.0)
        assert len(vecs) == 3

    def test_trigonal_planar_one_neighbor(self):
        vecs = get_missing_vectors(
            np.zeros(3), [np.array([1, 0, 0])], "trigonal_planar", 1.0
        )
        assert len(vecs) == 2

    def test_bent_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(center, [np.array([1, 0, 0])], "bent", 1.0)
        assert len(vecs) == 1

    def test_bent_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "bent", 1.0)
        assert len(vecs) == 2

    def test_trigonal_pyramidal_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "trigonal_pyramidal", 1.0)
        assert len(vecs) == 3

    def test_trigonal_pyramidal_one_neighbor(self):
        vecs = get_missing_vectors(
            np.zeros(3), [np.array([1, 0, 0])], "trigonal_pyramidal", 1.0
        )
        assert len(vecs) == 2

    def test_trigonal_pyramidal_two_neighbors(self):
        vecs = get_missing_vectors(
            np.zeros(3),
            [np.array([1, 0, 0]), np.array([0, 1, 0])],
            "trigonal_pyramidal",
            1.0,
        )
        assert len(vecs) == 1

    def test_planar_bisector_two_neighbors(self):
        center = np.zeros(3)
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([0.0, 1.0, 0.0])
        vecs = get_missing_vectors(center, [n1, n2], "planar_bisector", 1.0)
        assert len(vecs) == 1
        assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-6

    def test_octahedral_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "octahedral", 1.0)
        assert len(vecs) == 6

    def test_octahedral_one_neighbor(self):
        vecs = get_missing_vectors(
            np.zeros(3), [np.array([1, 0, 0])], "octahedral", 1.0
        )
        assert len(vecs) == 5

    def test_trigonal_bipyramidal_no_neighbors(self):
        vecs = get_missing_vectors(np.zeros(3), [], "trigonal_bipyramidal", 1.0)
        assert len(vecs) == 5

    def test_unknown_geometry(self):
        vecs = get_missing_vectors(np.zeros(3), [], "nonexistent", 1.0)
        assert len(vecs) == 0


class TestReduceSurfaceLattice:
    """Gauss reduction for surface lattice."""

    def test_already_reduced(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        lattice = np.eye(3) * 10.0
        r1, r2 = reduce_surface_lattice(v1, v2, lattice)
        np.testing.assert_allclose(np.linalg.norm(r1), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(r2), 1.0, atol=1e-10)

    def test_reduces_non_orthogonal(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([3.0, 1.0, 0.0])
        lattice = np.eye(3)
        r1, r2 = reduce_surface_lattice(v1, v2, lattice)
        assert np.linalg.norm(r1) <= np.linalg.norm(v2)
        assert np.linalg.norm(r2) <= np.linalg.norm(v2) + 1e-10


class TestDihedralAndAdjustment:
    """calculate_dihedral_and_adjustment."""

    def test_no_neighbors_returns_zero(self):
        result = calculate_dihedral_and_adjustment(
            np.array([0, 0, 0]), np.array([1, 0, 0]), [], []
        )
        assert result == 0.0

    def test_finite_result(self):
        result = calculate_dihedral_and_adjustment(
            np.array([0, 0, 0]),
            np.array([1.5, 0, 0]),
            [np.array([0, 1, 0])],
            [np.array([1.5, 1, 0])],
        )
        assert np.isfinite(result)
