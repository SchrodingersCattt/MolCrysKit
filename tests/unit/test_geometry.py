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
    volume_of_cell,
    calculate_center_of_mass,
    rotate_vector,
    get_missing_vectors,
)


class TestFracCart:
    """Fractional/cartesian coordinate conversion."""

    def test_frac_to_cart_center(self, cubic_lattice_5):
        lattice = cubic_lattice_5
        frac = np.array([0.5, 0.5, 0.5])
        cart = frac_to_cart(frac, lattice)
        np.testing.assert_allclose(cart, [2.5, 2.5, 2.5])

    def test_cart_to_frac_center(self, cubic_lattice_5):
        lattice = cubic_lattice_5
        cart = np.array([2.5, 2.5, 2.5])
        frac = cart_to_frac(cart, lattice)
        np.testing.assert_allclose(frac, [0.5, 0.5, 0.5])

    def test_roundtrip(self, cubic_lattice_5):
        lattice = cubic_lattice_5
        frac = np.array([0.1, 0.2, 0.3])
        cart = frac_to_cart(frac, lattice)
        back = cart_to_frac(cart, lattice)
        np.testing.assert_allclose(back, frac)


class TestNormalizeVector:
    """Vector normalization."""

    def test_unit_axis(self):
        v = np.array([3.0, 0.0, 0.0])
        n = normalize_vector(v)
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0])

    def test_zero_vector(self):
        n = normalize_vector(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(n, [0.0, 0.0, 0.0])

    def test_diagonal_unit_magnitude(self):
        v = np.array([1.0, 1.0, 1.0])
        n = normalize_vector(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10


class TestDistanceAngle:
    """Distance and angle helpers."""

    def test_distance_axial(self):
        d = distance_between_points(np.array([0, 0, 0]), np.array([3, 0, 0]))
        assert abs(d - 3.0) < 1e-10

    def test_distance_diagonal(self):
        d = distance_between_points(
            np.array([0, 0, 0]), np.array([1.0, 1.0, 1.0])
        )
        assert abs(d - np.sqrt(3.0)) < 1e-10

    def test_angle_parallel(self):
        a = angle_between_vectors(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        )
        assert abs(a) < 1e-10

    def test_angle_perpendicular(self):
        a = angle_between_vectors(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        assert abs(a - np.pi / 2) < 1e-10

    def test_dihedral_finite(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 1.0])
        d = dihedral_angle(p1, p2, p3, p4)
        assert np.isfinite(d)


class TestMinimumImage:
    """Minimum image distance."""

    def test_close_points(self, cubic_lattice_10):
        lattice = cubic_lattice_10
        f1, f2 = np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2])
        d = minimum_image_distance(f1, f2, lattice)
        direct = np.linalg.norm(frac_to_cart(f1 - f2, lattice))
        assert abs(d - direct) < 1e-10

    def test_wrap_opposite_edges(self, cubic_lattice_10):
        lattice = cubic_lattice_10
        f1 = np.array([0.1, 0.1, 0.1])
        f2 = np.array([0.9, 0.9, 0.9])
        d = minimum_image_distance(f1, f2, lattice)
        expected = np.linalg.norm(np.array([0.2, 0.2, 0.2]) * 10.0)
        assert abs(d - expected) < 1e-10


class TestVolumeCenterOfMass:
    """Volume and center-of-mass."""

    def test_volume_cubic(self, cubic_lattice_5):
        v = volume_of_cell(cubic_lattice_5)
        assert abs(v - 125.0) < 1e-10

    def test_volume_rectangular(self):
        lattice = np.diag([3.0, 4.0, 5.0])
        v = volume_of_cell(lattice)
        assert abs(v - 60.0) < 1e-10

    def test_com_equal_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        symbols = ["H", "H"]
        com = calculate_center_of_mass(coords, symbols)
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0])

    def test_com_different_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        symbols = ["C", "H"]
        com = calculate_center_of_mass(coords, symbols)
        np.testing.assert_allclose(com, [2.0 / 13.0, 0.0, 0.0], atol=0.01)


class TestRotateVector:
    """Vector rotation."""

    def test_rotate_z_90(self):
        v = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        r = rotate_vector(v, axis, 90.0)
        np.testing.assert_allclose(r, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotate_zero_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        r = rotate_vector(v, np.array([0, 1, 0]), 0.0)
        np.testing.assert_allclose(r, v, atol=1e-10)


class TestGetMissingVectors:
    """Missing coordination vectors."""

    def test_linear_no_neighbors(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(center, [], "linear", 1.0)
        assert len(vecs) == 2

    def test_linear_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(
            center, [np.array([1.0, 0.0, 0.0])], "linear", 1.0
        )
        assert len(vecs) == 1

    def test_tetrahedral_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(
            center, [np.array([1.0, 0.0, 0.0])], "tetrahedral", 1.0
        )
        assert len(vecs) == 3

    def test_trigonal_planar_two_neighbors(self):
        center = np.array([0.0, 0.0, 0.0])
        neighbors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        vecs = get_missing_vectors(center, neighbors, "trigonal_planar", 1.0)
        assert len(vecs) == 1

    def test_bent_one_neighbor(self):
        center = np.array([0.0, 0.0, 0.0])
        vecs = get_missing_vectors(
            center, [np.array([1.0, 0.0, 0.0])], "bent", 1.0
        )
        assert len(vecs) == 1
