#!/usr/bin/env python3
"""
Test the geometry module.
"""

import os
import sys
import numpy as np

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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


def test_frac_to_cart():
    """Test converting fractional to cartesian coordinates."""
    # Define a simple cubic lattice
    lattice = np.eye(3) * 5.0  # 5x5x5 Angstrom box

    # Test converting fractional to cartesian
    frac_coords = np.array([0.5, 0.5, 0.5])  # Center of the box
    cart_coords = frac_to_cart(frac_coords, lattice)

    expected_cart = np.array([2.5, 2.5, 2.5])
    assert np.allclose(
        cart_coords, expected_cart
    ), f"Expected {expected_cart}, got {cart_coords}"


def test_cart_to_frac():
    """Test converting cartesian to fractional coordinates."""
    # Define a simple cubic lattice
    lattice = np.eye(3) * 5.0  # 5x5x5 Angstrom box

    # Test converting cartesian to fractional
    cart_coords = np.array([2.5, 2.5, 2.5])  # Center of the box
    frac_coords = cart_to_frac(cart_coords, lattice)

    expected_frac = np.array([0.5, 0.5, 0.5])
    assert np.allclose(
        frac_coords, expected_frac
    ), f"Expected {expected_frac}, got {frac_coords}"


def test_normalize_vector():
    """Test normalizing vectors."""
    # Test normalizing a simple vector
    v = np.array([3.0, 0.0, 0.0])
    normalized = normalize_vector(v)

    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(normalized, expected), f"Expected {expected}, got {normalized}"

    # Test normalizing a zero vector
    v_zero = np.array([0.0, 0.0, 0.0])
    normalized_zero = normalize_vector(v_zero)

    expected_zero = np.array([0.0, 0.0, 0.0])
    assert np.allclose(
        normalized_zero, expected_zero
    ), f"Expected {expected_zero}, got {normalized_zero}"

    # Test normalizing a more complex vector
    v_complex = np.array([1.0, 1.0, 1.0])
    normalized_complex = normalize_vector(v_complex)

    magnitude = np.linalg.norm(normalized_complex)
    assert abs(magnitude - 1.0) < 1e-10, f"Magnitude should be 1.0, got {magnitude}"


def test_distance_between_points():
    """Test calculating distance between points."""
    # Test distance between two points
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 0.0, 0.0])

    dist = distance_between_points(p1, p2)
    expected = 3.0
    assert abs(dist - expected) < 1e-10, f"Expected {expected}, got {dist}"

    # Test distance between non-axial points
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([1.0, 1.0, 1.0])

    dist2 = distance_between_points(p3, p4)
    expected2 = np.sqrt(3.0)
    assert abs(dist2 - expected2) < 1e-10, f"Expected {expected2}, got {dist2}"


def test_angle_between_vectors():
    """Test calculating angle between vectors."""
    # Test angle between parallel vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    angle = angle_between_vectors(v1, v2)
    expected = 0.0  # 0 radians for parallel vectors
    assert abs(angle - expected) < 1e-10, f"Expected {expected}, got {angle}"

    # Test angle between perpendicular vectors
    v3 = np.array([1.0, 0.0, 0.0])
    v4 = np.array([0.0, 1.0, 0.0])

    angle2 = angle_between_vectors(v3, v4)
    expected2 = np.pi / 2  # 90 degrees in radians
    assert abs(angle2 - expected2) < 1e-10, f"Expected {expected2}, got {angle2}"

    # Test angle between vectors at 120 degrees
    v5 = np.array([1.0, 0.0, 0.0])
    v6 = np.array([-0.5, np.sqrt(3) / 2, 0.0])  # 120 degrees from v5

    angle3 = angle_between_vectors(v5, v6)
    expected3 = 2 * np.pi / 3  # 120 degrees in radians
    assert abs(angle3 - expected3) < 1e-10, f"Expected {expected3}, got {angle3}"


def test_dihedral_angle():
    """Test calculating dihedral angle."""
    # Create four points that form a dihedral angle of 180 degrees (planar)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])
    p4 = np.array([3.0, 0.0, 0.0])

    dihedral = dihedral_angle(p1, p2, p3, p4)
    # The dihedral angle for a straight line is undefined, but our function should handle it
    # For colinear points, it should return 0 or pi depending on implementation

    # Create four points that form a 60 degree dihedral angle
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, 1.0, 0.0])
    p4 = np.array([1.0, 1.0, 1.0])

    dihedral2 = dihedral_angle(p1, p2, p3, p4)
    # This is a more complex case, just test that it returns a finite value
    assert np.isfinite(dihedral2), f"Dihedral angle should be finite, got {dihedral2}"


def test_minimum_image_distance():
    """Test minimum image distance calculation."""
    # Define a simple cubic lattice
    lattice = np.eye(3) * 10.0  # 10x10x10 Angstrom box

    # Test minimum image distance for points close to each other
    frac1 = np.array([0.1, 0.1, 0.1])
    frac2 = np.array([0.2, 0.2, 0.2])

    min_dist = minimum_image_distance(frac1, frac2, lattice)
    # This should be the regular distance since they're not near opposite edges
    regular_dist = np.linalg.norm(frac_to_cart(frac1 - frac2, lattice))
    assert (
        abs(min_dist - regular_dist) < 1e-10
    ), f"Expected {regular_dist}, got {min_dist}"

    # Test minimum image distance for points near opposite edges (should wrap around)
    frac3 = np.array([0.1, 0.1, 0.1])
    frac4 = np.array([0.9, 0.9, 0.9])

    min_dist2 = minimum_image_distance(frac3, frac4, lattice)
    # These points are close when considering periodic boundaries
    # frac3 is close to (0.1,0.1,0.1) and frac4 is close to (0.9,0.9,0.9)
    # The minimum image distance would be between (0.1,0.1,0.1) and (-0.1,-0.1,-0.1) = (0.9,0.9,0.9) - (1,1,1)
    # So the distance is between (0.1,0.1,0.1) and (-0.1,-0.1,-0.1) = (0.2,0.2,0.2) = 0.2*sqrt(3)*10
    expected_dist = np.linalg.norm(np.array([0.2, 0.2, 0.2]) * 10.0)
    assert (
        abs(min_dist2 - expected_dist) < 1e-10
    ), f"Expected {expected_dist}, got {min_dist2}"


def test_volume_of_cell():
    """Test volume calculation of unit cell."""
    # Test with cubic lattice
    lattice = np.eye(3) * 5.0  # 5x5x5 Angstrom box
    volume = volume_of_cell(lattice)
    expected = 5.0**3  # 125 A^3
    assert abs(volume - expected) < 1e-10, f"Expected {expected}, got {volume}"

    # Test with rectangular lattice
    lattice2 = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    volume2 = volume_of_cell(lattice2)
    expected2 = 3.0 * 4.0 * 5.0  # 60 A^3
    assert abs(volume2 - expected2) < 1e-10, f"Expected {expected2}, got {volume2}"


def test_calculate_center_of_mass():
    """Test center of mass calculation."""
    # Test with equal masses at specific positions
    atom_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # Two atoms at 0 and 2A
    atom_symbols = ["H", "H"]  # Same mass for both

    com = calculate_center_of_mass(atom_coords, atom_symbols)
    expected = np.array([1.0, 0.0, 0.0])  # Midway between the atoms
    assert np.allclose(com, expected), f"Expected {expected}, got {com}"

    # Test with different masses
    atom_coords2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # Two atoms at 0 and 2A
    atom_symbols2 = ["C", "H"]  # Different masses

    com2 = calculate_center_of_mass(atom_coords2, atom_symbols2)
    # Center of mass should be closer to the carbon atom (more massive)
    # C mass ~ 12, H mass ~ 1, so COM should be at (0*12 + 2*1)/(12+1) = 2/13 ~ 0.154 for x
    expected2 = np.array([2.0 / 13.0, 0.0, 0.0])
    # Using a more relaxed tolerance since actual atomic masses might differ slightly
    assert np.allclose(com2, expected2, atol=0.01), f"Expected {expected2}, got {com2}"


def test_rotate_vector():
    """Test vector rotation."""
    # Test rotating a vector around z-axis by 90 degrees
    vector = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])  # z-axis
    angle_deg = 90.0

    rotated = rotate_vector(vector, axis, angle_deg)
    expected = np.array(
        [0.0, 1.0, 0.0]
    )  # Should point along y-axis after 90 deg rotation
    assert np.allclose(
        rotated, expected, atol=1e-10
    ), f"Expected {expected}, got {rotated}"

    # Test rotating by 0 degrees (should be unchanged)
    vector2 = np.array([1.0, 0.0, 0.0])
    axis2 = np.array([0.0, 1.0, 0.0])  # y-axis
    angle_deg2 = 0.0

    rotated2 = rotate_vector(vector2, axis2, angle_deg2)
    expected2 = np.array([1.0, 0.0, 0.0])  # Should be unchanged
    assert np.allclose(
        rotated2, expected2, atol=1e-10
    ), f"Expected {expected2}, got {rotated2}"


def test_get_missing_vectors():
    """Test calculation of missing vectors based on coordination geometry."""
    # Test linear geometry with no neighbors
    center = np.array([0.0, 0.0, 0.0])
    existing_neighbors = []
    vectors = get_missing_vectors(center, existing_neighbors, "linear", 1.0)
    assert (
        len(vectors) == 2
    ), f"Expected 2 vectors for linear with no neighbors, got {len(vectors)}"

    # Test linear geometry with one neighbor
    existing_neighbors = [np.array([1.0, 0.0, 0.0])]
    vectors = get_missing_vectors(center, existing_neighbors, "linear", 1.0)
    assert (
        len(vectors) == 1
    ), f"Expected 1 vector for linear with one neighbor, got {len(vectors)}"

    # Test tetrahedral geometry with one neighbor
    existing_neighbors = [np.array([1.0, 0.0, 0.0])]
    vectors = get_missing_vectors(center, existing_neighbors, "tetrahedral", 1.0)
    assert (
        len(vectors) == 3
    ), f"Expected 3 vectors for tetrahedral with one neighbor, got {len(vectors)}"

    # Test trigonal planar geometry with two neighbors
    existing_neighbors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    vectors = get_missing_vectors(center, existing_neighbors, "trigonal_planar", 1.0)
    assert (
        len(vectors) == 1
    ), f"Expected 1 vector for trigonal planar with two neighbors, got {len(vectors)}"

    # Test bent geometry with one neighbor
    existing_neighbors = [np.array([1.0, 0.0, 0.0])]
    vectors = get_missing_vectors(center, existing_neighbors, "bent", 1.0)
    assert (
        len(vectors) == 1
    ), f"Expected 1 vector for bent with one neighbor, got {len(vectors)}"


def run_tests():
    """Run all tests."""
    tests = [
        test_frac_to_cart,
        test_cart_to_frac,
        test_normalize_vector,
        test_distance_between_points,
        test_angle_between_vectors,
        test_dihedral_angle,
        test_minimum_image_distance,
        test_volume_of_cell,
        test_calculate_center_of_mass,
        test_rotate_vector,
        test_get_missing_vectors,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\nTests passed: {passed}")
    print(f"Tests failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
