"""
Geometry utilities for molecular crystals.

This module provides coordinate transformations and geometric calculations.
"""

import numpy as np
from typing import List
from ..constants import get_atomic_mass, has_atomic_mass


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Convert fractional coordinates to cartesian coordinates.

    Parameters
    ----------
    frac : np.ndarray
        Fractional coordinates.
    lattice : np.ndarray
        3x3 array of lattice vectors as rows.

    Returns
    -------
    np.ndarray
        Cartesian coordinates.
    """
    return np.dot(frac, lattice)


def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Convert cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    cart : np.ndarray
        Cartesian coordinates.
    lattice : np.ndarray
        3x3 array of lattice vectors as rows.

    Returns
    -------
    np.ndarray
        Fractional coordinates.
    """
    return np.dot(cart, np.linalg.inv(lattice))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    vector : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def distance_between_points(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : np.ndarray
        First point coordinates.
    point2 : np.ndarray
        Second point coordinates.

    Returns
    -------
    float
        Distance between the points.
    """
    return np.linalg.norm(point1 - point2)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in radians.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Angle between vectors in radians.
    """
    # Normalize vectors
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # Calculate dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    # Calculate angle
    return np.arccos(dot_product)


def dihedral_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """
    Calculate the dihedral angle between four points.

    Parameters
    ----------
    p1, p2, p3, p4 : np.ndarray
        Four points defining the dihedral angle.

    Returns
    -------
    float
        Dihedral angle in radians (-π to π).
    """
    # Calculate bond vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Calculate normal vectors to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Calculate angle between normals
    angle = angle_between_vectors(n1, n2)

    # Determine sign using the cross product
    sign_check = np.dot(n1, b3) * np.linalg.norm(b2)
    if sign_check < 0:
        angle = -angle

    return angle


def minimum_image_distance(
    frac1: np.ndarray, frac2: np.ndarray, lattice: np.ndarray
) -> float:
    """
    Calculate the minimum image distance between two fractional coordinates.

    Parameters
    ----------
    frac1 : np.ndarray
        First fractional coordinates.
    frac2 : np.ndarray
        Second fractional coordinates.
    lattice : np.ndarray
        3x3 array of lattice vectors as rows.

    Returns
    -------
    float
        Minimum image distance.
    """
    # Calculate distance vector
    delta = frac1 - frac2

    # Apply minimum image convention
    delta = delta - np.round(delta)

    # Convert to cartesian and calculate distance
    cart_delta = frac_to_cart(delta, lattice)
    return np.linalg.norm(cart_delta)


def volume_of_cell(lattice: np.ndarray) -> float:
    """
    Calculate the volume of a unit cell.

    Parameters
    ----------
    lattice : np.ndarray
        3x3 array of lattice vectors as rows.

    Returns
    -------
    float
        Volume of the unit cell.
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    return abs(np.dot(a, np.cross(b, c)))


def calculate_center_of_mass(atom_coords: np.ndarray, atom_symbols: list) -> np.ndarray:
    """
    Calculate the center of mass of a group of atoms.

    Parameters
    ----------
    atom_coords : np.ndarray
        Array of atomic coordinates.
    atom_symbols : list
        List of atomic symbols.

    Returns
    -------
    np.ndarray
        Center of mass coordinates.
    """
    if len(atom_coords) == 0 or len(atom_coords) != len(atom_symbols):
        return np.array([0.0, 0.0, 0.0])

    # Get atomic masses
    masses = np.array(
        [
            get_atomic_mass(symbol) if has_atomic_mass(symbol) else 1.0
            for symbol in atom_symbols
        ]
    )

    # Calculate mass-weighted center of mass
    total_mass = np.sum(masses)
    center_of_mass = np.sum(atom_coords * masses[:, np.newaxis], axis=0) / total_mass

    return center_of_mass


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a vector around an axis by an angle in degrees using Rodrigues' rotation formula.

    Parameters
    ----------
    vector : np.ndarray
        Vector to be rotated.
    axis : np.ndarray
        Axis of rotation (will be normalized).
    angle_deg : float
        Angle of rotation in degrees.

    Returns
    -------
    np.ndarray
        Rotated vector.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Get the rotation matrix
    rotation_matrix = get_rotation_matrix(axis, angle_rad)

    # Apply the rotation to the vector
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector


def get_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix using Rodrigues' rotation formula.

    Parameters
    ----------
    axis : np.ndarray
        The rotation axis (will be normalized).
    angle : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.
    """
    # Normalize the rotation axis
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    # Convert angle to radians if it's in degrees (assuming it's in radians based on the function signature)
    angle_rad = angle

    # Calculate trigonometric values
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Create the cross product matrix (skew-symmetric matrix)
    cross_matrix = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Apply Rodrigues' rotation formula to create rotation matrix
    rotation_matrix = (
        cos_angle * np.eye(3)
        + sin_angle * cross_matrix
        + (1 - cos_angle) * np.outer(axis, axis)
    )

    return rotation_matrix


def get_missing_vectors(
    center: np.ndarray,
    existing_neighbors: List[np.ndarray],
    geometry_type: str,
    bond_length: float = 1.0,
) -> List[np.ndarray]:
    """
    Calculate vectors for missing atoms based on coordination geometry.

    Parameters
    ----------
    center : np.ndarray
        Center atom position.
    existing_neighbors : List[np.ndarray]
        List of existing neighbor positions.
    geometry_type : str
        Type of coordination geometry ('linear', 'trigonal_planar', 'tetrahedral',
        'trigonal_bipyramidal', 'octahedral', 'bent', 'trigonal_pyramidal').
    bond_length : float
        Distance from center to new atoms.

    Returns
    -------
    List[np.ndarray]
        List of vectors from center to missing atoms.
    """
    # Calculate vectors from center to existing neighbors
    neighbor_vectors = [
        normalize_vector(neighbor - center) for neighbor in existing_neighbors
    ]

    if geometry_type == "linear":
        # Linear geometry (sp, coordination number = 2)
        if len(neighbor_vectors) == 0:
            # If no neighbors, return an arbitrary direction and its opposite
            v1 = np.array([1.0, 0.0, 0.0])
            v2 = -v1
            return [v1 * bond_length, v2 * bond_length]
        elif len(neighbor_vectors) == 1:
            # If one neighbor, return the opposite direction
            v = -neighbor_vectors[0]
            return [v * bond_length]
        else:
            # More than 1 neighbor for linear geometry - return empty list
            return []

    elif geometry_type == "trigonal_planar":
        # Trigonal planar geometry (sp2, coordination number = 3)
        if len(neighbor_vectors) == 0:
            # If no neighbors, return three vectors in a plane (e.g., xy-plane)
            angle = 2 * np.pi / 3  # 120 degrees
            v1 = np.array([1.0, 0.0, 0.0])
            v2 = rotate_vector(v1, np.array([0, 0, 1]), 120)
            v3 = rotate_vector(v1, np.array([0, 0, 1]), 240)
            return [v1 * bond_length, v2 * bond_length, v3 * bond_length]
        elif len(neighbor_vectors) == 1:
            # If one neighbor, define a plane containing the neighbor
            n = neighbor_vectors[0]
            # Find a perpendicular vector to n to define the plane
            if abs(n[2]) < 0.9:
                perp = np.cross(n, np.array([0, 0, 1]))
            else:
                perp = np.cross(n, np.array([1, 0, 0]))
            perp = normalize_vector(perp)

            # Create two new vectors at 120 degrees to the existing one
            # First, rotate n around perp by 120 degrees
            v1 = rotate_vector(n, perp, 120)
            v2 = rotate_vector(n, perp, -120)
            return [v1 * bond_length, v2 * bond_length]
        elif len(neighbor_vectors) == 2:
            # If two neighbors, return the third position in the plane
            # The third vector should be such that all three are 120 deg apart
            a = neighbor_vectors[0]
            b = neighbor_vectors[1]
            third_vector = -(a + b)
            third_vector = normalize_vector(third_vector)
            return [third_vector * bond_length]
        else:
            # More than 2 neighbors for trigonal planar - return empty list
            return []

    elif geometry_type == "tetrahedral":
        # Tetrahedral geometry (sp3, coordination number = 4)
        if len(neighbor_vectors) == 0:
            # If no neighbors, return standard tetrahedral directions
            # Vectors pointing to the vertices of a tetrahedron centered at origin
            v1 = normalize_vector(np.array([1, 1, 1]))
            v2 = normalize_vector(np.array([-1, -1, 1]))
            v3 = normalize_vector(np.array([-1, 1, -1]))
            v4 = normalize_vector(np.array([1, -1, -1]))
            return [
                v1 * bond_length,
                v2 * bond_length,
                v3 * bond_length,
                v4 * bond_length,
            ]
        elif len(neighbor_vectors) == 1:
            # If one neighbor, place the other 3 to form a tetrahedron
            n = neighbor_vectors[0]
            # Find two perpendicular axes to n
            if abs(n[2]) < 0.9:
                u = np.cross(n, np.array([0, 0, 1]))
            else:
                u = np.cross(n, np.array([1, 0, 0]))
            u = normalize_vector(u)
            w = normalize_vector(np.cross(n, u))

            # The ideal tetrahedral angle is ~109.47 degrees (109.47 = 180 - 70.53)
            # So we rotate the opposite of n by 70.53 degrees around u
            cone_angle = 180 - 109.47  # ~70.53 degrees
            v1 = rotate_vector(-n, u, cone_angle)
            v2 = rotate_vector(v1, n, 120)
            v3 = rotate_vector(v1, n, 240)
            return [v1 * bond_length, v2 * bond_length, v3 * bond_length]
        elif len(neighbor_vectors) == 2:
            # If two neighbors, find vectors for the remaining two positions
            # Correct approach: Calculate bisector and perpendicular normal
            n1 = neighbor_vectors[0]
            n2 = neighbor_vectors[1]

            # Calculate the bisector of the two existing vectors
            bisector = normalize_vector(n1 + n2)

            # Calculate the perpendicular normal to the plane of the two vectors
            perpendicular_normal = normalize_vector(np.cross(n1, n2))

            # Calculate the rotation axis that lies in the n1-n2 plane and is perpendicular to the bisector
            rotation_axis = normalize_vector(np.cross(bisector, perpendicular_normal))

            # Tetrahedral angle (109.47°) - angle between bonds in a tetrahedron
            # The angle between individual bonds and their bisector is half this: 54.7356°
            tetrahedral_angle_half = 109.47 / 2  # ~54.7356 degrees

            # The target vectors should be centered around the opposite of the bisector
            target_base = -bisector

            # Rotate the target base around the rotation_axis by the tetrahedral angle
            v1 = rotate_vector(target_base, rotation_axis, tetrahedral_angle_half)
            v2 = rotate_vector(target_base, rotation_axis, -tetrahedral_angle_half)

            return [v1 * bond_length, v2 * bond_length]
        elif len(neighbor_vectors) == 3:
            # If three neighbors, return the fourth position
            a, b, c = neighbor_vectors
            fourth_vector = -(a + b + c)
            fourth_vector = normalize_vector(fourth_vector)
            return [fourth_vector * bond_length]
        else:
            # More than 3 neighbors for tetrahedral - return empty list
            return []

    elif geometry_type == "trigonal_bipyramidal":
        # Trigonal bipyramidal geometry (sp3d, coordination number = 5)
        # 3 equatorial (in plane, 120 deg apart) + 2 axial (180 deg apart)
        if len(neighbor_vectors) == 0:
            # Default positions: 3 in xy plane (120 deg apart) and 2 axial
            v_equatorial1 = np.array([1, 0, 0])
            v_equatorial2 = rotate_vector(v_equatorial1, np.array([0, 0, 1]), 120)
            v_equatorial3 = rotate_vector(v_equatorial1, np.array([0, 0, 1]), 240)
            v_axial1 = np.array([0, 0, 1])
            v_axial2 = np.array([0, 0, -1])
            return [
                v_equatorial1 * bond_length,
                v_equatorial2 * bond_length,
                v_equatorial3 * bond_length,
                v_axial1 * bond_length,
                v_axial2 * bond_length,
            ]
        else:
            # For now, just return the remaining positions based on ideal geometry
            # This is a simplified approach; a more complex alignment would be needed for real cases
            # For now, return empty to indicate complexity
            return []

    elif geometry_type == "octahedral":
        # Octahedral geometry (sp3d2, coordination number = 6)
        # 6 directions along cartesian axes: ±x, ±y, ±z
        if len(neighbor_vectors) == 0:
            # All 6 positions available
            directions = [
                np.array([1, 0, 0]),
                np.array([-1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, 1]),
                np.array([0, 0, -1]),
            ]
            return [d * bond_length for d in directions]
        elif len(neighbor_vectors) == 1:
            # Return 5 positions orthogonal to the first
            n = neighbor_vectors[0]
            # Find 5 orthogonal directions to the existing one
            # This is a simplified approach
            if abs(n[2]) < 0.9:
                perp = np.cross(n, np.array([0, 0, 1]))
            else:
                perp = np.cross(n, np.array([1, 0, 0]))
            perp = normalize_vector(perp)

            # Find another perpendicular to create a coordinate system
            perp2 = normalize_vector(np.cross(n, perp))

            # In octahedral geometry, the orthogonal positions are along the perpendicular axes
            # We'll return positions based on the coordinate system defined by n, perp, perp2
            positions = []
            # Add positions along perp and perp2 directions
            positions.extend([perp * bond_length, -perp * bond_length])
            positions.extend([perp2 * bond_length, -perp2 * bond_length])

            # Add position opposite to n
            positions.append(-n * bond_length)

            return positions
        else:
            # More complex case - return empty for now
            return []

    elif geometry_type == "bent":
        # Bent geometry (like water, with 2 lone pairs)
        # This is for atoms with 2 bonds and 2 lone pairs (like oxygen in water)
        if len(neighbor_vectors) == 0:
            # If no neighbors, return two arbitrary directions with ~109.5 deg angle (tetrahedral)
            v1 = np.array([1.0, 0.0, 0.0])
            # Rotate second vector by ~109.5 degrees around z-axis
            v2 = rotate_vector(v1, np.array([0, 0, 1]), 109.5)
            return [v1 * bond_length, v2 * bond_length]
        elif len(neighbor_vectors) == 1:
            # If one neighbor, place the other atom with ~109.5 deg angle
            n = neighbor_vectors[0]
            # Find a perpendicular axis to rotate around
            if abs(n[2]) < 0.9:
                perp_axis = np.cross(n, np.array([0, 0, 1]))
            else:
                perp_axis = np.cross(n, np.array([1, 0, 0]))
            perp_axis = normalize_vector(perp_axis)

            # Rotate the first neighbor by ~109.5 degrees around the perpendicular axis
            v = rotate_vector(n, perp_axis, 109.5)
            return [v * bond_length]
        else:
            # More than 1 neighbor for bent geometry - return empty list
            return []

    elif geometry_type == "trigonal_pyramidal":
        # Trigonal pyramidal geometry (like ammonia, with 1 lone pair)
        # This is similar to tetrahedral but with one position occupied by a lone pair
        if len(neighbor_vectors) == 0:
            # If no neighbors, return three positions in a trigonal pyramidal arrangement
            # Pointing roughly in 3 directions with a vertical component
            v1 = normalize_vector(np.array([1, 0, -0.5]))
            v2 = rotate_vector(v1, np.array([0, 0, 1]), 120)
            v3 = rotate_vector(v1, np.array([0, 0, 1]), 240)
            return [v1 * bond_length, v2 * bond_length, v3 * bond_length]
        elif len(neighbor_vectors) == 1:
            # If one neighbor, place 2 more in trigonal pyramidal arrangement
            n = neighbor_vectors[0]
            # Find two perpendicular directions to the existing bond
            if abs(n[2]) < 0.9:
                perp1 = np.cross(n, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(n, np.array([1, 0, 0]))
            perp1 = normalize_vector(perp1)

            # Rotate around the axis to get the second and third positions
            v1 = rotate_vector(n, perp1, 109.5)  # Tetrahedral angle
            # Find axis perpendicular to both n and v1 to get the third vector
            perp2 = normalize_vector(np.cross(n, v1))
            v2 = rotate_vector(n, perp2, 109.5)
            return [v1 * bond_length, v2 * bond_length]
        elif len(neighbor_vectors) == 2:
            # If two neighbors, return the third position to complete the pyramid
            n1, n2 = neighbor_vectors
            # Find the third direction to complete the trigonal pyramid
            # Average the two vectors and reflect to get the third
            avg = normalize_vector((n1 + n2) / 2)
            # The third vector should be in the opposite direction, but adjusted for pyramidal geometry
            third_vector = -avg
            # Adjust to maintain proper geometry
            return [third_vector * bond_length]
        else:
            # More than 2 neighbors for trigonal pyramidal - return empty list
            return []

    # Unknown geometry type
    return []


def calculate_dihedral_and_adjustment(
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    neighbors_start: List[np.ndarray],
    neighbors_end: List[np.ndarray],
) -> float:
    """
    Calculate rotation needed to achieve staggered conformation for connected sp3 centers.

    Parameters
    ----------
    axis_start : np.ndarray
        Start position of the bond axis.
    axis_end : np.ndarray
        End position of the bond axis.
    neighbors_start : List[np.ndarray]
        Neighbor atoms connected to the start atom.
    neighbors_end : List[np.ndarray]
        Neighbor atoms connected to the end atom.

    Returns
    -------
    float
        Delta theta in degrees needed to reach optimal staggered conformation.
    """
    # Calculate the bond axis vector
    bond_axis = normalize_vector(axis_end - axis_start)

    # For each set of neighbors, we need to find the average direction of the
    # bonds for the dihedral angle calculation
    if not neighbors_start or not neighbors_end:
        return 0.0  # No neighbors to align

    # Calculate the average vector from axis_start to its neighbors
    avg_start_vector = np.zeros(3)
    for neighbor in neighbors_start:
        vec = neighbor - axis_start
        # Project onto plane perpendicular to bond axis
        projection = np.dot(vec, bond_axis) * bond_axis
        perp_vec = vec - projection
        avg_start_vector += (
            normalize_vector(perp_vec)
            if np.linalg.norm(perp_vec) > 1e-6
            else np.zeros(3)
        )

    # Calculate the average vector from axis_end to its neighbors
    avg_end_vector = np.zeros(3)
    for neighbor in neighbors_end:
        vec = neighbor - axis_end
        # Project onto plane perpendicular to bond axis
        projection = np.dot(vec, bond_axis) * bond_axis
        perp_vec = vec - projection
        avg_end_vector += (
            normalize_vector(perp_vec)
            if np.linalg.norm(perp_vec) > 1e-6
            else np.zeros(3)
        )

    # Normalize the average vectors
    if np.linalg.norm(avg_start_vector) > 1e-6:
        avg_start_vector = normalize_vector(avg_start_vector)
    else:
        # If no valid neighbors, use an arbitrary perpendicular direction
        if abs(bond_axis[2]) < 0.9:
            avg_start_vector = normalize_vector(
                np.cross(bond_axis, np.array([0, 0, 1]))
            )
        else:
            avg_start_vector = normalize_vector(
                np.cross(bond_axis, np.array([1, 0, 0]))
            )

    if np.linalg.norm(avg_end_vector) > 1e-6:
        avg_end_vector = normalize_vector(avg_end_vector)
    else:
        # If no valid neighbors, use an arbitrary perpendicular direction
        if abs(bond_axis[2]) < 0.9:
            avg_end_vector = normalize_vector(np.cross(bond_axis, np.array([0, 0, 1])))
        else:
            avg_end_vector = normalize_vector(np.cross(bond_axis, np.array([1, 0, 0])))

    # Calculate the current dihedral angle
    # Project both vectors onto a plane perpendicular to the bond axis
    current_angle = angle_between_vectors(avg_start_vector, avg_end_vector)

    # Calculate the angle needed to achieve 60 degrees (staggered)
    # We want the angle between the planes to be 60 degrees
    target_angle = np.radians(60.0)

    # Calculate the difference
    angle_diff = target_angle - current_angle

    # Convert to degrees
    return np.degrees(angle_diff)
