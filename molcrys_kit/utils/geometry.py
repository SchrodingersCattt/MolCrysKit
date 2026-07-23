"""
Geometry utilities for molecular crystals.

This module provides coordinate transformations and geometric calculations.
"""

import itertools
import numpy as np
from typing import List, Optional, Tuple
from ..constants import get_atomic_mass, has_atomic_mass


_GEOM_EPS = 1e-12


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


def skew_matrix(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix for a cross product vector."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def unskew_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the vector represented by a 3x3 skew-symmetric matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return np.array(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=float,
    )


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find the best active rotation aligning ``mobile`` to ``target``.

    The returned matrix ``R`` follows the MolCrysKit row-vector convention used
    throughout the operations layer: ``mobile @ R.T`` best approximates
    ``target``. Inputs are expected to be centered already.

    Parameters
    ----------
    mobile, target : np.ndarray
        Arrays with shape ``(n, 3)`` containing corresponding Cartesian points.

    Returns
    -------
    Tuple[np.ndarray, float]
        Active rotation matrix and RMSD after alignment.
    """
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            "mobile and target must both have shape (n, 3); "
            f"got {mobile.shape} and {target.shape}"
        )
    if len(mobile) == 0:
        raise ValueError("Cannot align empty point sets")

    covariance = mobile.T @ target
    u, _, vt = np.linalg.svd(covariance)
    det = np.linalg.det(vt.T @ u.T)
    correction = np.diag([1.0, 1.0, 1.0 if det >= 0.0 else -1.0])
    rotation = vt.T @ correction @ u.T
    aligned = mobile @ rotation.T
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, rmsd


def rotation_to_axis_angle(rotation: np.ndarray) -> Tuple[np.ndarray, float]:
    """Decompose a rotation matrix into an axis and angle in radians.

    The returned angle is in ``[0, π]``. For the identity rotation, the axis is
    the conventional ``[1, 0, 0]`` and the angle is zero.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")

    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_theta))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0

    if np.pi - angle < 1e-7:
        # Near 180 degrees the usual skew formula is ill-conditioned.  The
        # rotation axis is the eigenvector with eigenvalue +1.
        eigvals, eigvecs = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(np.real(eigvals) - 1.0)))
        axis = np.real(eigvecs[:, index])
        axis = normalize_vector(axis)
        if np.linalg.norm(axis) < _GEOM_EPS:
            axis = np.array([1.0, 0.0, 0.0])
        return axis, angle

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return normalize_vector(axis), angle


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm vector ``omega = axis * angle``."""
    axis, angle = rotation_to_axis_angle(rotation)
    return axis * angle


def rotation_exp_vector(omega: np.ndarray) -> np.ndarray:
    """Return ``exp([omega]x)`` as a 3x3 rotation matrix."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < _GEOM_EPS:
        return np.eye(3)
    return get_rotation_matrix(omega / theta, theta)


def _left_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3), used as the SE(3) translational V matrix."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    omega_hat = skew_matrix(omega)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    theta2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / theta2) * omega_hat
        + ((theta - np.sin(theta)) / (theta2 * theta)) * (omega_hat @ omega_hat)
    )


def _left_jacobian_so3_inverse(omega: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SO(3)."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    omega_hat = skew_matrix(omega)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0) * (omega_hat @ omega_hat)
    theta2 = theta * theta
    coefficient = (1.0 / theta2) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)


def se3_log(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Map an SE(3) transform to its Lie-algebra twist vector.

    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix.
    translation : np.ndarray
        Cartesian translation vector.

    Returns
    -------
    np.ndarray
        Six-vector ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]``.
    """
    omega = rotation_log_vector(rotation)
    translation = np.asarray(translation, dtype=float)
    v = _left_jacobian_so3_inverse(omega) @ translation
    return np.concatenate([omega, v])


def se3_exp(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map a Lie-algebra twist vector to an SE(3) transform.

    Parameters
    ----------
    xi : np.ndarray
        Six-vector ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Rotation matrix and Cartesian translation vector.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,):
        raise ValueError(f"xi must have shape (6,), got {xi.shape}")
    omega = xi[:3]
    v = xi[3:]
    rotation = rotation_exp_vector(omega)
    translation = _left_jacobian_so3(omega) @ v
    return rotation, translation


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized quaternion ``[w, x, y, z]``."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif idx == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_vector(quat)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion ``[w, x, y, z]`` to a rotation matrix."""
    q = normalize_vector(np.asarray(quaternion, dtype=float))
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical linear interpolation between normalized quaternions."""
    q0 = normalize_vector(np.asarray(q0, dtype=float))
    q1 = normalize_vector(np.asarray(q1, dtype=float))
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("q0 and q1 must both have shape (4,)")
    fraction = float(fraction)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_vector((1.0 - fraction) * q0 + fraction * q1)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * fraction
    scale0 = np.sin(theta_0 - theta) / sin_theta_0
    scale1 = np.sin(theta) / sin_theta_0
    return normalize_vector(scale0 * q0 + scale1 * q1)


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


def minimum_image_vector(frac_delta: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Vectorized Minimum Image Convention.
    Check 27 images to find the shortest Cartesian vector.
    """
    frac_delta = np.atleast_2d(frac_delta)  # Shape becomes (N, 3)

    # 1. First approximation: simple rounding to center at origin
    # This brings vectors roughly to [-0.5, 0.5]
    frac_delta = frac_delta - np.round(frac_delta)

    # 2. Check 27 neighbors
    # shifts shape: (27, 3)
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))

    # Broadcasting magic:
    # We want result shape (N, 27, 3)
    # frac_delta[:, None, :] is (N, 1, 3)
    # shifts[None, :, :]     is (1, 27, 3)
    candidates_frac = frac_delta[:, None, :] + shifts[None, :, :]

    # Convert to Cartesian: (N, 27, 3) dot (3, 3) -> (N, 27, 3)
    candidates_cart = np.dot(candidates_frac, lattice)

    # Calculate squared distances: (N, 27)
    dists_sq = np.sum(candidates_cart**2, axis=2)

    # Find index of min distance for each atom pair: (N,)
    min_indices = np.argmin(dists_sq, axis=1)

    # Select the best vectors
    # Advanced indexing to pick the right candidate for each row
    n_rows = frac_delta.shape[0]
    best_vectors = candidates_cart[np.arange(n_rows), min_indices]

    # If input was single vector, flatten back
    if best_vectors.shape[0] == 1:
        return best_vectors.flatten()

    return best_vectors


def minimum_image_distance(
    frac1: np.ndarray, frac2: np.ndarray, lattice: np.ndarray
) -> float:
    """
    Calculate the minimum image distance between two fractional coordinates.
    """
    delta = frac1 - frac2
    min_vector = minimum_image_vector(delta, lattice)
    return np.linalg.norm(min_vector)


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


def unwrap_positions_along_bonds(
    graph,
    indices,
    base_positions: np.ndarray,
    *,
    max_atoms: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Unwrap a bonded component by walking edge vectors in a graph.

    Parameters
    ----------
    graph
        Graph whose edges carry a ``vector`` attribute. The vector is expected
        to point from the smaller node id to the larger node id, matching the
        convention used by ASE ``neighbor_list("D")`` call sites in MolCrysKit.
    indices
        Node ids forming the component to unwrap. The returned positions follow
        this order.
    base_positions
        Cartesian positions indexed by node id.
    max_atoms
        Optional safety cap. Components larger than this are returned unchanged
        with ``completed=False`` so polymeric or framework-like components do
        not trigger unbounded traversal.

    Returns
    -------
    Tuple[np.ndarray, bool]
        ``(positions, completed)``. ``completed`` is false when the size cap is
        exceeded or the graph traversal does not reach every requested node.
    """
    indices = list(indices)
    base_positions = np.asarray(base_positions, dtype=float)
    if not indices:
        return np.zeros((0, 3), dtype=float), True

    if max_atoms is not None and len(indices) > int(max_atoms):
        return base_positions[indices].copy(), False

    local_of = {idx: pos for pos, idx in enumerate(indices)}
    out = base_positions[indices].copy()
    visited = {indices[0]}
    queue = [indices[0]]

    while queue:
        u = queue.pop(0)
        u_local = local_of[u]
        for v in graph.neighbors(u):
            if v not in local_of or v in visited:
                continue
            edge_data = graph.get_edge_data(u, v) or {}
            edge_vec = np.asarray(edge_data.get("vector", base_positions[v] - base_positions[u]), dtype=float)
            small, large = (u, v) if u < v else (v, u)
            shift = edge_vec if u == small else -edge_vec
            out[local_of[v]] = out[u_local] + shift
            visited.add(v)
            queue.append(v)

            if max_atoms is not None and len(visited) > int(max_atoms):
                return base_positions[indices].copy(), False

    return out, len(visited) == len(indices)


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
        'trigonal_bipyramidal', 'octahedral', 'bent', 'trigonal_pyramidal', 'planar_bisector').
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

    elif geometry_type == "planar_bisector":
        # Planar bisector geometry (for adding H in ring structures like pyrrole)
        # Input: Center with 2 neighbors in a plane
        # Logic: Find the vector that bisects the angle between neighbors, 
        # but lies in the same plane as the neighbors, pointing away from the ring
        if len(neighbor_vectors) == 2:
            n1 = neighbor_vectors[0]
            n2 = neighbor_vectors[1]
            
            # LOGIC FIX: n1 + n2 is the internal bisector (pointing between the neighbors).
            # We want the external bisector (pointing away from the ring).
            # So we simply take the negative sum.
            bisector = -(n1 + n2) 
            result_vector = normalize_vector(bisector)
            
            # Optional: Ensure rigorous planarity if the ring is slightly buckled
            # Project result_vector onto the plane defined by n1 and n2
            normal = np.cross(n1, n2)  # Normal to the neighbors
            if np.linalg.norm(normal) > 1e-3:  # Check to avoid division by zero if collinear
                normal = normalize_vector(normal)
                # Remove component along normal to ensure it's in plane
                result_vector = result_vector - np.dot(result_vector, normal) * normal
                result_vector = normalize_vector(result_vector)

            return [result_vector * bond_length]
        else:
            # Wrong number of neighbors for this geometry
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


def reduce_surface_lattice(
    v1: np.ndarray, v2: np.ndarray, lattice: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gauss reduction to find the shortest and most orthogonal basis vectors
    for the surface plane (2D LLL reduction).

    Parameters
    ----------
    v1 : np.ndarray
        First basis vector of the surface plane.
    v2 : np.ndarray
        Second basis vector of the surface plane.
    lattice : np.ndarray
        3x3 array of the original lattice vectors as rows (used for metric tensor).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reduced basis vectors v1 and v2.
    """

    # Calculate metric tensor elements (dot products)
    def dot_product(a, b):
        return np.dot(a, b)

    # Main loop of Gauss reduction
    while True:
        # Calculate the lengths of the current vectors
        v1_sq = dot_product(v1, v1)
        v2_sq = dot_product(v2, v2)

        # Calculate the dot product of the two vectors
        v1_dot_v2 = dot_product(v1, v2)

        # Check if vectors are orthogonal enough or if we should continue
        # If |v1|^2 > |v2|^2, try to reduce v1 using v2
        if v1_sq > v2_sq:
            # Swap v1 and v2
            v1, v2 = v2, v1
            v1_sq, v2_sq = v2_sq, v1_sq
            v1_dot_v2 = dot_product(v1, v2)

        # Calculate how much of v1 is in v2 (projection)
        # We want to subtract an integer multiple of v1 from v2
        m = round(v1_dot_v2 / v1_sq)

        if m == 0:
            # The vectors are now as reduced as possible
            break

        # Update v2 by subtracting m*v1
        v2 = v2 - m * v1

    return v1, v2


def orient_lattice(
    lattice: np.ndarray, target_axis: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a lattice so that row[0] aligns with the first in-plane Cartesian
    axis, row[1] lies in the in-plane subspace, and the surface normal (cross
    product of row[0] and row[1]) points along *target_axis*.

    The rotation is a proper rotation (det = +1) expressed as a 3×3 matrix
    for right-multiplication: ``rotated = original @ M``.

    For ``target_axis=2`` (default, Z), the result is the standard LAMMPS
    triclinic convention:

    - ``lattice[0] @ M`` → along X
    - ``lattice[1] @ M`` → in XY plane (y ≥ 0)
    - ``lattice[2] @ M`` → has positive z component

    Parameters
    ----------
    lattice : np.ndarray
        3×3 array of lattice vectors as rows.
    target_axis : int
        Cartesian axis index (0 = X, 1 = Y, 2 = Z) that the surface normal
        (i.e. the stacking direction, perpendicular to rows 0 and 1) should
        be aligned to.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(rotated_lattice, rotation_matrix)`` where *rotation_matrix* is a
        proper orthogonal matrix (det = +1). ``rotated_lattice`` equals
        ``lattice @ rotation_matrix`` **after** applying a tilt normalization
        step that ensures LAMMPS triclinic compatibility
        (``|xy| ≤ ax/2``, ``|xz| ≤ ax/2``, ``|yz| ≤ by/2``). The
        normalization adds integer multiples of in-plane vectors to the
        stacking vector — a lattice-equivalent transformation that does not
        change the physical crystal.

    Raises
    ------
    ValueError
        If *target_axis* is not 0, 1, or 2, or if the first two lattice
        rows are collinear.
    """
    if target_axis not in (0, 1, 2):
        raise ValueError(f"target_axis must be 0, 1, or 2, got {target_axis}")

    a = lattice[0]
    b = lattice[1]

    # Build orthonormal frame: x along a, y in ab-plane, z = x × y
    x_axis = a / np.linalg.norm(a)
    b_proj = b - np.dot(b, x_axis) * x_axis
    b_proj_norm = np.linalg.norm(b_proj)
    if b_proj_norm < _GEOM_EPS:
        raise ValueError("First two lattice rows are collinear; cannot orient.")
    y_axis = b_proj / b_proj_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Enforce key LAMMPS triclinic constraint for target_axis=Z:
    #   rotated[2] = (cx, cy, cz) with cz > 0
    # This ensures the stacking/shock direction has positive Z projection.
    # The normal direction (cross of rows 0,1) may point in ±Z — that's fine
    # for LAMMPS; the sign depends on the handedness of the in-plane basis.
    #
    # Additionally ensure b_y > 0 for conventional box shape.
    # With det(M)=+1, only two axes can be flipped simultaneously.
    c = lattice[2]

    # 1. Ensure c_z > 0 (stacking vector projection along frame normal)
    if np.dot(c, z_axis) < 0:
        y_axis = -y_axis
        z_axis = -z_axis

    # 2. Ensure b_y > 0 (projection of b onto in-plane y)
    #    If flipping is needed, negate x and y (keeps z, keeps det=+1).
    if np.dot(b, y_axis) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    # M_z: rotation that puts normal along Z (standard orientation)
    M_z = np.stack([x_axis, y_axis, z_axis], axis=1)

    if target_axis == 2:
        M = M_z
    else:
        # Apply a proper-rotation permutation P (det=+1) that maps the
        # Z-aligned normal to the target axis.
        if target_axis == 0:
            # Cyclic permutation mapping Z→X, X→Y, Y→Z
            P = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=float)
        else:  # target_axis == 1
            # −90° rotation around X-axis: maps Z→Y
            P = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, -1, 0]], dtype=float)
        M = M_z @ P

    rotated_lattice = lattice @ M

    # ------------------------------------------------------------------
    # Tilt normalization (LAMMPS triclinic convention):
    #   |xy| ≤ 0.5*ax,  |xz| ≤ 0.5*ax,  |yz| ≤ 0.5*by
    # This adds integer multiples of in-plane vectors to the stacking
    # vector — a lattice-equivalent transformation that does not change
    # the physical structure, only the cell description.
    # After this step, rotated_lattice may differ from lattice @ M by
    # an integer combination of rows 0 and 1 in row 2.
    # ------------------------------------------------------------------
    ax = rotated_lattice[0, 0]
    if abs(ax) > _GEOM_EPS:
        # xy: project of row[1] onto row[0] direction
        if abs(rotated_lattice[1, 0]) > 0.5 * abs(ax):
            n = round(rotated_lattice[1, 0] / ax)
            rotated_lattice[1] -= n * rotated_lattice[0]
        # xz: project of row[2] onto row[0] direction
        if abs(rotated_lattice[2, 0]) > 0.5 * abs(ax):
            n = round(rotated_lattice[2, 0] / ax)
            rotated_lattice[2] -= n * rotated_lattice[0]
    by = rotated_lattice[1, 1]
    if abs(by) > _GEOM_EPS:
        # yz: project of row[2] onto row[1] direction
        if abs(rotated_lattice[2, 1]) > 0.5 * abs(by):
            n = round(rotated_lattice[2, 1] / by)
            rotated_lattice[2] -= n * rotated_lattice[1]
    # Recheck xz after yz flip (yz flip adds row[1] which has xy component)
    if abs(ax) > _GEOM_EPS:
        if abs(rotated_lattice[2, 0]) > 0.5 * abs(ax):
            n = round(rotated_lattice[2, 0] / ax)
            rotated_lattice[2] -= n * rotated_lattice[0]

    return rotated_lattice, M


def min_distance_between_atom_sets(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray = None,
    pbc: tuple = None,
) -> float:
    """
    Calculate the minimum pairwise distance between two sets of atoms.

    When *lattice* and *pbc* are provided, the minimum-image convention is
    applied so that periodic copies are taken into account.  Otherwise plain
    Euclidean distances are used.

    Parameters
    ----------
    positions_a : np.ndarray
        Cartesian positions of atom set A, shape ``(N, 3)``.
    positions_b : np.ndarray
        Cartesian positions of atom set B, shape ``(M, 3)``.
    lattice : np.ndarray, optional
        3×3 lattice matrix (rows = lattice vectors) for PBC distance
        calculation.  Must be provided together with *pbc*.
    pbc : tuple of bool, optional
        Periodic boundary conditions ``(bool, bool, bool)``.

    Returns
    -------
    float
        Minimum distance (Å) between any atom in *A* and any atom in *B*.

    Notes
    -----
    Computational complexity is O(N×M).  For very large sets consider
    spatial partitioning, but for typical molecular-crystal manipulation
    (hundreds of atoms) this brute-force approach is fast enough.
    """
    positions_a = np.atleast_2d(positions_a)
    positions_b = np.atleast_2d(positions_b)

    use_pbc = (
        lattice is not None
        and pbc is not None
        and any(pbc)
    )

    if use_pbc:
        inv_lattice = np.linalg.inv(lattice)
        # Convert both sets to fractional coordinates
        frac_a = positions_a @ inv_lattice  # (N, 3)
        frac_b = positions_b @ inv_lattice  # (M, 3)

        # Difference matrix: (N, M, 3)
        delta_frac = frac_a[:, None, :] - frac_b[None, :, :]
        # Reshape to (N*M, 3) for vectorised MIC
        delta_frac_flat = delta_frac.reshape(-1, 3)

        mic_vectors = minimum_image_vector(delta_frac_flat, lattice)  # (N*M, 3) or (3,) if single
        mic_vectors = np.atleast_2d(mic_vectors)  # ensure always (K, 3)
        dists_sq = np.sum(mic_vectors ** 2, axis=1)
    else:
        # Simple Euclidean: broadcast (N,1,3) - (1,M,3) -> (N,M,3)
        diff = positions_a[:, None, :] - positions_b[None, :, :]
        dists_sq = np.sum(diff ** 2, axis=2).ravel()

    return float(np.sqrt(np.min(dists_sq)))


# ─────────────────────────────────────────────────────────────────────
# Variable-cell lattice interpolation utilities
# ─────────────────────────────────────────────────────────────────────


def lattice_deformation_logm(
    lattice_a: np.ndarray, lattice_b: np.ndarray
) -> np.ndarray:
    """Compute the matrix logarithm of the deformation gradient F = B @ inv(A).

    The result ``L = logm(F)`` is a real 3×3 matrix such that
    ``expm(λ * L) @ A`` smoothly deforms lattice A into lattice B as λ goes
    from 0 to 1 along the GL⁺(3) geodesic.

    Parameters
    ----------
    lattice_a, lattice_b : np.ndarray
        3×3 arrays with lattice vectors as rows.

    Returns
    -------
    np.ndarray
        3×3 real matrix log of the deformation gradient.

    Raises
    ------
    ValueError
        If the deformation gradient has non-positive determinant.
    """
    from scipy.linalg import logm

    A = np.asarray(lattice_a, dtype=float)
    B = np.asarray(lattice_b, dtype=float)
    F = B @ np.linalg.inv(A)
    det_F = np.linalg.det(F)
    if det_F <= 0:
        raise ValueError(
            f"Deformation gradient has non-positive determinant ({det_F:.6f}); "
            "cannot compute matrix logarithm for lattice interpolation."
        )
    L = logm(F)
    # logm may return complex with tiny imaginary part for real matrices
    if np.isrealobj(L):
        return L
    if np.max(np.abs(L.imag)) < 1e-10:
        return L.real
    raise ValueError(
        "Matrix logarithm of the deformation gradient has non-negligible "
        f"imaginary part (max |imag| = {np.max(np.abs(L.imag)):.2e}). "
        "The lattice transformation may not be continuously deformable."
    )


def lattice_at_lambda(
    lattice_a: np.ndarray, log_F: np.ndarray, lam: float
) -> np.ndarray:
    """Evaluate the interpolated lattice at parameter λ ∈ [0, 1].

    Uses the GL⁺(3) geodesic: ``lat(λ) = expm(λ * log_F) @ lattice_a``.

    Parameters
    ----------
    lattice_a : np.ndarray
        3×3 reference lattice (rows = vectors).
    log_F : np.ndarray
        3×3 matrix logarithm of the deformation gradient (from
        :func:`lattice_deformation_logm`).
    lam : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray
        3×3 interpolated lattice at the given λ.
    """
    from scipy.linalg import expm

    return expm(float(lam) * np.asarray(log_F, dtype=float)) @ np.asarray(
        lattice_a, dtype=float
    )