"""
Geometry utilities for molecular crystals.

This module provides coordinate transformations and geometric calculations.
"""

import numpy as np
from typing import Tuple
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


def dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
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


def minimum_image_distance(frac1: np.ndarray, frac2: np.ndarray, lattice: np.ndarray) -> float:
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
    masses = np.array([get_atomic_mass(symbol) if has_atomic_mass(symbol) 
                      else 1.0 for symbol in atom_symbols])
    
    # Calculate mass-weighted center of mass
    total_mass = np.sum(masses)
    center_of_mass = np.sum(atom_coords * masses[:, np.newaxis], axis=0) / total_mass
    
    return center_of_mass