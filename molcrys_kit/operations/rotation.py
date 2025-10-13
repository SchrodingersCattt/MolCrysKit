"""
Rotation operations for molecular crystals.

This module provides rigid-body rotation and translation operations for molecules.
"""

import numpy as np
from typing import Tuple
from ..structures.molecule import Molecule


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Generate a rotation matrix for rotating around an axis by an angle.
    
    Parameters
    ----------
    axis : np.ndarray
        3D vector representing the rotation axis.
    angle : float
        Rotation angle in radians.
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    # Normalize the axis
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_term = np.array([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
    
    rotation = cos_angle * np.eye(3) + sin_angle * cross_term + (1 - cos_angle) * np.outer(axis, axis)
    return rotation


def rotate_molecule(molecule: Molecule, axis: np.ndarray, angle: float) -> None:
    """
    Rotate a molecule around an axis by an angle.
    
    Parameters
    ----------
    molecule : Molecule
        The molecule to rotate.
    axis : np.ndarray
        3D vector representing the rotation axis.
    angle : float
        Rotation angle in radians.
    """
    # Generate rotation matrix
    rot_matrix = rotation_matrix(axis, angle)
    
    # Apply rotation to the molecule
    molecule.rotate(rot_matrix)


def translate_molecule(molecule: Molecule, vector: np.ndarray) -> None:
    """
    Translate a molecule by a vector.
    
    Parameters
    ----------
    molecule : Molecule
        The molecule to translate.
    vector : np.ndarray
        Translation vector in fractional coordinates.
    """
    molecule.translate(vector)


def euler_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Generate a rotation matrix from Euler angles (in ZXZ convention).
    
    Parameters
    ----------
    alpha : float
        First rotation angle around Z axis (radians).
    beta : float
        Second rotation angle around X axis (radians).
    gamma : float
        Third rotation angle around Z axis (radians).
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    # Rotation matrices around principal axes
    rz_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                         [np.sin(alpha), np.cos(alpha), 0],
                         [0, 0, 1]])
    
    rx_beta = np.array([[1, 0, 0],
                        [0, np.cos(beta), -np.sin(beta)],
                        [0, np.sin(beta), np.cos(beta)]])
    
    rz_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                         [np.sin(gamma), np.cos(gamma), 0],
                         [0, 0, 1]])
    
    # Combined rotation: R = Rz(gamma) * Rx(beta) * Rz(alpha)
    return rz_gamma @ rx_beta @ rz_alpha