"""
Rotation operations for molecular crystals.

This module provides functions for rotating molecules and crystals.
"""

import numpy as np
from typing import Tuple
from ..structures.molecule import CrystalMolecule


def rotate_molecule_at_center(molecule: CrystalMolecule, axis: np.ndarray, angle: float) -> None:
    """
    Rotate a molecule around its centroid by a specified angle around a given axis.
    
    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to rotate.
    axis : np.ndarray
        The rotation axis as a 3D vector.
    angle : float
        The rotation angle in degrees.
    """
    # Normalize the rotation axis
    axis = np.array(axis) / np.linalg.norm(axis)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Get the centroid of the molecule
    centroid = molecule.get_centroid()
    
    # Get current positions
    positions = molecule.get_positions()
    
    # Translate molecule to origin (centered at centroid)
    translated_positions = positions - centroid
    
    # Create rotation matrix using Rodrigues' rotation formula
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    cross_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = (
        cos_angle * np.eye(3) +
        sin_angle * cross_matrix +
        (1 - cos_angle) * np.outer(axis, axis)
    )
    
    # Apply rotation
    rotated_positions = np.dot(translated_positions, rotation_matrix.T)
    
    # Translate back to original position
    new_positions = rotated_positions + centroid
    
    # Update molecule positions
    molecule.set_positions(new_positions)


def rotate_molecule_at_com(molecule: CrystalMolecule, axis: np.ndarray, angle: float) -> None:
    """
    Rotate a molecule around its center of mass by a specified angle around a given axis.
    
    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to rotate.
    axis : np.ndarray
        The rotation axis as a 3D vector.
    angle : float
        The rotation angle in degrees.
    """
    # Normalize the rotation axis
    axis = np.array(axis) / np.linalg.norm(axis)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Get the center of mass of the molecule
    center_of_mass = molecule.get_center_of_mass()
    
    # Get current positions
    positions = molecule.get_positions()
    
    # Translate molecule to origin (centered at center of mass)
    translated_positions = positions - center_of_mass
    
    # Create rotation matrix using Rodrigues' rotation formula
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    cross_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = (
        cos_angle * np.eye(3) +
        sin_angle * cross_matrix +
        (1 - cos_angle) * np.outer(axis, axis)
    )
    
    # Apply rotation
    rotated_positions = np.dot(translated_positions, rotation_matrix.T)
    
    # Translate back to original position
    new_positions = rotated_positions + center_of_mass
    
    # Update molecule positions
    molecule.set_positions(new_positions)