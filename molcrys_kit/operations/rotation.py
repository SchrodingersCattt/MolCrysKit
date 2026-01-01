"""
Rotation operations for molecular crystals.

This module provides functions for rotating molecules and crystals.
"""

import numpy as np
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import get_rotation_matrix


def _rotate_molecule_around_point(molecule: CrystalMolecule, point: np.ndarray, axis: np.ndarray, angle: float) -> None:
    """
    Rotate a molecule around a specified point by a specified angle around a given axis.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to rotate.
    point : np.ndarray
        The point around which to rotate the molecule.
    axis : np.ndarray
        The rotation axis as a 3D vector.
    angle : float
        The rotation angle in degrees.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Get the rotation matrix using the utility function
    rotation_matrix = get_rotation_matrix(axis, angle_rad)

    # Get current positions
    positions = molecule.get_positions()

    # Translate molecule to origin (centered at the specified point)
    translated_positions = positions - point

    # Apply rotation
    rotated_positions = np.dot(translated_positions, rotation_matrix.T)

    # Translate back to original position
    new_positions = rotated_positions + point

    # Update molecule positions
    molecule.set_positions(new_positions)


def rotate_molecule_at_center(
    molecule: CrystalMolecule, axis: np.ndarray, angle: float
) -> None:
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
    centroid = molecule.get_centroid()
    _rotate_molecule_around_point(molecule, centroid, axis, angle)


def rotate_molecule_at_com(
    molecule: CrystalMolecule, axis: np.ndarray, angle: float
) -> None:
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
    center_of_mass = molecule.get_center_of_mass()
    _rotate_molecule_around_point(molecule, center_of_mass, axis, angle)