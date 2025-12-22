"""
Generator for molecular disorder in crystals.

This module provides functionality for generating different orientations
of molecules to model orientational disorder in molecular crystals.
"""

import numpy as np
from typing import List
from ...structures.molecule import CrystalMolecule
from ...operations.rotation import rotate_molecule_at_center


def generate_orientational_disorder(
    molecule: CrystalMolecule,
    num_orientations: int = 4,
    rotation_axis: np.ndarray = None,
) -> List[CrystalMolecule]:
    """
    Generate multiple orientations of a molecule by rotation.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to generate orientations for.
    num_orientations : int, default=4
        Number of different orientations to generate.
    rotation_axis : np.ndarray, optional
        Axis around which to rotate the molecule. If None, a random axis is chosen.

    Returns
    -------
    List[CrystalMolecule]
        List of molecules in different orientations.
    """
    # If no rotation axis specified, choose a random one
    if rotation_axis is None:
        rotation_axis = np.random.randn(3)
        rotation_axis /= np.linalg.norm(rotation_axis)

    # Create list to store orientations
    orientations = []

    # Add original orientation
    orientations.append(molecule.copy())

    # Generate additional orientations
    for i in range(num_orientations - 1):
        # Create a copy of the molecule
        new_orientation = molecule.copy()

        # Calculate rotation angle
        angle = (360.0 / num_orientations) * (i + 1)

        # Apply rotation
        rotate_molecule_at_center(new_orientation, rotation_axis, angle)

        # Add to list
        orientations.append(new_orientation)

    return orientations


def generate_random_orientations(
    molecule: CrystalMolecule, num_orientations: int = 10
) -> List[CrystalMolecule]:
    """
    Generate random orientations of a molecule.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to generate orientations for.
    num_orientations : int, default=10
        Number of random orientations to generate.

    Returns
    -------
    List[CrystalMolecule]
        List of molecules in random orientations.
    """
    # Create list to store orientations
    orientations = []

    # Generate random orientations
    for i in range(num_orientations):
        # Create a copy of the molecule
        new_orientation = molecule.copy()

        # Generate random rotation axis
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)

        # Generate random rotation angle
        angle = np.random.uniform(0, 360)

        # Apply rotation
        rotate_molecule_at_center(new_orientation, axis, angle)

        # Add to list
        orientations.append(new_orientation)

    return orientations
