"""
Operations for applying perturbations to molecular crystals.

This module provides functions for applying various types of displacements
and perturbations to molecular crystals and individual molecules.
"""

import numpy as np
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import get_rotation_matrix


def apply_gaussian_displacement_molecule(
    molecule: CrystalMolecule, sigma: float
) -> None:
    """
    Apply random Gaussian displacement to a molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule to perturb.
    sigma : float
        Standard deviation of the Gaussian distribution (in fractional coordinates).
    """
    # Get current positions
    positions = molecule.get_positions()

    # Generate random displacements
    displacements = np.random.normal(0, sigma, positions.shape)

    # Apply displacements
    new_positions = positions + displacements
    molecule.set_positions(new_positions)


def apply_gaussian_displacement_crystal(crystal, sigma: float) -> None:
    """
    Apply random Gaussian displacement to all molecules in a crystal.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to perturb.
    sigma : float
        Standard deviation of the Gaussian displacement (in Angstroms).
    """
    for molecule in crystal.molecules:
        apply_gaussian_displacement_molecule(molecule, sigma)


def apply_directional_displacement(
    molecule: CrystalMolecule, direction: np.ndarray, magnitude: float
) -> None:
    """
    Apply directed displacement to a molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule to displace.
    direction : np.ndarray
        Direction vector for displacement.
    magnitude : float
        Magnitude of displacement (in fractional coordinates).
    """
    # Normalize direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Calculate displacement vector
    displacement = direction * magnitude

    # Get current positions
    positions = molecule.get_positions()

    # Apply displacement
    new_positions = positions + displacement
    molecule.set_positions(new_positions)


def apply_random_rotation(molecule: CrystalMolecule, max_angle: float = 10.0) -> None:
    """
    Apply a random rotation to a molecule.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to rotate.
    max_angle : float, default=10.0
        Maximum rotation angle in degrees.
    """
    # Generate random rotation axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    # Generate random rotation angle
    angle = np.random.uniform(0, max_angle)

    # Convert to radians
    angle_rad = np.radians(angle)

    # Get the rotation matrix using the utility function
    rotation_matrix = get_rotation_matrix(axis, angle_rad)

    # Get molecule centroid
    centroid = molecule.get_centroid()

    # Get current positions
    positions = molecule.get_positions()

    # Translate molecule to origin
    translated_positions = positions - centroid

    # Apply rotation
    rotated_positions = np.dot(translated_positions, rotation_matrix.T)

    # Translate back
    new_positions = rotated_positions + centroid

    # Update positions
    molecule.set_positions(new_positions)