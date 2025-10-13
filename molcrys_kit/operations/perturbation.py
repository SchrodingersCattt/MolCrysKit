"""
Perturbation operations for molecular crystals.

This module applies random or directed perturbations to atoms or molecules.
"""

import numpy as np
from typing import Tuple
from ..structures.atom import Atom
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


def apply_gaussian_displacement_atom(atom: Atom, sigma: float) -> None:
    """
    Apply random Gaussian displacement to an atom.
    
    Parameters
    ----------
    atom : Atom
        The atom to perturb.
    sigma : float
        Standard deviation of the Gaussian distribution (in fractional coordinates).
    """
    # Generate random displacement
    displacement = np.random.normal(0, sigma, 3)
    
    # Apply displacement
    atom.frac_coords += displacement


def apply_gaussian_displacement_molecule(molecule: Molecule, sigma: float) -> None:
    """
    Apply random Gaussian displacement to a molecule.
    
    Parameters
    ----------
    molecule : Molecule
        The molecule to perturb.
    sigma : float
        Standard deviation of the Gaussian distribution (in fractional coordinates).
    """
    # Generate random displacement
    displacement = np.random.normal(0, sigma, 3)
    
    # Apply displacement to all atoms in the molecule
    molecule.translate(displacement)


def apply_gaussian_displacement_crystal(crystal: MolecularCrystal, sigma: float) -> None:
    """
    Apply random Gaussian displacement to all atoms in a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to perturb.
    sigma : float
        Standard deviation of the Gaussian distribution (in fractional coordinates).
    """
    # Apply displacement to all atoms
    for molecule in crystal.molecules:
        apply_gaussian_displacement_molecule(molecule, sigma)


def apply_anisotropic_displacement(atom: Atom, sigma_x: float, sigma_y: float, sigma_z: float) -> None:
    """
    Apply anisotropic Gaussian displacement to an atom.
    
    Parameters
    ----------
    atom : Atom
        The atom to perturb.
    sigma_x : float
        Standard deviation along x-axis.
    sigma_y : float
        Standard deviation along y-axis.
    sigma_z : float
        Standard deviation along z-axis.
    """
    # Generate random displacements for each direction
    dx = np.random.normal(0, sigma_x)
    dy = np.random.normal(0, sigma_y)
    dz = np.random.normal(0, sigma_z)
    
    # Apply displacement
    atom.frac_coords += np.array([dx, dy, dz])


def apply_directional_displacement(molecule: Molecule, direction: np.ndarray, magnitude: float) -> None:
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
    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)
    
    # Apply displacement
    displacement = direction * magnitude
    molecule.translate(displacement)