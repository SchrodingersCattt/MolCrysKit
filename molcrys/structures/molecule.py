"""
Molecule representation for molecular crystals.

This module defines the Molecule class which represents a rigid body of atoms.
"""

import numpy as np
from typing import List, Tuple, Optional
from .atom import Atom
from ..constants import get_atomic_mass, has_atomic_mass, get_atomic_radius, has_atomic_radius


class Molecule:
    """
    Represents a molecule as a rigid body of atoms.
    
    Attributes
    ----------
    atoms : List[Atom]
        List of atoms that make up the molecule.
    center_of_mass : np.ndarray
        Center of mass of the molecule in fractional coordinates.
    rotation_matrix : np.ndarray or None
        3x3 rotation matrix applied to the molecule.
    lattice : np.ndarray or None
        3x3 array representing the lattice vectors (needed for proper distance calculations).
    """
    
    def __init__(self, atoms: List[Atom], center_of_mass: np.ndarray = None, 
                 rotation_matrix: np.ndarray = None, lattice: Optional[np.ndarray] = None):
        """
        Initialize a Molecule.
        
        Parameters
        ----------
        atoms : List[Atom]
            List of atoms that make up the molecule.
        center_of_mass : np.ndarray, optional
            Center of mass of the molecule. If not provided, it will be computed.
        rotation_matrix : np.ndarray, optional
            3x3 rotation matrix applied to the molecule.
        lattice : np.ndarray, optional
            3x3 array representing the lattice vectors.
        """
        self.atoms = atoms
        self.lattice = lattice
        if center_of_mass is not None:
            self.center_of_mass = np.array(center_of_mass)
        else:
            self.center_of_mass = self.compute_center_of_mass()
            
        self.rotation_matrix = rotation_matrix if rotation_matrix is not None else np.eye(3)
    
    def translate(self, vector: np.ndarray) -> None:
        """
        Translate all atoms in the molecule by a vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Translation vector in fractional coordinates.
        """
        for atom in self.atoms:
            atom.frac_coords += vector
        self.center_of_mass += vector
    
    def rotate(self, matrix: np.ndarray) -> None:
        """
        Rotate the molecule using a rotation matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            3x3 rotation matrix.
        """
        # Update the rotation matrix
        self.rotation_matrix = np.dot(matrix, self.rotation_matrix)
        
        # Rotate all atoms around the center of mass
        for atom in self.atoms:
            # Translate atom coords relative to center of mass
            rel_coords = atom.frac_coords - self.center_of_mass
            # Apply rotation
            atom.frac_coords = np.dot(matrix, rel_coords) + self.center_of_mass
    
    def compute_center_of_mass(self) -> np.ndarray:
        """
        Compute the center of mass of the molecule using atomic masses.
        
        Returns
        -------
        np.ndarray
            Center of mass in fractional coordinates.
        """
        if not self.atoms:
            return np.array([0.0, 0.0, 0.0])
        
        # Get coordinates and masses
        coords = np.array([atom.frac_coords for atom in self.atoms])
        masses = np.array([get_atomic_mass(atom.symbol) if has_atomic_mass(atom.symbol) 
                          else 1.0 for atom in self.atoms])
        
        # Calculate mass-weighted center of mass
        total_mass = np.sum(masses)
        center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        
        return center_of_mass
    
    def get_bonds(self, threshold_factor: float = 1.2) -> List[Tuple[int, int, float]]:
        """
        Identify bonds within the molecule based on distance criteria.
        
        Parameters
        ----------
        threshold_factor : float, default=1.2
            Factor to multiply covalent radii with to determine bonding threshold.
            
        Returns
        -------
        List[Tuple[int, int, float]]
            List of tuples containing (atom1_index, atom2_index, distance).
        """
        if self.lattice is None:
            raise ValueError("Lattice information is required for bond detection")
        
        bonds = []
        
        # Distance-based bond detection using atomic radii
        for i, atom1 in enumerate(self.atoms):
            radius1 = get_atomic_radius(atom1.symbol) if has_atomic_radius(atom1.symbol) else 0.5
            for j, atom2 in enumerate(self.atoms[i+1:], i+1):
                radius2 = get_atomic_radius(atom2.symbol) if has_atomic_radius(atom2.symbol) else 0.5
                
                # Calculate distance in fractional coordinates with periodic boundary conditions
                delta = atom1.frac_coords - atom2.frac_coords
                # Apply minimum image convention
                delta = delta - np.round(delta)
                # Convert to Cartesian coordinates
                cart_delta = np.dot(delta, self.lattice)
                distance = np.linalg.norm(cart_delta)
                
                # Use covalent radii sum as threshold
                threshold = (radius1 + radius2) * threshold_factor
                
                if distance < threshold and distance > 0.01:  # Avoid self-interaction
                    bonds.append((i, j, distance))
                    
        return bonds
