"""
Molecule representation for molecular crystals.

This module defines the Molecule class which represents a rigid body of atoms.
"""

import numpy as np
from typing import List, Tuple
from .atom import Atom


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
    """
    
    def __init__(self, atoms: List[Atom], center_of_mass: np.ndarray = None, 
                 rotation_matrix: np.ndarray = None):
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
        """
        self.atoms = atoms
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
        Compute the center of mass of the molecule.
        
        Returns
        -------
        np.ndarray
            Center of mass in fractional coordinates.
        """
        if not self.atoms:
            return np.array([0.0, 0.0, 0.0])
        
        # Simple center of mass calculation (assuming all atoms have equal mass)
        coords = np.array([atom.frac_coords for atom in self.atoms])
        return np.mean(coords, axis=0)
    
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
        # This is a simplified implementation
        # In a full implementation, we would use covalent radii from a database
        bonds = []
        
        # Simple distance-based bond detection
        # Using fixed threshold for demonstration
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms[i+1:], i+1):
                distance = np.linalg.norm(atom1.frac_coords - atom2.frac_coords)
                # Simplified bonding criteria - in reality would use covalent radii
                if distance < 0.2 * threshold_factor:  
                    bonds.append((i, j, distance))
                    
        return bonds