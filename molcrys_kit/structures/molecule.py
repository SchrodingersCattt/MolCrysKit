"""
Molecule representation for molecular crystals.

This module defines the Molecule class which represents a rigid body of atoms.
"""

import numpy as np
from typing import List, Tuple, Optional
from .atom import Atom
from ..constants import get_atomic_mass, has_atomic_mass, get_atomic_radius, has_atomic_radius, is_metal_element, METAL_THRESHOLD_FACTOR, NON_METAL_THRESHOLD_FACTOR

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder for type hints


class EnhancedMolecule:
    """
    An enhanced wrapper for ASE Atoms objects with additional geometric properties and methods.
    
    This class wraps an ASE Atoms object and provides additional methods for computing
    molecular properties such as ellipsoid radii, centroids, and other geometric characteristics.
    
    Attributes
    ----------
    atoms : Atoms
        The underlying ASE Atoms object representing the molecule.
    """
    
    def __init__(self, atoms: Atoms):
        """
        Initialize an EnhancedMolecule.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object representing the molecule.
        """
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required for EnhancedMolecule. Please install it with 'pip install ase'")
        # Create a wrapped version of the atoms to handle PBC correctly
        self.atoms = atoms.copy()
        self._adjust_positions_for_pbc()
    
    def _adjust_positions_for_pbc(self):
        """
        Adjust atomic positions to be contiguous for molecules that span periodic boundaries.
        """
        if len(self.atoms) <= 1:
            return
        
        # Get positions
        positions = self.atoms.get_positions()
        
        # Use the first atom as reference
        reference_pos = positions[0]
        
        # Adjust all other atoms to be close to the reference atom considering PBC
        cell = self.atoms.get_cell()
        for i in range(1, len(positions)):
            # Calculate the difference vector
            diff = positions[i] - reference_pos
            
            # Apply minimum image convention
            # Transform to fractional coordinates
            frac_diff = np.linalg.solve(cell.T, diff)
            
            # Wrap fractional coordinates to [-0.5, 0.5]
            frac_diff = frac_diff - np.round(frac_diff)
            
            # Transform back to Cartesian coordinates
            diff_wrapped = np.dot(frac_diff, cell)
            
            # Update position
            positions[i] = reference_pos + diff_wrapped
        
        # Set the adjusted positions
        self.atoms.set_positions(positions)
    
    @property
    def positions(self):
        """Get atomic positions."""
        return self.atoms.get_positions()
    
    @property
    def symbols(self):
        """Get chemical symbols."""
        return self.atoms.get_chemical_symbols()
    
    @property
    def numbers(self):
        """Get atomic numbers."""
        return self.atoms.get_atomic_numbers()
    
    def get_centroid(self) -> np.ndarray:
        """
        Calculate the centroid (geometric center) of the molecule.
        
        Returns
        -------
        np.ndarray
            Centroid coordinates (x, y, z).
        """
        return np.mean(self.positions, axis=0)
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate the center of mass of the molecule.
        
        Returns
        -------
        np.ndarray
            Center of mass coordinates (x, y, z).
        """
        masses = self.atoms.get_masses()
        return np.average(self.positions, axis=0, weights=masses)
    
    def get_ellipsoid_radii(self) -> Tuple[float, float, float]:
        """
        Calculate the radii of the ellipsoid that best fits the molecule.
        
        This method computes the principal axes based on the distribution of atoms
        in 3D space using singular value decomposition.
        
        Returns
        -------
        Tuple[float, float, float]
            The three radii (semi-axes) of the fitted ellipsoid, sorted in descending order.
        """
        # Handle single atom case
        if len(self.atoms) == 1:
            # For a single atom, use its atomic radius for all three axes
            symbol = self.symbols[0]
            atomic_radius = get_atomic_radius(symbol) if has_atomic_radius(symbol) else 0.5
            return (atomic_radius, atomic_radius, atomic_radius)
        
        # Get positions relative to centroid
        centroid = self.get_centroid()
        rel_positions = self.positions - centroid
        
        # Perform singular value decomposition
        # This gives us the principal axes and their scales
        try:
            _, singular_values, _ = np.linalg.svd(rel_positions, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback in case of SVD failure
            return (0.01, 0.01, 0.01)
        
        # The singular values represent the scale (spread) along each principal axis
        # We convert them to effective radii by scaling with a factor that accounts
        # for the number of atoms and provides a more realistic size estimate
        n_atoms = len(self.atoms)
        # Use sqrt(5) factor to get a more reasonable estimate of the molecular extent
        # This is based on the idea that atoms are roughly distributed in a sphere
        radii = singular_values * np.sqrt(5 / n_atoms)
        
        # Sort in descending order
        sorted_radii = np.sort(radii)[::-1]
        
        # Ensure we always return 3 values
        while len(sorted_radii) < 3:
            sorted_radii = np.append(sorted_radii, 0.01)
            
        return tuple(sorted_radii[:3])
    
    def get_principal_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the principal axes of the molecule.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The three principal axes as normalized vectors.
        """
        # Handle single atom case
        if len(self.atoms) == 1:
            # For a single atom, return arbitrary orthogonal axes
            return (np.array([1.0, 0.0, 0.0]), 
                    np.array([0.0, 1.0, 0.0]), 
                    np.array([0.0, 0.0, 1.0]))
        
        # Get positions relative to centroid
        centroid = self.get_centroid()
        rel_positions = self.positions - centroid
        
        # Perform singular value decomposition to get principal axes
        try:
            u, _, vh = np.linalg.svd(rel_positions, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback in case of SVD failure
            return (np.array([1.0, 0.0, 0.0]), 
                    np.array([0.0, 1.0, 0.0]), 
                    np.array([0.0, 0.0, 1.0]))
        
        # The principal axes are the rows of vh (V transpose)
        # They are already normalized by SVD
        axes = vh
        # Ensure we always return 3 axes
        while len(axes) < 3:
            axes = np.append(axes, [np.array([0.0, 0.0, 0.0])], axis=0)
            
        return tuple(axes[i] if np.linalg.norm(axes[i]) > 0 else np.array([1.0, 0.0, 0.0]) 
                    for i in range(3))


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
    
    def __init__(self, atoms: List[Atom], center_of_mass: Optional[np.ndarray] = None, 
                 rotation_matrix: Optional[np.ndarray] = None, lattice: Optional[np.ndarray] = None):
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
    
    def get_bonds(self) -> List[Tuple[int, int, float]]:
        """
        Identify bonds within the molecule based on distance criteria.
        
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
            is_metal1 = is_metal_element(atom1.symbol)
            for j, atom2 in enumerate(self.atoms[i+1:], i+1):
                radius2 = get_atomic_radius(atom2.symbol) if has_atomic_radius(atom2.symbol) else 0.5
                is_metal2 = is_metal_element(atom2.symbol)
                
                # Calculate distance in fractional coordinates with periodic boundary conditions
                delta = atom1.frac_coords - atom2.frac_coords
                # Apply minimum image convention
                delta = delta - np.round(delta)
                # Convert to Cartesian coordinates
                cart_delta = np.dot(delta, self.lattice)
                distance = np.linalg.norm(cart_delta)
                
                # Determine threshold factor based on element types
                if is_metal1 and is_metal2:  # Metal-Metal
                    factor = METAL_THRESHOLD_FACTOR
                elif not is_metal1 and not is_metal2:  # Non-metal-Non-metal
                    factor = NON_METAL_THRESHOLD_FACTOR
                else:  # Metal-Non-metal
                    factor = (METAL_THRESHOLD_FACTOR + NON_METAL_THRESHOLD_FACTOR) / 2
                
                threshold = (radius1 + radius2) * factor
                
                if distance < threshold and distance > 0.01:  # Avoid self-interaction
                    bonds.append((i, j, distance))
                    
        return bonds