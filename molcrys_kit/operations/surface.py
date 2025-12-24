"""
Surface generation module for molecular crystals.

This module provides tools for generating surface slabs from molecular crystals
while preserving molecular topology during the cutting process.
"""

import numpy as np
from typing import Tuple, List, Optional
from math import gcd
from functools import reduce

# Import internal modules
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import cart_to_frac, frac_to_cart


def _gcd_multiple(numbers):
    """Calculate the GCD of multiple numbers."""
    return reduce(gcd, numbers)


class TopologicalSlabGenerator:
    """
    Generates surface slabs from molecular crystals while preserving molecular topology.
    
    This class generates surface slabs based on molecular topology, ensuring that
    no intramolecular bonds are broken during the cutting process. Molecules are
    treated as rigid units, and their inclusion in a layer is determined by their
    centroid position.
    """
    
    def __init__(self, crystal: MolecularCrystal):
        """
        Initialize the TopologicalSlabGenerator with a crystal structure.
        
        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to generate the surface slab from.
        """
        self.crystal = crystal
    
    def _get_primitive_surface_vectors(self, h: int, k: int, l: int) -> np.ndarray:
        """
        Derives the integer basis transformation matrix (3x3) for the surface.
        
        Given Miller indices (h, k, l), this method finds two in-plane lattice
        vectors (u, v) that lie in the plane and a third vector (w) that is
        perpendicular to the plane (stacking direction).
        
        Parameters
        ----------
        h, k, l : int
            Miller indices of the surface plane.
        
        Returns
        -------
        np.ndarray
            3x3 transformation matrix where rows are the new basis vectors
            in terms of the original lattice coordinates.
        
        Raises
        ------
        ValueError
            If all Miller indices are zero.
        """
        if h == 0 and k == 0 and l == 0:
            raise ValueError("Miller indices cannot all be zero")
        
        # Create a copy of the lattice for calculations
        lattice = self.crystal.lattice
        
        # Find two vectors in the plane (hkl)
        # If h and k are not both 0
        if abs(h) > 0 or abs(k) > 0:
            # v1 is perpendicular to [h, k, l]
            v1 = np.array([k, -h, 0], dtype=int)
            # v2 is also perpendicular to [h, k, l] and independent of v1
            # v2 = [l*h, l*k, -(h^2 + k^2)]
            v2 = np.array([l * h, l * k, -(h * h + k * k)], dtype=int)
        else:  # h=0, k=0 (plane is (001))
            v1 = np.array([1, 0, 0], dtype=int)
            v2 = np.array([0, 1, 0], dtype=int)
        
        # Reduce v1 and v2 to primitive vectors by dividing by their GCDs
        v1_gcd = _gcd_multiple(np.abs(v1))
        if v1_gcd > 0:
            v1 = v1 // v1_gcd
        else:
            v1 = np.array([1, 0, 0], dtype=int)  # fallback
            
        v2_gcd = _gcd_multiple(np.abs(v2))
        if v2_gcd > 0:
            v2 = v2 // v2_gcd
        else:
            v2 = np.array([0, 1, 0], dtype=int)  # fallback
        
        # Find the stacking vector (w) such that it's not in the plane
        # Try simple vectors [1,0,0], [0,1,0], [0,0,1] to find one that's not perpendicular to [h,k,l]
        stacking_vector = None
        min_det = float('inf')
        best_w = None
        
        # Try various simple vectors as possible stacking directions
        for w_test in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]:
            w_test = np.array(w_test)
            # Check if w_test is not in the plane (h*k_test[0] + k*w_test[1] + l*w_test[2] != 0)
            dot_product = h * w_test[0] + k * w_test[1] + l * w_test[2]
            if dot_product != 0:
                # Calculate determinant of matrix [v1, v2, w_test]
                test_matrix = np.array([v1, v2, w_test]).T
                det = abs(np.linalg.det(test_matrix))
                if det < min_det:
                    min_det = det
                    best_w = w_test.copy()
        
        # If we couldn't find a suitable vector, default to [0, 0, 1]
        if best_w is None:
            best_w = np.array([0, 0, 1], dtype=int)
        
        # Construct the transformation matrix (as column vectors)
        transformation_matrix = np.array([v1, v2, best_w]).T
        
        return transformation_matrix
    
    def build(self, miller_indices: Tuple[int, int, int], layers: int, vacuum: float) -> MolecularCrystal:
        """
        Build a surface slab with the specified Miller indices, number of layers, and vacuum.
        
        Parameters
        ----------
        miller_indices : Tuple[int, int, int]
            Miller indices (h, k, l) of the surface.
        layers : int
            Number of layers in the slab.
        vacuum : float
            Thickness of vacuum region to add above the slab (in Angstroms).
        
        Returns
        -------
        MolecularCrystal
            The generated surface slab as a MolecularCrystal object.
        """
        h, k, l = miller_indices
        
        # Get the transformation matrix
        transformation_matrix = self._get_primitive_surface_vectors(h, k, l)
        
        # Calculate the new lattice
        old_lattice = self.crystal.lattice
        new_lattice = transformation_matrix.T @ old_lattice  # New basis vectors in Cartesian
        
        # Get unwrapped molecules to handle periodic boundary conditions correctly
        unwrapped_molecules = self.crystal.get_unwrapped_molecules()
        
        # Transform all molecular centroids to the new fractional coordinate system
        new_lattice_inv = np.linalg.inv(new_lattice)
        transformed_molecules = []
        
        for mol in unwrapped_molecules:
            # Get the centroid of the molecule in Cartesian coordinates
            centroid_cart = mol.get_centroid()
            
            # Convert to new fractional coordinates
            centroid_frac_new = centroid_cart @ new_lattice_inv
            
            # Store the transformed centroid along with a COPY of the molecule
            transformed_molecules.append((mol.copy(), centroid_frac_new))
        
        # Shift molecules to the fundamental layer (0 <= z < 1)
        final_molecules = []
        for mol, centroid_frac_new in transformed_molecules:
            # Shift only the z-coordinate (index 2) to be in [0, 1)
            z_shift = -centroid_frac_new[2] // 1  # How many unit cells to shift
            adjusted_centroid = centroid_frac_new.copy()
            adjusted_centroid[2] += z_shift + 1  # Shift to [0, 1) range
            
            # Shift all atoms in the molecule by the same amount
            positions = mol.get_positions()
            shift_vector = (z_shift + 1) * new_lattice[2]  # Only z-direction shift
            mol.positions = positions + shift_vector
            
            final_molecules.append((mol, adjusted_centroid))
        
        # Now stack the molecules to create multiple layers
        all_atoms_list = []
        
        for layer_idx in range(layers):
            for mol, _ in transformed_molecules:
                # Create a copy of the molecule for this layer
                layer_mol = mol.copy()
                
                # Shift the z-coordinate by layer_idx unit cells
                shift_vector = layer_idx * new_lattice[2]  # Only z-direction shift
                layer_mol.positions = layer_mol.get_positions() + shift_vector
                
                all_atoms_list.append(layer_mol)
        
        # Calculate the normal vector to the surface (a × b)
        cross_product = np.cross(new_lattice[0], new_lattice[1])
        normal_vector = cross_product / np.linalg.norm(cross_product)
        
        # Modify the c vector to add vacuum in the direction normal to the surface
        vacuum_vector = vacuum * normal_vector
        final_lattice = new_lattice.copy()
        final_lattice[2] = layers * new_lattice[2] + vacuum_vector
        
        # Create the final molecular crystal with the new lattice
        final_molecules = all_atoms_list
        
        # Create a new MolecularCrystal with the new lattice and molecules
        # Set PBC to (True, True, False) for a surface, or (True, True, True) with vacuum
        # Here we'll use (True, True, True) since we're adding vacuum
        slab = MolecularCrystal(
            lattice=final_lattice,
            molecules=[mol for mol in final_molecules],  # All molecules in the stacked layers
            pbc=(True, True, True)
        )
        
        return slab


def generate_topological_slab(
    crystal: MolecularCrystal,
    miller_indices: Tuple[int, int, int],
    layers: int = 3,
    vacuum: float = 10.0
) -> MolecularCrystal:
    """
    Public API wrapper to generate a topological surface slab.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to generate the surface slab from.
    miller_indices : Tuple[int, int, int]
        Miller indices (h, k, l) of the surface.
    layers : int, optional
        Number of layers in the slab (default: 3).
    vacuum : float, optional
        Thickness of vacuum region to add above the slab (in Angstroms, default: 10.0).
    
    Returns
    -------
    MolecularCrystal
        The generated surface slab as a MolecularCrystal object.
    """
    generator = TopologicalSlabGenerator(crystal)
    return generator.build(miller_indices, layers, vacuum)