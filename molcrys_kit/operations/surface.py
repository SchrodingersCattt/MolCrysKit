"""
Surface generation module for molecular crystals.

This module provides tools for generating surface slabs from molecular crystals
while preserving molecular topology during the cutting process.
"""

import numpy as np
from typing import Tuple
from math import gcd
from functools import reduce

# Import internal modules
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import frac_to_cart, cart_to_frac


def _extended_gcd(a, b):
    """
    Extended Euclidean Algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)
    """
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


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
        
        # Reduce Miller indices to be coprime
        g = _gcd_multiple([h, k, l])
        h, k, l = h // g, k // g, l // g
        
        # Handle special case where plane is parallel to z-axis (001)
        if h == 0 and k == 0:
            v1 = np.array([1, 0, 0], dtype=int)
            v2 = np.array([0, 1, 0], dtype=int)
        else:
            # General case using Extended Euclidean Algorithm
            g_hk, p, q = _extended_gcd(h, k)
            # v1 is perpendicular to [h, k, l] and primitive along its direction
            v1 = np.array([k // g_hk, -h // g_hk, 0], dtype=int)
            # v2 completes the primitive basis for the plane
            v2 = np.array([p * l, q * l, -g_hk], dtype=int)
        
        # Find the stacking vector (v3) such that h*v3[0] + k*v3[1] + l*v3[2] = 1 (Bezout's identity)
        # We need to solve h*u + k*v + l*w = 1 for integers u, v, w
        # Since gcd(h, k, l) = 1, a solution exists
        stacking_vector = None
        for w in range(abs(l) + 1):  # Try different values of w
            # Now solve h*u + k*v = 1 - l*w
            rhs = 1 - l * w
            if rhs == 0 and h == 0 and k == 0:
                # This case shouldn't happen since h, k, l are coprime and not all zero
                continue
            elif h == 0 and k == 0:
                # Then l*w = 1, so l=1, w=1 or l=-1, w=-1
                if l * w == 1:
                    stacking_vector = np.array([0, 0, w], dtype=int)
                    break
            else:
                # Solve h*u + k*v = rhs - l*w for u and v
                # Using the extended Euclidean algorithm approach
                if h == 0:
                    if rhs % k == 0:
                        stacking_vector = np.array([0, rhs // k, w], dtype=int)
                        break
                elif k == 0:
                    if rhs % h == 0:
                        stacking_vector = np.array(rhs // h, 0, w, dtype=int)
                        break
                else:
                    # Use extended Euclidean to find a particular solution
                    g_hk, p_hk, q_hk = _extended_gcd(h, k)
                    if (rhs % g_hk) == 0:  # Check if solution exists
                        # Scale the solution
                        p_hk *= rhs // g_hk
                        q_hk *= rhs // g_hk
                        stacking_vector = np.array([p_hk, q_hk, w], dtype=int)
                        break
        
        # If we couldn't find a suitable vector with the above method, try brute force
        if stacking_vector is None:
            # Try simple vectors [1,0,0], [0,1,0], [0,0,1] and their negatives
            for w_test in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]:
                w_test = np.array(w_test)
                dot_product = h * w_test[0] + k * w_test[1] + l * w_test[2]
                if dot_product == 1:
                    stacking_vector = w_test
                    break
                elif dot_product == -1:
                    stacking_vector = -w_test  # This would give dot product of 1
                    break

        if stacking_vector is None:
            raise ValueError(f"Could not find a suitable stacking vector for plane ({h}, {k}, {l})")
        
        # Construct the transformation matrix (as column vectors)
        transformation_matrix = np.array([v1, v2, stacking_vector]).T
        
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
        transformed_molecules = []
        
        for mol in unwrapped_molecules:
            # Get the centroid of the molecule in Cartesian coordinates
            centroid_cart = mol.get_centroid()
            
            # Convert to new fractional coordinates using the geometry utility
            centroid_frac_new = cart_to_frac(centroid_cart, new_lattice)
            
            # Store the transformed centroid along with the molecule data
            transformed_molecules.append((mol, centroid_frac_new))
        
        # Shift molecules to the fundamental layer (0 <= z < 1)
        shifted_molecules = []
        for mol, centroid_frac_new in transformed_molecules:
            # Shift only the z-coordinate (index 2) to be in [0, 1)
            # Use modulo to map to [0, 1) range
            z_new = centroid_frac_new[2] % 1.0
            adjusted_centroid = centroid_frac_new.copy()
            adjusted_centroid[2] = z_new

            # Shift all atoms in the molecule by the same amount
            positions = mol.get_positions()
            # Calculate shift in Cartesian coordinates
            shift_vector = (z_new - centroid_frac_new[2]) * new_lattice[2]
            new_positions = positions + shift_vector
            
            # Store molecule with new positions and adjusted centroid
            shifted_molecules.append((mol, new_positions, adjusted_centroid))
        
        # Now stack the molecules to create multiple layers
        all_atoms_list = []
        
        for layer_idx in range(layers):
            for mol, positions, _ in shifted_molecules:
                # Calculate the z-shift for this layer
                layer_shift = layer_idx * new_lattice[2]  # Only z-direction shift
                layer_positions = positions + layer_shift
                
                # Create a copy of the molecule with the new positions
                layer_mol = mol.copy()
                layer_mol.positions = layer_positions
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