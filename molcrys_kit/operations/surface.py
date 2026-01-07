"""
Surface generation module for molecular crystals.

This module provides tools for generating surface slabs from molecular crystals
while preserving molecular topology during the cutting process.
"""

import numpy as np
import math
from typing import Tuple
from math import gcd
from functools import reduce

# Import internal modules
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import reduce_surface_lattice


def _extended_gcd(a, b):
    """
    Extended Euclidean Algorithm.
    Returns (g, x, y) such that a*x + b*y = gcd(a, b)
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
        Now includes orthogonalization of the stacking vector.
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
            v1 = np.array([k // g_hk, -h // g_hk, 0], dtype=int)
            v2 = np.array([p * l, q * l, -g_hk], dtype=int)

        # Find the initial stacking vector (v3)
        stacking_vector = None
        for w in range(max(abs(l), g_hk) + 1):
            rhs = 1 - l * w
            if rhs == 0 and h == 0 and k == 0:
                continue
            elif h == 0 and k == 0:
                if l * w == 1:
                    stacking_vector = np.array([0, 0, w], dtype=int)
                    break
            else:
                if h == 0:
                    if rhs % k == 0:
                        stacking_vector = np.array([0, rhs // k, w], dtype=int)
                        break
                elif k == 0:
                    if rhs % h == 0:
                        stacking_vector = np.array([rhs // h, 0, w], dtype=int)
                        break
                else:
                    if (rhs % g_hk) == 0:
                        p_hk = p * (rhs // g_hk)
                        q_hk = q * (rhs // g_hk)
                        stacking_vector = np.array([p_hk, q_hk, w], dtype=int)
                        break

        if stacking_vector is None:
            raise ValueError(
                f"Could not find a suitable stacking vector for plane ({h}, {k}, {l})"
            )

        # Get the original lattice
        old_lattice = self.crystal.lattice
        
        # Convert vectors to Cartesian
        v1_cart = np.dot(v1, old_lattice)
        v2_cart = np.dot(v2, old_lattice)
        
        # 1. Reduce surface vectors (v1, v2)
        v1_reduced, v2_reduced = reduce_surface_lattice(v1_cart, v2_cart, old_lattice)
        
        # Convert reduced v1, v2 back to integer lattice coordinates
        inv_lattice = np.linalg.inv(old_lattice)
        v1_int = np.round(np.dot(v1_reduced, inv_lattice)).astype(int)
        v2_int = np.round(np.dot(v2_reduced, inv_lattice)).astype(int)

        # 2. Orthogonalize Stacking Vector (v3) relative to v1, v2
        # This is CRITICAL for Monoclinic/Triclinic systems to avoid highly sheared slabs
        stacking_cart = np.dot(stacking_vector, old_lattice)
        
        # Update Cartesian versions of the final integer v1/v2
        v1_c = np.dot(v1_int, old_lattice)
        v2_c = np.dot(v2_int, old_lattice)

        # Iteratively subtract v1 or v2 from stacking_vector if it reduces the length
        # This is a simple greedy approach to 3D lattice reduction focusing on v3
        improved = True
        current_stacking = stacking_vector.copy()
        current_cart = stacking_cart.copy()

        while improved:
            improved = False
            best_norm = np.linalg.norm(current_cart)
            best_op = None # (coefficient for v1, coefficient for v2)

            # Try shifting by combination of v1 and v2
            # We check a small range of coefficients
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    if n1 == 0 and n2 == 0:
                        continue
                    
                    # Calculate candidate vector
                    # Since v1_int and v2_int are in the plane, adding them 
                    # doesn't change the volume constraint (Miller index projection)
                    candidate_stacking = current_stacking + n1 * v1_int + n2 * v2_int
                    candidate_cart = np.dot(candidate_stacking, old_lattice)
                    candidate_norm = np.linalg.norm(candidate_cart)

                    if candidate_norm < best_norm - 1e-5: # Use tolerance for float comparison
                        best_norm = candidate_norm
                        best_op = (n1, n2)
                        
            if best_op:
                n1, n2 = best_op
                current_stacking += n1 * v1_int + n2 * v2_int
                current_cart = np.dot(current_stacking, old_lattice)
                improved = True

        stacking_vector = current_stacking

        # Construct the transformation matrix
        transformation_matrix = np.array([v1_int, v2_int, stacking_vector]).T

        return transformation_matrix

    def build(
        self,
        miller_indices: Tuple[int, int, int],
        layers: int = 3,
        min_thickness: float = None,
        vacuum: float = 10.0,
    ) -> MolecularCrystal:
        """
        Build a surface slab with the specified Miller indices, number of layers, and vacuum.

        Parameters
        ----------
        miller_indices : Tuple[int, int, int]
            Miller indices (h, k, l) of the surface.
        layers : int, optional
            Number of unit planes in the slab. If not provided, min_thickness will be used to calculate layers.
            Defaults to 3.
        min_thickness : float, optional
            Minimum thickness of the slab in Angstroms. If provided along with layers, layers will be used.
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
        new_lattice = (
            transformation_matrix.T @ old_lattice
        )  # New basis vectors in Cartesian

        # Pre-calculate the inverse of the new lattice to avoid repeated matrix inversions
        inv_new_lattice = np.linalg.inv(new_lattice)

        # Calculate d_spacing (thickness of a single layer)
        cross_product = np.cross(new_lattice[0], new_lattice[1])
        normal_vector = cross_product / np.linalg.norm(cross_product)
        d_spacing = abs(np.dot(new_lattice[2], normal_vector))

        # Determine number of layers based on parameters
        if min_thickness is not None:
            layers = max(1, math.ceil(min_thickness / d_spacing))

        # Get unwrapped molecules to handle periodic boundary conditions correctly
        unwrapped_molecules = self.crystal.get_unwrapped_molecules()

        # Transform all molecular centroids to the new fractional coordinate system
        # Using pre-calculated inverse matrix for efficiency
        transformed_molecules = []

        for mol in unwrapped_molecules:
            # Get the centroid of the molecule in Cartesian coordinates
            centroid_cart = mol.get_centroid()

            # Convert to new fractional coordinates using the pre-calculated inverse
            centroid_frac_new = np.dot(centroid_cart, inv_new_lattice)

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

        # Now stack the molecules to create multiple layers in Cartesian coordinates
        all_molecules_list = []

        for layer_idx in range(layers):
            for mol, positions, _ in shifted_molecules:
                # Calculate the z-shift for this layer (applying the tilted stacking offset)
                layer_shift = layer_idx * new_lattice[2]  # Only z-direction shift
                layer_positions = positions + layer_shift

                # Create a copy of the molecule with the new positions
                layer_mol = mol.copy()
                layer_mol.positions = layer_positions
                all_molecules_list.append(layer_mol)

        # Calculate the normal vector to the surface (a × b)
        cross_product = np.cross(new_lattice[0], new_lattice[1])
        normal_vector = cross_product / np.linalg.norm(cross_product)

        # Calculate the projected slab thickness
        total_stacking_vector = layers * new_lattice[2]
        slab_thickness = abs(np.dot(total_stacking_vector, normal_vector))

        # Define the final output lattice where a and b are reduced surface vectors
        # and c is the orthogonal vacuum vector
        output_lattice = new_lattice.copy()
        output_lattice[2] = (
            slab_thickness + vacuum
        ) * normal_vector  # Replace the c vector with the orthogonal one

        # Calculate the minimum z-coordinate of all atoms in the slab
        all_positions = []
        for mol in all_molecules_list:
            all_positions.extend(mol.get_positions())
        all_positions = np.array(all_positions)

        # Calculate minimum z coordinate
        min_z = np.min(all_positions[:, 2]) if len(all_positions) > 0 else 0.0

        # Apply "slab-at-bottom" shift to ensure all z-coordinates are positive
        shift_vector = np.array([0.0, 0.0, -min_z + 0.05])  # small margin to avoid atoms at exactly 0
        for mol in all_molecules_list:
            mol.positions += shift_vector

        # Pre-calculate the inverse of the output lattice for efficiency
        inv_output_lattice = np.linalg.inv(output_lattice)

        # Apply selective wrapping to maintain molecular integrity
        for mol in all_molecules_list:
            # Get positions of all atoms in the molecule
            positions = mol.get_positions()
            
            # Calculate molecular centroid
            centroid_cart = mol.get_centroid()
            
            # Convert centroid to fractional coordinates of the output lattice
            centroid_frac = np.dot(centroid_cart, inv_output_lattice)
            
            # Apply wrapping only to X and Y components (indices 0 and 1)
            wrapped_centroid_frac = centroid_frac.copy()
            wrapped_centroid_frac[0] = wrapped_centroid_frac[0] - np.floor(wrapped_centroid_frac[0])
            wrapped_centroid_frac[1] = wrapped_centroid_frac[1] - np.floor(wrapped_centroid_frac[1])
            # Do NOT wrap Z component (index 2) - leave it as is
            
            # Calculate the shift vector to move the centroid to the wrapped position in X and Y
            target_centroid_cart = np.dot(wrapped_centroid_frac, output_lattice)
            shift_vector = target_centroid_cart - centroid_cart
            
            # Apply the same shift to all atoms in the molecule to maintain molecular integrity
            new_positions = positions + shift_vector
            mol.set_positions(new_positions)

        # Create the final molecular crystal with the output lattice and processed molecules
        slab = MolecularCrystal(
            lattice=output_lattice,
            molecules=all_molecules_list,  # Use the processed molecules
            pbc=(True, True, False),  # PBC is False in the surface normal (z) direction
        )

        return slab


def generate_topological_slab(
    crystal: MolecularCrystal,
    miller_indices: Tuple[int, int, int],
    layers: int = None,
    min_thickness: float = None,
    vacuum: float = 10.0,
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
        Number of unit planes in the slab. If not provided, min_thickness will be used to calculate layers.
    min_thickness : float, optional
        Minimum thickness of the slab in Angstroms. If provided along with layers, layers will be used.
        If neither is provided, defaults to 3 layers.
    vacuum : float, optional
        Thickness of vacuum region to add above the slab (in Angstroms, default: 10.0).

    Returns
    -------
    MolecularCrystal
        The generated surface slab as a MolecularCrystal object.
    """
    generator = TopologicalSlabGenerator(crystal)
    return generator.build(miller_indices, layers, min_thickness, vacuum)