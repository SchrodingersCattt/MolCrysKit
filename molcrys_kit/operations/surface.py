"""
Surface generation module for molecular crystals.

This module provides tools for generating surface slabs from molecular crystals
using a robust 'Region Cropping' method that handles non-orthogonal systems correctly.
"""

import numpy as np
import math
import itertools
from typing import Tuple, List
from math import gcd
from functools import reduce

# Import internal modules
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import reduce_surface_lattice


def _extended_gcd(a, b):
    """Extended Euclidean Algorithm."""
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
    Generates surface slabs from molecular crystals using a Region Cropping approach.
    This method effectively 'slices' the crystal physically, avoiding shear artifacts
    common in monoclinic/triclinic systems.
    """

    def __init__(self, crystal: MolecularCrystal):
        self.crystal = crystal

    def _get_surface_basis(self, h: int, k: int, l: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the primitive surface vectors (v1, v2) and the surface normal.
        """
        if h == 0 and k == 0 and l == 0:
            raise ValueError("Miller indices cannot all be zero")

        # Reduce Miller indices
        g = _gcd_multiple([h, k, l])
        h, k, l = h // g, k // g, l // g

        # Find two vectors in the plane: h*u + k*v + l*w = 0
        if h == 0 and k == 0:  # (001)
            v1_int = np.array([1, 0, 0])
            v2_int = np.array([0, 1, 0])
        elif h == 0 and l == 0:  # (010)
            v1_int = np.array([0, 0, 1])
            v2_int = np.array([1, 0, 0])
        elif k == 0 and l == 0:  # (100)
            v1_int = np.array([0, 1, 0])
            v2_int = np.array([0, 0, 1])
        else:
            # General solution
            g_hk, p, q = _extended_gcd(h, k)
            if g_hk != 0:
                v1_int = np.array([k // g_hk, -h // g_hk, 0], dtype=int)
                v2_int = np.array([p * l, q * l, -g_hk], dtype=int)
            else:
                # Fallback for (0, 0, 1) if not caught above
                v1_int = np.array([1, 0, 0])
                v2_int = np.array([0, 1, 0])

        # Convert to Cartesian
        old_lattice = self.crystal.lattice
        v1_cart = np.dot(v1_int, old_lattice)
        v2_cart = np.dot(v2_int, old_lattice)

        # Apply 2D Lattice Reduction to get the most orthogonal surface cell possible
        v1_reduced, v2_reduced = reduce_surface_lattice(v1_cart, v2_cart, old_lattice)
        
        # Calculate Surface Normal
        normal = np.cross(v1_reduced, v2_reduced)
        normal = normal / np.linalg.norm(normal)
        
        return v1_reduced, v2_reduced, normal

    def build(
        self,
        miller_indices: Tuple[int, int, int],
        layers: int = 3,
        min_thickness: float = None,
        vacuum: float = 10.0,
    ) -> MolecularCrystal:
        """
        Builds the slab by cropping molecules that fall within the surface boundaries.
        """
        h, k, l = miller_indices
        
        # 1. Get Geometry
        v1, v2, normal = self._get_surface_basis(h, k, l)
        
        # Calculate d-spacing (interplanar spacing) for layer calculation
        # d = 2*pi / |G|, where G = h*b1 + k*b2 + l*b3
        # Reciprocal lattice vectors (without 2pi factor for simpler G calculation)
        # But standard formula involves 2pi. Let's use simple geometric projection.
        # d_spacing is the projection of any lattice vector traversing the plane onto the normal.
        # But simpler: d = Volume / Base_Area (for primitive step).
        # We need the SMALLEST step that repeats the plane pattern perpendicular to it.
        # Let's use reciprocal lattice logic to be safe.
        recip_lattice = 2 * np.pi * np.linalg.inv(self.crystal.lattice).T
        G_vec = h * recip_lattice[0] + k * recip_lattice[1] + l * recip_lattice[2]
        d_spacing = 2 * np.pi / np.linalg.norm(G_vec)

        # Determine target thickness
        if min_thickness is not None:
            # Round up to nearest integer layer to ensure stoichiometry/periodicity
            n_layers = max(1, math.ceil(min_thickness / d_spacing))
        else:
            n_layers = layers
            
        slab_thickness = n_layers * d_spacing

        # 2. Define the "Crop Box" Basis
        # We need to express any point P as: P = c1*v1 + c2*v2 + cn*normal
        # So we need the inverse of the matrix M = [v1, v2, normal]
        crop_basis_matrix = np.vstack([v1, v2, normal]).T
        inv_crop_basis = np.linalg.inv(crop_basis_matrix)

        # 3. Determine Search Range in Original Lattice
        # We project the 8 corners of our target slab box back to the original fractional coordinates
        # to find the min/max indices of unit cells we need to search.
        
        # Corners of the slab in Cartesian:
        # Combinations of (0, v1), (0, v2), (0, slab_thickness*normal)
        corners_cart = []
        for c1, c2, c3 in itertools.product([0, 1], [0, 1], [0, 1]):
            pt = c1 * v1 + c2 * v2 + c3 * (slab_thickness * normal)
            corners_cart.append(pt)
        
        # Convert corners to original fractional coordinates
        inv_old_lattice = np.linalg.inv(self.crystal.lattice)
        corners_frac_old = np.dot(corners_cart, inv_old_lattice)
        
        # Find bounds (add padding to be safe)
        min_indices = np.floor(np.min(corners_frac_old, axis=0)).astype(int) - 1
        max_indices = np.ceil(np.max(corners_frac_old, axis=0)).astype(int) + 1
        
        # 4. Search and Select Molecules
        # Use get_unwrapped_molecules to fix internal bonds first (CRITICAL: Requires fixed crystal.py)
        base_molecules = self.crystal.get_unwrapped_molecules()
        selected_molecules = []
        
        # Iterate over the grid of original unit cells
        ranges = [range(min_indices[i], max_indices[i] + 1) for i in range(3)]
        
        for n_a, n_b, n_c in itertools.product(*ranges):
            shift_vec = np.dot(np.array([n_a, n_b, n_c]), self.crystal.lattice)
            
            for mol in base_molecules:
                # Shift molecule to this periodic image
                new_mol = mol.copy()
                new_mol.positions += shift_vec
                
                # Check Centroid
                centroid = new_mol.get_centroid()
                
                # Convert centroid to "Crop Basis" coordinates (c1, c2, dist_normal)
                coeffs = np.dot(inv_crop_basis, centroid)
                
                c1, c2, cn = coeffs[0], coeffs[1], coeffs[2]
                
                # Selection Criteria:
                # 1. Inside the parallelogram base (0 <= c1 < 1, 0 <= c2 < 1)
                # 2. Inside the thickness (0 <= cn < slab_thickness)
                # Epsilon used to handle float precision at boundaries
                eps = 1e-4
                if (-eps <= c1 < 1.0 - eps) and \
                   (-eps <= c2 < 1.0 - eps) and \
                   (-eps <= cn < slab_thickness - eps):
                    
                    selected_molecules.append(new_mol)

        # 5. Construct Output Lattice
        # The output lattice is strictly orthogonal in Z: [v1, v2, (thick+vac)*normal]
        c_out = (slab_thickness + vacuum) * normal
        output_lattice = np.array([v1, v2, c_out])
        
        # 6. Final Adjustment (Wrapping & Centering)
        inv_output = np.linalg.inv(output_lattice)
        
        for mol in selected_molecules:
            centroid = mol.get_centroid()
            
            # Check fractional coords in output lattice
            frac = np.dot(inv_output, centroid)
            
            # Wrap X and Y (indices 0, 1)
            shift_frac = np.zeros(3)
            shift_frac[0] = -np.floor(frac[0])
            shift_frac[1] = -np.floor(frac[1])
            # Do NOT shift Z
            
            # Convert shift to Cartesian and apply
            shift_cart = np.dot(shift_frac, output_lattice) 
            mol.positions += shift_cart

        # Center in Z (Slab at bottom with small buffer)
        if selected_molecules:
            all_pos = np.vstack([m.get_positions() for m in selected_molecules])
            min_z = np.min(all_pos[:, 2]) if len(all_pos) > 0 else 0
            # Shift everything so min_z is at ~0.5 Angstrom
            z_shift = np.array([0, 0, -min_z + 0.5])
            for mol in selected_molecules:
                mol.positions += z_shift

        return MolecularCrystal(
            lattice=output_lattice,
            molecules=selected_molecules,
            pbc=(True, True, False)
        )


def generate_topological_slab(
    crystal: MolecularCrystal,
    miller_indices: Tuple[int, int, int],
    layers: int = None,
    min_thickness: float = None,
    vacuum: float = 10.0,
) -> MolecularCrystal:
    """
    Public API wrapper to generate a topological surface slab.
    """
    generator = TopologicalSlabGenerator(crystal)
    return generator.build(miller_indices, layers, min_thickness, vacuum)