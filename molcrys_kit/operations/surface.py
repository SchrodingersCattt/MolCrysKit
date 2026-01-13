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

    @staticmethod
    def _get_standard_rotation_matrix(lattice: np.ndarray) -> np.ndarray:
        """
        Returns a rotation matrix M such that:
        - lattice[0] @ M aligns with X axis
        - lattice[1] @ M lies in XY plane (Y >= 0)
        - lattice[2] @ M points generally +Z
        All input/output are row vectors. Use right-multiplication: rotated = original @ M
        """
        a = lattice[0]
        b = lattice[1]
        # Normalize a to X
        x_axis = a / np.linalg.norm(a)
        # Remove x component from b, then normalize to get Y
        b_proj = b - np.dot(b, x_axis) * x_axis
        y_axis = b_proj / np.linalg.norm(b_proj)
        # Z is right-handed
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        # Ensure z points generally +Z (not -Z)
        if z_axis[2] < 0:
            y_axis = -y_axis
            z_axis = -z_axis
        # Compose rotation matrix (columns are new axes)
        M = np.stack([x_axis, y_axis, z_axis], axis=1)
        return M

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
            # (001) surface
            v1 = np.array([1, 0, 0], dtype=int)
            v2 = np.array([0, 1, 0], dtype=int)
            stacking_vector = np.array([0, 0, 1 if l > 0 else -1], dtype=int)

            transformation_matrix = np.array([v1, v2, stacking_vector]).T
            return transformation_matrix
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
        for w in range(
            max(abs(l), g_hk) + 1
        ):  # Changed from abs(l) + 1 to max(abs(l), g_hk) + 1 to ensure we check enough values
            # Now solve h*u + k*v = 1 - l*w
            rhs = 1 - l * w
            # Solve h*u + k*v = rhs - l*w for u and v
            # Using the extended Euclidean algorithm approach
            if h == 0:
                if rhs % k == 0:
                    stacking_vector = np.array([0, rhs // k, w], dtype=int)
                    break
            elif k == 0:
                if rhs % h == 0:
                    stacking_vector = np.array([rhs // h, 0, w], dtype=int)
                    break
            else:
                # Use extended Euclidean to find a particular solution
                # g_hk was already calculated earlier, no need to recalculate
                if (rhs % g_hk) == 0:  # Check if solution exists
                    # Scale the solution (p and q were calculated earlier)
                    p_hk = p * (rhs // g_hk)
                    q_hk = q * (rhs // g_hk)
                    stacking_vector = np.array([p_hk, q_hk, w], dtype=int)
                    break

        if stacking_vector is None:
            raise ValueError(
                f"Could not find a suitable stacking vector for plane ({h}, {k}, {l})"
            )

        # Get the original lattice to use for surface lattice reduction
        old_lattice = self.crystal.lattice

        # Convert the initial v1 and v2 vectors to Cartesian coordinates
        v1_cart = np.dot(v1, old_lattice)
        v2_cart = np.dot(v2, old_lattice)

        # Apply Gauss reduction to get more orthogonal surface vectors
        v1_reduced, v2_reduced = reduce_surface_lattice(v1_cart, v2_cart, old_lattice)

        # Convert the reduced vectors back to lattice coordinates
        inv_lattice = np.linalg.inv(old_lattice)
        v1_reduced_lat = np.dot(v1_reduced, inv_lattice)
        v2_reduced_lat = np.dot(v2_reduced, inv_lattice)

        # Round to integers to get the transformation matrix
        v1_int = np.round(v1_reduced_lat).astype(int)
        v2_int = np.round(v2_reduced_lat).astype(int)

        # Construct the transformation matrix (as column vectors)
        transformation_matrix = np.array([v1_int, v2_int, stacking_vector]).T

        return transformation_matrix

    def build(
        self,
        miller_indices: Tuple[int, int, int],
        layers: int = None,  # 修改默认值为 None，以便区分是否由用户指定
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
            Number of unit planes in the slab. If provided, it takes precedence.
        min_thickness : float, optional
            Minimum thickness of the slab in Angstroms. Used to calculate layers if layers is None.
        vacuum : float
            Thickness of vacuum region to add above the slab (in Angstroms).

        Returns
        -------
        MolecularCrystal
            The generated surface slab as a MolecularCrystal object.
        """
        h, k, l = miller_indices

        # 1. Get primitive surface transformation matrix
        transformation_matrix = self._get_primitive_surface_vectors(h, k, l)
        old_lattice = self.crystal.lattice
        raw_surface_lattice = (
            transformation_matrix.T @ old_lattice
        )  # shape (3,3), row vectors

        # 2. Rotate to standard orientation
        M = self._get_standard_rotation_matrix(raw_surface_lattice)
        rotated_lattice = raw_surface_lattice @ M  # shape (3,3), row vectors

        # 3. Get stacking vector in rotated frame
        stacking_vector = rotated_lattice[2]        
        
        # Calculate d_spacing (slab thickness of 1 layer)
        # Since we rotated the lattice such that a and b are in the XY plane, 
        # the normal is simply the Z-axis (or very close to it).
        # We calculate it strictly using cross product for robustness.
        a_vec, b_vec = rotated_lattice[0], rotated_lattice[1]
        normal = np.cross(a_vec, b_vec)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
             raise ValueError("Surface lattice vectors are collinear.")
        normal /= normal_norm
        
        d_spacing = abs(np.dot(stacking_vector, normal))
        
        # Determine number of layers
        if layers is None:
            if min_thickness is not None:
                if d_spacing < 1e-5:
                    raise ValueError(f"d_spacing ({d_spacing}) is too small to build a slab.")
                layers = int(np.ceil(min_thickness / d_spacing))
                # Ensure at least 1 layer
                layers = max(1, layers)
            else:
                raise ValueError("Either layers or min_thickness must be specified.")

        # 4. Get unwrapped molecules and rotate their positions
        unwrapped_molecules = self.crystal.get_unwrapped_molecules()
        rotated_mols = []
        for mol in unwrapped_molecules:
            positions = mol.get_positions() @ M  # right-mult
            mol_rot = mol.copy()
            mol_rot.positions = positions
            rotated_mols.append(mol_rot)

        # 5. Compute inverse lattice for fractional coordinates
        inv_rotated_lattice = np.linalg.inv(rotated_lattice)

        # 6. Shift all molecules to fundamental layer using rotated stacking vector
        shifted_mols = []
        for mol in rotated_mols:
            centroid = mol.get_centroid()
            frac = centroid @ inv_rotated_lattice
            z_frac = frac[2]
            shift_vec = -np.floor(z_frac) * stacking_vector
            mol_shift = mol.copy()
            mol_shift.positions = mol_shift.get_positions() + shift_vec
            shifted_mols.append(mol_shift)

        # 7. Stack layers in rotated frame
        all_mols = []
        for i in range(layers):
            layer_shift = i * stacking_vector
            for mol in shifted_mols:
                mol_layer = mol.copy()
                mol_layer.positions = mol_layer.get_positions() + layer_shift
                all_mols.append(mol_layer)

        # 8. Compute slab thickness
        # (d_spacing was calculated earlier, so we just use it)
        slab_thickness = layers * d_spacing

        # 9. Define final orthogonal lattice: a, b as before, c = [0,0,slab_thickness+vacuum]
        output_lattice = rotated_lattice.copy()
        output_lattice[2] = np.array([0, 0, slab_thickness + vacuum])

        # 10. Center slab in XY: move geometric center to (0.5, 0.5) fractional
        all_positions = np.vstack([mol.get_positions() for mol in all_mols])
        if len(all_positions) > 0:
            xy_cart_center = np.mean(all_positions[:, :2], axis=0)
            # Convert to fractional
            inv_ab = np.linalg.inv(output_lattice[:2, :2])
            xy_frac_center = xy_cart_center @ inv_ab
            shift_frac = np.array([0.5, 0.5]) - xy_frac_center
            shift_cart = shift_frac @ output_lattice[:2, :2]
            for mol in all_mols:
                mol.positions[:, :2] += shift_cart

        # 11. Rigid body wrapping in X/Y only
        inv_output_lattice = np.linalg.inv(output_lattice)
        for mol in all_mols:
            centroid = mol.get_centroid()
            frac = centroid @ inv_output_lattice
            wrapped_frac = frac.copy()
            wrapped_frac[0] = wrapped_frac[0] % 1.0
            wrapped_frac[1] = wrapped_frac[1] % 1.0
            # Z unchanged
            target_centroid = wrapped_frac @ output_lattice
            shift = target_centroid - centroid
            mol.set_positions(mol.get_positions() + shift)

        # 12. Shift slab so minimum z is at 0.05 Å
        all_positions = np.vstack([mol.get_positions() for mol in all_mols])
        min_z = np.min(all_positions[:, 2]) if all_positions.size > 0 else 0.0
        z_shift = 0.05 - min_z
        for mol in all_mols:
            mol.positions[:, 2] += z_shift

        # 13. Assemble final MolecularCrystal
        slab = MolecularCrystal(
            lattice=output_lattice,
            molecules=all_mols,
            pbc=(True, True, False),
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
