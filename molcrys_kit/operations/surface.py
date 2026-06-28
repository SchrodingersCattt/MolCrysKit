"""
Surface generation module for molecular crystals.

This module provides tools for generating surface slabs from molecular crystals
while preserving molecular topology during the cutting process.  It also
provides termination enumeration and Tasker-aware termination selection.
"""

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from math import gcd
from functools import reduce

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


def _cluster_z_fracs(z_fracs: List[float], threshold_frac: float = 0.1) -> List[List[int]]:
    """
    Cluster fractional z-coordinates (in [0, 1)) into discrete layers.

    Parameters
    ----------
    z_fracs : list of float
        Fractional z-coordinates, expected in [0, 1).
    threshold_frac : float
        Maximum gap (in fractional units) within a cluster.

    Returns
    -------
    list of list of int
        Each sub-list contains the indices of molecules belonging to that
        cluster, sorted by ascending z-coordinate.
    """
    if not z_fracs:
        return []

    indexed = sorted(enumerate(z_fracs), key=lambda x: x[1])
    clusters: List[List[int]] = [[indexed[0][0]]]
    prev_z = indexed[0][1]

    for idx, zf in indexed[1:]:
        if zf - prev_z < threshold_frac:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev_z = zf

    return clusters


@dataclass
class TerminationInfo:
    """
    Metadata describing a single surface termination.

    Attributes
    ----------
    miller_index : tuple of int
        Miller indices of the surface plane.
    termination_index : int
        Zero-based index of this termination among the unique set.
    shift : float
        Fractional shift applied along the stacking direction, in [0, 1).
    topo_signature : str
        Sorted molecule-type signatures of the topmost surface layer,
        used for topo-based de-duplication.
    tasker_type : str
        Tasker classification: ``"TypeI_like"``, ``"TypeII_like"``,
        ``"TypeIII_like"``, or ``"unknown"``.
    layer_charges : list of float
        Net formal charge of each molecular layer in the fundamental unit
        cell (one layer = one repeat of d_spacing).
    dipole_per_area : float
        Magnitude of the layer-charge dipole moment divided by surface
        area (in e·Å / Å²).
    is_polar : bool
        ``True`` if *dipole_per_area* exceeds the polar tolerance.
    is_tasker_preferred : bool
        ``True`` for TypeI_like or TypeII_like non-polar terminations.
    charge_source : str
        Origin of the charge data: ``"user_map"``, ``"auto_guess"``,
        ``"none"``, or ``"neutral"`` (all charges zero, fast path).
    tasker2_corrected : bool
        ``True`` if the top-layer molecules were moved to the bottom to reduce
        the perpendicular dipole moment (Tasker Type II correction).
    """

    miller_index: Tuple[int, int, int]
    termination_index: int
    shift: float
    topo_signature: str
    tasker_type: str
    layer_charges: List[float] = field(default_factory=list)
    dipole_per_area: float = 0.0
    is_polar: bool = False
    is_tasker_preferred: bool = True
    charge_source: str = "neutral"
    tasker2_corrected: bool = False


@dataclass
class _FrameData:
    """Internal container for the surface-oriented coordinate frame."""

    rotation_matrix: np.ndarray       # (3, 3): right-multiply to rotate
    rotated_lattice: np.ndarray       # (3, 3): row vectors in rotated frame
    stacking_vector: np.ndarray       # (3,): rotated_lattice[2]
    d_spacing: float                  # Angstrom
    surface_area: float               # Angstrom^2
    shifted_mols: list                # molecules with centroid z_frac in [0, 1)
    inv_rotated_lattice: np.ndarray   # (3, 3)


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

    def _prepare_frame(self, h: int, k: int, l: int) -> "_FrameData":
        """
        Compute the surface-oriented coordinate frame for Miller plane (h, k, l).

        This encapsulates steps 1-6 of the original build() pipeline:
        lattice transformation, standard rotation, unwrapping molecules, and
        shifting all molecular centroids into the fundamental unit cell
        (z_frac in [0, 1)).

        Parameters
        ----------
        h, k, l : int
            Miller indices.

        Returns
        -------
        _FrameData
            Container holding the rotation matrix, rotated lattice, stacking
            vector, d_spacing, surface_area, shifted molecules, and the
            inverse rotated lattice.
        """
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
        a_vec, b_vec = rotated_lattice[0], rotated_lattice[1]
        normal = np.cross(a_vec, b_vec)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            raise ValueError("Surface lattice vectors are collinear.")
        normal /= normal_norm
        d_spacing = abs(np.dot(stacking_vector, normal))

        # 4. Surface area |a x b|
        surface_area = np.linalg.norm(np.cross(a_vec, b_vec))

        # 5. Get unwrapped molecules and rotate their positions
        unwrapped_molecules = self.crystal.get_unwrapped_molecules()
        rotated_mols = []
        for mol in unwrapped_molecules:
            positions = mol.get_positions() @ M  # right-mult
            mol_rot = mol.copy()
            mol_rot.positions = positions
            rotated_mols.append(mol_rot)

        # 6. Compute inverse lattice for fractional coordinates
        inv_rotated_lattice = np.linalg.inv(rotated_lattice)

        # 7. Shift all molecules to fundamental layer using rotated stacking vector
        shifted_mols = []
        for mol in rotated_mols:
            centroid = mol.get_centroid()
            frac = centroid @ inv_rotated_lattice
            z_frac = frac[2]
            shift_vec = -np.floor(z_frac) * stacking_vector
            mol_shift = mol.copy()
            mol_shift.positions = mol_shift.get_positions() + shift_vec
            shifted_mols.append(mol_shift)

        return _FrameData(
            rotation_matrix=M,
            rotated_lattice=rotated_lattice,
            stacking_vector=stacking_vector,
            d_spacing=d_spacing,
            surface_area=surface_area,
            shifted_mols=shifted_mols,
            inv_rotated_lattice=inv_rotated_lattice,
        )

    def _build_with_shift(
        self,
        frame: "_FrameData",
        shift: float = 0.0,
        layers: Optional[int] = None,
        min_thickness: Optional[float] = None,
        vacuum: float = 10.0,
        center_slab: bool = False,
    ) -> MolecularCrystal:
        """
        Build a slab for the given surface frame and fractional shift.

        Applying a non-zero shift translates all molecules by
        ``-shift * stacking_vector`` then re-wraps centroids to
        z_frac in [0, 1), effectively changing which molecular layer
        is at the top/bottom surface.

        Parameters
        ----------
        frame : _FrameData
            Pre-computed surface frame from :meth:`_prepare_frame`.
        shift : float
            Fractional shift along the stacking direction, in [0, 1).
        layers : int, optional
            Explicit number of unit planes.  Takes priority over
            *min_thickness*.
        min_thickness : float, optional
            Minimum slab thickness in Angstroms.  Used when *layers* is None.
        vacuum : float
            Vacuum thickness in Angstroms.
        center_slab : bool
            If True, vertically center the slab in the vacuum cell
            instead of anchoring the bottom at 0.05 Angstrom.

        Returns
        -------
        MolecularCrystal
            The generated slab.
        """
        stacking_vector = frame.stacking_vector
        d_spacing = frame.d_spacing
        inv_rotated_lattice = frame.inv_rotated_lattice

        # Determine number of layers
        if layers is None:
            if min_thickness is not None:
                if d_spacing < 1e-5:
                    raise ValueError(f"d_spacing ({d_spacing:.6f}) is too small.")
                layers = max(1, int(np.ceil(min_thickness / d_spacing)))
            else:
                raise ValueError("Either layers or min_thickness must be specified.")

        # Apply shift and re-wrap centroid z_frac to [0, 1)
        applied_mols = []
        for mol in frame.shifted_mols:
            mol_copy = mol.copy()
            mol_copy.positions -= shift * stacking_vector
            centroid = mol_copy.get_centroid()
            frac = centroid @ inv_rotated_lattice
            z_frac = frac[2]
            mol_copy.positions += -np.floor(z_frac) * stacking_vector
            applied_mols.append(mol_copy)

        # Stack layers
        all_mols = []
        for i in range(layers):
            layer_shift = i * stacking_vector
            for mol in applied_mols:
                mol_layer = mol.copy()
                mol_layer.positions = mol_layer.get_positions() + layer_shift
                all_mols.append(mol_layer)

        # Compute slab thickness
        slab_thickness = layers * d_spacing

        # Define final orthogonal lattice: a, b as before, c = [0,0,slab_thickness+vacuum]
        output_lattice = frame.rotated_lattice.copy()
        output_lattice[2] = np.array([0, 0, slab_thickness + vacuum])

        # Center slab in XY: move geometric center to (0.5, 0.5) fractional
        all_positions = np.vstack([mol.get_positions() for mol in all_mols])
        if len(all_positions) > 0:
            xy_cart_center = np.mean(all_positions[:, :2], axis=0)
            inv_ab = np.linalg.inv(output_lattice[:2, :2])
            xy_frac_center = xy_cart_center @ inv_ab
            shift_frac = np.array([0.5, 0.5]) - xy_frac_center
            shift_cart = shift_frac @ output_lattice[:2, :2]
            for mol in all_mols:
                mol.positions[:, :2] += shift_cart

        # Rigid body wrapping in X/Y only
        inv_output_lattice = np.linalg.inv(output_lattice)
        for mol in all_mols:
            centroid = mol.get_centroid()
            frac = centroid @ inv_output_lattice
            wrapped_frac = frac.copy()
            wrapped_frac[0] = wrapped_frac[0] % 1.0
            wrapped_frac[1] = wrapped_frac[1] % 1.0
            # Z unchanged
            target_centroid = wrapped_frac @ output_lattice
            shift_vec = target_centroid - centroid
            mol.set_positions(mol.get_positions() + shift_vec)

        # Z positioning
        all_positions = np.vstack([mol.get_positions() for mol in all_mols])
        min_z = np.min(all_positions[:, 2]) if all_positions.size > 0 else 0.0
        if center_slab:
            max_z = np.max(all_positions[:, 2])
            slab_z_center = (min_z + max_z) / 2.0
            cell_z_center = output_lattice[2, 2] / 2.0
            z_shift = cell_z_center - slab_z_center
        else:
            z_shift = 0.05 - min_z
        for mol in all_mols:
            mol.positions[:, 2] += z_shift

        # Assemble final MolecularCrystal
        slab = MolecularCrystal(
            lattice=output_lattice,
            molecules=all_mols,
            pbc=(True, True, False),
        )
        return slab

    def build(
        self,
        miller_indices: Tuple[int, int, int],
        layers: int = None,  # default None, distinguish user-specified vs not
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
        frame = self._prepare_frame(h, k, l)
        return self._build_with_shift(
            frame,
            shift=0.0,
            layers=layers,
            min_thickness=min_thickness,
            vacuum=vacuum,
            center_slab=False,
        )


def _get_termination_topo_signature(
    shifted_mols: list,
    candidate_shift: float,
    inv_rotated_lattice: np.ndarray,
    stacking_vector: np.ndarray,
    d_spacing: float,
    threshold_frac: float,
    z_fracs: Optional[List[float]] = None,
) -> str:
    """
    Compute the topo signature of the topmost molecular layer for a given shift.

    Parameters
    ----------
    shifted_mols : list of CrystalMolecule
        Molecules with centroid z_frac in [0, 1) in the fundamental cell.
    candidate_shift : float
        Fractional shift to apply.
    inv_rotated_lattice : np.ndarray
        Inverse of the rotated lattice matrix.
    stacking_vector : np.ndarray
        Stacking direction vector.
    d_spacing : float
        Spacing between layers in Angstrom.
    threshold_frac : float
        Cluster width threshold in fractional units.
    z_fracs : list of float, optional
        Pre-computed z-fractional coordinates. If None, will be computed.

    Returns
    -------
    str
        Pipe-joined sorted list of molecule topo signatures for the top layer.
    """
    from ..analysis.charge import compute_topo_signature

    if z_fracs is None:
        new_z_fracs = []
        for mol in shifted_mols:
            centroid = mol.get_centroid()
            frac = centroid @ inv_rotated_lattice
            z_frac = frac[2]
            new_z_fracs.append((z_frac - candidate_shift) % 1.0)
    else:
        new_z_fracs = [(z - candidate_shift) % 1.0 for z in z_fracs]

    clusters = _cluster_z_fracs(new_z_fracs, threshold_frac=threshold_frac)
    if not clusters:
        return ""

    cluster_means = [
        sum(new_z_fracs[i] for i in c) / len(c) for c in clusters
    ]
    top_cluster_idx = int(np.argmax(cluster_means))
    top_mol_indices = clusters[top_cluster_idx]

    mol_sigs = sorted(
        [compute_topo_signature(shifted_mols[i]) for i in top_mol_indices]
    )
    return "|".join(mol_sigs)


def _evaluate_tasker(
    shifted_mols: list,
    candidate_shift: float,
    frame: _FrameData,
    charge_results: Dict,
    threshold_frac: float,
    tasker_polar_tol: float,
    charge_tol: float,
    z_fracs: Optional[List[float]] = None,
) -> Dict:
    """
    Evaluate Tasker classification for a surface termination.

    Parameters
    ----------
    shifted_mols : list
        Molecules with centroid z_frac in [0, 1).
    candidate_shift : float
        Fractional shift to apply along the stacking direction.
    frame : _FrameData
        Surface coordinate frame.
    charge_results : dict
        Mapping topo_signature -> MolChargeResult from assign_mol_formal_charges.
    threshold_frac : float
        Layer clustering threshold in fractional units.
    tasker_polar_tol : float
        Threshold for dipole_per_area to classify as polar.
    charge_tol : float
        Threshold for layer_charge to classify as neutral.
    z_fracs : list of float, optional
        Pre-computed z-fractional coordinates. If None, will be computed.

    Returns
    -------
    dict
        Keys: layer_charges, dipole_per_area, is_polar, tasker_type,
        charge_source, is_tasker_preferred.
    """
    from ..analysis.charge import compute_topo_signature

    if z_fracs is None:
        new_z_fracs = []
        for mol in shifted_mols:
            centroid = mol.get_centroid()
            frac = centroid @ frame.inv_rotated_lattice
            z_frac = frac[2]
            new_z_fracs.append((z_frac - candidate_shift) % 1.0)
    else:
        new_z_fracs = [(z - candidate_shift) % 1.0 for z in z_fracs]

    clusters = _cluster_z_fracs(new_z_fracs, threshold_frac=threshold_frac)
    if not clusters:
        return {
            "layer_charges": [],
            "dipole_per_area": 0.0,
            "is_polar": False,
            "tasker_type": "unknown",
            "charge_source": "none",
            "is_tasker_preferred": False,
        }

    # Pre-compute signatures for all molecules once
    mol_signatures = [compute_topo_signature(mol) for mol in shifted_mols]

    # Determine overall charge source (worst priority among all molecules).
    # Start from the best possible source; any worse source found will override.
    source_priority = {"user_map": 0, "auto_guess": 1, "neutral": 2, "none": 3}
    overall_source = "user_map"
    for sig in mol_signatures:
        if sig in charge_results:
            src = charge_results[sig].source
            if source_priority.get(src, 99) > source_priority.get(overall_source, 99):
                overall_source = src

    # Compute per-layer charges and z-centers
    layer_charges = []
    layer_z_centers = []
    for cluster_indices in clusters:
        q = sum(
            charge_results[mol_signatures[i]].formal_charge
            if mol_signatures[i] in charge_results
            else 0.0
            for i in cluster_indices
        )
        z_center = (
            sum(new_z_fracs[i] * frame.d_spacing for i in cluster_indices)
            / len(cluster_indices)
        )
        layer_charges.append(q)
        layer_z_centers.append(z_center)

    # Fast path: all layers neutral
    all_neutral = all(abs(q) < charge_tol for q in layer_charges)
    if all_neutral:
        return {
            "layer_charges": layer_charges,
            "dipole_per_area": 0.0,
            "is_polar": False,
            "tasker_type": "TypeI_like",
            "charge_source": overall_source,
            "is_tasker_preferred": True,
        }

    # Compute dipole per unit area
    dipole = sum(q * z for q, z in zip(layer_charges, layer_z_centers))
    dipole_per_area = (
        abs(dipole) / frame.surface_area if frame.surface_area > 0 else 0.0
    )
    is_polar = dipole_per_area > tasker_polar_tol

    if overall_source == "none":
        tasker_type = "unknown"
        is_preferred = False
    elif is_polar:
        tasker_type = "TypeIII_like"
        is_preferred = False
    else:
        tasker_type = "TypeII_like"
        is_preferred = True

    return {
        "layer_charges": layer_charges,
        "dipole_per_area": dipole_per_area,
        "is_polar": is_polar,
        "tasker_type": tasker_type,
        "charge_source": overall_source,
        "is_tasker_preferred": is_preferred,
    }


def enumerate_terminations(
    crystal: MolecularCrystal,
    miller_index: Tuple[int, int, int],
    unique_terminations: str = "topo",
    termination_resolution: Optional[float] = None,
    symmetry_reduction: bool = False,
    mol_charge_map: Optional[Dict[str, int]] = None,
    tasker_polar_tol: float = 1e-3,
    charge_tol: float = 0.05,
    _precomputed_frame=None,
) -> List[TerminationInfo]:
    """
    Enumerate topologically unique surface terminations for a given Miller plane.

    Candidate shifts are determined by clustering molecular centroids along the
    stacking direction, then taking the midpoint of each inter-layer gap.
    De-duplication is performed at the level of molecule types (topo signatures)
    present in the topmost surface layer.

    Tasker classification is applied to each unique termination using molecular
    formal charges (hybrid strategy: user_map -> pymatgen guess -> zero).

    Parameters
    ----------
    crystal : MolecularCrystal
        The bulk crystal to analyse.
    miller_index : tuple of int
        Miller indices (h, k, l) of the desired surface plane.
    unique_terminations : str
        ``"topo"`` (default) de-duplicates by surface-layer molecule type.
        ``"none"`` keeps all candidate shifts.
    termination_resolution : float, optional
        Minimum inter-layer gap (in Angstrom) below which two molecular layers
        are considered the same layer.  Defaults to 0.5 Angstrom.
    symmetry_reduction : bool
        If True, treat terminations whose top-layer signature matches another's
        bottom-layer signature as equivalent.
    mol_charge_map : dict, optional
        Formula -> formal charge map passed to assign_mol_formal_charges.
    tasker_polar_tol : float
        Dipole-per-area threshold (e*Ang/Ang^2) for polar classification.
    charge_tol : float
        Layer-charge threshold (e) below which a layer is considered neutral.

    Returns
    -------
    list of TerminationInfo
        Unique terminations sorted by Tasker preference (preferred first),
        then by shift value.
    """
    from ..analysis.charge import assign_mol_formal_charges

    h, k, l = miller_index
    if _precomputed_frame is not None:
        frame = _precomputed_frame
    else:
        gen = TopologicalSlabGenerator(crystal)
        frame = gen._prepare_frame(h, k, l)

    # Default resolution: 0.5 Ang in fractional units
    if termination_resolution is None:
        termination_resolution = 0.5
    threshold_frac = (
        termination_resolution / frame.d_spacing if frame.d_spacing > 0 else 0.05
    )

    # Compute z_fracs for all shifted molecules
    z_fracs = []
    for mol in frame.shifted_mols:
        centroid = mol.get_centroid()
        frac = centroid @ frame.inv_rotated_lattice
        z_fracs.append(frac[2])

    clusters = _cluster_z_fracs(z_fracs, threshold_frac=threshold_frac)
    n_clusters = len(clusters)

    # Cluster centres (mean z_frac), sorted ascending
    cluster_centers_raw = [
        sum(z_fracs[i] for i in c) / len(c) for c in clusters
    ]
    sorted_centers = sorted(cluster_centers_raw)

    # Candidate shifts: midpoint between consecutive clusters (periodic)
    candidate_shifts = []
    for i in range(n_clusters):
        z_a = sorted_centers[i]
        if i < n_clusters - 1:
            z_b = sorted_centers[i + 1]
            shift = (z_a + z_b) / 2.0
        else:
            # Periodic gap: between last cluster and next period of first cluster
            z_b = sorted_centers[0]
            shift = ((z_a + z_b + 1.0) / 2.0) % 1.0
        candidate_shifts.append(shift)

    # Assign formal charges once for the whole crystal
    charge_results = assign_mol_formal_charges(crystal, mol_charge_map)

    # Fast path check: all formal charges are 0
    all_neutral = all(
        abs(r.formal_charge) < charge_tol for r in charge_results.values()
    )

    seen_topo_sigs: set = set()
    seen_bottom_sigs: set = set()
    result_list: List[TerminationInfo] = []

    for shift in candidate_shifts:
        topo_sig = _get_termination_topo_signature(
            frame.shifted_mols,
            shift,
            frame.inv_rotated_lattice,
            frame.stacking_vector,
            frame.d_spacing,
            threshold_frac,
            z_fracs=z_fracs,
        )

        if unique_terminations == "topo" and topo_sig in seen_topo_sigs:
            continue

        # Optional symmetry reduction
        if symmetry_reduction:
            from ..analysis.charge import compute_topo_signature

            new_z_fracs_shift = [
                (z - shift) % 1.0 for z in z_fracs
            ]
            bottom_clusters = _cluster_z_fracs(
                new_z_fracs_shift, threshold_frac=threshold_frac
            )
            if bottom_clusters:
                bot_cluster_means = [
                    sum(new_z_fracs_shift[i] for i in c) / len(c)
                    for c in bottom_clusters
                ]
                bot_idx = int(np.argmin(bot_cluster_means))
                bottom_mol_sigs = sorted(
                    [
                        compute_topo_signature(frame.shifted_mols[i])
                        for i in bottom_clusters[bot_idx]
                    ]
                )
                bottom_sig = "|".join(bottom_mol_sigs)
                if bottom_sig in seen_bottom_sigs:
                    continue
                seen_bottom_sigs.add(bottom_sig)

        seen_topo_sigs.add(topo_sig)

        # Evaluate Tasker classification
        if all_neutral:
            # Determine the actual charge source even when all charges happen to be 0.
            # "user_map" or "auto_guess" override the generic "neutral" label so callers
            # can distinguish explicit user input from the pure-fallback fast path.
            has_user_map = any(
                r.source == "user_map" for r in charge_results.values()
            )
            has_auto_guess = any(
                r.source == "auto_guess" for r in charge_results.values()
            )
            if has_user_map:
                fast_path_source = "user_map"
            elif has_auto_guess:
                fast_path_source = "auto_guess"
            else:
                fast_path_source = "neutral"  # all fell back to 0 (source="none")
            # Compute actual layer count for this shift so layer_charges is non-empty.
            _shifted_z = [(z - shift) % 1.0 for z in z_fracs]
            _shift_clusters = _cluster_z_fracs(_shifted_z, threshold_frac=threshold_frac)
            tasker_info = {
                "layer_charges": [0.0] * len(_shift_clusters),
                "dipole_per_area": 0.0,
                "is_polar": False,
                "tasker_type": "TypeI_like",
                "charge_source": fast_path_source,
                "is_tasker_preferred": True,
            }
        else:
            tasker_info = _evaluate_tasker(
                frame.shifted_mols,
                shift,
                frame,
                charge_results,
                threshold_frac,
                tasker_polar_tol,
                charge_tol,
                z_fracs=z_fracs,
            )

        info = TerminationInfo(
            miller_index=miller_index,
            termination_index=len(result_list),
            shift=shift,
            topo_signature=topo_sig,
            tasker_type=tasker_info["tasker_type"],
            layer_charges=tasker_info["layer_charges"],
            dipole_per_area=tasker_info["dipole_per_area"],
            is_polar=tasker_info["is_polar"],
            is_tasker_preferred=tasker_info["is_tasker_preferred"],
            charge_source=tasker_info["charge_source"],
        )
        result_list.append(info)

    # Sort: preferred first, then by shift
    result_list.sort(key=lambda ti: (not ti.is_tasker_preferred, ti.shift))

    return result_list


def _apply_tasker2_correction(
    slab: MolecularCrystal,
    ti: "TerminationInfo",
    charge_results: Dict,
    threshold_frac: float = 0.05,
) -> Optional[MolecularCrystal]:
    """
    Attempt a Tasker Type II dipole correction on a slab.

    Moves the topmost molecular layer to the bottom of the slab (by
    translating those molecules by ``-slab_c`` along the stacking axis).
    If the resulting perpendicular dipole moment is strictly smaller than
    the original, the corrected slab is returned; otherwise ``None`` is
    returned to signal that the correction did not help.

    Parameters
    ----------
    slab : MolecularCrystal
        The slab to correct.  Its ``lattice[2, 2]`` is taken as the cell
        height (``slab_c``).
    ti : TerminationInfo
        Termination metadata for *slab*, used to obtain ``layer_charges``
        and ``dipole_per_area``.
    charge_results : dict
        Mapping topo_signature -> MolChargeResult (from
        :func:`assign_mol_formal_charges`).
    threshold_frac : float
        Fractional z-clustering threshold used to identify the top layer.

    Returns
    -------
    MolecularCrystal or None
        Corrected slab if the dipole was reduced, else ``None``.
    """
    from ..analysis.charge import compute_topo_signature

    slab_c = float(slab.lattice[2, 2])
    if slab_c < 1e-5:
        return None

    # Compute fractional z of each molecule centroid within the slab cell
    mols = list(slab.molecules)
    z_fracs = [float(mol.get_centroid()[2]) / slab_c for mol in mols]

    clusters = _cluster_z_fracs(z_fracs, threshold_frac=threshold_frac)
    if not clusters:
        return None

    # Identify the topmost layer
    cluster_means = [
        sum(z_fracs[i] for i in c) / len(c) for c in clusters
    ]
    top_idx = int(np.argmax(cluster_means))
    top_mol_indices = set(clusters[top_idx])

    # Build corrected molecule list: move top layer to bottom
    corrected_mols = []
    for j, mol in enumerate(mols):
        mol_copy = mol.copy()
        if j in top_mol_indices:
            mol_copy.positions[:, 2] -= slab_c
        corrected_mols.append(mol_copy)

    # Recompute layer z-fracs and charges for the corrected slab
    new_z_fracs = [float(mol.get_centroid()[2]) / slab_c for mol in corrected_mols]
    new_clusters = _cluster_z_fracs(new_z_fracs, threshold_frac=threshold_frac)

    # Pre-compute signatures for all molecules once
    mol_signatures = [compute_topo_signature(mol) for mol in mols]
    corrected_mol_signatures = [compute_topo_signature(mol) for mol in corrected_mols]

    def _dipole_from_clusters(mol_list, mol_sigs, zf_list, clust_list):
        dipole = 0.0
        for c in clust_list:
            q = sum(
                charge_results[mol_sigs[i]].formal_charge
                if mol_sigs[i] in charge_results
                else 0.0
                for i in c
            )
            z_center = sum(zf_list[i] * slab_c for i in c) / len(c)
            dipole += q * z_center
        return dipole

    surface_area = float(np.linalg.norm(
        np.cross(slab.lattice[0], slab.lattice[1])
    ))
    if surface_area < 1e-10:
        return None

    dipole_old = abs(_dipole_from_clusters(mols, mol_signatures, z_fracs, clusters)) / surface_area
    dipole_new = abs(_dipole_from_clusters(corrected_mols, corrected_mol_signatures, new_z_fracs, new_clusters)) / surface_area

    if dipole_new >= dipole_old:
        return None

    # Recompute actual slab extent after correction
    all_z_corrected = np.concatenate([mol.get_positions()[:, 2] for mol in corrected_mols])
    min_z_corr = np.min(all_z_corrected)
    max_z_corr = np.max(all_z_corrected)

    # Recover the original vacuum from the original slab
    original_slab_z = np.concatenate([mol.get_positions()[:, 2] for mol in mols])
    original_min_z = np.min(original_slab_z)
    original_max_z = np.max(original_slab_z)
    original_slab_extent = original_max_z - original_min_z
    original_vacuum = slab.lattice[2, 2] - original_slab_extent

    # Build new lattice with same vacuum
    new_lattice = slab.lattice.copy()
    new_lattice[2, 2] = (max_z_corr - min_z_corr) + original_vacuum

    # Re-anchor: shift all molecules so min_z = same small offset as original
    z_shift = original_min_z - min_z_corr
    for mol in corrected_mols:
        mol.positions[:, 2] += z_shift

    corrected_slab = MolecularCrystal(
        lattice=new_lattice,
        molecules=corrected_mols,
        pbc=slab.pbc,
    )
    return corrected_slab


def generate_slabs_with_terminations(
    structure_or_crystal: MolecularCrystal,
    miller_index: Tuple[int, int, int],
    min_slab_size: Optional[float] = None,
    layers: Optional[int] = None,
    min_vacuum_size: float = 10.0,
    center_slab: bool = True,
    unique_terminations: str = "topo",
    termination_resolution: Optional[float] = None,
    symmetry_reduction: bool = False,
    term_selection: str = "tasker_preferred",
    termination_indices: Optional[List[int]] = None,
    mol_charge_map: Optional[Dict[str, int]] = None,
    tasker_polar_tol: float = 1e-3,
    charge_tol: float = 0.05,
    correct_tasker2: bool = False,
) -> List[Tuple[MolecularCrystal, TerminationInfo]]:
    """
    Generate surface slabs for one or more terminations of a Miller plane.

    Parameters
    ----------
    structure_or_crystal : MolecularCrystal
        Bulk crystal input.
    miller_index : tuple of int
        Miller indices (h, k, l).
    min_slab_size : float, optional
        Minimum slab thickness in Angstrom.  Used when *layers* is None.
    layers : int, optional
        Explicit number of unit planes.  Overrides *min_slab_size*.
    min_vacuum_size : float
        Vacuum thickness in Angstrom (default 10.0).
    center_slab : bool
        Centre the slab vertically within the vacuum cell (default True).
    unique_terminations : str
        Passed to :func:`enumerate_terminations`.
    termination_resolution : float, optional
        Passed to :func:`enumerate_terminations`.
    symmetry_reduction : bool
        Passed to :func:`enumerate_terminations`.
    term_selection : str
        ``"tasker_preferred"`` (default) -- return only Tasker-preferred
        terminations; falls back to all with a warning if none qualify.
        ``"all"`` -- return all unique terminations.
        ``"by_index"`` -- return terminations whose index is in
        *termination_indices*.
    termination_indices : list of int, optional
        Required when *term_selection* is ``"by_index"``.
    mol_charge_map : dict, optional
        Formula -> formal charge map.
    tasker_polar_tol : float
        Passed to :func:`enumerate_terminations`.
    charge_tol : float
        Passed to :func:`enumerate_terminations`.
    correct_tasker2 : bool
        If ``True``, attempt to reduce the perpendicular dipole of
        ``TypeII_like`` slabs by moving the topmost molecular layer to the
        bottom.  When the correction reduces the dipole, the returned slab
        is the corrected version and ``TerminationInfo.tasker2_corrected``
        is set to ``True``.  Has no effect on TypeI_like or TypeIII_like
        terminations.  Default ``False``.

    Returns
    -------
    list of (MolecularCrystal, TerminationInfo)
        Each element is a (slab, info) pair, ordered as determined by
        *term_selection*.
    """
    from ..analysis.charge import assign_mol_formal_charges

    if layers is None and min_slab_size is None:
        min_slab_size = 8.0

    # Build frame once and reuse for both enumerate_terminations and slab building
    h, k, l = miller_index
    gen = TopologicalSlabGenerator(structure_or_crystal)
    frame = gen._prepare_frame(h, k, l)

    termination_infos = enumerate_terminations(
        crystal=structure_or_crystal,
        miller_index=miller_index,
        unique_terminations=unique_terminations,
        termination_resolution=termination_resolution,
        symmetry_reduction=symmetry_reduction,
        mol_charge_map=mol_charge_map,
        tasker_polar_tol=tasker_polar_tol,
        charge_tol=charge_tol,
        _precomputed_frame=frame,
    )

    if not termination_infos:
        return []

    # Filter terminations by selection strategy
    if term_selection == "tasker_preferred":
        selected = [ti for ti in termination_infos if ti.is_tasker_preferred]
        if not selected:
            warnings.warn(
                "No Tasker-preferred terminations found; returning all terminations.",
                UserWarning,
                stacklevel=2,
            )
            selected = termination_infos
    elif term_selection == "all":
        selected = termination_infos
    elif term_selection == "by_index":
        if termination_indices is None:
            raise ValueError(
                "termination_indices must be specified when term_selection='by_index'."
            )
        idx_set = set(termination_indices)
        selected = [ti for ti in termination_infos if ti.termination_index in idx_set]
    else:
        raise ValueError(
            f"Unknown term_selection='{term_selection}'. "
            "Choose 'tasker_preferred', 'all', or 'by_index'."
        )

    # Pre-compute charge results once (needed for TypeII correction)
    charge_results = assign_mol_formal_charges(structure_or_crystal, mol_charge_map) \
        if correct_tasker2 else {}

    # Build slab for each selected termination (reuse the same frame)

    results = []
    for ti in selected:
        slab = gen._build_with_shift(
            frame=frame,
            shift=ti.shift,
            layers=layers,
            min_thickness=min_slab_size,
            vacuum=min_vacuum_size,
            center_slab=center_slab,
        )

        # Attempt TypeII dipole correction when requested
        if correct_tasker2 and ti.tasker_type == "TypeII_like":
            corrected = _apply_tasker2_correction(slab, ti, charge_results)
            if corrected is not None:
                slab = corrected
                ti.tasker2_corrected = True

        results.append((slab, ti))

    return results


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
