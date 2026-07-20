"""
Crystal reorientation module.

Provides utilities for reorienting a molecular crystal so that a specified
crystallographic direction (Miller indices) is aligned with a target
Cartesian axis.  This is useful for setting up MSST shock simulations,
strain loading along arbitrary directions, and other applications requiring
a specific axis alignment.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..structures.crystal import MolecularCrystal
from ..utils.geometry import orient_lattice
from .surface import get_surface_basis


_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


@dataclass
class ReorientationInfo:
    """
    Metadata from a crystal reorientation operation.

    Attributes
    ----------
    rotation_matrix : np.ndarray
        3×3 rotation matrix (right-multiply convention: ``coords @ M``).
    transformation_matrix : np.ndarray
        3×3 integer Miller-to-supercell basis transformation matrix.
    rotated_lattice : np.ndarray
        3×3 array of reoriented lattice vectors as rows.
    d_spacing : float
        Layer spacing along the target axis (Å).
    surface_area : float
        Area of the in-plane unit cell (Å²).
    supercell_factor : int
        Ratio of reoriented cell volume to original cell volume
        (``abs(det(transformation_matrix))``).  For the Bezout-based
        stacking construction this is always 1 (primitive cell preserved);
        the field is retained for future multi-layer stacking variants.
    """

    rotation_matrix: np.ndarray
    transformation_matrix: np.ndarray
    rotated_lattice: np.ndarray
    d_spacing: float
    surface_area: float
    supercell_factor: int


def reorient_crystal(
    crystal: MolecularCrystal,
    direction: tuple[int, int, int],
    target_axis: str = "z",
    reduce_2d: bool = True,
) -> tuple[MolecularCrystal, ReorientationInfo]:
    """
    Reorient a molecular crystal so that a given crystallographic direction
    is aligned with a Cartesian axis.

    The operation constructs a (generally supercell) unit cell whose stacking
    vector points along *target_axis*, with in-plane vectors spanning the
    complementary plane.  All molecular positions are rigidly rotated; no
    bonds are broken.

    Parameters
    ----------
    crystal : MolecularCrystal
        Input crystal structure.
    direction : tuple of int
        Miller indices (h, k, l) defining the crystallographic plane whose
        normal should be aligned with *target_axis*.
    target_axis : str
        Cartesian axis to align to: ``'x'``, ``'y'``, or ``'z'`` (default).
    reduce_2d : bool
        If True (default), apply 2D Gauss lattice reduction to the in-plane
        vectors for near-orthogonality.

    Returns
    -------
    Tuple[MolecularCrystal, ReorientationInfo]
        The reoriented crystal and associated metadata.

    Raises
    ------
    ValueError
        If *direction* is (0, 0, 0) or *target_axis* is invalid.

    Examples
    --------
    >>> from molcrys_kit.io import load_crystal
    >>> from molcrys_kit.operations import reorient_crystal
    >>> crystal = load_crystal("NaCl.cif")
    >>> reoriented, info = reorient_crystal(crystal, (1, 1, 0), target_axis='z')
    >>> print(f"d-spacing: {info.d_spacing:.3f} Å, factor: {info.supercell_factor}x")
    """
    if target_axis not in _AXIS_MAP:
        raise ValueError(
            f"target_axis must be 'x', 'y', or 'z', got {target_axis!r}"
        )

    h, k, l = direction
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller direction cannot be (0, 0, 0).")

    axis_idx = _AXIS_MAP[target_axis]

    # 1. Get integer basis transformation from Miller indices.
    #    get_surface_basis applies Gauss reduction internally when reduce_2d
    #    is True; when False we use raw (unreduced) in-plane vectors.
    transformation_matrix = get_surface_basis(
        h, k, l, crystal.lattice, reduce_2d=reduce_2d
    )
    supercell_factor = int(round(abs(np.linalg.det(transformation_matrix))))

    # 2. Build the surface-oriented lattice
    raw_lattice = transformation_matrix.T @ crystal.lattice  # row vectors

    # 3. Rotate so that the surface normal aligns with target_axis
    rotated_lattice, M = orient_lattice(raw_lattice, target_axis=axis_idx)

    # 4. Compute geometric properties
    a_vec, b_vec = rotated_lattice[0], rotated_lattice[1]
    cross_ab = np.cross(a_vec, b_vec)
    surface_area = np.linalg.norm(cross_ab)
    normal = cross_ab / surface_area
    stacking_vector = rotated_lattice[2]
    d_spacing = abs(np.dot(stacking_vector, normal))

    # 5. Apply rotation to all molecule positions
    unwrapped_mols = crystal.get_unwrapped_molecules()
    rotated_mols = []
    for mol in unwrapped_mols:
        mol_copy = mol.copy()
        mol_copy.positions = mol.get_positions() @ M
        rotated_mols.append(mol_copy)

    # 6. Wrap molecular centroids into the new unit cell [0, 1)
    inv_lattice = np.linalg.inv(rotated_lattice)
    wrapped_mols = []
    for mol in rotated_mols:
        centroid = mol.get_centroid()
        frac = centroid @ inv_lattice
        # Shift whole molecule so centroid fractional coords are in [0, 1)
        shift = -np.floor(frac) @ rotated_lattice
        mol_wrapped = mol.copy()
        mol_wrapped.positions = mol.get_positions() + shift
        wrapped_mols.append(mol_wrapped)

    # 7. Construct reoriented crystal
    reoriented = MolecularCrystal(
        lattice=rotated_lattice,
        molecules=wrapped_mols,
        pbc=(True, True, True),
    )

    info = ReorientationInfo(
        rotation_matrix=M,
        transformation_matrix=transformation_matrix,
        rotated_lattice=rotated_lattice,
        d_spacing=d_spacing,
        surface_area=surface_area,
        supercell_factor=supercell_factor,
    )

    return reoriented, info
