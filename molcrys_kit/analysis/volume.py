"""
Volume analysis and solvent-accessible boundary computation.

This module provides van der Waals volume calculations and solvent-accessible
surface point generation for molecular clusters, primarily intended for:

- Estimating cluster volume in solvent box construction
- Generating surface boundary for solvent placement clash detection
- Computing minimum distance from new atoms to the accessible boundary

The boundary algorithm is a simplified Shrake-Rupley sphere sampling method:
for each atom, surface test points are generated on a sphere of radius
(r_vdw + probe_radius); points buried inside neighbouring spheres are removed.
The surviving points form the solvent-accessible boundary.

Notes
-----
- This is **not** a high-precision SASA implementation; it is optimised for
  fast geometric clash detection during structure generation.
- For overlap-corrected volume, a 3-D occupancy grid is used. The default
  voxel size (0.2 Å) provides adequate accuracy for box-filling calculations.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..constants import get_atomic_radius, get_vdw_radius, has_vdw_radius, has_atomic_radius


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_radius(symbol: str, radii_type: str = "vdw") -> float:
    """Get atomic radius with fallback chain.

    Fallback order for ``radii_type="vdw"``:
      1. VdW radius from constants table
      2. Covalent radius × 1.2
      3. 1.5 Å (safe default) + warning

    For ``radii_type="covalent"``, only steps 2-3 apply.
    """
    if radii_type == "vdw":
        if has_vdw_radius(symbol):
            return get_vdw_radius(symbol)
        # Fallback to scaled covalent radius
        if has_atomic_radius(symbol):
            r = get_atomic_radius(symbol) * 1.2
            warnings.warn(
                f"VdW radius unavailable for '{symbol}'; "
                f"using covalent×1.2 = {r:.3f} Å",
                stacklevel=3,
            )
            return r
    elif radii_type == "covalent":
        if has_atomic_radius(symbol):
            return get_atomic_radius(symbol)

    # Final fallback
    warnings.warn(
        f"No radius data for '{symbol}'; using fallback 1.5 Å",
        stacklevel=3,
    )
    return 1.5


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate approximately uniform points on a unit sphere.

    Uses the Fibonacci spiral / golden-section method.

    Parameters
    ----------
    n_points : int
        Number of points to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) with unit-sphere coordinates.
    """
    indices = np.arange(n_points, dtype=float)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    theta = 2.0 * np.pi * indices / golden_ratio
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_points)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_atomic_volumes(
    atoms,
    radii_type: str = "vdw",
) -> np.ndarray:
    """Calculate per-atom spherical volume using the specified radius type.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure (positions are not used; only chemical symbols).
    radii_type : str, optional
        ``"vdw"`` (default) or ``"covalent"``.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with volume of each atom in ų.
    """
    symbols = atoms.get_chemical_symbols()
    radii = np.array([_get_radius(s, radii_type) for s in symbols])
    return (4.0 / 3.0) * np.pi * radii ** 3


def calculate_total_volume(
    atoms,
    radii_type: str = "vdw",
    overlap_correction: bool = False,
    voxel_size: float = 0.2,
) -> float:
    """Calculate total volume occupied by atomic spheres.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    radii_type : str, optional
        ``"vdw"`` (default) or ``"covalent"``.
    overlap_correction : bool, optional
        If False (default), returns the sum of individual sphere volumes.
        If True, uses a 3-D occupancy grid to account for overlaps, giving
        a more accurate (smaller) volume estimate.
    voxel_size : float, optional
        Grid spacing in Å for overlap correction. Default 0.2 Å.

    Returns
    -------
    float
        Total volume in ų.
    """
    if not overlap_correction:
        return float(np.sum(calculate_atomic_volumes(atoms, radii_type)))

    # Grid-based overlap correction
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    radii = np.array([_get_radius(s, radii_type) for s in symbols])

    if len(positions) == 0:
        return 0.0

    # Determine grid bounds (expand by max radius on each side)
    r_max = radii.max()
    pos_min = positions.min(axis=0) - r_max - voxel_size
    pos_max = positions.max(axis=0) + r_max + voxel_size

    # Build grid axes
    axes = [
        np.arange(pos_min[d], pos_max[d] + voxel_size, voxel_size)
        for d in range(3)
    ]
    nz = len(axes[2])

    # For memory efficiency, iterate over z-slices
    voxel_vol = voxel_size ** 3
    occupied_count = 0

    # Pre-compute squared radii
    radii_sq = radii ** 2

    # Hoist meshgrid outside z-loop (grid_xy is identical for every slice)
    xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])  # (nx*ny, 2)
    n_grid = len(grid_xy)

    for iz in range(nz):
        z = axes[2][iz]

        # Pre-filter: only atoms whose z-extent overlaps this slice
        dz_all = np.abs(z - positions[:, 2])
        nearby_mask = dz_all <= radii
        nearby_indices = np.where(nearby_mask)[0]

        if len(nearby_indices) == 0:
            continue

        # Check if any nearby atom covers these grid points
        occupied = np.zeros(n_grid, dtype=bool)
        for i in nearby_indices:
            dx = grid_xy[:, 0] - positions[i, 0]
            dy = grid_xy[:, 1] - positions[i, 1]
            dz = z - positions[i, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            occupied |= dist_sq <= radii_sq[i]

        occupied_count += occupied.sum()

    return float(occupied_count) * voxel_vol


def calculate_accessible_boundary(
    atoms,
    probe_radius: float = 1.4,
    radii_type: str = "vdw",
    n_sphere_points: int = 50,
) -> np.ndarray:
    """Generate solvent-accessible boundary points.

    For each atom, generates test points on a sphere of radius
    ``(r_atom + probe_radius)`` using a Fibonacci spiral. Points that fall
    inside any neighbouring atom's extended sphere are removed.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure (non-periodic; for periodic systems, extract the
        cluster first).
    probe_radius : float, optional
        Probe radius in Å (default 1.4, typical for water).
    radii_type : str, optional
        ``"vdw"`` (default) or ``"covalent"``.
    n_sphere_points : int, optional
        Number of test points per atom on the unit sphere (default 50).

    Returns
    -------
    np.ndarray
        Array of shape (M, 3) with Cartesian coordinates of the accessible
        surface points.
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    n_atoms = len(positions)

    if n_atoms == 0:
        return np.empty((0, 3), dtype=float)

    radii = np.array([_get_radius(s, radii_type) for s in symbols])
    extended_radii = radii + probe_radius

    # Generate unit sphere template
    unit_sphere = _fibonacci_sphere(n_sphere_points)

    # Build KD-tree over atom positions for neighbour lookup
    # Max interaction distance: largest extended_radius + largest extended_radius
    max_interaction = 2.0 * extended_radii.max()

    tree = cKDTree(positions)

    all_surface_points = []

    for i in range(n_atoms):
        # Generate surface points for atom i
        points = positions[i] + extended_radii[i] * unit_sphere  # (n_sphere_points, 3)

        # Find atoms close enough to potentially occlude these points
        neighbour_indices = tree.query_ball_point(positions[i], max_interaction)

        # Check each point against neighbours
        keep = np.ones(n_sphere_points, dtype=bool)
        for j in neighbour_indices:
            if j == i:
                continue
            # Distance from each surface point to atom j
            diff = points - positions[j]
            dist_sq = np.sum(diff * diff, axis=1)
            # Point is buried if inside atom j's extended sphere
            keep &= dist_sq > extended_radii[j] ** 2

        if keep.any():
            all_surface_points.append(points[keep])

    if not all_surface_points:
        return np.empty((0, 3), dtype=float)

    return np.vstack(all_surface_points)


def min_distance_to_boundary(
    new_positions: np.ndarray,
    boundary_points: np.ndarray,
    lattice: Optional[np.ndarray] = None,
    pbc: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Compute minimum distance from each new position to boundary points.

    Parameters
    ----------
    new_positions : np.ndarray
        Positions to query, shape (K, 3).
    boundary_points : np.ndarray
        Surface/boundary points, shape (M, 3).
    lattice : np.ndarray, optional
        3×3 lattice matrix (rows are lattice vectors). Required if pbc is set.
    pbc : sequence of bool, optional
        Periodic boundary conditions along each axis. If None or all False,
        uses non-periodic distance calculation.

    Returns
    -------
    np.ndarray
        Array of shape (K,) with the minimum distance from each query point
        to the closest boundary point. Returns ``inf`` if boundary is empty.
    """
    new_positions = np.atleast_2d(new_positions)
    boundary_points = np.atleast_2d(boundary_points)

    if len(boundary_points) == 0:
        return np.full(len(new_positions), np.inf)

    use_pbc = pbc is not None and any(pbc)

    if not use_pbc:
        # Fast path: non-periodic — use KD-tree
        tree = cKDTree(boundary_points)
        distances, _ = tree.query(new_positions)
        return distances

    # PBC path: minimum image convention
    # For moderate boundary sizes (< ~50k points), brute-force with images
    # is simpler and fast enough.
    if lattice is None:
        raise ValueError("lattice is required when pbc is set")

    lattice = np.asarray(lattice, dtype=float)
    inv_lattice = np.linalg.inv(lattice)

    # Convert to fractional
    frac_new = new_positions @ inv_lattice
    frac_boundary = boundary_points @ inv_lattice

    min_dists = np.full(len(new_positions), np.inf)

    # Process in chunks to limit memory for large boundary sets
    chunk_size = 5000
    n_boundary = len(frac_boundary)

    for start in range(0, n_boundary, chunk_size):
        end = min(start + chunk_size, n_boundary)
        frac_chunk = frac_boundary[start:end]  # (C, 3)

        # Compute all pairwise fractional deltas: (K, C, 3)
        delta = frac_new[:, None, :] - frac_chunk[None, :, :]

        # Apply minimum image convention per PBC axis
        for d in range(3):
            if pbc[d]:
                delta[:, :, d] -= np.round(delta[:, :, d])

        # Convert to Cartesian
        # delta shape: (K, C, 3); lattice shape: (3, 3)
        cart_delta = np.einsum("ijk,kl->ijl", delta, lattice)
        dist_sq = np.sum(cart_delta ** 2, axis=2)  # (K, C)

        chunk_min = np.sqrt(dist_sq.min(axis=1))  # (K,)
        min_dists = np.minimum(min_dists, chunk_min)

    return min_dists
