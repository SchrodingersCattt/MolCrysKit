"""Geometric smoothing for interpolated crystal paths.

This module provides an IDPP-style smoother for removing short contacts from
crystal interpolation paths.  It is a geometric post-processing step only: it is
not a physical relaxation, not NEB, and does not estimate transition barriers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..structures.crystal import MolecularCrystal
from ..utils.geometry import minimum_image_vector


# Guards against div-by-zero in distances and degenerate zero-distance targets.
_IDPP_EPS = 1e-8


def _pair_indices(n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(n_atoms), k=1)


def _mic_distances(frac: np.ndarray, lattice: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    i, j = pairs
    delta = frac[j] - frac[i]
    vec = minimum_image_vector(delta, lattice)
    vec = np.atleast_2d(vec)
    return np.linalg.norm(vec, axis=1)


def _shortest_frac_path(frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    return frac_a + (frac_b - frac_a - np.round(frac_b - frac_a))


def _copy_crystal_with_scaled_positions(
    crystal: MolecularCrystal, scaled_positions: np.ndarray
) -> MolecularCrystal:
    atoms = crystal.to_ase()
    atoms.set_scaled_positions(np.asarray(scaled_positions, dtype=float) % 1.0)
    return MolecularCrystal.from_ase_atoms(atoms)


def _idpp_energy_and_gradient(
    frac: np.ndarray,
    lattice: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    i, j = pairs
    delta = frac[j] - frac[i]
    vec = minimum_image_vector(delta, lattice)
    vec = np.atleast_2d(vec)
    dist = np.linalg.norm(vec, axis=1)
    safe_dist = np.maximum(dist, _IDPP_EPS)
    residual = dist - target
    energy = float(np.sum(weights * residual * residual))

    # Constant weights (1/target^4); this differs from canonical IDPP
    # (1/d^4) but is equivalent near convergence and avoids an extra
    # gradient term.
    # grad_frac = 2*w*(d-target)/d * (vec_cart @ L^T)
    # [chain rule: grad_s = grad_cart @ L^T]
    coeff = 2.0 * weights * residual / safe_dist
    grad_pair = coeff[:, None] * (vec @ lattice.T)
    grad = np.zeros_like(frac, dtype=float)
    np.add.at(grad, j, grad_pair)
    np.add.at(grad, i, -grad_pair)
    return energy, grad


def smooth_interpolation_idpp(
    images: Sequence[MolecularCrystal],
    *,
    max_steps: int = 200,
    fmax: float = 0.001,
    step_size: float = 0.1,
) -> list[MolecularCrystal]:
    """Smooth an interpolated crystal path with a fractional-coordinate IDPP.

    Endpoints are kept unchanged.  Each intermediate image keeps its own lattice
    fixed while its atomic fractional coordinates are optimized so that pairwise
    MIC distances approach linearly interpolated endpoint distances.

    This is a geometric short-contact remover, not a physical relaxation or NEB.

    Parameters
    ----------
    images : sequence of MolecularCrystal
        Crystal path to smooth.  The first and last images are frozen.
    max_steps : int, default=200
        Maximum gradient-descent steps for each intermediate image.
    fmax : float, default=0.001
        Stop when the largest Cartesian-gradient norm is below this threshold.
    step_size : float, default=0.1
        Fractional-coordinate gradient-descent step size.

    Returns
    -------
    list of MolecularCrystal
        Smoothed path with unchanged endpoints and unchanged per-image lattices.

    Notes
    -----
    This routine evaluates all atom pairs, so each intermediate-image smoothing
    step scales as O(N²) in the number of atoms.
    """
    if len(images) < 3:
        return list(images)

    atoms_a = images[0].to_ase()
    atoms_b = images[-1].to_ase()
    if len(atoms_a) != len(atoms_b):
        raise ValueError("IDPP smoothing requires endpoint images with identical atom counts")

    n_atoms = len(atoms_a)
    pairs = _pair_indices(n_atoms)
    frac_a = atoms_a.get_scaled_positions()
    frac_b = _shortest_frac_path(frac_a, atoms_b.get_scaled_positions())
    dist_a = _mic_distances(frac_a, np.asarray(images[0].lattice, dtype=float), pairs)
    dist_b = _mic_distances(frac_b, np.asarray(images[-1].lattice, dtype=float), pairs)

    smoothed: list[MolecularCrystal] = [images[0]]
    n_intervals = len(images) - 1

    for image_index, image in enumerate(images[1:-1], start=1):
        lam = image_index / n_intervals
        lattice = np.asarray(image.lattice, dtype=float)
        target = (1.0 - lam) * dist_a + lam * dist_b
        target = np.maximum(target, _IDPP_EPS)
        weights = 1.0 / target**4

        frac = image.to_ase().get_scaled_positions()
        for _ in range(int(max_steps)):
            _, grad = _idpp_energy_and_gradient(frac, lattice, pairs, target, weights)
            cart_grad = grad @ np.linalg.inv(lattice).T
            max_grad = float(np.max(np.linalg.norm(cart_grad, axis=1)))
            if max_grad < float(fmax):
                break
            frac = (frac - float(step_size) * grad) % 1.0

        smoothed.append(_copy_crystal_with_scaled_positions(image, frac))

    smoothed.append(images[-1])
    return smoothed


__all__ = ["smooth_interpolation_idpp"]
