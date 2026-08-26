"""Bonding-distance utilities shared by topology and interaction analysis."""

import numpy as np

from ...constants import (
    METAL_NON_METAL_THRESHOLD_FACTOR,
    METAL_THRESHOLD_FACTOR,
    NON_METAL_THRESHOLD_FACTOR,
)


def get_bonding_thresholds(
    radius_i: np.ndarray | float,
    radius_j: np.ndarray | float,
    is_metal_i: np.ndarray | bool,
    is_metal_j: np.ndarray | bool,
) -> np.ndarray:
    """Return bonding thresholds for broadcast-compatible array inputs."""
    first_metal = np.asarray(is_metal_i, dtype=bool)
    second_metal = np.asarray(is_metal_j, dtype=bool)
    factors = np.where(
        first_metal & second_metal,
        METAL_THRESHOLD_FACTOR,
        np.where(
            first_metal | second_metal,
            METAL_NON_METAL_THRESHOLD_FACTOR,
            NON_METAL_THRESHOLD_FACTOR,
        ),
    )
    return (np.asarray(radius_i) + np.asarray(radius_j)) * factors


def get_bonding_threshold(
    radius_i: float, radius_j: float, is_metal_i: bool, is_metal_j: bool
) -> float:
    """
    Return the distance cutoff used to infer a bond between two atoms.

    The cutoff is the sum of the two atomic radii multiplied by an element-class
    factor.  Metal-metal, nonmetal-nonmetal, and mixed metal-nonmetal pairs use
    separate calibrated factors because coordination bonds require different
    distance tolerance from ordinary covalent bonds.  This is a heuristic
    connectivity cutoff, not a bond-order assignment.

    Parameters
    ----------
    radius_i : float
        Atomic radius of the first atom, in Å.
    radius_j : float
        Atomic radius of the second atom, in Å.
    is_metal_i : bool
        Whether the first atom is a metal.
    is_metal_j : bool
        Whether the second atom is a metal.

    Returns
    -------
    float
        The bonding threshold distance, in Å.
    """
    return float(get_bonding_thresholds(radius_i, radius_j, is_metal_i, is_metal_j))
