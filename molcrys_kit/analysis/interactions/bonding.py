"""Bonding-distance utilities shared by topology and interaction analysis."""

from ...constants import (
    METAL_NON_METAL_THRESHOLD_FACTOR,
    METAL_THRESHOLD_FACTOR,
    NON_METAL_THRESHOLD_FACTOR,
)


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
    if is_metal_i and is_metal_j:
        factor = METAL_THRESHOLD_FACTOR
    elif not is_metal_i and not is_metal_j:
        factor = NON_METAL_THRESHOLD_FACTOR
    else:
        factor = METAL_NON_METAL_THRESHOLD_FACTOR

    return (radius_i + radius_j) * factor
