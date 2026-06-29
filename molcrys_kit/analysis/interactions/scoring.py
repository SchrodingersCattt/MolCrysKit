"""Continuous scoring kernels for weak interaction geometries.

The detector modules use hard geometric filters as inexpensive pre-filters, then
assign a continuous score in ``(0, 1]`` to accepted contacts.  Distance terms use
Lorentzian/Cauchy kernels to keep a physically useful long tail, while angular
terms use Gaussian kernels to preserve strong directionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Literal, Sequence

from ...constants import VDW_RADII

KernelKind = Literal["lorentzian", "gaussian"]
ScoreDimension = tuple[float, float, float, KernelKind]


@dataclass(frozen=True)
class ScoringParams:
    """Default continuous-scoring parameters for interaction detectors.

    Distance dimensions in Å use Lorentzian kernels.  Normalized distances use
    the dimensionless ratio ``d / (r_vdw_i + r_vdw_j)``.  Angles in degrees use
    Gaussian kernels.  ``prefilter_factor`` widens legacy hard cutoffs before
    score-threshold filtering.
    """

    prefilter_factor: float = 1.2
    score_threshold: float = 0.01

    hbond_d_norm0: float = 0.7
    hbond_d_norm_sigma: float = 0.20
    hbond_angle0_deg: float = 180.0
    hbond_angle_sigma_deg: float = 30.0

    halogen_d_norm0: float = 0.85
    halogen_d_norm_sigma: float = 0.10
    halogen_angle0_deg: float = 180.0
    halogen_angle_sigma_deg: float = 10.0

    pi_centroid_distance0_A: float = 3.4
    pi_centroid_distance_sigma_A: float = 0.4
    pi_t_shape_distance0_A: float = 5.0
    pi_t_shape_distance_sigma_A: float = 1.0
    pi_t_shape_approach0_A: float = 0.0
    pi_t_shape_approach_sigma_A: float = 1.5
    pi_parallel_angle0_deg: float = 0.0
    pi_t_shape_angle0_deg: float = 90.0
    pi_angle_sigma_deg: float = 15.0

    ch_pi_h_centroid_distance0_A: float = 2.6
    ch_pi_h_centroid_distance_sigma_A: float = 0.3
    ch_pi_angle0_deg: float = 180.0
    ch_pi_angle_sigma_deg: float = 25.0

    # H···H close-contact scores describe closeness to the vdW contact shell,
    # not attractive interaction strength. Shorter steric contacts can be more
    # severe but are intentionally not assigned larger positive scores here.
    close_contact_distance_sigma_A: float = 0.3


DEFAULT_SCORING_PARAMS = ScoringParams()


def lorentzian_kernel(x: float, x0: float, sigma: float) -> float:
    """Return a Lorentzian kernel value for one scalar dimension."""
    sigma = _positive_sigma(sigma)
    delta = (float(x) - float(x0)) / sigma
    return 1.0 / (1.0 + delta * delta)


def gaussian_kernel(x: float, x0: float, sigma: float) -> float:
    """Return a Gaussian kernel value for one scalar dimension."""
    sigma = _positive_sigma(sigma)
    delta = (float(x) - float(x0)) / sigma
    return float(exp(-(delta * delta)))


def composite_score(dimensions: Sequence[ScoreDimension]) -> float:
    """Return product score over Lorentzian and Gaussian dimensions."""
    score = 1.0
    for value, x0, sigma, kernel in dimensions:
        if kernel == "lorentzian":
            score *= lorentzian_kernel(value, x0, sigma)
        elif kernel == "gaussian":
            score *= gaussian_kernel(value, x0, sigma)
        else:  # pragma: no cover - defensive for dynamic callers
            raise ValueError(f"unknown scoring kernel: {kernel!r}")
    return float(score)


def normalized_vdw_distance(distance_A: float, symbol1: str, symbol2: str) -> float:
    """Return distance normalized by the sum of two van der Waals radii."""
    return float(distance_A) / vdw_radius_sum(symbol1, symbol2)


def vdw_radius_sum(symbol1: str, symbol2: str) -> float:
    """Return the sum of two element vdW radii in Å.

    Raises
    ------
    KeyError
        If either element has no vdW-radius entry.
    """
    return float(VDW_RADII[symbol1]) + float(VDW_RADII[symbol2])


def scaled_cutoff(cutoff: float, params: ScoringParams | None = None) -> float:
    """Return a legacy cutoff widened by ``prefilter_factor``."""
    params = params or DEFAULT_SCORING_PARAMS
    return float(cutoff) * float(params.prefilter_factor)


def _positive_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("scoring sigma must be positive")
    return sigma
