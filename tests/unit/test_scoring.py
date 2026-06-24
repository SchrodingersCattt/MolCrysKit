"""Tests for continuous weak-interaction scoring kernels."""

import pytest

from molcrys_kit.analysis.interactions import (
    ScoringParams,
    composite_score,
    gaussian_kernel,
    lorentzian_kernel,
    normalized_vdw_distance,
    scaled_cutoff,
    vdw_radius_sum,
)


def test_lorentzian_kernel_has_long_tail():
    assert lorentzian_kernel(3.4, 3.4, 0.4) == pytest.approx(1.0)
    assert lorentzian_kernel(4.5, 3.4, 0.4) > gaussian_kernel(4.5, 3.4, 0.4)
    assert lorentzian_kernel(4.5, 3.4, 0.4) == pytest.approx(0.1168, rel=1e-3)


def test_gaussian_kernel_sharp_angular_decay():
    assert gaussian_kernel(180.0, 180.0, 10.0) == pytest.approx(1.0)
    assert gaussian_kernel(150.0, 180.0, 10.0) == pytest.approx(0.0001234, rel=1e-3)


def test_composite_score_multiplies_kernels():
    score = composite_score(
        (
            (0.9, 0.9, 0.15, "lorentzian"),
            (180.0, 180.0, 30.0, "gaussian"),
        )
    )
    assert score == pytest.approx(1.0)


def test_normalized_vdw_distance_is_element_adaptive():
    cl_o = normalized_vdw_distance(3.0, "Cl", "O")
    i_n = normalized_vdw_distance(3.0, "I", "N")
    assert cl_o > i_n
    assert vdw_radius_sum("H", "H") == pytest.approx(2.4)


def test_scaled_cutoff_uses_configurable_prefilter_factor():
    assert scaled_cutoff(3.5, ScoringParams(prefilter_factor=1.1)) == pytest.approx(3.85)
