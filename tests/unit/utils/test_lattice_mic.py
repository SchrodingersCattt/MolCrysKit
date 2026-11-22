"""Regression tests for exact minimum-image lattice search."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
from ase.geometry import cellpar_to_cell

from molcrys_kit.utils.geometry import frac_to_cart, minimum_image_vector
from molcrys_kit.utils.lattice_mic import (
    minimum_image_cartesian,
    periodic_mask,
    translations_within_cartesian,
)


def _exhaustive_minimum_cartesian(delta, cell, pbc):
    """Independent shortest-image oracle with an interplanar shift bound."""
    delta = np.asarray(delta, dtype=float).reshape(3)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    periodic = np.asarray(pbc, dtype=bool)
    ranges = []
    raw = float(np.linalg.norm(delta))
    for axis in range(3):
        if not periodic[axis]:
            ranges.append([0])
            continue
        other = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        height = abs(float(np.dot(cell[axis], other))) / (
            float(np.linalg.norm(other)) + 1e-15
        )
        count = int(np.ceil(raw / max(height, 1e-8))) + 1
        ranges.append(range(-count, count + 1))
    best_vector = delta.copy()
    best_length = raw
    best_shift = (0, 0, 0)
    for shift in itertools.product(*ranges):
        vector = delta + np.asarray(shift, dtype=float) @ cell
        length = float(np.linalg.norm(vector))
        if length < best_length - 1e-12 or (
            abs(length - best_length) <= 1e-12 and shift < best_shift
        ):
            best_vector = vector
            best_length = length
            best_shift = shift
    return best_vector, best_length


def _oracle_from_fractional(frac_delta, lattice):
    cart = frac_to_cart(frac_delta, lattice)
    return _exhaustive_minimum_cartesian(cart, lattice, np.ones(3, dtype=bool))


def test_issue_237_skew_flat_cell_reproduction():
    lattice = cellpar_to_cell([5.108, 2.895, 4.291, 73.19, 79.94, 153.04])
    frac_delta = np.array([0.161, 1.476, 1.494])
    mic = minimum_image_vector(frac_delta, lattice)
    expected, expected_len = _oracle_from_fractional(frac_delta, lattice)
    assert np.linalg.norm(mic) == pytest.approx(expected_len, abs=1e-8)
    np.testing.assert_allclose(mic, expected, atol=1e-8)
    assert np.linalg.norm(mic) < 0.35
    assert np.linalg.norm(mic) == pytest.approx(0.18965033508813575, rel=0, abs=1e-6)


def test_large_unwrapped_fractional_displacement_matches_oracle():
    lattice = cellpar_to_cell([4.0, 5.0, 6.0, 78.0, 82.0, 98.0])
    frac_delta = np.array([18.37, -12.4, 7.15])
    mic = minimum_image_vector(frac_delta, lattice)
    expected, _ = _oracle_from_fractional(frac_delta, lattice)
    np.testing.assert_allclose(mic, expected, atol=1e-8)


def test_vectorized_batch_shapes_and_oracle():
    lattice = np.diag([8.0, 9.0, 10.0])
    deltas = np.array([[0.8, 0.0, 0.0], [0.0, -0.9, 0.0], [4.2, 3.3, -5.1]])
    batch = minimum_image_vector(deltas, lattice)
    assert batch.shape == (3, 3)
    for row, frac_delta in enumerate(deltas):
        single = minimum_image_vector(frac_delta, lattice)
        np.testing.assert_allclose(batch[row], single, atol=1e-10)
        expected, _ = _oracle_from_fractional(frac_delta, lattice)
        np.testing.assert_allclose(batch[row], expected, atol=1e-8)


def test_equal_length_tie_breaks_on_lexicographic_shift():
    lattice = np.diag([10.0, 10.0, 10.0])
    delta = np.array([7.5, 0.0, 0.0])
    left = minimum_image_cartesian(delta, lattice, np.ones(3, dtype=bool))
    right = minimum_image_cartesian(-delta, lattice, np.ones(3, dtype=bool))
    np.testing.assert_allclose(left, [-2.5, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(right, [2.5, 0.0, 0.0], atol=1e-10)
    assert abs(np.linalg.norm(left) - np.linalg.norm(right)) <= 1e-10


def test_partial_pbc_matches_path_core_contract():
    cell = np.eye(3) * 10.0
    delta = np.array([-8.0, -8.0, 0.0])
    mask = periodic_mask((False, True, False))
    vector = minimum_image_cartesian(delta, cell, mask)
    np.testing.assert_allclose(vector, [-8.0, 2.0, 0.0], atol=1e-10)


def test_degenerate_cell_row_skips_wrap_along_that_axis():
    cell = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    delta = np.array([3.0, 0.0, 4.0])
    mask = np.array([True, True, False])
    vector = minimum_image_cartesian(delta, cell, mask)
    np.testing.assert_allclose(vector, [-2.0, 0.0, 4.0], atol=1e-10)


def test_enumeration_lists_only_strict_ball_members():
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.0, 0.0], [0.2, -0.4, 5.0]])
    delta = np.array([1.5, -0.5, 0.3])
    radius = 2.5
    pbc = np.array([True, True, True])
    shifts = translations_within_cartesian(delta, cell, pbc, radius)
    for shift in shifts:
        length = float(np.linalg.norm(delta + shift @ cell))
        assert length < radius


def test_minimum_image_batch_performance_smoke():
    lattice = cellpar_to_cell([13.2, 13.1, 13.8, 80.0, 75.0, 62.0])
    rng = np.random.default_rng(237)
    deltas = rng.normal(size=(256, 3))
    started = time.perf_counter()
    minimum_image_vector(deltas, lattice)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"256 skew-cell MIC queries took {elapsed:.2f}s"