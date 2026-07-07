"""Unit tests for molcrys_kit.analysis.volume module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.volume import (
    _fibonacci_sphere,  # Private — tested for coverage
    _get_radius,  # Private — tested for coverage
    calculate_accessible_boundary,
    calculate_atomic_volumes,
    calculate_total_volume,
    min_distance_to_boundary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def water():
    """Water molecule."""
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ],
    )


@pytest.fixture
def methane():
    """Methane molecule (tetrahedral)."""
    return Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    )


@pytest.fixture
def single_carbon():
    """Single carbon atom at origin."""
    return Atoms("C", positions=[[0.0, 0.0, 0.0]])


@pytest.fixture
def two_overlapping_carbons():
    """Two carbon atoms close enough for VdW overlap (distance=2.0 < 2*1.7)."""
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


@pytest.fixture
def cubic_cell():
    """10 Å cubic lattice."""
    return np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])


# ---------------------------------------------------------------------------
# Tests: _get_radius
# ---------------------------------------------------------------------------


class TestGetRadius:
    def test_vdw_known_element(self):
        """VdW radius for common organic elements should be available."""
        r_c = _get_radius("C", "vdw")
        assert 1.5 < r_c < 2.0  # C vdw ~1.7

    def test_covalent_known_element(self):
        r_c = _get_radius("C", "covalent")
        assert 0.5 < r_c < 1.0  # C covalent ~0.76

    def test_vdw_fallback_to_covalent(self, monkeypatch):
        """Element with covalent but no vdw should use covalent×1.2."""
        import molcrys_kit.analysis.volume as vol_mod
        # Temporarily make has_vdw_radius return False for "C"
        original_has_vdw = vol_mod.has_vdw_radius
        monkeypatch.setattr(vol_mod, "has_vdw_radius", lambda s: False if s == "C" else original_has_vdw(s))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = _get_radius("C", "vdw")

        # Should return covalent × 1.2
        from molcrys_kit.constants import get_atomic_radius
        expected = get_atomic_radius("C") * 1.2
        np.testing.assert_allclose(r, expected, rtol=1e-10)
        assert len(w) == 1
        assert "covalent" in str(w[0].message).lower()

    def test_final_fallback_warning(self):
        """Unknown element should produce warning and return 1.5."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = _get_radius("Xx", "vdw")
            assert r == 1.5
            assert len(w) == 1
            assert "fallback" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Tests: _fibonacci_sphere
# ---------------------------------------------------------------------------


class TestFibonacciSphere:
    def test_shape(self):
        pts = _fibonacci_sphere(100)
        assert pts.shape == (100, 3)

    def test_unit_sphere(self):
        """All points should lie on the unit sphere."""
        pts = _fibonacci_sphere(200)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_distribution(self):
        """Points should be roughly uniformly distributed (z-coords span [-1, 1])."""
        pts = _fibonacci_sphere(500)
        assert pts[:, 2].min() < -0.95
        assert pts[:, 2].max() > 0.95


# ---------------------------------------------------------------------------
# Tests: calculate_atomic_volumes
# ---------------------------------------------------------------------------


class TestCalculateAtomicVolumes:
    def test_known_elements(self, water):
        """Verify volumes match hand calculation."""
        vols = calculate_atomic_volumes(water, radii_type="vdw")
        assert vols.shape == (3,)

        # Oxygen VdW radius ~1.52 Å → volume = 4/3 * π * 1.52³ ≈ 14.71 ų
        r_o = _get_radius("O", "vdw")
        expected_o = (4.0 / 3.0) * np.pi * r_o ** 3
        np.testing.assert_allclose(vols[0], expected_o, rtol=1e-10)

        # Hydrogen VdW radius ~1.20 Å → volume = 4/3 * π * 1.20³ ≈ 7.24 ų
        r_h = _get_radius("H", "vdw")
        expected_h = (4.0 / 3.0) * np.pi * r_h ** 3
        np.testing.assert_allclose(vols[1], expected_h, rtol=1e-10)
        np.testing.assert_allclose(vols[2], expected_h, rtol=1e-10)

    def test_covalent_type(self, methane):
        """Covalent volumes should be smaller than VdW volumes."""
        v_vdw = calculate_atomic_volumes(methane, radii_type="vdw")
        v_cov = calculate_atomic_volumes(methane, radii_type="covalent")
        assert np.all(v_cov < v_vdw)

    def test_empty_atoms(self):
        """Empty Atoms should return empty array."""
        atoms = Atoms()
        vols = calculate_atomic_volumes(atoms)
        assert vols.shape == (0,)


# ---------------------------------------------------------------------------
# Tests: calculate_total_volume
# ---------------------------------------------------------------------------


class TestCalculateTotalVolume:
    def test_sum_equals_no_overlap(self, water):
        """Without overlap correction, total = sum of individual."""
        total = calculate_total_volume(water, overlap_correction=False)
        individual = calculate_atomic_volumes(water)
        np.testing.assert_allclose(total, individual.sum(), rtol=1e-10)

    def test_overlap_correction_smaller(self, two_overlapping_carbons):
        """With overlap, corrected volume should be less than simple sum."""
        total_sum = calculate_total_volume(
            two_overlapping_carbons, overlap_correction=False
        )
        total_corrected = calculate_total_volume(
            two_overlapping_carbons, overlap_correction=True, voxel_size=0.1
        )
        # Overlap correction should give a smaller value
        assert total_corrected < total_sum
        # But still positive and substantial (two C atoms at 2 Å apart
        # with r_vdw=1.7 have significant overlap)
        assert total_corrected > 0.5 * total_sum

    def test_non_overlapping_equals_sum(self, single_carbon):
        """Single atom: corrected volume ≈ simple sum (within grid error)."""
        total_sum = calculate_total_volume(single_carbon, overlap_correction=False)
        total_corrected = calculate_total_volume(
            single_carbon, overlap_correction=True, voxel_size=0.1
        )
        # Should be close (grid discretization error ~few %)
        np.testing.assert_allclose(total_corrected, total_sum, rtol=0.05)


# ---------------------------------------------------------------------------
# Tests: calculate_accessible_boundary
# ---------------------------------------------------------------------------


class TestCalculateAccessibleBoundary:
    def test_single_atom_full_sphere(self, single_carbon):
        """Single atom should retain all surface points (nothing to occlude)."""
        boundary = calculate_accessible_boundary(
            single_carbon, probe_radius=1.4, n_sphere_points=100
        )
        # All 100 points should survive
        assert boundary.shape == (100, 3)

    def test_single_atom_distance(self, single_carbon):
        """All boundary points should be at distance r_vdw + probe from center."""
        probe = 1.4
        boundary = calculate_accessible_boundary(
            single_carbon, probe_radius=probe, n_sphere_points=100
        )
        r_c = _get_radius("C", "vdw")
        expected_dist = r_c + probe
        actual_dists = np.linalg.norm(boundary, axis=1)
        np.testing.assert_allclose(actual_dists, expected_dist, rtol=1e-6)

    def test_two_atoms_fewer_points(self, two_overlapping_carbons):
        """Two overlapping atoms should have fewer total points than 2×n_sphere."""
        n_pts = 100
        boundary = calculate_accessible_boundary(
            two_overlapping_carbons, probe_radius=1.4, n_sphere_points=n_pts
        )
        # Some points between the atoms are occluded
        assert boundary.shape[0] < 2 * n_pts
        # But most should survive (they overlap but don't fully bury each other)
        assert boundary.shape[0] > n_pts

    def test_empty_atoms(self):
        """Empty Atoms should return empty boundary."""
        boundary = calculate_accessible_boundary(Atoms(), probe_radius=1.4)
        assert boundary.shape == (0, 3)

    def test_point_count_monotonic(self, water):
        """More sphere points → more total boundary points (monotonic)."""
        b1 = calculate_accessible_boundary(water, n_sphere_points=20)
        b2 = calculate_accessible_boundary(water, n_sphere_points=100)
        assert b2.shape[0] >= b1.shape[0]


# ---------------------------------------------------------------------------
# Tests: min_distance_to_boundary
# ---------------------------------------------------------------------------


class TestMinDistanceToBoundary:
    def test_non_periodic_basic(self):
        """Non-periodic: distance from (5,0,0) to boundary at (0,0,0) and (3,0,0)."""
        boundary = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        query = np.array([[5.0, 0.0, 0.0]])
        dists = min_distance_to_boundary(query, boundary)
        np.testing.assert_allclose(dists, [2.0], atol=1e-10)

    def test_non_periodic_multiple_queries(self):
        """Multiple query points, non-periodic."""
        boundary = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        query = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        dists = min_distance_to_boundary(query, boundary)
        np.testing.assert_allclose(dists, [1.0, 1.0], atol=1e-10)

    def test_empty_boundary(self):
        """Empty boundary → inf distances."""
        boundary = np.empty((0, 3))
        query = np.array([[1.0, 2.0, 3.0]])
        dists = min_distance_to_boundary(query, boundary)
        assert np.all(np.isinf(dists))

    def test_periodic_wrapping(self, cubic_cell):
        """PBC: point at (9.5,0,0) should be close to boundary at (0.5,0,0) via wrapping."""
        boundary = np.array([[0.5, 0.0, 0.0]])
        query = np.array([[9.5, 0.0, 0.0]])
        dists = min_distance_to_boundary(
            query, boundary, lattice=cubic_cell, pbc=[True, True, True]
        )
        # Distance via wrapping: 10 - 9.5 + 0.5 = 1.0
        np.testing.assert_allclose(dists, [1.0], atol=1e-10)

    def test_periodic_vs_non_periodic(self, cubic_cell):
        """PBC distance should be ≤ non-periodic distance."""
        boundary = np.array([[0.5, 5.0, 5.0]])
        query = np.array([[9.5, 5.0, 5.0]])

        d_non_pbc = min_distance_to_boundary(query, boundary)
        d_pbc = min_distance_to_boundary(
            query, boundary, lattice=cubic_cell, pbc=[True, True, True]
        )
        assert d_pbc[0] <= d_non_pbc[0]

    def test_periodic_requires_lattice(self):
        """Setting pbc without lattice should raise ValueError."""
        boundary = np.array([[0.0, 0.0, 0.0]])
        query = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="lattice is required"):
            min_distance_to_boundary(query, boundary, pbc=[True, True, True])

    def test_triclinic_cell(self):
        """Triclinic cell should still produce correct minimum image distances."""
        # 60-degree sheared cell
        lattice = np.array([
            [10.0, 0.0, 0.0],
            [5.0, 8.66, 0.0],
            [0.0, 0.0, 10.0],
        ])
        boundary = np.array([[0.0, 0.0, 0.0]])
        query = np.array([[9.5, 0.0, 0.0]])  # fractional ~ (0.95, 0, 0)

        dists = min_distance_to_boundary(
            query, boundary, lattice=lattice, pbc=[True, True, True]
        )
        # In fractional: delta = (0.95, 0, 0) → rounded to (-0.05, 0, 0)
        # Cartesian: -0.05 * [10, 0, 0] = [-0.5, 0, 0] → distance = 0.5
        np.testing.assert_allclose(dists, [0.5], atol=1e-10)
