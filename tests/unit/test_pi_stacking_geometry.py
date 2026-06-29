"""Regression tests for pi-stacking geometry rework and profile cleanup.

Tests the subtype-specific filtering (interplane h for parallel,
centroid d for T-shape), the coarse spatial prefilter for parallel
subtypes, and the removal of ch_pi / h_h_contact from
interaction_profile.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.analysis.interactions import find_pi_stacking, interaction_profile
from molcrys_kit.analysis.interactions.pi_stacking import PiStackingCriteria


def _make_benzene(center, normal_axis="z"):
    """Create a planar 6-C ring (benzene-like) centered at `center`."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r = 1.4  # C-C bond length
    if normal_axis == "z":
        positions = np.column_stack([
            center[0] + r * np.cos(angles),
            center[1] + r * np.sin(angles),
            np.full(6, center[2]),
        ])
    elif normal_axis == "x":
        positions = np.column_stack([
            np.full(6, center[0]),
            center[1] + r * np.cos(angles),
            center[2] + r * np.sin(angles),
        ])
    else:
        raise ValueError(f"Unsupported normal_axis: {normal_axis}")
    return CrystalMolecule(Atoms("C6", positions=positions))


def _make_ring(center, n_atoms, r, normal_axis="z", element="C"):
    """Create a planar n-atom ring of radius `r` centered at `center`."""
    angles = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
    if normal_axis == "z":
        positions = np.column_stack([
            center[0] + r * np.cos(angles),
            center[1] + r * np.sin(angles),
            np.full(n_atoms, center[2]),
        ])
    elif normal_axis == "x":
        positions = np.column_stack([
            np.full(n_atoms, center[0]),
            center[1] + r * np.cos(angles),
            center[2] + r * np.sin(angles),
        ])
    else:
        raise ValueError(f"Unsupported normal_axis: {normal_axis}")
    formula = f"{element}{n_atoms}"
    return CrystalMolecule(Atoms(formula, positions=positions))


class TestParallelUsesInterplaneDistance:
    """Parallel stacking should filter by interplane distance h, not centroid d."""

    def test_displaced_parallel_accepted(self):
        """Two parallel rings with h=3.5 and lateral offset within ring radius.

        Old code rejected this (d > 4.5). New code should accept (h < 4.0)
        when the centroid projection falls inside the target ring.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        # l=1.2 < benzene circumradius ≈ 1.4 → projection check passes
        ring2 = _make_benzene([1.2, 0, 3.5], normal_axis="z")  # h=3.5, l=1.2, d≈3.70
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "displaced_parallel"

    def test_face_centered_parallel(self):
        """Two coplanar rings directly stacked at h=3.4."""
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([0.3, 0, 3.4], normal_axis="z")  # h=3.4, l=0.3
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "face_centered_parallel"


class TestTShapeUsesLargerCutoff:
    """T-shape should use centroid distance with larger cutoff (6.5 Å)."""

    def test_t_shape_at_5A_accepted(self):
        """Ring2 perpendicular to ring1, positioned above (h~4.8, l~1.0, d~4.9)."""
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([1.0, 0, 4.8], normal_axis="x")
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "T_shape"

    def test_t_shape_at_6A_accepted(self):
        """Ring2 perpendicular at larger distance (h~5.5, l~1.0, d~5.59).

        Projection of ring2 centroid onto ring1 plane must fall within
        ring1 circumradius (~1.4) — l=1.0 satisfies this.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([1.0, 0, 5.5], normal_axis="x")  # l=1.0, d≈5.59
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "T_shape"


class TestProfileContract:
    """interaction_profile includes the correct keys after cleanup."""

    def test_profile_has_expected_keys(self):
        """Profile must include hydrogen_bond, halogen_bond, pi_stacking."""
        mol1 = CrystalMolecule(
            Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        )
        mol2 = CrystalMolecule(
            Atoms("OHH", positions=[[2.8, 0, 0], [3.76, 0, 0], [2.8, 0.96, 0]])
        )
        profile = interaction_profile([mol1, mol2])
        assert "hydrogen_bond" in profile.summaries
        assert "halogen_bond" in profile.summaries
        assert "pi_stacking" in profile.summaries

    def test_profile_excludes_deprecated_keys(self):
        """Profile must NOT include ch_pi or close_contact."""
        mol1 = CrystalMolecule(
            Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        )
        mol2 = CrystalMolecule(
            Atoms("OHH", positions=[[2.8, 0, 0], [3.76, 0, 0], [2.8, 0.96, 0]])
        )
        profile = interaction_profile([mol1, mol2])
        assert "ch_pi" not in profile.summaries
        assert "close_contact" not in profile.summaries


class TestTShapeOffCenter:
    """T-shape projection check must NOT reject valid off-center geometries.

    In edge-over-face T-shape contacts the stem ring's centroid
    legitimately projects well outside the face ring's footprint.
    The coarse spatial prefilter applies only to parallel subtypes.
    """

    def test_t_shape_off_center_accepted(self):
        """Stem centroid lateral offset (l=2.0) exceeds face ring circumradius (~1.4).

        This is a geometrically valid T-shape: the stem ring's edge
        atoms point at the face ring, but the centroid is displaced.
        Must NOT be rejected.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")  # face ring
        ring2 = _make_benzene([2.0, 0, 4.8], normal_axis="x")  # stem ring, l=2.0
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "T_shape"

    def test_t_shape_far_off_center_rejected_by_lateral_cutoff(self):
        """Lateral offset l=3.5 exceeds max_t_shape_lateral_offset_A=3.0.

        Should be rejected by the existing lateral offset cutoff, NOT
        by any projection check.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([3.5, 0, 4.0], normal_axis="x")  # l=3.5 > 3.0
        results = find_pi_stacking([ring1, ring2])
        assert len(results) == 0

    def test_t_shape_centroid_beyond_6A_rejected(self):
        """Centroid distance d ≈ 7.0 exceeds max_t_shape_centroid_distance_A=6.5.

        Should be rejected by centroid distance, not projection.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([1.0, 0, 6.9], normal_axis="x")  # d ≈ 6.97
        results = find_pi_stacking([ring1, ring2])
        assert len(results) == 0


class TestParallelProjectionBoundary:
    """Parallel prefilter boundary: positive near circumradius, negative outside."""

    def test_parallel_at_boundary_inside(self):
        """Lateral offset l=1.3, just inside benzene circumradius (~1.4)."""
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([1.3, 0, 3.4], normal_axis="z")  # h=3.4, l=1.3
        results = find_pi_stacking([ring1, ring2])
        assert len(results) >= 1
        assert results[0].subtype == "displaced_parallel"

    def test_parallel_outside_circumradius_rejected(self):
        """Lateral offset l=2.0 > benzene circumradius (~1.4), outside for both.

        Both rings are the same benzene, so the bidirectional check
        also fails: both projections fall outside circumradius.
        The lateral_offset 2.0 is still within max_parallel_lateral_offset_A=2.5,
        but the projection prefilter should reject it.
        """
        ring1 = _make_benzene([0, 0, 0], normal_axis="z")
        ring2 = _make_benzene([2.0, 0, 3.5], normal_axis="z")  # l=2.0 > 1.4
        results = find_pi_stacking([ring1, ring2])
        assert len(results) == 0


class TestAsymmetricRingSizes:
    """The projection prefilter must handle rings of different sizes correctly."""

    def test_small_over_large_parallel_accepted(self):
        """5-atom ring (r=1.1) stacked over 6-atom ring (r=1.4).

        Lateral offset l=1.2 is within the large ring's circumradius (1.4)
        but outside the small ring's circumradius (1.1).  The bidirectional
        check should accept via the large ring direction.
        """
        large_ring = _make_ring([0, 0, 0], n_atoms=6, r=1.4, normal_axis="z")
        small_ring = _make_ring([1.2, 0, 3.4], n_atoms=5, r=1.1, normal_axis="z")
        results = find_pi_stacking([large_ring, small_ring])
        assert len(results) >= 1
        assert results[0].subtype in ("face_centered_parallel", "displaced_parallel")

    def test_large_over_small_parallel_rejected_when_both_outside(self):
        """Both projections fall outside their respective circumradii.

        Large ring (r=1.4) over small ring (r=0.9), lateral offset l=1.5.
        l > 1.4 (large circumradius) and reverse projection also outside
        0.9 (small circumradius) → rejected.
        """
        small_ring = _make_ring([0, 0, 0], n_atoms=5, r=0.9, normal_axis="z")
        large_ring = _make_ring([1.5, 0, 3.4], n_atoms=6, r=1.4, normal_axis="z")
        results = find_pi_stacking([small_ring, large_ring])
        # l=1.5 > large circumradius=1.4 and reverse l=1.5 > small circumradius=0.9
        assert len(results) == 0

    def test_asymmetric_t_shape_not_affected(self):
        """T-shape between a 5-atom ring and a 6-atom ring.

        Projection check is skipped for T-shape, so ring size asymmetry
        does not affect acceptance — only angle and distance matter.
        """
        ring6 = _make_ring([0, 0, 0], n_atoms=6, r=1.4, normal_axis="z")
        ring5 = _make_ring([1.5, 0, 4.5], n_atoms=5, r=1.1, normal_axis="x")
        results = find_pi_stacking([ring6, ring5])
        assert len(results) >= 1
        assert results[0].subtype == "T_shape"
