"""Regression tests for pi-stacking geometry rework and profile cleanup.

Tests the subtype-specific filtering (interplane h for parallel,
centroid d for T-shape) and the removal of ch_pi / h_h_contact
from interaction_profile.
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
