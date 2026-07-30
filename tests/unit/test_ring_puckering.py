"""Tests for general N-membered ring puckering coordinates."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations.ring_conformation import (
    DegenerateRingGeometryError,
    InvalidRingOrderError,
    PuckeringCoordinates,
    find_ring_systems,
    puckering_coordinates,
    reconstruct_z_from_modes,
)
from molcrys_kit.structures.molecule import CrystalMolecule


def _regular_polygon_molecule(n: int, radius: float = None) -> CrystalMolecule:
    """Planar regular N-gon — should have zero puckering.

    Uses a radius that ensures consecutive atoms are within bonding threshold.
    """
    if radius is None:
        # Ensure edge length ~1.5 Å which is below C-C bonding threshold
        radius = 0.75 / np.sin(np.pi / n) if n >= 3 else 1.4
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)]
    )
    return CrystalMolecule(Atoms("C" * n, positions=positions, pbc=False), check_pbc=False)


def _puckered_ring(n: int, z_displacements: np.ndarray, radius: float = None) -> CrystalMolecule:
    """Ring with prescribed out-of-plane displacements."""
    if radius is None:
        radius = 0.75 / np.sin(np.pi / n) if n >= 3 else 1.4
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), z_displacements]
    )
    return CrystalMolecule(Atoms("C" * n, positions=positions, pbc=False), check_pbc=False)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 12])
class TestPlanarRings:
    """Planar regular polygons should have zero puckering."""

    def test_zero_total_amplitude(self, n):
        mol = _regular_polygon_molecule(n)
        pc = puckering_coordinates(mol, list(range(n)))
        assert pc.total_amplitude < 1e-12
        np.testing.assert_allclose(pc.z_displacements, 0.0, atol=1e-12)

    def test_all_amplitudes_zero(self, n):
        mol = _regular_polygon_molecule(n)
        pc = puckering_coordinates(mol, list(range(n)))
        np.testing.assert_allclose(pc.amplitudes, 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 12])
class TestSingleModeRoundTrip:
    """A single Fourier mode can be reconstructed exactly."""

    def test_first_paired_mode(self, n):
        # Inject only mode m=2 with known amplitude and phase
        j_arr = np.arange(n)
        q_target = 0.3
        phi_target = np.pi / 4.0
        z_input = q_target * np.sqrt(2.0 / n) * np.cos(
            2.0 * np.pi * 2 * j_arr / n + phi_target
        )
        mol = _puckered_ring(n, z_input)
        pc = puckering_coordinates(mol, list(range(n)))

        # First paired mode should match
        assert abs(pc.amplitudes[0] - q_target) < 1e-10
        # Phase comparison modulo 2pi
        delta_phi = (pc.phases[0] - phi_target + np.pi) % (2 * np.pi) - np.pi
        assert abs(delta_phi) < 1e-10

        # Reconstruction
        z_recon = reconstruct_z_from_modes(n, pc.amplitudes, pc.phases)
        np.testing.assert_allclose(z_recon, z_input, atol=1e-10)


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 12])
class TestMultiModeRoundTrip:
    """Random multi-mode puckering round-trips through forward/inverse."""

    def test_random_modes(self, n):
        rng = np.random.default_rng(seed=42 + n)
        # Generate random z displacements (zero-mean by construction)
        z_raw = rng.normal(0, 0.2, size=n)
        z_raw -= z_raw.mean()  # Ensure mean is zero for valid CP coords
        mol = _puckered_ring(n, z_raw)
        pc = puckering_coordinates(mol, list(range(n)))
        # Round-trip: reconstruction must match the CP-computed z, not raw input
        # (they differ slightly because the SVD mean plane may not be z=0)
        z_recon = reconstruct_z_from_modes(n, pc.amplitudes, pc.phases)
        np.testing.assert_allclose(z_recon, pc.z_displacements, atol=1e-8)


@pytest.mark.parametrize("n", [5, 6, 7, 8])
class TestTranslationRotationInvariance:
    """Puckering coordinates should not change under rigid body motion."""

    def test_translation(self, n):
        rng = np.random.default_rng(seed=100 + n)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        pc_before = puckering_coordinates(mol, list(range(n)))

        # Translate
        shifted = mol.copy()
        shifted.set_positions(shifted.get_positions() + np.array([5.0, -3.0, 7.0]))
        pc_after = puckering_coordinates(shifted, list(range(n)))

        np.testing.assert_allclose(pc_after.amplitudes, pc_before.amplitudes, atol=1e-10)
        # Phases that correspond to nonzero amplitude should match (skip NaN terminal)
        for i in range(len(pc_before.phases)):
            if np.isnan(pc_before.phases[i]):
                continue
            if pc_before.amplitudes[i] > 1e-8:
                delta = (pc_after.phases[i] - pc_before.phases[i] + np.pi) % (2 * np.pi) - np.pi
                assert abs(delta) < 1e-8

    def test_rotation(self, n):
        rng = np.random.default_rng(seed=200 + n)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        pc_before = puckering_coordinates(mol, list(range(n)))

        # Apply arbitrary rotation
        theta = 0.7
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        rotated = mol.copy()
        rotated.set_positions(mol.get_positions() @ rot.T)
        pc_after = puckering_coordinates(rotated, list(range(n)))

        np.testing.assert_allclose(pc_after.amplitudes, pc_before.amplitudes, atol=1e-10)
        assert abs(pc_after.total_amplitude - pc_before.total_amplitude) < 1e-10


class TestCyclicRelabeling:
    """Cyclic permutation of ring atoms transforms modes predictably."""

    def test_relabeling_preserves_amplitudes(self):
        n = 6
        rng = np.random.default_rng(seed=300)
        z = rng.normal(0, 0.2, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        pc_original = puckering_coordinates(mol, list(range(n)))

        # Cyclic shift by 1
        shifted_order = [(i + 1) % n for i in range(n)]
        pc_shifted = puckering_coordinates(mol, shifted_order)

        # Paired mode amplitudes should be identical
        # Terminal mode (even N) flips sign under odd cyclic shift, so compare abs
        n_paired = (n - 1) // 2 - 1  # number of paired modes (m=2..floor((N-1)/2))
        if n_paired > 0:
            np.testing.assert_allclose(
                pc_shifted.amplitudes[:n_paired],
                pc_original.amplitudes[:n_paired],
                atol=1e-10,
            )
        # Terminal mode: absolute value preserved
        if n % 2 == 0:
            np.testing.assert_allclose(
                abs(pc_shifted.amplitudes[-1]),
                abs(pc_original.amplitudes[-1]),
                atol=1e-10,
            )
        # Total amplitude preserved
        assert abs(pc_shifted.total_amplitude - pc_original.total_amplitude) < 1e-10


class TestReversal:
    """Ring direction reversal preserves amplitudes."""

    def test_reversal_preserves_amplitudes(self):
        n = 7
        rng = np.random.default_rng(seed=400)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        ring_forward = list(range(n))
        ring_reversed = [0] + list(range(n - 1, 0, -1))

        pc_forward = puckering_coordinates(mol, ring_forward)
        pc_reversed = puckering_coordinates(mol, ring_reversed)

        np.testing.assert_allclose(pc_reversed.amplitudes, pc_forward.amplitudes, atol=1e-10)


class TestErrorCases:
    """Invalid inputs raise appropriate errors."""

    def test_too_few_atoms(self):
        mol = CrystalMolecule(Atoms("CC", positions=[[0, 0, 0], [1, 0, 0]], pbc=False), check_pbc=False)
        with pytest.raises(InvalidRingOrderError, match="at least 3"):
            puckering_coordinates(mol, [0, 1])

    def test_repeated_atoms(self):
        mol = _regular_polygon_molecule(5)
        with pytest.raises(InvalidRingOrderError, match="unique"):
            puckering_coordinates(mol, [0, 1, 2, 3, 3])

    def test_out_of_range(self):
        mol = _regular_polygon_molecule(5)
        with pytest.raises(InvalidRingOrderError, match="out of range"):
            puckering_coordinates(mol, [0, 1, 2, 3, 99])

    def test_non_bonded_consecutive(self):
        # 5-atom chain, not a ring — atom 0 and 4 not bonded
        positions = np.array([[i * 1.4, 0, 0] for i in range(5)])
        mol = CrystalMolecule(Atoms("C" * 5, positions=positions, pbc=False), check_pbc=False)
        with pytest.raises(InvalidRingOrderError, match="not bonded"):
            puckering_coordinates(mol, [0, 1, 2, 3, 4])

    def test_collinear_atoms(self):
        positions = np.array([[0, 0, i * 0.5] for i in range(4)])
        mol = CrystalMolecule(Atoms("C" * 4, positions=positions, pbc=False), check_pbc=False)
        # These form a linear chain; will hit InvalidRingOrderError (no closing
        # bond) or DegenerateRingGeometryError depending on bond perception
        with pytest.raises((InvalidRingOrderError, DegenerateRingGeometryError)):
            puckering_coordinates(mol, [0, 1, 2, 3])


class TestFindRingSystems:
    """Ring system detection proposes cycles with classification."""

    def test_single_ring(self):
        mol = _regular_polygon_molecule(6)
        systems = find_ring_systems(mol)
        assert len(systems) == 1
        assert systems[0].ring_size == 6
        assert systems[0].is_simple is True
        assert systems[0].classification == "simple"

    def test_no_ring(self):
        # Linear chain
        positions = np.array([[i * 1.4, 0, 0] for i in range(5)])
        mol = CrystalMolecule(Atoms("C" * 5, positions=positions, pbc=False), check_pbc=False)
        systems = find_ring_systems(mol)
        assert len(systems) == 0


class TestReconstructZInvalidInput:
    """Invalid inputs to reconstruction raise errors."""

    def test_ring_size_too_small(self):
        with pytest.raises(InvalidRingOrderError):
            reconstruct_z_from_modes(2, np.array([]), np.array([]))
