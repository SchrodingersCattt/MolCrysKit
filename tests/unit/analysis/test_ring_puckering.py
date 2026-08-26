"""Tests for general N-membered ring puckering coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx
import pytest
from ase import Atoms

from molcrys_kit.io import read_mol_crystal
from molcrys_kit.analysis.ring_conformation import (
    DegenerateRingGeometryError,
    InvalidRingOrderError,
    RingConformationError,
    RingCycleLimitError,
    find_ring_systems,
    puckering_coordinates,
    reconstruct_z_from_modes,
)
from molcrys_kit.structures.molecule import CrystalMolecule


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CIF_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cif"


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
    return CrystalMolecule(
        Atoms("C" * n, positions=positions, pbc=False), check_pbc=False
    )


def _puckered_ring(
    n: int, z_displacements: np.ndarray, radius: float = None
) -> CrystalMolecule:
    """Ring with prescribed out-of-plane displacements."""
    if radius is None:
        radius = 0.75 / np.sin(np.pi / n) if n >= 3 else 1.4
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), z_displacements]
    )
    return CrystalMolecule(
        Atoms("C" * n, positions=positions, pbc=False), check_pbc=False
    )


def _molecule_with_graph(graph: nx.Graph) -> CrystalMolecule:
    """Build a molecule whose explicit graph isolates topology tests."""
    n = max(graph.nodes) + 1
    positions = np.column_stack([np.arange(n) * 10.0, np.zeros((n, 2))])
    molecule = CrystalMolecule(
        Atoms("C" * n, positions=positions, pbc=False), check_pbc=False
    )
    molecule._graph = graph.copy()
    return molecule


def _phase_delta(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


def _oracle_chordless_cycles(graph: nx.Graph, max_size: int) -> set[tuple[int, ...]]:
    """Brute-force independent oracle for small graph-atlas fixtures."""
    nodes = sorted(graph.nodes)
    cycles = set()
    for size in range(3, min(max_size, len(nodes)) + 1):
        from itertools import combinations, permutations

        for subset in combinations(nodes, size):
            start = min(subset)
            for tail in permutations(node for node in subset if node != start):
                cycle = (start, *tail)
                if cycle[1] > cycle[-1]:
                    continue
                cycle_edges = {
                    frozenset((cycle[i], cycle[(i + 1) % size])) for i in range(size)
                }
                induced_edges = {
                    frozenset(edge) for edge in graph.subgraph(subset).edges
                }
                if induced_edges == cycle_edges:
                    cycles.add(cycle)
    return cycles


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
        rotation = np.array(
            [
                [0.36, 0.48, -0.8],
                [-0.8, 0.60, 0.0],
                [0.48, 0.64, 0.60],
            ]
        )
        mol.set_positions(
            mol.get_positions() @ rotation.T + np.array([1.0e5, -2.0e5, 3.0e5])
        )
        pc = puckering_coordinates(mol, list(range(n)))
        np.testing.assert_allclose(pc.amplitudes, 0.0, atol=1e-10)
        paired_count = max(0, (n - 1) // 2 - 1)
        zero_modes = pc.amplitudes[:paired_count] == 0.0
        np.testing.assert_array_equal(pc.phases[:paired_count][zero_modes], 0.0)


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 12])
class TestSingleModeRoundTrip:
    """A single Fourier mode can be reconstructed exactly."""

    def test_first_paired_mode(self, n):
        # Inject only mode m=2 with known amplitude and phase
        j_arr = np.arange(n)
        q_target = 0.3
        phi_target = np.pi / 4.0
        z_input = (
            q_target
            * np.sqrt(2.0 / n)
            * np.cos(2.0 * np.pi * 2 * j_arr / n + phi_target)
        )
        mol = _puckered_ring(n, z_input)
        pc = puckering_coordinates(mol, list(range(n)))

        # First paired mode should match
        assert abs(pc.amplitudes[0] - q_target) < 1e-10

        # Reconstruction
        z_recon = reconstruct_z_from_modes(n, pc.amplitudes, pc.phases)
        np.testing.assert_allclose(z_recon, pc.z_displacements, atol=1e-10)

    def test_large_translation_preserves_nonzero_phase(self, n):
        j_arr = np.arange(n)
        q_target = 1.0e-3
        phi_target = 1.1
        z_input = (
            q_target
            * np.sqrt(2.0 / n)
            * np.cos(2.0 * np.pi * 2 * j_arr / n + phi_target)
        )
        mol = _puckered_ring(n, z_input)
        before = puckering_coordinates(mol, list(range(n)))
        mol.set_positions(mol.get_positions() + 1.0e12)
        after = puckering_coordinates(mol, list(range(n)))

        assert after.amplitudes[0] > 0.0
        np.testing.assert_allclose(after.amplitudes[0], before.amplitudes[0], rtol=0.15)
        assert abs(_phase_delta(after.phases[:1], before.phases[:1])[0]) < 0.15
        np.testing.assert_allclose(
            reconstruct_z_from_modes(n, after.amplitudes, after.phases),
            after.z_displacements,
            atol=1e-12,
        )


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
        # (they differ slightly because the CP reference plane may not be z=0)
        z_recon = reconstruct_z_from_modes(n, pc.amplitudes, pc.phases)
        np.testing.assert_allclose(z_recon, pc.z_displacements, atol=1e-12)

    def test_tiny_mode_at_large_scale_is_preserved(self, n):
        scale = 1.0e7
        j_arr = np.arange(n)
        z = 5.0e-7 * np.sqrt(2.0 / n) * np.sin(4.0 * np.pi * j_arr / n)
        mol = _puckered_ring(n, z, radius=scale)
        mol._graph = nx.cycle_graph(n)
        pc = puckering_coordinates(mol, list(range(n)))

        reconstructed = reconstruct_z_from_modes(n, pc.amplitudes, pc.phases)
        assert pc.amplitudes[0] > 0.0
        np.testing.assert_allclose(reconstructed, pc.z_displacements, atol=1e-14)
        assert pc.total_amplitude == pytest.approx(np.linalg.norm(reconstructed))
        assert pc.total_amplitude**2 == pytest.approx(float(np.sum(pc.amplitudes**2)))


def test_nonplanar_four_ring_terminal_mode_round_trip():
    z = np.array([0.2, -0.2, 0.2, -0.2])
    mol = _puckered_ring(4, z)
    pc = puckering_coordinates(mol, [0, 1, 2, 3])
    assert pc.amplitudes.shape == (1,)
    assert np.isnan(pc.phases[0])
    np.testing.assert_allclose(
        reconstruct_z_from_modes(4, pc.amplitudes, pc.phases),
        pc.z_displacements,
    )


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

        np.testing.assert_allclose(
            pc_after.amplitudes, pc_before.amplitudes, atol=1e-10
        )
        # Phases that correspond to nonzero amplitude should match (skip NaN terminal)
        for i in range(len(pc_before.phases)):
            if np.isnan(pc_before.phases[i]):
                continue
            if pc_before.amplitudes[i] > 1e-8:
                delta = (pc_after.phases[i] - pc_before.phases[i] + np.pi) % (
                    2 * np.pi
                ) - np.pi
                assert abs(delta) < 1e-8

    def test_rotation(self, n):
        rng = np.random.default_rng(seed=200 + n)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        pc_before = puckering_coordinates(mol, list(range(n)))

        # Apply arbitrary rotation
        theta = 0.7
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        rotated = mol.copy()
        rotated.set_positions(mol.get_positions() @ rot.T)
        pc_after = puckering_coordinates(rotated, list(range(n)))

        np.testing.assert_allclose(
            pc_after.amplitudes, pc_before.amplitudes, atol=1e-10
        )
        assert abs(pc_after.total_amplitude - pc_before.total_amplitude) < 1e-10

    def test_signed_coordinates_are_covariant_across_global_xy_plane(self, n):
        rng = np.random.default_rng(seed=250 + n)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        pc_before = puckering_coordinates(mol, list(range(n)))

        rotation = np.diag([1.0, -1.0, -1.0])
        rotated = mol.copy()
        rotated.set_positions(mol.get_positions() @ rotation.T)
        pc_after = puckering_coordinates(rotated, list(range(n)))

        np.testing.assert_allclose(
            pc_after.mean_plane_normal,
            pc_before.mean_plane_normal @ rotation.T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            pc_after.mean_plane_center,
            pc_before.mean_plane_center @ rotation.T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            pc_after.z_displacements, pc_before.z_displacements, atol=1e-12
        )
        np.testing.assert_allclose(
            pc_after.amplitudes, pc_before.amplitudes, atol=1e-12
        )
        paired_count = max(0, (n - 1) // 2 - 1)
        phase_delta = (
            pc_after.phases[:paired_count] - pc_before.phases[:paired_count] + np.pi
        ) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(phase_delta, 0.0, atol=1e-12)

    @pytest.mark.parametrize("scale", [1.0e-7, 1.0e7])
    def test_uniform_scaling(self, n, scale):
        rng = np.random.default_rng(seed=280 + n)
        z = rng.normal(0, 0.15, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        before = puckering_coordinates(mol, list(range(n)))
        scaled = mol.copy()
        scaled.set_positions(mol.get_positions() * scale)
        after = puckering_coordinates(scaled, list(range(n)))

        np.testing.assert_allclose(
            after.z_displacements, before.z_displacements * scale
        )
        np.testing.assert_allclose(after.amplitudes, before.amplitudes * scale)
        paired_count = max(0, (n - 1) // 2 - 1)
        np.testing.assert_allclose(
            _phase_delta(after.phases[:paired_count], before.phases[:paired_count]),
            0.0,
            atol=1e-10,
        )


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

    @pytest.mark.parametrize("n", [7, 8])
    @pytest.mark.parametrize("shift", [1, 3])
    def test_exact_cyclic_shift_transform(self, n, shift):
        rng = np.random.default_rng(seed=500 + n)
        z = rng.normal(0, 0.2, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        original = puckering_coordinates(mol, list(range(n)))
        shifted = puckering_coordinates(mol, [(j + shift) % n for j in range(n)])

        np.testing.assert_allclose(
            shifted.mean_plane_normal, original.mean_plane_normal
        )
        np.testing.assert_allclose(
            shifted.z_displacements, np.roll(original.z_displacements, -shift)
        )
        paired_count = max(0, (n - 1) // 2 - 1)
        np.testing.assert_allclose(
            shifted.amplitudes[:paired_count], original.amplitudes[:paired_count]
        )
        assert shifted.total_amplitude == pytest.approx(original.total_amplitude)
        np.testing.assert_allclose(
            shifted.mean_plane_center, original.mean_plane_center, atol=1e-14
        )
        modes = np.arange(2, 2 + paired_count)
        expected_phases = original.phases[:paired_count] + 2 * np.pi * modes * shift / n
        np.testing.assert_allclose(
            _phase_delta(shifted.phases[:paired_count], expected_phases),
            0.0,
            atol=1e-10,
        )
        if n % 2 == 0:
            assert shifted.amplitudes[-1] == pytest.approx(
                (-1) ** shift * original.amplitudes[-1]
            )
        np.testing.assert_allclose(
            reconstruct_z_from_modes(n, shifted.amplitudes, shifted.phases),
            shifted.z_displacements,
        )


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

        np.testing.assert_allclose(
            pc_reversed.amplitudes, pc_forward.amplitudes, atol=1e-10
        )

    @pytest.mark.parametrize("n", [7, 8])
    @pytest.mark.parametrize("anchor", [0, 3])
    def test_exact_reversal_transform(self, n, anchor):
        rng = np.random.default_rng(seed=600 + n)
        z = rng.normal(0, 0.2, size=n)
        z -= z.mean()
        mol = _puckered_ring(n, z)
        original = puckering_coordinates(mol, list(range(n)))
        order = [(anchor - j) % n for j in range(n)]
        reversed_pc = puckering_coordinates(mol, order)

        np.testing.assert_allclose(
            reversed_pc.mean_plane_normal, -original.mean_plane_normal
        )
        np.testing.assert_allclose(
            reversed_pc.z_displacements,
            -original.z_displacements[order],
        )
        paired_count = max(0, (n - 1) // 2 - 1)
        modes = np.arange(2, 2 + paired_count)
        expected_phases = (
            np.pi - original.phases[:paired_count] - 2 * np.pi * modes * anchor / n
        )
        np.testing.assert_allclose(
            _phase_delta(reversed_pc.phases[:paired_count], expected_phases),
            0.0,
            atol=1e-10,
        )
        if n % 2 == 0:
            assert reversed_pc.amplitudes[-1] == pytest.approx(
                (-1) ** (anchor + 1) * original.amplitudes[-1]
            )
        np.testing.assert_allclose(
            reversed_pc.amplitudes[:paired_count],
            original.amplitudes[:paired_count],
        )
        assert reversed_pc.total_amplitude == pytest.approx(original.total_amplitude)
        np.testing.assert_allclose(
            reversed_pc.mean_plane_center, original.mean_plane_center, atol=1e-14
        )
        np.testing.assert_allclose(
            reconstruct_z_from_modes(n, reversed_pc.amplitudes, reversed_pc.phases),
            reversed_pc.z_displacements,
        )


class TestErrorCases:
    """Invalid inputs raise appropriate errors."""

    def test_too_few_atoms(self):
        mol = CrystalMolecule(
            Atoms("CC", positions=[[0, 0, 0], [1, 0, 0]], pbc=False), check_pbc=False
        )
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
        mol = CrystalMolecule(
            Atoms("C" * 5, positions=positions, pbc=False), check_pbc=False
        )
        with pytest.raises(InvalidRingOrderError, match="not bonded"):
            puckering_coordinates(mol, [0, 1, 2, 3, 4])

    def test_collinear_atoms(self):
        graph = nx.cycle_graph(4)
        mol = _molecule_with_graph(graph)
        with pytest.raises(DegenerateRingGeometryError):
            puckering_coordinates(mol, [0, 1, 2, 3])

    def test_tiny_valid_ring_is_not_degenerate(self):
        mol = _regular_polygon_molecule(5, radius=1.0e-7)
        pc = puckering_coordinates(mol, list(range(5)))
        assert pc.ring_size == 5

    @pytest.mark.parametrize("invalid_index", [True, 1.9, "1", None])
    def test_non_integer_atom_index(self, invalid_index):
        mol = _regular_polygon_molecule(5)
        with pytest.raises(InvalidRingOrderError, match="integer atom indices"):
            puckering_coordinates(mol, [0, invalid_index, 2, 3, 4])


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
        mol = CrystalMolecule(
            Atoms("C" * 5, positions=positions, pbc=False), check_pbc=False
        )
        systems = find_ring_systems(mol)
        assert len(systems) == 0

    @pytest.mark.parametrize(
        "filename, expected_atoms, expected_molecules, expected_rings",
        [
            ("Acetaminophen_HXACAN.cif", 160, 8, [(6, "simple")]),
            ("ISATIN.cif", 44, 4, [(5, "fused"), (6, "fused")]),
        ],
    )
    def test_real_cif_ring_topology(
        self,
        filename,
        expected_atoms,
        expected_molecules,
        expected_rings,
    ):
        crystal = read_mol_crystal(str(CIF_DATA_DIR / filename))
        assert crystal.get_total_nodes() == expected_atoms
        assert len(crystal.molecules) == expected_molecules
        for molecule in crystal.molecules:
            systems = find_ring_systems(molecule, max_ring_size=8)
            assert [
                (system.ring_size, system.classification) for system in systems
            ] == expected_rings

    def test_fused_ring(self):
        graph = nx.Graph()
        graph.add_edges_from(nx.cycle_graph(4).edges)
        graph.add_edges_from([(1, 4), (4, 5), (5, 2)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        assert {system.classification for system in systems} == {"fused"}

    def test_spiro_ring(self):
        graph = nx.Graph()
        graph.add_edges_from(nx.cycle_graph(4).edges)
        graph.add_edges_from([(0, 4), (4, 5), (5, 0)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        assert {system.classification for system in systems} == {"spiro"}

    def test_linked_ring_is_not_spiro(self):
        graph = nx.cycle_graph(4)
        graph.add_edges_from([(0, 4), (4, 5), (5, 6), (6, 7), (7, 5)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        four_ring = next(system for system in systems if system.ring_size == 4)
        assert four_ring.classification == "simple"

    def test_nonadjacent_bridgeheads_are_bridged(self):
        graph = nx.cycle_graph(6)
        graph.add_edges_from([(0, 6), (6, 3)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        assert {system.classification for system in systems} == {"bridged"}

    def test_middle_ring_with_two_fused_neighbors_is_fused(self):
        graph = nx.cycle_graph(6)
        graph.add_edges_from([(0, 6), (6, 7), (7, 1)])
        graph.add_edges_from([(3, 8), (8, 9), (9, 4)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        six_ring = next(system for system in systems if system.ring_size == 6)
        assert six_ring.classification == "fused"

    def test_adjacent_fused_paths_are_not_bridged(self):
        graph = nx.cycle_graph(6)
        graph.add_edges_from([(0, 6), (6, 1), (1, 7), (7, 2)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        six_ring = next(system for system in systems if system.ring_size == 6)
        assert six_ring.classification == "fused"

    def test_filtered_out_partner_still_affects_classification(self):
        graph = nx.cycle_graph(3)
        graph.add_edges_from([(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)])
        systems = find_ring_systems(_molecule_with_graph(graph), max_ring_size=3)
        assert len(systems) == 1
        assert systems[0].classification == "spiro"

    def test_chorded_square_returns_only_chordless_triangles(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        systems = find_ring_systems(_molecule_with_graph(graph))
        assert [system.ring_atoms for system in systems] == [(0, 1, 2), (0, 2, 3)]

    def test_max_ring_size_is_inclusive(self):
        graph = nx.cycle_graph(4)
        assert find_ring_systems(_molecule_with_graph(graph), max_ring_size=3) == []
        assert len(find_ring_systems(_molecule_with_graph(graph), max_ring_size=4)) == 1

    def test_cycle_results_ignore_edge_insertion_order(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4), (4, 5), (5, 0)]
        records = []
        for inserted in [edges, list(reversed(edges)), edges[3:] + edges[:3]]:
            graph = nx.Graph()
            graph.add_edges_from(inserted)
            records.append(
                [
                    (system.ring_atoms, system.classification)
                    for system in find_ring_systems(_molecule_with_graph(graph))
                ]
            )
        assert records[1:] == records[:1] * 2

    @pytest.mark.parametrize("atlas_index", [7, 16, 38, 52, 125])
    def test_graph_atlas_matches_independent_chordless_oracle(self, atlas_index):
        graph = nx.graph_atlas_g()[atlas_index]
        if not graph.nodes:
            return
        systems = find_ring_systems(_molecule_with_graph(graph), max_ring_size=6)
        assert {system.ring_atoms for system in systems} == _oracle_chordless_cycles(
            graph, 6
        )

    def test_cycle_budget_raises_typed_error(self):
        graph = nx.complete_bipartite_graph(8, 8)
        with pytest.raises(RingCycleLimitError, match="max_cycles"):
            find_ring_systems(
                _molecule_with_graph(graph), max_ring_size=4, max_cycles=10
            )
        with pytest.raises(RingCycleLimitError, match="max_search_states"):
            find_ring_systems(
                _molecule_with_graph(graph),
                max_ring_size=4,
                max_cycles=10_000,
                max_search_states=5,
            )

    @pytest.mark.parametrize("invalid", [True, 3.5, 2])
    def test_invalid_max_ring_size(self, invalid):
        with pytest.raises(RingConformationError, match="max_ring_size"):
            find_ring_systems(_regular_polygon_molecule(5), max_ring_size=invalid)


class TestReconstructZInvalidInput:
    """Invalid inputs to reconstruction raise errors."""

    def test_ring_size_too_small(self):
        with pytest.raises(InvalidRingOrderError):
            reconstruct_z_from_modes(2, np.array([]), np.array([]))

    @pytest.mark.parametrize(
        "ring_size, amplitudes, phases",
        [
            (6, np.array([0.2]), np.array([0.1])),
            (5, np.array([0.2, 0.3]), np.array([0.1, 0.2])),
            (5, np.array([0.2]), np.array([])),
            (5, np.array([[0.2]]), np.array([0.1])),
        ],
    )
    def test_mode_shapes_must_match_exactly(self, ring_size, amplitudes, phases):
        with pytest.raises(RingConformationError, match="shape"):
            reconstruct_z_from_modes(ring_size, amplitudes, phases)

    def test_mode_values_must_be_finite(self):
        with pytest.raises(RingConformationError, match="finite"):
            reconstruct_z_from_modes(5, np.array([np.inf]), np.array([0.0]))
        with pytest.raises(RingConformationError, match="finite"):
            reconstruct_z_from_modes(5, np.array([0.2]), np.array([np.nan]))

    def test_even_terminal_phase_must_be_nan(self):
        with pytest.raises(RingConformationError, match="must be NaN"):
            reconstruct_z_from_modes(6, np.array([0.2, -0.1]), np.array([0.3, 0.0]))
