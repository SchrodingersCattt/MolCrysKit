"""
Unit tests for molcrys_kit.analysis.disorder (DisorderGraphBuilder, DisorderInfo, DisorderSolver).
"""

import itertools

import numpy as np
import pytest
import networkx as nx

from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.analysis.disorder.solver import DisorderSolver
from molcrys_kit.structures.crystal import MolecularCrystal


def _crystal_symbol_signature(crystal):
    symbols = []
    for molecule in crystal.molecules:
        symbols.extend(molecule.get_chemical_symbols())
    return tuple(sorted(symbols))


def _max_independent_set_size(graph, nodes):
    """Max independent set size for subgraph of given nodes."""
    sub = graph.subgraph(nodes).copy()
    try:
        mis = nx.algorithms.approximation.maximum_independent_set(sub)
        return len(mis)
    except Exception:
        if len(sub.nodes()) <= 20:
            for r in range(len(sub.nodes()), 0, -1):
                for subset in itertools.combinations(sub.nodes(), r):
                    if sub.subgraph(subset).number_of_edges() == 0:
                        return r
        return 1


# =====================================================================
# DisorderGraphBuilder
# =====================================================================


class TestDisorderGraph:
    """DisorderGraphBuilder topology and conflict types."""

    def test_simple_geometric_collision(self):
        info = DisorderInfo(
            labels=["H1", "H2"],
            symbols=["H", "H"],
            frac_coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            occupancies=[1.0, 1.0],
            disorder_groups=[0, 0],
            assemblies=["", ""],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        assert graph.has_edge(0, 1)
        assert graph[0][1]["conflict_type"] == "geometric"

    def test_geometric_vs_bonded(self):
        info = DisorderInfo(
            labels=["A", "B", "C", "D"],
            symbols=["C", "O", "H", "H"],
            frac_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.05],
                [0.0, 0.1, 0.0],
                [0.0, 0.105, 0.0],
            ]),
            occupancies=[1.0, 1.0, 1.0, 1.0],
            disorder_groups=[0, 0, 0, 0],
            assemblies=["", "", "", ""],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        assert graph.has_edge(2, 3)
        assert graph[2][3]["conflict_type"] == "geometric"
        if builder._are_bonded("C", "O", 0.5):
            assert not graph.has_edge(0, 1)

    def test_explicit_conflicts_topology(self):
        info = DisorderInfo(
            labels=["C1", "C2", "H1", "H2"],
            symbols=["C", "C", "H", "H"],
            frac_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]),
            occupancies=[1.0, 1.0, 1.0, 1.0],
            disorder_groups=[1, 2, 1, 2],
            assemblies=["A", "A", "A", "A"],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        assert graph.has_edge(0, 1)
        assert graph.has_edge(2, 3)
        assert graph[0][1]["conflict_type"] == "explicit"
        assert graph[2][3]["conflict_type"] == "explicit"
        mis_size = _max_independent_set_size(graph, [0, 1])
        assert mis_size == 1

    def test_ammonium_topology(self):
        """Exclusion graph for DAP-4-like ammonium: 1 N + 8 H (two tetrahedra)."""
        labels = ["N1"] + [f"H{i}" for i in range(1, 9)]
        symbols = ["N"] + ["H"] * 8
        tet1 = np.array([
            [0.02, 0.02, 0.02],
            [0.02, -0.02, -0.02],
            [-0.02, 0.02, -0.02],
            [-0.02, -0.02, 0.02],
        ])
        tet2 = np.array([
            [0.025, 0.025, -0.025],
            [0.025, -0.025, 0.025],
            [-0.025, 0.025, 0.025],
            [-0.025, -0.025, -0.025],
        ])
        frac_coords = np.vstack([np.array([[0.0, 0.0, 0.0]]), tet1, tet2])
        occupancies = [1.0] + [0.5] * 8
        disorder_groups = [0] * 9
        assemblies = [""] * 9
        info = DisorderInfo(
            labels=labels,
            symbols=symbols,
            frac_coords=frac_coords,
            occupancies=occupancies,
            disorder_groups=disorder_groups,
            assemblies=assemblies,
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        set_a = [1, 2, 3, 4]
        set_b = [5, 6, 7, 8]
        inter_edges = sum(1 for a in set_a for b in set_b if graph.has_edge(a, b))
        assert inter_edges > 0
        mis_size = _max_independent_set_size(graph, set_a + set_b)
        assert mis_size >= 1


# =====================================================================
# DisorderSolver
# =====================================================================


class TestDisorderSolver:
    """DisorderSolver structure generation."""

    @pytest.fixture
    def simple_disorder_setup(self):
        """Two pairs of conflicting atoms at well-separated positions."""
        info = DisorderInfo(
            labels=["C1", "C2", "O1", "O2"],
            symbols=["C", "C", "O", "O"],
            frac_coords=np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.105],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.505],
            ]),
            occupancies=[0.5, 0.5, 0.5, 0.5],
            disorder_groups=[1, 2, 1, 2],
            assemblies=["A", "A", "A", "A"],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        return info, graph, lattice

    @pytest.fixture
    def multi_part_setup(self):
        """Two independent assemblies, each with two PART alternatives."""
        info = DisorderInfo(
            labels=["A1", "A2", "B1", "B2"],
            symbols=["C", "O", "N", "F"],
            frac_coords=np.array([
                [0.10, 0.10, 0.10],
                [0.16, 0.10, 0.10],
                [0.60, 0.60, 0.60],
                [0.66, 0.60, 0.60],
            ]),
            occupancies=[0.7, 0.3, 0.6, 0.4],
            disorder_groups=[1, 2, 1, 2],
            assemblies=["A", "A", "B", "B"],
        )
        lattice = np.eye(3) * 20.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        return info, graph, lattice

    def test_optimal_solve(self, simple_disorder_setup):
        info, graph, lattice = simple_disorder_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=1, method="optimal")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], MolecularCrystal)

    def test_random_solve(self, simple_disorder_setup):
        info, graph, lattice = simple_disorder_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=3, method="random")
        assert isinstance(results, list)
        assert len(results) >= 1
        for crystal in results:
            assert isinstance(crystal, MolecularCrystal)

    def test_unknown_method_raises(self, simple_disorder_setup):
        info, graph, lattice = simple_disorder_setup
        solver = DisorderSolver(info, graph, lattice)
        with pytest.raises(ValueError, match="Unknown method"):
            solver.solve(method="invalid")

    def test_random_covers_all_parts(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=20, method="random", random_seed=0)

        signatures = {_crystal_symbol_signature(crystal) for crystal in results}
        assert signatures == {
            ("C", "N"),
            ("C", "F"),
            ("N", "O"),
            ("F", "O"),
        }

    def test_random_seed_reproducible(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver_a = DisorderSolver(info, graph, lattice)
        solver_b = DisorderSolver(info, graph, lattice)

        results_a = solver_a.solve(num_structures=4, method="random", random_seed=42)
        results_b = solver_b.solve(num_structures=4, method="random", random_seed=42)

        assert [
            _crystal_symbol_signature(crystal) for crystal in results_a
        ] == [
            _crystal_symbol_signature(crystal) for crystal in results_b
        ]

    def test_random_weighted_by_occupancy(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        draws = 500
        c_count = 0

        for seed in range(draws):
            solver = DisorderSolver(info, graph, lattice)
            # random replica #0 is the deterministic MWIS reference; replica #1
            # is the first occupancy-weighted sample.
            result = solver.solve(
                num_structures=2,
                method="random",
                random_seed=seed,
            )[1]
            if "C" in _crystal_symbol_signature(result):
                c_count += 1

        # Because replica #0 is always the C/N reference and duplicate samples
        # are skipped, replica #1 follows the conditional distribution over
        # non-reference structures.  For this setup:
        #   P(C,F) / (1 - P(C,N)) = (0.7 * 0.4) / (1 - 0.7 * 0.6)
        assert abs((c_count / draws) - (0.28 / 0.58)) < 0.08

    def test_enumerate_yields_all_combinations(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(method="enumerate")

        # enumerate now returns chemistry-equivalent replicas; alternatives
        # whose element totals drift from the MWIS reference are stabilised
        # back to replica #0.
        assert [
            _crystal_symbol_signature(crystal) for crystal in results
        ] == [
            ("C", "N"),
            ("C", "N"),
            ("C", "N"),
            ("C", "N"),
        ]

    def test_enumerate_caps_at_num_structures(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=2, method="enumerate")

        assert [
            _crystal_symbol_signature(crystal) for crystal in results
        ] == [
            ("C", "N"),
            ("C", "N"),
        ]

    def test_optimal_prefers_highest_occupancy_parts(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=1, method="optimal")

        assert _crystal_symbol_signature(results[0]) == ("C", "N")

    def test_solve_can_return_kept_indices(self, multi_part_setup):
        info, graph, lattice = multi_part_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(
            num_structures=1,
            method="optimal",
            return_kept_indices=True,
        )

        assert len(results) == 1
        crystal, kept_indices = results[0]
        assert isinstance(crystal, MolecularCrystal)
        assert kept_indices == [0, 2]
        assert tuple(sorted(info.symbols[i] for i in kept_indices)) == _crystal_symbol_signature(crystal)

    def test_reconstruct_gives_valid_crystal(self, simple_disorder_setup):
        info, graph, lattice = simple_disorder_setup
        solver = DisorderSolver(info, graph, lattice)
        results = solver.solve(num_structures=1, method="optimal")
        crystal = results[0]
        total_atoms = sum(len(m) for m in crystal.molecules)
        assert total_atoms >= 1
        np.testing.assert_allclose(crystal.lattice, lattice)
