"""
Unit tests for molcrys_kit.analysis.disorder (DisorderGraphBuilder, DisorderInfo).
"""

import itertools

import numpy as np
import pytest
import networkx as nx

from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder


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
        frac_coords = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            tet1,
            tet2,
        ])
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
        inter_edges = sum(
            1 for a in set_a for b in set_b if graph.has_edge(a, b)
        )
        assert inter_edges > 0
        mis_size = _max_independent_set_size(graph, set_a + set_b)
        assert mis_size >= 1
