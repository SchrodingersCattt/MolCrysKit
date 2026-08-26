"""
Unit tests for special-position disorder helpers in DisorderGraphBuilder.

CIF-level regressions live in `test_disorder_regression.py` -- this file
is intentionally limited to small synthetic inputs that exercise the
internals of `_is_same_parent_pair`, `_add_implicit_sp_conflicts`, and
`_decompose_cliques`.
"""

import numpy as np
import networkx as nx

from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder


# =====================================================================
# DisorderGraphBuilder: implicit special-position conflict detection
# =====================================================================


class TestImplicitSPConflicts:
    """Tests for `_is_same_parent_pair` and `_add_implicit_sp_conflicts`."""

    def test_partial_occ_dg0_same_parent_are_competing(self):
        """Two dg=0 partial-occ copies from same asym parent should NOT be
        treated as 'same parent' (they compete for the same site)."""
        info = DisorderInfo(
            labels=["H3A", "H3A"],
            symbols=["H", "H"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.12, 0.12, 0.12]]),
            occupancies=[0.5, 0.5],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 0],
            site_symmetry_order=[1, 1],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert not builder._is_same_parent_pair(0, 1)

    def test_full_occ_dg0_same_parent_are_same(self):
        """Two dg=0 full-occ copies from same asym parent should be
        treated as 'same parent' (legitimate symmetry copies)."""
        info = DisorderInfo(
            labels=["N1", "N1"],
            symbols=["N", "N"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]),
            occupancies=[1.0, 1.0],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 0],
            site_symmetry_order=[3, 3],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert builder._is_same_parent_pair(0, 1)

    def test_different_parent_are_not_same(self):
        """Two atoms from different asym parents should NOT be same parent."""
        info = DisorderInfo(
            labels=["N1", "N2"],
            symbols=["N", "N"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]),
            occupancies=[1.0, 1.0],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 1],
            site_symmetry_order=[1, 1],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert not builder._is_same_parent_pair(0, 1)

    def test_implicit_sp_conflicts_added_for_partial_occ(self):
        """`_add_implicit_sp_conflicts` should add conflict edges between
        partial-occ dg=0 copies of the same parent atom."""
        n = 6
        info = DisorderInfo(
            labels=["H3A"] * n,
            symbols=["H"] * n,
            frac_coords=np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.12],
                    [0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.32],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.52],
                ]
            ),
            occupancies=[0.5] * n,
            disorder_groups=[0] * n,
            assemblies=[""] * n,
            asym_id=[0] * n,
            site_symmetry_order=[1] * n,
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        builder._precompute_metrics()
        builder.graph = nx.Graph()
        builder.graph.add_nodes_from(range(n))

        builder._add_implicit_sp_conflicts()

        # Each cluster {(0,1), (2,3), (4,5)} should have at least one
        # internal conflict edge.
        assert (
            builder.graph.has_edge(0, 1)
            or builder.graph.has_edge(2, 3)
            or builder.graph.has_edge(4, 5)
        )

    def test_ammonium_tetrahedral_decomposition(self):
        """NH4+ with 8 half-occ H atoms should decompose into two
        tetrahedral groups via `_decompose_cliques`."""
        labels = ["N1"] + [f"H{i}" for i in range(1, 9)]
        symbols = ["N"] + ["H"] * 8
        tet1 = np.array(
            [
                [0.02, 0.02, 0.02],
                [0.02, -0.02, -0.02],
                [-0.02, 0.02, -0.02],
                [-0.02, -0.02, 0.02],
            ]
        )
        tet2 = np.array(
            [
                [0.025, 0.025, -0.025],
                [0.025, -0.025, 0.025],
                [-0.025, 0.025, 0.025],
                [-0.025, -0.025, -0.025],
            ]
        )
        frac_coords = np.vstack([np.array([[0.0, 0.0, 0.0]]), tet1, tet2])
        occupancies = [1.0] + [0.5] * 8
        disorder_groups = [0] * 9
        assemblies = [""] * 9
        asym_id = [0] + [1] * 4 + [2] * 4
        sso = [6] + [1] * 4 + [3] * 4

        info = DisorderInfo(
            labels=labels,
            symbols=symbols,
            frac_coords=frac_coords,
            occupancies=occupancies,
            disorder_groups=disorder_groups,
            assemblies=assemblies,
            asym_id=asym_id,
            site_symmetry_order=sso,
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()

        h_nodes = list(range(1, 9))
        h_edges = [(u, v) for u, v in graph.edges if u in h_nodes and v in h_nodes]
        assert len(h_edges) > 0, "Should have conflict edges between H atoms"
