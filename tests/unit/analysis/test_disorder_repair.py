"""Focused unit tests for disorder-solver repair/stabilisation helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np

from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.solver import DisorderSolver


def _solver(symbols, frac_coords, atom_groups):
    labels = [f"{symbol}{idx}" for idx, symbol in enumerate(symbols)]
    info = DisorderInfo(
        labels=labels,
        symbols=list(symbols),
        frac_coords=np.array(frac_coords, dtype=float),
        occupancies=[0.5] * len(symbols),
        disorder_groups=[0] * len(symbols),
        assemblies=[""] * len(symbols),
        sym_op_indices=[0] * len(symbols),
        asym_id=list(range(len(symbols))),
        site_symmetry_order=[1] * len(symbols),
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    solver = DisorderSolver(info, graph, np.eye(3) * 10.0)
    solver.atom_groups = [list(group) for group in atom_groups]
    return solver


def test_repair_motifs_completes_partial_water_group():
    solver = _solver(
        symbols=["O", "H", "H"],
        frac_coords=[
            [0.50, 0.50, 0.50],
            [0.58, 0.50, 0.50],
            [0.50, 0.58, 0.50],
        ],
        atom_groups=[[0, 1, 2]],
    )

    assert solver._repair_motifs_in_set([0, 1]) == [0, 1, 2]


def test_complete_nh4_count_rejects_partial_motif():
    solver = _solver(
        symbols=["N", "H", "H", "H", "H"],
        frac_coords=[
            [0.50, 0.50, 0.50],
            [0.60, 0.50, 0.50],
            [0.50, 0.60, 0.50],
            [0.50, 0.50, 0.60],
            [0.42, 0.42, 0.42],
        ],
        atom_groups=[[0, 1, 2, 3, 4]],
    )

    assert solver._selected_complete_motif_count([0, 1, 2, 3, 4], "N") == 1
    assert solver._selected_complete_motif_count([0, 1, 2, 3], "N") == 0


def test_too_close_contact_detector_uses_minimum_image_distance():
    solver = _solver(
        symbols=["C", "H", "H"],
        frac_coords=[
            [0.01, 0.50, 0.50],
            [0.99, 0.50, 0.50],
            [0.50, 0.50, 0.50],
        ],
        atom_groups=[[0], [1], [2]],
    )

    assert solver._has_too_close_contact([0, 1])
    assert not solver._has_too_close_contact([0, 2])
