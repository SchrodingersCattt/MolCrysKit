"""Tests for graph-based PBC unwrapping helpers."""

import networkx as nx
import numpy as np
from ase import Atoms

from molcrys_kit.io.cif import identify_molecules
from molcrys_kit.utils.geometry import unwrap_positions_along_bonds


def test_unwrap_positions_along_bonds_crosses_boundary():
    graph = nx.Graph()
    graph.add_edge(0, 1, vector=np.array([0.4, 0.0, 0.0]))
    positions = np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]])

    unwrapped, completed = unwrap_positions_along_bonds(graph, [0, 1], positions)

    assert completed is True
    np.testing.assert_allclose(unwrapped, [[9.8, 0.0, 0.0], [10.2, 0.0, 0.0]])


def test_unwrap_positions_along_bonds_respects_max_atoms():
    graph = nx.Graph()
    graph.add_edge(0, 1, vector=np.array([1.0, 0.0, 0.0]))
    graph.add_edge(1, 2, vector=np.array([1.0, 0.0, 0.0]))
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    unwrapped, completed = unwrap_positions_along_bonds(
        graph,
        [0, 1, 2],
        positions,
        max_atoms=2,
    )

    assert completed is False
    np.testing.assert_allclose(unwrapped, positions)


def test_identify_molecules_uses_unwrapped_positions():
    atoms = Atoms(
        symbols=["H", "H"],
        positions=[[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    molecules = identify_molecules(atoms)

    assert len(molecules) == 1
    np.testing.assert_allclose(molecules[0].get_positions()[1], [10.2, 0.0, 0.0])
    assert molecules[0].info["unwrap_completed"] is True


def test_identify_molecules_marks_cap_overflow():
    atoms = Atoms(
        symbols=["H", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    molecules = identify_molecules(atoms, max_atoms=2)

    assert len(molecules) == 1
    assert molecules[0].info["unwrap_completed"] is False
