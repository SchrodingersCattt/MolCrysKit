#!/usr/bin/env python3
"""
Test graph-based molecule identification logic.
"""

import os
import sys

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ase import Atoms

from molcrys_kit.io.cif import identify_molecules


def test_simple_molecular_system():
    """Test identify_molecules with a simple system of bonded and isolated atoms."""
    # Create a simple system with:
    # 1. Two bonded atoms (H and O, forming a fragment of a molecule)
    # 2. One isolated atom (Cl)
    atoms = Atoms(
        symbols=["H", "O", "Cl"],
        positions=[
            [0.0, 0.0, 0.0],  # H
            [0.9, 0.0, 0.0],  # O (bonded to H, ~0.9 Å distance)
            [5.0, 5.0, 5.0],  # Cl (isolated)
        ],
    )

    # Identify molecules
    molecules = identify_molecules(atoms)

    # Should identify 2 molecules:
    # 1. H-O molecule (2 atoms)
    # 2. Cl atom (1 atom)
    assert len(molecules) == 2, f"Expected 2 molecules, got {len(molecules)}"

    # Sort by number of atoms to ensure consistent ordering
    molecules_sorted = sorted(molecules, key=len)

    # First molecule should be the Cl atom (1 atom)
    assert (
        len(molecules_sorted[0]) == 1
    ), f"Expected first molecule to have 1 atom, got {len(molecules_sorted[0])}"
    assert molecules_sorted[0].get_chemical_symbols()[0] == "Cl"

    # Second molecule should be the H-O molecule (2 atoms)
    assert (
        len(molecules_sorted[1]) == 2
    ), f"Expected second molecule to have 2 atoms, got {len(molecules_sorted[1])}"
    symbols = sorted(molecules_sorted[1].get_chemical_symbols())
    assert symbols == ["H", "O"], f"Expected H and O atoms, got {symbols}"

    # Check that the graph is correctly built for the H-O molecule
    graph = molecules_sorted[1].graph
    assert (
        len(graph.nodes()) == 2
    ), f"Expected 2 nodes in graph, got {len(graph.nodes())}"
    assert (
        len(graph.edges()) == 1
    ), f"Expected 1 edge in graph, got {len(graph.edges())}"

    # Check that the isolated Cl atom has no bonds
    graph_cl = molecules_sorted[0].graph
    assert (
        len(graph_cl.nodes()) == 1
    ), f"Expected 1 node in Cl graph, got {len(graph_cl.nodes())}"
    assert (
        len(graph_cl.edges()) == 0
    ), f"Expected 0 edges in Cl graph, got {len(graph_cl.edges())}"


def test_more_complex_system():
    """Test identify_molecules with a more complex system."""
    # Create a system with:
    # 1. Water molecule (O, H, H)
    # 2. Ammonia molecule (N, H, H, H)
    # 3. Isolated atom (Ne)
    atoms = Atoms(
        symbols=["O", "H", "H", "N", "H", "H", "H", "Ne"],
        positions=[
            # Water molecule
            [0.0, 0.0, 0.0],  # O
            [0.95, 0.0, 0.0],  # H1
            [0.0, 0.95, 0.0],  # H2
            # Ammonia molecule
            [5.0, 5.0, 5.0],  # N
            [5.95, 5.0, 5.0],  # H1
            [5.0, 5.95, 5.0],  # H2
            [5.0, 5.0, 5.95],  # H3
            # Isolated atom
            [10.0, 10.0, 10.0],  # Ne
        ],
    )

    # Identify molecules
    molecules = identify_molecules(atoms)

    # Should identify 3 molecules:
    # 1. Water (3 atoms)
    # 2. Ammonia (4 atoms)
    # 3. Neon atom (1 atom)
    assert len(molecules) == 3, f"Expected 3 molecules, got {len(molecules)}"

    # Sort by number of atoms
    molecules_sorted = sorted(molecules, key=len)

    # Check the isolated Ne atom
    assert len(molecules_sorted[0]) == 1
    assert molecules_sorted[0].get_chemical_symbols()[0] == "Ne"

    # Check for water and ammonia molecules
    molecule_sizes = [len(mol) for mol in molecules_sorted]
    assert molecule_sizes == [
        1,
        3,
        4,
    ], f"Expected molecule sizes [1, 3, 4], got {molecule_sizes}"

    # Check that graphs are correctly built
    # Water molecule (3 atoms)
    water = molecules_sorted[1]
    water_graph = water.graph
    assert len(water_graph.nodes()) == 3
    # Should have 2 bonds (O-H, O-H)
    assert len(water_graph.edges()) == 2

    # Ammonia molecule (4 atoms)
    ammonia = molecules_sorted[2]
    ammonia_graph = ammonia.graph
    assert len(ammonia_graph.nodes()) == 4
    # Should have 3 bonds (N-H, N-H, N-H)
    assert len(ammonia_graph.edges()) == 3


def run_tests():
    """Run all tests."""
    tests = [test_simple_molecular_system, test_more_complex_system]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\nTests passed: {passed}")
    print(f"Tests failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
