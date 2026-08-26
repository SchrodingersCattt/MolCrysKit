from __future__ import annotations

import numpy as np
from ase import Atoms

from molcrys_kit.structures import (
    VerletBondTracker,
    build_bond_candidates,
    candidate_list_needs_rebuild,
    infer_bond_pairs,
)
from molcrys_kit.analysis.interactions import get_bonding_threshold
from molcrys_kit.constants import (
    get_atomic_radius,
    is_metal_element,
)
from molcrys_kit.structures import CrystalMolecule


def _threshold(first: str, second: str) -> float:
    return get_bonding_threshold(
        get_atomic_radius(first),
        get_atomic_radius(second),
        is_metal_element(first),
        is_metal_element(second),
    )


def test_bond_pairs_strictly_match_existing_molecule_graph() -> None:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    expected = {
        tuple(sorted((int(first), int(second))))
        for first, second in CrystalMolecule(
            atoms,
            check_pbc=False,
        ).graph.edges
    }
    batch = infer_bond_pairs(atoms.positions, atoms.numbers)
    assert {tuple(pair) for pair in batch.pairs.tolist()} == expected
    assert batch.pairs.dtype == np.int32
    assert batch.vectors.dtype == np.float32
    assert batch.distances.dtype == np.float32
    assert batch.pairs.flags.c_contiguous
    assert not batch.pairs.flags.writeable


def test_verlet_reuses_candidates_for_bond_formation_then_rebuilds() -> None:
    limit = _threshold("C", "H")
    numbers = np.asarray([6, 1], dtype=np.uint8)
    positions = np.asarray([[0.0, 0.0, 0.0], [limit + 0.20, 0.0, 0.0]])
    tracker = VerletBondTracker(skin=0.5)

    assert len(tracker.update(positions, numbers).pairs) == 0
    assert tracker.rebuild_count == 1

    positions[1, 0] = limit - 0.03
    bonded = tracker.update(positions, numbers)
    assert bonded.pairs.tolist() == [[0, 1]]
    assert tracker.rebuild_count == 1

    positions[1, 0] = limit + 1.0
    assert len(tracker.update(positions, numbers).pairs) == 0
    assert tracker.rebuild_count == 2


def test_cell_change_and_skin_displacement_trigger_rebuild() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    numbers = np.asarray([6, 6])
    cell = np.diag([5.0, 5.0, 5.0])
    candidates = build_bond_candidates(
        positions,
        numbers,
        cell=cell,
        pbc=(True, True, True),
        skin=0.4,
    )
    small_move = positions.copy()
    small_move[1, 0] += 0.19
    assert not candidate_list_needs_rebuild(
        candidates,
        small_move,
        cell=cell,
        pbc=(True, True, True),
    )
    large_move = positions.copy()
    large_move[1, 0] += 0.21
    assert candidate_list_needs_rebuild(
        candidates,
        large_move,
        cell=cell,
        pbc=(True, True, True),
    )
    changed_cell = cell.copy()
    changed_cell[0, 0] += 0.01
    assert candidate_list_needs_rebuild(
        candidates,
        positions,
        cell=changed_cell,
        pbc=(True, True, True),
    )


def test_mixed_periodic_minimum_image_vector() -> None:
    positions = np.asarray([[0.1, 1.0, 1.0], [9.9, 1.0, 1.0]])
    batch = infer_bond_pairs(
        positions,
        np.asarray([6, 1]),
        cell=np.diag([10.0, 4.0, 4.0]),
        pbc=(True, False, False),
    )
    assert batch.pairs.tolist() == [[0, 1]]
    np.testing.assert_allclose(batch.vectors[0], [-0.2, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(batch.distances[0], 0.2, atol=1.0e-6)


def test_triclinic_periodic_candidate_fallback_is_correct() -> None:
    cell = np.asarray([[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    fractional = np.asarray([[0.05, 0.5, 0.5], [0.95, 0.5, 0.5]])
    positions = fractional @ cell
    batch = infer_bond_pairs(
        positions,
        np.asarray([6, 6]),
        cell=cell,
        pbc=(True, True, True),
    )
    assert batch.pairs.tolist() == [[0, 1]]
    np.testing.assert_allclose(batch.distances[0], 0.3, atol=1.0e-6)
