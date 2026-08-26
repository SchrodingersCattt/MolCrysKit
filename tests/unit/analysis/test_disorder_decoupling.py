from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

import numpy as np

from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.analysis.disorder.solver import DisorderSolver


CIF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")
)


def _structure_signature(crystal):
    atoms = crystal.to_ase()
    symbols = atoms.get_chemical_symbols()
    coords = atoms.get_scaled_positions()
    return tuple(
        sorted(
            (sym, tuple(round(float(x), 4) for x in coord))
            for sym, coord in zip(symbols, coords)
        )
    )


def _element_totals(crystal):
    totals = Counter()
    for molecule in crystal.molecules:
        totals.update(molecule.get_chemical_symbols())
    return dict(totals)


def _formula_counts(crystal):
    formulas = Counter()
    for molecule in crystal.molecules:
        counts = Counter(molecule.get_chemical_symbols())
        formulas["".join(f"{el}{counts[el]}" for el in sorted(counts))] += 1
    return formulas


@lru_cache(maxsize=None)
def _solve_fixture(cif_name, method, generate_count, coupled):
    return generate_ordered_replicas_from_disordered_sites(
        os.path.join(CIF_DATA_DIR, cif_name),
        method=method,
        generate_count=generate_count,
        coupled=coupled,
    )


def test_1htp_explicit_decoupled_enumerate_count_geq_4():
    crystals = _solve_fixture("1-HTP.cif", "enumerate", 8, False)

    assert len({_structure_signature(crystal) for crystal in crystals}) >= 4


def test_1htp_explicit_coupled_enumerate_count_eq_2():
    crystals = _solve_fixture("1-HTP.cif", "enumerate", 8, True)

    assert len({_structure_signature(crystal) for crystal in crystals}) == 2


def test_1htp_explicit_decoupled_element_totals_invariant():
    crystals = _solve_fixture("1-HTP.cif", "enumerate", 8, False)
    totals = [_element_totals(crystal) for crystal in crystals]

    assert len(totals) >= 4
    assert all(total == totals[0] for total in totals)


def test_dap4_implicit_decoupled_enumerate_multiple_orientations():
    crystals = _solve_fixture("DAP-4.cif", "enumerate", 4, False)

    assert len({_structure_signature(crystal) for crystal in crystals}) >= 2
    assert [_formula_counts(crystal).get("H4N1", 0) for crystal in crystals] == [
        8,
        8,
        8,
        8,
    ]


def test_dap4_implicit_coupled_legacy_equivalent():
    crystals = _solve_fixture("DAP-4.cif", "enumerate", 4, True)

    assert len({_structure_signature(crystal) for crystal in crystals}) == 1
    assert [_formula_counts(crystal).get("H4N1", 0) for crystal in crystals] == [
        8,
        8,
        8,
        8,
    ]


def test_explicit_no_sym_info_unchanged():
    info = DisorderInfo(
        labels=["C1", "O1", "N1", "F1"],
        symbols=["C", "O", "N", "F"],
        frac_coords=np.array([
            [0.10, 0.10, 0.10],
            [0.35, 0.10, 0.10],
            [0.10, 0.35, 0.10],
            [0.35, 0.35, 0.10],
        ]),
        occupancies=[0.5, 0.5, 0.5, 0.5],
        disorder_groups=[1, 2, 1, 2],
        assemblies=["A", "A", "A", "A"],
        sym_op_indices=None,
    )
    lattice = np.eye(3) * 20.0
    graph = DisorderGraphBuilder(info, lattice, coupled=False).build()
    solver = DisorderSolver(info, graph, lattice, coupled=False)
    solver.atom_groups = []
    solver._identify_atom_groups()
    group_graph = solver._build_group_conflict_graph()

    assert len(solver.atom_groups) == 2
    assert group_graph.number_of_edges() == 1
