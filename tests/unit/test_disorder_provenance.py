from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

from ase import Atoms

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.structures.crystal import MolecularCrystal


CIF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cif")
)


@lru_cache(maxsize=None)
def _fixture_path(cif_name: str) -> str:
    return os.path.join(CIF_DATA_DIR, cif_name)


@lru_cache(maxsize=None)
def _solve_fixture(cif_name: str, method: str, generate_count: int, coupled: bool):
    return generate_ordered_replicas_from_disordered_sites(
        _fixture_path(cif_name),
        method=method,
        generate_count=generate_count,
        coupled=coupled,
    )


def _element_totals(crystal):
    totals = Counter()
    for molecule in crystal.molecules:
        totals.update(molecule.get_chemical_symbols())
    return dict(totals)


def test_provenance_attribute_present_on_solved_crystal():
    crystal = _solve_fixture("1-HTP.cif", "optimal", 1, False)[0]

    assert crystal.disorder_provenance is not None
    assert crystal.disorder_provenance.kept_indices


def test_provenance_kept_dropped_complement_full_source():
    crystal = _solve_fixture("1-HTP.cif", "optimal", 1, False)[0]
    provenance = crystal.disorder_provenance
    source_count = len(scan_cif_disorder(_fixture_path("1-HTP.cif")).labels)

    assert set(provenance.kept_indices).isdisjoint(provenance.dropped_indices)
    assert set(provenance.kept_indices) | set(provenance.dropped_indices) == set(
        range(source_count)
    )


def test_provenance_records_method_and_coupled():
    for method, coupled in (("optimal", False), ("optimal", True), ("enumerate", False), ("enumerate", True)):
        crystal = _solve_fixture("1-HTP.cif", method, 2, coupled)[0]
        provenance = crystal.disorder_provenance

        assert provenance.method == method
        assert provenance.coupled is coupled


def test_provenance_supercell_drops_provenance():
    crystal = _solve_fixture("1-HTP.cif", "optimal", 1, False)[0]

    supercell = crystal.get_supercell(2, 1, 1)

    assert supercell.disorder_provenance is None


def test_provenance_from_ase_has_no_provenance():
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    crystal = MolecularCrystal.from_ase(atoms)

    assert crystal.disorder_provenance is None


def test_return_kept_indices_tuple_still_matches_crystal_provenance():
    results = generate_ordered_replicas_from_disordered_sites(
        _fixture_path("1-HTP.cif"),
        method="optimal",
        return_kept_indices=True,
        coupled=False,
    )
    crystal, kept_indices = results[0]

    assert kept_indices == crystal.disorder_provenance.kept_indices


def test_decoupled_replicas_share_source_atoms_for_neb():
    crystals = _solve_fixture("1-HTP.cif", "enumerate", 8, False)
    kept_sets = [set(crystal.disorder_provenance.kept_indices) for crystal in crystals]
    unique_sets = []
    for kept in kept_sets:
        if kept not in unique_sets:
            unique_sets.append(kept)

    assert len(unique_sets) >= 2
    start, end = unique_sets[0], unique_sets[1]
    assert start & end
    assert start ^ end
    assert _element_totals(crystals[0]) == _element_totals(crystals[1])


def test_provenance_atom_indices_consistent_with_kept_indices():
    crystal = _solve_fixture("1-HTP.cif", "optimal", 1, False)[0]
    local_indices = [
        int(index)
        for molecule in crystal.molecules
        for index in molecule.info["atom_indices"]
    ]
    atom_count = len(crystal.to_ase())

    assert sorted(local_indices) == list(range(atom_count))
    assert len(crystal.disorder_provenance.kept_indices) == atom_count
