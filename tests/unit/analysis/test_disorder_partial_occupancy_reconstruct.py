"""Regression for GitHub issue #226: ordered-replica occupancy metadata."""

from __future__ import annotations

import os

import pytest

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.analysis.structure_summary import summarize_structure
from molcrys_kit.constants.config import PARTIAL_OCCUPANCY_TOLERANCE
from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.operations.disorder_input import disorder_input_stats

CIF_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")
)

LONE_PARTIAL_CIF = os.path.join(CIF_DIR, "issue226_lone_partial_na.cif")
LONE_PARTIAL_NONZERO_GROUP_CIF = os.path.join(
    CIF_DIR, "issue226_lone_partial_nonzero_group.cif"
)
EXPLICIT_ALT_CIF = os.path.join(CIF_DIR, "issue226_explicit_alternatives.cif")
GROUP0_GEOM_OVERLAP_CIF = os.path.join(
    CIF_DIR, "issue226_group0_geometric_overlap.cif"
)
BONDED_PARTIAL_CIF = os.path.join(CIF_DIR, "issue226_bonded_partial_water.cif")
INDEPENDENT_ASM_CIF = os.path.join(CIF_DIR, "issue226_independent_assemblies.cif")
GROUP0_VS_PART_CIF = os.path.join(CIF_DIR, "issue226_group0_vs_dropped_part.cif")
DAP4_CIF = os.path.join(CIF_DIR, "DAP-4.cif")


def test_lone_group_zero_partial_occupancy_survives_resolve_disorder():
    """Isolated partial site (group 0) must not be promoted to occupancy 1."""
    refined = read_mol_crystal(LONE_PARTIAL_CIF, resolve_disorder=False)
    resolved = read_mol_crystal(LONE_PARTIAL_CIF, resolve_disorder=True)

    refined_sites = {(r.symbol, round(r.occupancy, 5)) for r in refined.get_site_records()}
    assert ("Na", 0.5) in refined_sites

    resolved_sites = resolved.get_site_records()
    assert len(resolved_sites) == 2
    na_site = next(r for r in resolved_sites if r.symbol == "Na")
    cl_site = next(r for r in resolved_sites if r.symbol == "Cl")
    assert na_site.occupancy == pytest.approx(0.5)
    assert na_site.disorder_group == 0
    assert cl_site.occupancy == pytest.approx(1.0)

    disorder = summarize_structure(resolved)["disorder"]
    assert disorder["partial_occupancy_sites"] == 1
    assert disorder["groups"] == []


def test_lone_nonzero_group_partial_without_competitor_keeps_refined():
    """Lone PART-tagged partial (group != 0) is not auto-filled to 1.0."""
    resolved = read_mol_crystal(LONE_PARTIAL_NONZERO_GROUP_CIF, resolve_disorder=True)
    sites = resolved.get_site_records()
    assert len(sites) == 2
    na_site = next(r for r in sites if r.symbol == "Na")
    assert na_site.disorder_group == 1
    assert na_site.occupancy == pytest.approx(0.5)


def test_group_zero_geometric_overlap_keeps_winner_refined_occupancy():
    """Geometric de-overlap alone does not promote occupancy to 1.0."""
    resolved = read_mol_crystal(GROUP0_GEOM_OVERLAP_CIF, resolve_disorder=True)
    sites = resolved.get_site_records()
    assert len(sites) == 1
    site = sites[0]
    assert site.disorder_group == 0
    assert site.occupancy == pytest.approx(0.65)


def test_explicit_disorder_alternatives_resolve_to_full_occupancy():
    resolved = read_mol_crystal(EXPLICIT_ALT_CIF, resolve_disorder=True)
    sites = resolved.get_site_records()
    assert len(sites) == 1
    site = sites[0]
    assert site.occupancy == pytest.approx(1.0)
    assert site.disorder_group in {1, 2}
    assert site.occupancy > 1.0 - PARTIAL_OCCUPANCY_TOLERANCE


def test_bonded_partial_molecule_attachments_preserve_refined_occupancy():
    resolved = read_mol_crystal(BONDED_PARTIAL_CIF, resolve_disorder=True)
    o_site = next(r for r in resolved.get_site_records() if r.symbol == "O")
    assert o_site.occupancy == pytest.approx(0.5)
    hydrogens = [r for r in resolved.get_site_records() if r.symbol == "H"]
    occs = sorted(r.occupancy for r in hydrogens)
    assert occs == [pytest.approx(0.25), pytest.approx(0.5)]
    assert all(r.occupancy < 1.0 - PARTIAL_OCCUPANCY_TOLERANCE for r in hydrogens)


def test_independent_assemblies_explicit_winners_are_full_occupancy():
    for independent in (False, True):
        resolved = read_mol_crystal(
            INDEPENDENT_ASM_CIF,
            resolve_disorder=True,
            independent_assemblies=independent,
        )
        for site in resolved.get_site_records():
            assert site.occupancy == pytest.approx(1.0)
            assert site.disorder_group in {1, 2}


def test_group_zero_partial_wins_over_dropped_part_without_occ_promotion():
    """Group-0 partial vs dropped non-zero PART: geometric clash only."""
    resolved = read_mol_crystal(GROUP0_VS_PART_CIF, resolve_disorder=True)
    sites = resolved.get_site_records()
    assert len(sites) == 1
    assert sites[0].symbol == "O"
    assert sites[0].occupancy == pytest.approx(0.6)
    assert sites[0].disorder_group == 0


def test_resolved_replica_with_part_labels_is_still_ordered():
    path = os.path.join(CIF_DIR, "1-HTP.cif")
    resolved = read_mol_crystal(path, resolve_disorder=True)
    stats = disorder_input_stats(resolved)
    assert stats["ordered_replica"] is True
    assert stats["all_atom_ordered"] is True
    assert stats["retained_disorder_group_label_count"] > 0


def test_dap4_resolved_replica_ordered_with_partial_sites():
    unresolved = read_mol_crystal(DAP4_CIF, resolve_disorder=False)
    resolved = read_mol_crystal(DAP4_CIF, resolve_disorder=True)
    assert disorder_input_stats(unresolved)["all_atom_ordered"] is False
    stats = disorder_input_stats(resolved)
    assert stats["ordered_replica"] is True
    assert stats["all_atom_ordered"] is True
    assert stats["nonunit_occupancy_count"] == 32
    info = scan_cif_disorder(DAP4_CIF)
    graph = DisorderGraphBuilder(info, info.lattice_matrix).build()
    kept = resolved.disorder_provenance.kept_indices
    dropped = set(resolved.disorder_provenance.dropped_indices)
    partial_kept = [i for i in kept if info.occupancies[i] < 0.999]
    assert len(partial_kept) == 32
    for index in partial_kept:
        for other in dropped:
            if graph.has_edge(index, other):
                ct = graph[index][other].get("conflict_type", "")
                assert ct not in {
                    "explicit",
                    "logical_alternative",
                    "implicit_sp",
                }


def test_crystal_carrier_path_preserves_partial_occupancy():
    """Carrier/core path: resolve from in-memory MolecularCrystal, not only CIF."""
    carrier = read_mol_crystal(LONE_PARTIAL_CIF, resolve_disorder=False)
    replicas = generate_ordered_replicas_from_disordered_sites(
        crystal=carrier,
        method="optimal",
        generate_count=1,
    )
    resolved = replicas[0]
    na_site = next(r for r in resolved.get_site_records() if r.symbol == "Na")
    assert na_site.occupancy == pytest.approx(0.5)
    assert len(resolved.get_site_records()) == 2