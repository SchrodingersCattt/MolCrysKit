"""Regression for GitHub issue #226: ordered-replica occupancy metadata."""

from __future__ import annotations

import os

import pytest

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.analysis.structure_summary import summarize_structure
from molcrys_kit.constants.config import PARTIAL_OCCUPANCY_TOLERANCE
from molcrys_kit.io.cif import read_mol_crystal

CIF_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")
)

LONE_PARTIAL_CIF = os.path.join(CIF_DIR, "issue226_lone_partial_na.cif")
EXPLICIT_ALT_CIF = os.path.join(CIF_DIR, "issue226_explicit_alternatives.cif")


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


def test_explicit_disorder_alternatives_resolve_to_full_occupancy():
    resolved = read_mol_crystal(EXPLICIT_ALT_CIF, resolve_disorder=True)
    sites = resolved.get_site_records()
    assert len(sites) == 1
    site = sites[0]
    assert site.occupancy == pytest.approx(1.0)
    assert site.disorder_group in {1, 2}
    assert site.occupancy > 1.0 - PARTIAL_OCCUPANCY_TOLERANCE


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