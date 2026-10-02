"""Tests for periodic-cell transforms and integrity reports."""

import numpy as np

from molcrys_kit.analysis import check_cell_integrity


def test_cell_integrity_report_accepts_rigid_transform(simple_crystal):
    transformed = simple_crystal.transform_cell(
        new_lattice=np.diag([12.0, 10.0, 10.0]),
        position_mode="rigid_molecule",
        wrap_mode="centroid",
    )
    report = check_cell_integrity(transformed, reference=simple_crystal)
    assert report.passed is True, report.to_dict()
    assert report.checks["molecular_inventory"] is True
    assert report.checks["topology"] is True
    assert report.checks["per_atom_metadata"] is True
    assert report.details["output_hash"] == transformed.metadata["cell_transform"]["output_hash"]


def test_cell_integrity_rejects_left_handed_lattice(simple_crystal):
    simple_crystal.lattice = np.diag([-10.0, 10.0, 10.0])
    report = check_cell_integrity(simple_crystal)
    assert report.passed is False
    assert report.checks["lattice_right_handed"] is False
