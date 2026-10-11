"""Tests for periodic-cell transforms and integrity reports."""

import numpy as np
from ase import Atoms

from molcrys_kit.analysis import check_cell_integrity
from molcrys_kit.structures import MolecularCrystal


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


def test_cell_integrity_accepts_unwrapped_transform_output():
    source = MolecularCrystal(
        np.diag([10.0, 10.0, 10.0]),
        [Atoms("C", positions=[[12.0, 5.0, 5.0]], cell=np.diag([10.0] * 3), pbc=True)],
    )
    transformed = source.transform_cell(matrix=np.eye(3), wrap_mode="none")
    report = check_cell_integrity(transformed, reference=source)

    assert report.passed is True, report.to_dict()
    assert report.checks["molecule_centroids_in_cell"] is True
    assert report.details["centroids_wrapping_required"] is False


def test_cell_integrity_identity_preserves_nan_metadata(simple_crystal):
    """Missing numeric metadata remains equal across an identity transform."""
    for molecule in simple_crystal.molecules:
        molecule.set_array(
            "thermal_u",
            np.array([np.nan, 0.02], dtype=float),
        )
    simple_crystal.extra_arrays["thermal_scale"] = np.array(
        [np.nan, 1.0], dtype=float
    )

    transformed = simple_crystal.transform_cell(
        matrix=np.eye(3),
        position_mode="rigid_molecule",
        wrap_mode="none",
    )
    report = check_cell_integrity(transformed, reference=simple_crystal)

    assert report.passed is True, report.to_dict()
    assert report.checks["per_atom_metadata"] is True


def test_cell_integrity_report_is_json_native():
    """Public reports contain only JSON-native check flags."""
    import json

    crystal = MolecularCrystal(
        np.diag([10.0, 10.0, 10.0]),
        [Atoms("C", positions=[[1.0, 1.0, 1.0]])],
    )
    payload = check_cell_integrity(crystal).to_dict()
    encoded = json.dumps(payload)
    assert '"lattice_shape": true' in encoded


def test_cell_transform_provenance_survives_cif_round_trip(tmp_path):
    """CIF export keeps transform metadata and its stable output hash."""
    from molcrys_kit.analysis import check_cell_integrity
    from molcrys_kit.io import read_mol_crystal, write_cif

    lattice = np.array(
        [[8.123456789, 0.1, 0.0], [0.0, 9.234567891, 0.2], [0.0, 0.0, 10.345678912]]
    )
    source = MolecularCrystal(
        lattice,
        [Atoms("C", positions=[[1.123456789, 2.234567891, 3.345678912]])],
    )
    transformed = source.transform_cell(matrix=np.eye(3), wrap_mode="none")
    output = tmp_path / "transformed.cif"
    write_cif(transformed, str(output))

    restored = read_mol_crystal(str(output))
    report = check_cell_integrity(restored)
    assert report.passed is True, report.to_dict()
    assert restored.metadata["cell_transform"] == transformed.metadata["cell_transform"]
