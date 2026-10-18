"""Calibration schema and uncertainty-band regressions."""

import json
from pathlib import Path

import pytest

from molcrys_kit.chemistry.aromaticity_calibration import (
    CalibrationError,
    CalibrationPolicy,
    ELEMENT_PAIRS,
    calibrate_aromaticity_geometry,
    read_records,
    runtime_overrides,
    write_snapshot,
)


FIXTURE = Path(__file__).parents[2] / "data" / "chemistry" / "aromaticity_calibration_fixture.json"


def _fixture_records():
    return read_records(FIXTURE)


def _policy():
    return CalibrationPolicy(min_pair_observations=2, min_ring_observations=3)


def test_calibration_reports_pair_coverage_and_explicit_metric_bands():
    snapshot = calibrate_aromaticity_geometry(
        _fixture_records(),
        source={"database": "fixture", "sample_size": 5},
        policy=_policy(),
    )

    assert snapshot["status"] == "calibrated"
    quality = snapshot["quality"]
    assert quality["invalid_rows"] == 0
    assert set(quality["pair_coverage"]) == {"-".join(pair) for pair in ELEMENT_PAIRS}
    assert all(count == 2 for count in quality["pair_coverage"].values())

    bands = snapshot["geometry_bands"]
    planarity = bands["planarity"]
    assert planarity["aromatic_max"] < planarity["non_aromatic_min"]
    assert planarity["uncertain"] == [
        planarity["aromatic_max"],
        planarity["non_aromatic_min"],
    ]
    bond_metric = bands["bond_metric"]
    assert bond_metric["aromatic_max"] < bond_metric["saturated_min"]
    assert bond_metric["uncertain"] == [
        bond_metric["aromatic_max"],
        bond_metric["saturated_min"],
    ]
    alternation = bands["alternation_gap"]
    assert alternation["aromatic_max"] < alternation["non_aromatic_min"]

    overrides = runtime_overrides(snapshot)
    assert overrides["AROMATIC_RING_PLANARITY_MAX"] == bands["planarity"]["aromatic_max"]
    assert overrides["AROMATIC_RING_BOND_METRIC_AROMATIC_MAX"] == bond_metric["aromatic_max"]
    assert overrides["AROMATIC_RING_SINGLE_BOND_LENGTHS"][("C", "C")] == 1.505


def test_calibration_snapshot_round_trips_as_deterministic_json(tmp_path):
    snapshot = calibrate_aromaticity_geometry(_fixture_records(), policy=_policy())
    path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == snapshot
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_jsonl_input_is_supported(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = _fixture_records()[:3]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert read_records(path) == rows


def test_missing_pair_refuses_to_emit_a_snapshot():
    records = [
        record
        for record in _fixture_records()
        if record.get("pair") != ["S", "S"]
    ]
    with pytest.raises(CalibrationError, match="S-S"):
        calibrate_aromaticity_geometry(records, policy=_policy())


def test_overlapping_ring_metric_tails_refuse_calibration():
    records = _fixture_records()
    for record in records:
        if record.get("ring_class") == "non_aromatic":
            record["bond_metric"] = 0.92
    with pytest.raises(CalibrationError, match="bond metrics overlap"):
        calibrate_aromaticity_geometry(records, policy=_policy())
