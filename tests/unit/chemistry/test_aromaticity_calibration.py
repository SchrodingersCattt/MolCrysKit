"""Calibration schema and uncertainty-band regressions."""

import json
from pathlib import Path
import os
import subprocess
import sys

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
    assert quality["aromatic_rings"] == 5
    assert quality["saturated_rings"] == 5
    assert quality["conjugated_non_aromatic_rings"] == 5
    assert set(quality["pair_coverage"]) == {"-".join(pair) for pair in ELEMENT_PAIRS}
    assert all(count == 2 for count in quality["pair_coverage"].values())

    bands = snapshot["geometry_bands"]
    planarity = bands["planarity"]
    assert planarity["aromatic_max"] < planarity["saturated_min"]
    assert planarity["conjugated_non_aromatic"]["q05"] < planarity["saturated_min"]
    assert planarity["uncertain"] == [
        planarity["aromatic_max"],
        planarity["saturated_min"],
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
    assert snapshot["references_A"]["C-C"] == 1.505


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
        if record.get("ring_class") == "saturated":
            record["bond_metric"] = 0.92
    with pytest.raises(CalibrationError, match="bond metrics overlap"):
        calibrate_aromaticity_geometry(records, policy=_policy())


def test_conjugated_planarity_is_reported_without_setting_planarity_band():
    records = _fixture_records()
    for record in records:
        if record.get("ring_class") == "conjugated_non_aromatic":
            record["planarity"] = 0.0
    snapshot = calibrate_aromaticity_geometry(records, policy=_policy())
    planarity = snapshot["geometry_bands"]["planarity"]
    assert planarity["saturated_min"] > planarity["aromatic_max"]
    assert planarity["conjugated_non_aromatic"]["q05"] == 0.0


def test_metricless_and_unknown_rows_are_counted_and_excluded():
    records = _fixture_records()
    records[20].pop("bond_metric")
    records.append({"ring_class": "typo", "planarity": 0.2})
    snapshot = calibrate_aromaticity_geometry(records, policy=_policy())
    assert snapshot["quality"]["invalid_rows"] == 2
    assert snapshot["quality"]["aromatic_rings"] == 4


def test_insufficient_valid_ring_metrics_are_refused():
    records = _fixture_records()
    aromatic = [r for r in records if r.get("ring_class") == "aromatic"]
    for record in aromatic[:3]:
        record.pop("alternation_gap")
    with pytest.raises(CalibrationError, match=r"insufficient ring metrics \(n < 3\)"):
        calibrate_aromaticity_geometry(records, policy=_policy())


def test_runtime_overrides_rejects_reference_mismatch_and_missing_pair():
    snapshot = calibrate_aromaticity_geometry(_fixture_records(), policy=_policy())
    mismatch = json.loads(json.dumps(snapshot))
    mismatch["references_A"]["C-C"] = 1.6
    with pytest.raises(CalibrationError, match="references_A"):
        runtime_overrides(mismatch)
    missing = json.loads(json.dumps(snapshot))
    del missing["element_pair_references_A"]["S-S"]
    with pytest.raises(CalibrationError, match="missing element-pair references"):
        runtime_overrides(missing)


def test_reference_median_comparison_uses_angstrom_tolerance():
    records = _fixture_records()
    for record in records:
        references = record.get("references_A")
        if isinstance(references, dict):
            references["C-C"] += 5e-5
    snapshot = calibrate_aromaticity_geometry(records, policy=_policy())
    assert snapshot["references_A"]["C-C"] == 1.505

    outside_tolerance = _fixture_records()
    for record in outside_tolerance:
        references = record.get("references_A")
        if isinstance(references, dict):
            references["C-C"] += 2e-4
    with pytest.raises(CalibrationError, match="references_A"):
        calibrate_aromaticity_geometry(outside_tolerance, policy=_policy())


def test_cli_smoke_uses_policy_instance_defaults(tmp_path):
    fixture_rows = _fixture_records()
    single_rows = [row for row in fixture_rows if row.get("bond_class") == "single"]
    ring_rows = [row for row in fixture_rows if row.get("ring_class")]
    rows = single_rows * 10 + ring_rows * 5
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "snapshot.json"
    input_path.write_text(json.dumps(rows), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[3]))
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_aromaticity.py", str(input_path), str(output_path)],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "calibrated"
