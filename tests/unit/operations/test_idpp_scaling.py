"""Smoke test for the IDPP scaling probe (shape and loose wall time only).

This is not a performance regression gate: see ``benchmarks/idpp_scaling.py``.
"""

from benchmarks.idpp_scaling import time_idpp_grid


def test_idpp_scaling_probe_reports_atom_and_frame_counts():
    rows = time_idpp_grid(atom_counts=(4,), frame_counts=(3,))
    assert len(rows) == 1
    row = rows[0]
    assert row["atom_count"] == 4
    assert row["frame_count"] == 3
    assert row["pair_count"] == 6
    assert row["interior_frames"] == 1
    assert row["seconds"] < 30.0
