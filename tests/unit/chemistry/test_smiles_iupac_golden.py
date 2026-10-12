"""Acceptance tests for the fixed SMILES-to-IUPAC golden cases.

The expected names are frozen reference strings.  The timing assertion starts
immediately before each strict conversion and gives every individual case a
one-second budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from molcrys_kit import smiles_to_iupac


_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "chemistry_golden"
    / "smiles_iupac_golden_10.json"
)
_CASES: list[dict[str, Any]] = json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_smiles_to_reference_iupac(case: dict[str, Any]) -> None:
    start = perf_counter()
    result = None
    error: Exception | None = None
    try:
        result = smiles_to_iupac(case["smiles"], strict=True)
    except Exception as exc:  # Keep the independent timing gate active.
        error = exc
    elapsed = perf_counter() - start

    assert elapsed <= case["max_seconds"], (
        f"{case['id']}: {elapsed:.6f}s exceeds {case['max_seconds']:.3f}s"
    )
    assert error is None, f"{case['id']}: strict conversion raised {error!r}"
    assert result is not None
    assert result.name == case["expected_iupac"]
