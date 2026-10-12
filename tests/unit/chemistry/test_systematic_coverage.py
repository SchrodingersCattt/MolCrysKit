from __future__ import annotations

from pathlib import Path

import pytest

from molcrys_kit import iupac_to_smiles, smiles_to_iupac
from molcrys_kit.chemistry import notations_equivalent


_COVERAGE = tuple(
    line.strip()
    for line in (
        Path(__file__).resolve().parents[2]
        / "data"
        / "chemistry_golden"
        / "coverage_smiles.txt"
    ).read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)


@pytest.mark.parametrize("smiles", _COVERAGE)
def test_coverage_smiles_have_strict_reversible_general_names(smiles: str) -> None:
    result = smiles_to_iupac(smiles, strict=True)
    assert not result.name.startswith("molecular entity ")
    # A von Baeyer marker must describe a named ring system.  The former
    # generic fallback used ``bicyclo[generic]-molecule-<hex MCK-LN>``; that
    # payload is an internal serialization, not a systematic name.
    assert not result.name.startswith("bicyclo[generic]-molecule-")
    assert not result.name.startswith("substituted-molecule-")
    assert "MCK-LN" not in result.name
    rebuilt = iupac_to_smiles(result.name)
    assert rebuilt.lossless is True
    assert notations_equivalent(smiles, rebuilt.value) is True
