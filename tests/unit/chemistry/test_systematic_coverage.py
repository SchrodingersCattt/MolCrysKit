from __future__ import annotations

from pathlib import Path

import pytest

from molcrys_kit import iupac_to_smiles, smiles_to_iupac
from molcrys_kit.chemistry import NamingKind, notations_equivalent


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
    rebuilt = iupac_to_smiles(result.name)
    assert rebuilt.lossless is True
    assert notations_equivalent(smiles, rebuilt.value) is True


def test_paclitaxel_snapshot_is_general_and_ring_marked() -> None:
    smiles = _COVERAGE[-1]
    result = smiles_to_iupac(smiles, strict=True)
    assert result.kind is NamingKind.GENERAL_IUPAC_NAME
    assert result.preferred is False
    assert "paclitaxel" not in result.name.lower()
    assert "bicyclo" in result.name
