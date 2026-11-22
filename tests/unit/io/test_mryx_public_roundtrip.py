"""Independent validation against traceable public Mercury .mryx exports."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read

from molcrys_kit.constants.mryx import MRYX_COORDINATE_TOLERANCE_A
from molcrys_kit.io import read_extxyz, read_mryx, read_mryx_atoms, write_extxyz

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "mryx" / "public_manifest.json"


def _public_fixtures():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["fixtures"]


def _fetch_public_fixture(entry: dict, dest: Path) -> Path:
    request = urllib.request.Request(
        entry["url"],
        headers={"User-Agent": "MolCrysKit-pytest-mryx/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise AssertionError(f"sha256 mismatch for {entry['id']}")
    return dest


@pytest.mark.parametrize("entry", _public_fixtures(), ids=lambda item: item["id"])
def test_public_mryx_tier_a_roundtrip(tmp_path, entry):
    """Real Mercury export → MolCrysKit → ExtXYZ → ASE, with manifest pinning."""
    path = tmp_path / entry["filename"].replace(",", "_")
    try:
        _fetch_public_fixture(entry, path)
    except urllib.error.URLError as exc:
        pytest.skip(f"public fixture download unavailable: {exc}")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        atoms = read_mryx_atoms(path)
        crystal = read_mryx(path)

    expectations = entry["tier_a_expectations"]
    assert len(atoms) == expectations["n_atoms"]
    np.testing.assert_allclose(
        atoms.cell.cellpar(),
        expectations["cellpar_deg"],
        atol=expectations.get("cellpar_atol", 0.01),
    )

    out = tmp_path / "roundtrip.extxyz"
    write_extxyz(crystal, str(out))
    ase_atoms = ase_read(str(out))
    pos_atol = expectations.get("position_atol_A", MRYX_COORDINATE_TOLERANCE_A)
    np.testing.assert_allclose(
        atoms.get_positions(),
        ase_atoms.get_positions(),
        atol=pos_atol,
    )
    assert atoms.get_chemical_symbols() == ase_atoms.get_chemical_symbols()

    rt = read_extxyz(str(out)).to_ase()
    np.testing.assert_allclose(atoms.cell.array, rt.cell.array, atol=1e-6)