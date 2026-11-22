"""Networked tier-(A) check against pinned public Mercury exports.

Excluded from the default ``pytest tests/unit`` job. CI runs this file in the
``mryx-public-validation`` workflow job, which fails closed on download errors.
"""

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

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "mryx" / "public_manifest.json"


def _fixtures():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"]


def _download(entry: dict, dest: Path) -> None:
    request = urllib.request.Request(
        entry["url"],
        headers={"User-Agent": "MolCrysKit-integration-mryx/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise AssertionError(f"sha256 mismatch for {entry['id']}")


@pytest.mark.parametrize("entry", _fixtures(), ids=lambda item: item["id"])
def test_public_mryx_tier_a_integration(tmp_path, entry):
    """Authentic Mercury bytes → stored-site import → ExtXYZ → ASE (cross-format)."""
    path = tmp_path / entry["filename"].replace(",", "_")
    try:
        _download(entry, path)
    except urllib.error.URLError as exc:
        pytest.fail(f"public fixture download required for integration gate: {exc}")

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