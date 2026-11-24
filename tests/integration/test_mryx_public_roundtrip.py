"""Offline tier-(A) check against git-bundled public Mercury exports."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read

from molcrys_kit.constants.mryx import MRYX_COORDINATE_TOLERANCE_A
from molcrys_kit.io import read_extxyz, read_mryx, read_mryx_atoms, write_extxyz

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "mryx" / "public_manifest.json"


def _fixture_paths():
    paths_file = MANIFEST.parent / "fixture_paths.py"
    spec = importlib.util.spec_from_file_location("mryx_fixture_paths", paths_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {paths_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixtures():
    return _fixture_paths().load_manifest()["fixtures"]


@pytest.mark.parametrize("entry", _fixtures(), ids=lambda item: item["id"])
def test_public_mryx_tier_a_integration(tmp_path, entry, monkeypatch):
    """Bundled Mercury bytes → stored-site import → ExtXYZ → ASE (no network)."""

    def _forbid_network(*_args, **_kwargs):
        raise AssertionError("tier-(A) integration must not perform HTTP downloads")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        _forbid_network,
        raising=False,
    )

    path = _fixture_paths().resolve_bundled_fixture(entry)

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

    meta = atoms.info["mryx"]
    assert meta.get("stored_spacegroup_name") == "P-1"
    assert meta.get("stored_spacegroup_number") == 2
    assert meta.get("label") == expectations.get("structure_label")
    assert meta.get("label") not in entry["anonymization"]["removed_identifiers"]