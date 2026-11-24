#!/usr/bin/env python3
"""Run tier-(A) round-trip validation on git-bundled public Mercury exports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from molcrys_kit.constants.mryx import MRYX_COORDINATE_TOLERANCE_A
from molcrys_kit.io import read_extxyz, read_mryx, read_mryx_atoms, write_extxyz

MANIFEST = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "mryx" / "public_manifest.json"
)


def _fixture_paths_module():
    paths_file = MANIFEST.parent / "fixture_paths.py"
    spec = importlib.util.spec_from_file_location("mryx_fixture_paths", paths_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture resolver from {paths_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tier_a(path: Path, expectations: dict) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        atoms = read_mryx_atoms(path)
        crystal = read_mryx(path)

    assert len(atoms) == expectations["n_atoms"]

    cellpar = atoms.cell.cellpar()
    np.testing.assert_allclose(
        cellpar,
        expectations["cellpar_deg"],
        atol=expectations.get("cellpar_atol", 0.01),
    )

    meta = atoms.info["mryx"]
    expected_label = expectations.get("structure_label")
    if expected_label is not None:
        assert meta.get("label") == expected_label

    out = path.with_suffix(".extxyz")
    write_extxyz(crystal, str(out))
    ase_atoms = ase_read(str(out))
    rt = read_extxyz(str(out)).to_ase()

    pos_atol = expectations.get("position_atol_A", MRYX_COORDINATE_TOLERANCE_A)
    np.testing.assert_allclose(atoms.get_positions(), ase_atoms.get_positions(), atol=pos_atol)
    np.testing.assert_allclose(atoms.cell.array, ase_atoms.cell.array, atol=1e-6)
    np.testing.assert_allclose(atoms.get_positions(), rt.get_positions(), atol=pos_atol)
    assert atoms.get_chemical_symbols() == ase_atoms.get_chemical_symbols()

    return {
        "fixture_path": str(path),
        "n_atoms": len(atoms),
        "structure_label": meta.get("label"),
        "stored_spacegroup_name": meta.get("stored_spacegroup_name"),
        "stored_spacegroup_number": meta.get("stored_spacegroup_number"),
        "extxyz_path": str(out),
        "max_position_delta_A": float(
            np.max(np.abs(atoms.get_positions() - ase_atoms.get_positions()))
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="public_manifest.json path",
    )
    parser.add_argument(
        "--fixture-id",
        default=None,
        help="run one manifest id (default: all fixtures)",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolve = _fixture_paths_module().resolve_bundled_fixture

    entries = manifest["fixtures"]
    if args.fixture_id:
        entries = [item for item in entries if item["id"] == args.fixture_id]
        if not entries:
            raise SystemExit(f"unknown fixture id: {args.fixture_id}")

    reports = []
    for entry in entries:
        path = resolve(entry)
        reports.append({"id": entry["id"], **_tier_a(path, entry["tier_a_expectations"])})

    print(json.dumps({"tier": "A", "bundled_only": True, "results": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())