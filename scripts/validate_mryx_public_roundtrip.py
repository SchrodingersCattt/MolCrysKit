#!/usr/bin/env python3
"""Download public Mercury .mryx exports and run tier-(A) round-trip validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from molcrys_kit.constants.mryx import MRYX_COORDINATE_TOLERANCE_A
from molcrys_kit.io import read_extxyz, read_mryx, read_mryx_atoms, write_extxyz

MANIFEST = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "mryx" / "public_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MolCrysKit-mryx-validation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            dest.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"download failed: {url}\n{exc}") from exc


def _load_fixture(entry: dict, cache_dir: Path) -> Path:
    filename = entry["filename"].replace(",", "_")
    path = cache_dir / filename
    if path.exists() and _sha256(path) == entry["sha256"]:
        return path
    _download(entry["url"], path)
    if _sha256(path) != entry["sha256"]:
        raise SystemExit(f"sha256 mismatch for {entry['id']}: {path}")
    return path


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
        "--cache-dir",
        type=Path,
        default=Path("/tmp/molcrys_kit_mryx_cache"),
        help="directory for downloaded .mryx files",
    )
    parser.add_argument(
        "--fixture-id",
        default=None,
        help="run one manifest id (default: all fixtures)",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["fixtures"]
    if args.fixture_id:
        entries = [item for item in entries if item["id"] == args.fixture_id]
        if not entries:
            raise SystemExit(f"unknown fixture id: {args.fixture_id}")

    reports = []
    for entry in entries:
        path = _load_fixture(entry, args.cache_dir)
        reports.append({"id": entry["id"], **_tier_a(path, entry["tier_a_expectations"])})

    print(json.dumps({"tier": "A", "results": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())