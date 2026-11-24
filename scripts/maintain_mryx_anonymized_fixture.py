#!/usr/bin/env python3
"""Regenerate the vendored anonymized Zenodo tier-(A) fixture (maintainer-only).

Not invoked in CI. Requires a local upstream ``.mryx`` whose SHA-256 matches
``public_manifest.json`` ``upstream.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

MANIFEST = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "mryx" / "public_manifest.json"
)
DEFAULT_FIXTURE_ID = "zenodo-5aa-triclinic"


def _load_derive_module():
    derive_file = MANIFEST.parent / "derive_fixture.py"
    spec = importlib.util.spec_from_file_location("mryx_derive_fixture", derive_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {derive_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_mryx",
        type=Path,
        help="path to authentic upstream .mryx (verified against manifest upstream.sha256)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="public_manifest.json path",
    )
    parser.add_argument(
        "--fixture-id",
        default=DEFAULT_FIXTURE_ID,
        help="manifest fixture id to refresh",
    )
    args = parser.parse_args(argv)

    derive = _load_derive_module()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["fixtures"] if item["id"] == args.fixture_id)
    upstream = entry["upstream"]
    if _sha256(args.upstream_mryx) != upstream["sha256"]:
        raise SystemExit(
            f"upstream sha256 mismatch (expected {upstream['sha256']}); "
            "refusing to derive a new bundled fixture"
        )

    old_label = entry["anonymization"]["removed_identifiers"][0]
    new_label = entry["anonymization"]["replaces_structure_label"]
    out_path = args.manifest.parent / entry["bundled_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived = derive.anonymize_upstream_bytes(
        args.upstream_mryx.read_bytes(),
        old_label,
        new_label,
    )
    out_path.write_bytes(derived)
    derived_sha = _sha256(out_path)
    print(f"wrote {out_path}")
    print(f"derived_sha256={derived_sha}")
    if derived_sha != entry["derived_sha256"]:
        print("Update public_manifest.json, ATTRIBUTION.md, and docs if this value changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())