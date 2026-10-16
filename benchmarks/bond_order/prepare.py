"""Freeze and validate issue #174 benchmark manifests.

Examples
--------
``python -m benchmarks.bond_order.prepare --verify-manifest``
``python -m benchmarks.bond_order.prepare --manifest data/manifest.json --describe``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import MANIFEST_PATH, load_manifest, verify_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--verify-manifest", action="store_true", help="verify frozen checksum and fixture IDs")
    parser.add_argument("--describe", action="store_true", help="print a compact manifest summary")
    parser.add_argument("--write-verified", metavar="PATH", help="write a byte-for-byte canonical copy after verification")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except OSError as exc:
        print(f"ERROR: cannot read manifest: {exc}", file=sys.stderr)
        return 2
    ok, detail = verify_manifest(manifest)
    if args.verify_manifest or not (args.describe or args.write_verified):
        print(f"manifest={manifest.get('manifest_id')} sha256={manifest.get('manifest_sha256')}")
        print(f"checksum={'PASS' if ok else 'FAIL'} detail={detail}")
    if args.describe:
        print(json.dumps({
            "manifest_id": manifest.get("manifest_id"),
            "frozen_date": manifest.get("frozen_date"),
            "cases": len(manifest.get("cases", [])),
            "case_ids": [c["case_id"] for c in manifest.get("cases", [])],
            "corpora": manifest.get("corpora", []),
            "openbabel_baseline": manifest.get("openbabel_baseline", {}),
        }, indent=2))
    if args.write_verified:
        if not ok:
            print("ERROR: refusing to write an unverified manifest", file=sys.stderr)
            return 2
        Path(args.write_verified).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
