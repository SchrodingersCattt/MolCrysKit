#!/usr/bin/env python3
"""Build an aromaticity geometry calibration snapshot from CSD-exported rows.

The CSD Python API is intentionally not imported here.  Run a licensed CSD
job that exports JSON/JSONL rows following the schema in
``docs/chemistry_perception_design.md``, then invoke this script with the input
and output paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molcrys_kit.chemistry.aromaticity_calibration import (
    CalibrationPolicy,
    calibrate_aromaticity_geometry,
    read_records,
    write_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSD ring observations (.json or .jsonl)")
    parser.add_argument("output", type=Path, help="calibration snapshot (.json)")
    parser.add_argument("--source", type=Path, help="JSON metadata for the CSD sample")
    parser.add_argument(
        "--min-pair-observations",
        type=int,
        default=CalibrationPolicy.min_pair_observations,
    )
    parser.add_argument(
        "--min-ring-observations",
        type=int,
        default=CalibrationPolicy.min_ring_observations,
    )
    args = parser.parse_args()

    source = {}
    if args.source:
        source = json.loads(args.source.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            parser.error("--source must contain a JSON object")
    policy = CalibrationPolicy(
        min_pair_observations=args.min_pair_observations,
        min_ring_observations=args.min_ring_observations,
    )
    snapshot = calibrate_aromaticity_geometry(
        read_records(args.input), source=source, policy=policy
    )
    write_snapshot(snapshot, args.output)
    print(json.dumps(snapshot["quality"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
