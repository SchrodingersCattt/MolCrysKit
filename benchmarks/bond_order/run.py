"""Run candidate/comparator engines and retain raw, auditable case results."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .common import ENGINE_CALLS, environment, load_manifest, timed_case, verify_manifest


def _engines(value: str) -> list[str]:
    names = ["mck", "ob_cpp", "ob_python"] if value == "all" else [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(names) - set(ENGINE_CALLS))
    if unknown: raise ValueError(f"unknown engine(s): {', '.join(unknown)}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("ci", "release"), default="ci")
    parser.add_argument("--engine", default="all", help="mck, ob_cpp, ob_python, or all")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True, help="result JSON path or directory")
    parser.add_argument("--repeats", type=int, default=3, help="warm measurements per case")
    parser.add_argument("--timeout", type=float, default=0.0, help="per-case timeout in seconds (0 disables)")
    parser.add_argument("--processes", type=int, default=1, help="independent processes represented by this run")
    parser.add_argument("--case", action="append", dest="cases", help="restrict to a case ID (repeatable)")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.processes < 1:
        parser.error("--repeats and --processes must be positive")
    try: engines = _engines(args.engine)
    except ValueError as exc: parser.error(str(exc))
    manifest = load_manifest(args.manifest)
    valid, detail = verify_manifest(manifest)
    if not valid:
        print(f"ERROR: refusing to run with invalid manifest: {detail}", file=sys.stderr)
        return 2
    cases = manifest.get("cases", [])
    if args.cases:
        wanted = set(args.cases)
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            print(f"ERROR: unknown case(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["case_id"] in wanted]
    # Release completeness is a report gate; retaining this fact in metadata
    # prevents a reduced run from being misrepresented as the full release suite.
    missing_corpora = [c for c in manifest.get("corpora", []) if c.get("status") != "available"]
    rows = []
    for case in cases:
        for engine in engines:
            result = timed_case(engine, case, repeats=args.repeats, timeout_s=args.timeout)
            rows.append({"case_id": case["case_id"], "family": case.get("family"), "engine": engine, **result})
            print(f"{engine:10s} {case['case_id']:34s} {result.get('status')}")
    out = Path(args.output)
    if out.suffix.lower() != ".json":
        out = out / f"{args.suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "suite": args.suite,
        "engines": engines,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        # This adapter executes in one interpreter.  Keep requested vs observed
        # process counts separate so --processes cannot manufacture a release claim.
        "environment": {**environment(processes=1), "requested_processes": args.processes, "processes_mode": "single-process adapter"},
        "requested_repeats": args.repeats,
        "rss_measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss (kB on Linux; platform-native otherwise)",
        "selected_case_count": len(cases),
        "full_manifest_case_count": len(manifest.get("cases", [])),
        "missing_corpora": missing_corpora,
        "raw_results": rows,
        "notes": [
            "Unavailable, incomplete, error and timeout rows are retained and count against strict completeness.",
            "This file contains observations only; no score or superiority claim is produced by run.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
