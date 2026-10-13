"""Render a scorecard and enforce issue #174's conservative release gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="scorecard JSON from score.py")
    p.add_argument("--strict-gates", action="store_true")
    p.add_argument("--json", action="store_true", help="print machine-readable scorecard")
    args = p.parse_args(argv)
    try: report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read scorecard: {exc}", file=sys.stderr); return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"manifest={report.get('manifest_id')} suite={report.get('suite')}")
        for gate in report.get("scorecard", []):
            print(f"{gate['status']:16s} {gate['metric']}: {gate.get('reason','')}")
        print("completeness:")
        for key, value in report.get("completeness", {}).items(): print(f"  {key}: {value}")
        print(f"strict_pass={report.get('strict_pass', False)}")
    return 0 if (not args.strict_gates or report.get("strict_pass") is True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
