"""Coordinate-free scorecard, retaining every frozen case in denominators."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json

from .common import ROOT, STRATA, config, read_json, verify_manifest, write_json


def summarize(manifest, rows):
    if len({r.get("harness_sha256") for r in rows}) > 1:
        raise ValueError("mixed harness versions in results")
    by_ref = {}
    for row in rows:
        if row["refcode"] in by_ref:
            raise ValueError("duplicate result row")
        if row["manifest_sha256"] != manifest["sha256"]:
            raise ValueError("result manifest mismatch")
        by_ref[row["refcode"]] = row
    if set(by_ref) - {c["refcode"] for c in manifest["cases"]}:
        raise ValueError("unexpected result refcode")
    completed = []
    for case in manifest["cases"]:
        completed.append(
            by_ref.get(
                case["refcode"],
                {
                    **case,
                    "status": "indeterminate",
                    "record_match": False,
                    "stages": {
                        "execution": {"status": "indeterminate", "code": "not_executed"}
                    },
                },
            )
        )
    groups = {}
    for stratum in (*STRATA, "targeted", "all"):
        group = [
            r
            for r in completed
            if (
                r.get("targeted")
                if stratum == "targeted"
                else not r.get("targeted") and r["stratum"] == stratum
                if stratum != "all"
                else True
            )
        ]
        groups[stratum] = {
            "total": len(group),
            "status": dict(Counter(r["status"] for r in group)),
            "record_match": sum(bool(r.get("record_match")) for r in group),
            "geometry_roundtrip_match": sum(
                bool(r.get("geometry_roundtrip_match")) for r in group
            ),
            "memory_geometry_pass": sum(
                r.get("stages", {}).get("memory_geometry", {}).get("status") == "pass"
                for r in group
            ),
            "memory_average_pass": sum(
                r.get("stages", {}).get("memory_average", {}).get("status") == "pass"
                for r in group
            ),
        }
    codes = Counter()
    examples = defaultdict(list)
    stages = defaultdict(Counter)
    for row in completed:
        for name, stage in row["stages"].items():
            stages[name][stage["status"]] += 1
            if stage["status"] != "pass":
                key = name + ":" + stage["code"]
                codes[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append(row["refcode"])
    return completed, {
        "schema": "csd-roundtrip.summary.v1",
        "manifest_sha256": manifest["sha256"],
        "target_commit": config()["target_commit"],
        "observed": len(rows),
        "frozen_total": len(completed),
        "groups": groups,
        "stages": {k: dict(v) for k, v in stages.items()},
        "failure_codes": dict(codes),
        "examples": dict(examples),
    }


def generate(run_id):
    folder = ROOT / "reports" / run_id
    manifest = read_json(ROOT / "manifest.json")
    verify_manifest(manifest)
    rows = [
        json.loads(line)
        for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed, summary = summarize(manifest, rows)
    write_json(folder / "summary.json", summary)
    with (folder / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "refcode",
                "stratum",
                "targeted",
                "status",
                "record_match",
                "geometry_roundtrip_match",
                "molecular_completeness",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in writer.fieldnames} for r in completed)
    lines = [
        "# CSD molecular-crystal round-trip benchmark",
        "",
        f"Run: `{run_id}`. Target: `{config()['target_commit']}`.",
        f"Manifest SHA-256: `{manifest['sha256']}`.",
        "",
        f"Results received: **{len(rows)}/{len(completed)}**. Missing results remain indeterminate in the denominator.",
        "",
        "## Match contract",
        "",
        "No rigid fitting, cell scaling or origin fitting. Molecules permit only coherent integer lattice translations,",
        "valid atom permutations and whole-molecule permutations. Position max 1e-4 Å; cell lengths 1e-6 Å;",
        "angles 1e-5 degrees; operation translations 1e-7 fractional; occupancy 1e-5.",
        "",
        "Overall pass requires all applicable stages, including known chemical identity, both CIF representations,",
        "and reconstructable whole-molecule disorder where applicable. Source-missing coordinates are reported separately;",
        "no atoms are added. Marginal average coordinates do not establish a lossless disorder round trip.",
        "",
        "## Fixed-test-set results",
        "",
        "| Stratum | N | Pass | Mismatch | Unsupported | Error | Timeout | Indeterminate | Record match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["groups"].items():
        c = values["status"]
        lines.append(
            f"| {key} | {values['total']} | "
            + " | ".join(
                str(c.get(x, 0))
                for x in (
                    "pass",
                    "mismatch",
                    "unsupported",
                    "error",
                    "timeout",
                    "indeterminate",
                )
            )
            + f" | {values['record_match']} |"
        )
    lines += [
        "",
        "## Geometric recovery",
        "",
        "| Stratum | Ordered memory geometry | Disorder memory average | Complete ordered geometry round trip |",
        "|---|---:|---:|---:|",
    ]
    for key, values in summary["groups"].items():
        lines.append(
            f"| {key} | {values['memory_geometry_pass']} | {values['memory_average_pass']} | {values['geometry_roundtrip_match']} |"
        )
    lines += ["", "## Stage results", "", "| Stage | Status counts |", "|---|---|"]
    for key, value in sorted(summary["stages"].items()):
        lines.append(f"| {key} | {json.dumps(value, sort_keys=True)} |")
    lines += [
        "",
        "## Failure categories",
        "",
        "| Stage and code | N | Example refcodes |",
        "|---|---:|---|",
    ]
    for code, count in sorted(
        summary["failure_codes"].items(), key=lambda x: (-x[1], x[0])
    ):
        lines.append(f"| {code} | {count} | {', '.join(summary['examples'][code])} |")
    lines += [
        "",
        "## Interpretation and reproducibility",
        "",
        "- The five strata have fixed quotas; totals describe this test set, not a population-weighted CSD success rate.",
        "- Export/CIF extraction defects are separate from constructor failures. An earlier failure can prevent later stages;",
        "  stage denominators and absent stages must be read with the overall status.",
        "- CCDC supplies original chemical annotations. The pinned constructor accepts untyped connectivity; known bond-order",
        "  information that cannot be represented is an unsupported chemistry gate, even if coordinates match.",
        "- CSD coordinates, CIFs, templates, mappings and raw logs remain under ignored `_local/`.",
        "- Use the frozen manifest and recorded environment; `run --refcodes REFCODE` reproduces a case.",
        "- Source correlations absent from CIF stay unknown. Bounded enumeration is not experimental joint-distribution evidence.",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    generate(parser.parse_args().run_id)
