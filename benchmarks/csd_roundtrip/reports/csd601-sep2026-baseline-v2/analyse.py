"""Rebuild the readable synthesis from public metadata only, without CCDC.

From the repository root:
python -c "import runpy;runpy.run_path('benchmarks/csd_roundtrip/reports/csd601-sep2026-baseline-v2/analyse.py',run_name='__main__')"
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from benchmarks.csd_roundtrip.common import (
    ROOT,
    STRATA,
    config,
    public_result,
    read_json,
    verify_manifest,
    write_json,
)

GEOMETRY_GATES = (
    "export",
    "memory_geometry",
    "memory_cell",
    "memory_symmetry",
    "p1_ccdc_geometry",
    "p1_cell",
    "p1_pymatgen",
    "p1_mck_readback",
    "p1_source_operations",
    "asu_ccdc_geometry",
    "asu_cell",
    "asu_operations",
    "asu_pymatgen",
    "asu_mck_readback",
)


def geometric_gates_pass(row):
    return all(
        row.get("stages", {}).get(k, {}).get("status") == "pass" for k in GEOMETRY_GATES
    )


def main():
    folder = Path(__file__).resolve().parent
    manifest = read_json(ROOT / "manifest.json")
    verify_manifest(manifest)
    rows = [
        json.loads(line)
        for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(manifest["cases"]) == 1004
    assert {r["refcode"] for r in rows} == {c["refcode"] for c in manifest["cases"]}
    environment = read_json(folder / "environment.json")
    assert {r["harness_sha256"] for r in rows} == {environment["harness_sha256"]}
    for row in rows:
        assert row == public_result(row), row["refcode"]
        assert row["manifest_sha256"] == manifest["sha256"]
        for stage in row["stages"].values():
            for key in ("code", "status", "error_type"):
                if key in stage:
                    assert re.fullmatch(r"[A-Za-z0-9_:-]+", stage[key]), (
                        row["refcode"],
                        key,
                    )
    coverage = read_json(folder / "coverage_tags.json")
    validation = read_json(folder / "validation.json")
    audit = read_json(folder / "input_audit.json")
    assert (
        audit["cached_cases"] == 1004
        and not audit["periodic_duplicate_reference_cases"]
    )
    ordered = [r for r in rows if "disordered" not in r["tags"]]
    disordered = [r for r in rows if "disordered" in r["tags"]]
    late_geometry = [
        r["refcode"]
        for r in rows
        if geometric_gates_pass(r) and r["status"] == "timeout"
    ]
    stage_counts = {}
    for name in sorted({k for r in rows for k in r["stages"]}):
        stage_counts[name] = dict(
            Counter(
                r["stages"].get(name, {}).get("status", "not_reached") for r in rows
            )
        )
    strata = {}
    for name in (*STRATA, "targeted", "all"):
        selected = [
            r
            for r in rows
            if name == "all"
            or (
                r["targeted"]
                if name == "targeted"
                else not r["targeted"] and r["stratum"] == name
            )
        ]
        strata[name] = {
            "n": len(selected),
            "overall_pass": sum(r["status"] == "pass" for r in selected),
            "ordered_memory_geometry": sum(
                r["stages"].get("memory_geometry", {}).get("status") == "pass"
                for r in selected
            ),
            "all_ordered_geometry_gates": sum(
                geometric_gates_pass(r) for r in selected
            ),
            "disorder_memory_average": sum(
                r["stages"].get("memory_average", {}).get("status") == "pass"
                for r in selected
            ),
        }
    synthesis = {
        "schema": "csd-roundtrip.synthesis.v1",
        "cases": len(rows),
        "ordered_cases": len(ordered),
        "disordered_cases": len(disordered),
        "target_commit": config()["target_commit"],
        "harness_sha256": environment["harness_sha256"],
        "manifest_sha256": manifest["sha256"],
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "strata": strata,
        "stages": stage_counts,
        "molecular_completeness": dict(
            Counter(r.get("molecular_completeness", "not_reached") for r in rows)
        ),
        "late_timeout_geometry_already_verified": late_geometry,
        "public_result_projection_checked": len(rows),
        "coverage": coverage["tag_counts"],
        "isotope_loss_cases": sorted(
            {
                r["refcode"]
                for r in rows
                if any(
                    "isotope" in s.get("missing_annotations", [])
                    for s in r["stages"].values()
                )
            }
        ),
    }
    write_json(folder / "analysis.json", synthesis)
    n_pass = strata["all"]["overall_pass"]
    n_memory = strata["all"]["ordered_memory_geometry"]
    n_geometry = strata["all"]["all_ordered_geometry_gates"]
    n_average = strata["all"]["disorder_memory_average"]
    lines = [
        "# CSD round-trip benchmark: interpretation",
        "",
        "The frozen corpus has **1004/1004 executed entries**: 1000 stratified entries plus four targeted references.",
        f"The package tree is `{config()['target_commit']}`; benchmark-only commits change no production code.",
        "",
        "## Provenance and primary results",
        "",
        "- CSD 6.01, active February/May/September 2026 updates; 1,469,017 entries in the source pool.",
        "- Licensed CCDC API 3.7.2 in its own Python environment; environment versions and active pool are recorded in `environment.json`.",
        "- Seed 42; each of the five frozen strata contains 200 entries. Family suffixes remain distinct exact refcodes.",
        f"- Strict source-record preservation: **{n_pass}/1004**.",
        f"- Ordered in-memory geometric reconstruction: **{n_memory}/{len(ordered)}**.",
        f"- All required ordered geometric round-trip gates satisfied: **{n_geometry}/{len(ordered)}**.",
        f"- Disorder raw marginal-site inventory match in the documented selected-state seed path: **{n_average}/{len(disordered)}**.",
        "",
        "The strict score includes known chemical annotations and lossless whole-molecule disorder restoration. The geometric score",
        "checks the original cell, origin, molecular inventory, graph connectivity and coordinates with coherent molecular lattice images,",
        "plus both P1 and ASU readback. It does not claim typed chemical-annotation preservation.",
        "",
        "| Stratum | Entries | Strict pass | Ordered memory geometry | All ordered geometry gates | Disorder memory average |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, x in strata.items():
        lines.append(
            f"| {name} | {x['n']} | {x['overall_pass']} | {x['ordered_memory_geometry']} | {x['all_ordered_geometry_gates']} | {x['disorder_memory_average']} |"
        )
    lines += [
        "",
        "## Interpretation of non-passes",
        "",
        "1. Known source bond orders cannot be supplied as typed connectivity through this template interface. The missing",
        "   annotation gate remains separate from successful geometry; CIF output may also assert changed chemical annotations.",
        "2. The synthetic I-43m methane example independently demonstrates ASU molecular grouping loss: two CH4 molecules",
        "   become one C2H8 object in MolCrysKit, while pymatgen retains the correct ten-atom inventory. See `findings.md`.",
        "3. The explicit enantiopure general-position P-1 example is accepted and copies R labels onto inversion mates; this",
        "   confirms the corresponding #198 concern without using any CSD coordinates.",
        "4. Statistical-average P1 data are insufficient to restore whole-molecule alternatives. The pinned target has no lossless",
        "   companion reader and no verified true-ASU statistical path. Full disorder round trips therefore remain unestablished.",
        "5. Disorder marginal mismatches in the selected-state seed experiment can involve source occupancy/group-weight",
        "   differences, split/merged site representations and non-symmetry-related alternatives. These are not all attributed to constructor defects. Source",
        "   correlations absent from CIF remain unknown; no physical joint distribution is invented.",
        "6. Parser errors and 300-second evaluation deadlines remain in the denominator. Stage counts include not-reached cases",
        "   explicitly in `analysis.json`; an absent stage is not a demonstrated structural mismatch.",
        "",
        f"Late evaluation timeouts with all geometric gates already recorded: **{len(late_geometry)}**. These remain overall timeouts;",
        "the geometric synthesis counts their completed checks. The worker-completion boolean in the primary CSV can be more conservative.",
        "",
        "## Coverage, completeness and integrity",
        "",
        f"Source completeness flags: `{json.dumps(synthesis['molecular_completeness'], sort_keys=True)}`.",
        "Completeness means coordinate presence for API-listed atoms, not an independently established protonation/valence model.",
        "Water, solvent-formula and neutral-multispecies tags are coverage candidates, not verified crystallization roles.",
        f"All 1004 private source hashes and CIF byte hashes were verified; all {audit['ordered_reference_cases_checked']} ordered",
        "packing references were checked for repeated whole-molecule lattice images and none were found.",
        "The public JSON projection was checked for every case. Coordinates, CIFs, templates, mappings, raw logs and companions remain ignored/local.",
        "",
        "## Validation",
        "",
        f"- Full tests: {validation['full_pytest']['passed']} passed, {validation['full_pytest']['skipped']} skipped.",
        f"- Focused forward/CIF/benchmark tests: {validation['focused']['passed']} passed; {validation['synthetic_benchmark_tests']} new synthetic benchmark tests.",
        "- Changed-file Ruff, format check and compile checks passed. No licensed retrieval runs in ordinary CI.",
        "- Windows fixture checksum validation used exact pinned Git-blob bytes to avoid checkout CRLF conversion.",
        "",
        "The fixed-quota totals describe this corpus. They are not a CSD-population success-rate estimate or evidence of stable packing energies.",
        "",
    ]
    (folder / "analysis.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status_counts": synthesis["status_counts"],
                "ordered_memory": n_memory,
                "ordered_geometry": n_geometry,
                "disorder_average": n_average,
                "late_geometry_timeouts": len(late_geometry),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
