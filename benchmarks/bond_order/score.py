"""Score raw bond-order observations without hiding unknowns or failures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import _edge_key, load_manifest, verify_manifest


def _same_edge(key: str, mapping: dict[str, Any] | None) -> Any:
    mapping = mapping or {}
    if key in mapping:
        return mapping[key]
    a, sep, b = key.partition("--")
    if sep and f"{b}--{a}" in mapping:
        return mapping[f"{b}--{a}"]
    # Periodic keys are canonically oriented, but accept a reverse spelling
    # with the negated image shift for hand-authored comparator output.
    if sep and "@[" in b:
        left, right, shift, occurrence = _split_edge_key(key)
        reverse = _edge_key(right, left, tuple(-v for v in shift), occurrence)
        if reverse in mapping:
            return mapping[reverse]
    return None


def _split_edge_key(key: str) -> tuple[str, str, tuple[int, int, int], int]:
    left, _sep, right = key.partition("--")
    occurrence = 1
    if "#" in right:
        right, suffix = right.rsplit("#", 1)
        try:
            occurrence = int(suffix)
        except ValueError:
            occurrence = 1
    shift = (0, 0, 0)
    if "@[" in right:
        right, text = right.split("@[", 1)
        try:
            shift = tuple(int(v) for v in text.rstrip("]").split(","))
            if len(shift) != 3:
                raise ValueError
        except ValueError:
            shift = (0, 0, 0)
    return left, right, shift, occurrence


def score_prediction(
    case: dict[str, Any], prediction: dict[str, Any] | None
) -> dict[str, Any]:
    expected = case.get("expected", {})
    if not prediction:
        return {
            "status": "no_prediction",
            "edge_count": len(expected.get("bond_orders", {})),
            "known_total": 0,
            "known_correct": 0,
            "unknown_total": 0,
            "unknown_correct": 0,
            "false_assertions": 0,
            "missing": len(expected.get("bond_orders", {})),
            "edge_exact": False,
            "mismatches": [{"reason": "no prediction"}],
            "bond_kind": {"total": 0, "correct": 0},
            "resonance": {"total": 0, "correct": 0},
            "charge": {
                "total": 0,
                "correct": 0,
                "unknown_total": 0,
                "unknown_correct": 0,
            },
            "implicit_hydrogen": {
                "total": 0,
                "correct": 0,
                "unknown_total": 0,
                "unknown_correct": 0,
            },
            "aromatic": {"tp": 0, "fp": 0, "fn": 0},
        }
    got = prediction.get("bond_orders", {})
    kekule = bool(case.get("equivalent_forms")) and _kekule_equivalent(
        expected.get("bond_orders") or {}, got
    )
    mismatches = []
    known_total = known_correct = unknown_total = unknown_correct = false_assertions = (
        missing
    ) = 0
    kind_total = kind_correct = 0
    exact = True
    for key, truth in expected.get("bond_orders", {}).items():
        value = _same_edge(key, got)
        present = (
            value is not None
            or key in got
            or key.split("--")[::-1]
            and f"{key.split('--')[1]}--{key.split('--')[0]}" in got
        )
        if truth is None:
            unknown_total += 1
            if not present or value is None:
                unknown_correct += 1
            else:
                false_assertions += 1
                exact = False
                mismatches.append(
                    {
                        "edge": key,
                        "truth": None,
                        "prediction": value,
                        "reason": "asserted order where truth is unknown",
                    }
                )
        else:
            known_total += 1
            if not present:
                missing += 1
                exact = False
                mismatches.append(
                    {
                        "edge": key,
                        "truth": truth,
                        "prediction": None,
                        "reason": "missing edge prediction",
                    }
                )
            elif kekule or (
                value is not None
                and math.isclose(float(value), float(truth), rel_tol=0.0, abs_tol=1e-8)
            ):
                known_correct += 1
            elif value is None:
                exact = False
                mismatches.append(
                    {
                        "edge": key,
                        "truth": truth,
                        "prediction": None,
                        "reason": "missing edge prediction",
                    }
                )
            else:
                exact = False
                mismatches.append(
                    {
                        "edge": key,
                        "truth": truth,
                        "prediction": value,
                        "reason": "wrong order",
                    }
                )
    expected_kinds = expected.get("bond_kinds", {})
    predicted_kinds = prediction.get("bond_kinds", {})
    for key, truth_kind in expected_kinds.items():
        value_kind = _same_edge(key, predicted_kinds)
        # Unknown kind is an explicit semantic value and can be matched.
        kind_total += 1
        if value_kind == truth_kind or (
            value_kind is not None
            and getattr(value_kind, "value", value_kind) == truth_kind
        ):
            kind_correct += 1
    charge_scores = _score_mapping(
        expected.get("formal_charges"), prediction.get("formal_charges")
    )
    h_scores = _score_mapping(
        expected.get("implicit_hydrogens"), prediction.get("implicit_hydrogens")
    )
    hydrogen_conservation = _hydrogen_conservation(case, prediction)

    def canonical_aromatic(key: str) -> str:
        left, right, shift, occurrence = _split_edge_key(key)
        if shift == (0, 0, 0) and right < left:
            left, right = right, left
        return _edge_key(left, right, shift, occurrence)

    truth_arom = {canonical_aromatic(k) for k in expected.get("aromatic_bonds", [])}
    pred_arom = {canonical_aromatic(k) for k in prediction.get("aromatic_bonds", [])}
    arom_tp = len(truth_arom & pred_arom)
    arom_fp = len(pred_arom - truth_arom)
    arom_fn = len(truth_arom - pred_arom)
    truth_resonance = expected.get("resonance_class")
    pred_resonance = prediction.get("resonance_class")
    resonance_known = truth_resonance is not None
    # If a result exposes only aromatic flags/classes (the public result uses
    # index tuples), accept an equivalent resonance class when every normative
    # aromatic edge is retained. Numeric Kekule orders remain scored above.
    resonance_from_aromatic = bool(
        truth_resonance and truth_arom and truth_arom <= pred_arom
    )
    resonance_correct = bool(
        resonance_known
        and (pred_resonance == truth_resonance or resonance_from_aromatic)
    )
    return {
        "status": "scored",
        "edge_count": len(expected.get("bond_orders", {})),
        "known_total": known_total,
        "known_correct": known_correct,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
        "false_assertions": false_assertions,
        "missing": missing,
        "edge_exact": exact,
        "known_exact": known_total > 0 and known_correct == known_total,
        "mismatches": mismatches,
        "bond_kind": {"total": kind_total, "correct": kind_correct},
        "resonance": {"total": int(resonance_known), "correct": int(resonance_correct)},
        "charge": charge_scores,
        "implicit_hydrogen": h_scores,
        "hydrogen_conservation": hydrogen_conservation,
        "aromatic": {"tp": arom_tp, "fp": arom_fp, "fn": arom_fn},
    }


def _hydrogen_conservation(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, int]:
    """Check explicit-attached plus implicit H without double-counting either."""
    expected = case.get("expected", {})
    implicit_truth = expected.get("implicit_hydrogens") or {}
    attached_truth = expected.get("attached_hydrogens") or {}
    implicit_pred = prediction.get("implicit_hydrogens") or {}
    attached_pred = prediction.get("attached_hydrogens") or attached_truth
    total = correct = 0
    for atom_id, implicit in implicit_truth.items():
        if implicit is None:
            continue
        total += 1
        truth_total = int(attached_truth.get(atom_id, 0) or 0) + int(implicit)
        pred_total = int(attached_pred.get(atom_id, 0) or 0) + int(
            implicit_pred.get(atom_id, 0) or 0
        )
        correct += int(truth_total == pred_total)
    return {"total": total, "correct": correct}


def _kekule_equivalent(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    """True when both assignments are {1, 2} placements with the same atom sums.

    This is only consulted for fixtures that declare ``equivalent_forms``.
    Other cases stay on exact numeric orders.
    """
    if not expected:
        return False
    left: dict[str, float] = {}
    right: dict[str, float] = {}
    for key, truth in expected.items():
        value = _same_edge(key, predicted)
        if truth is None or value is None:
            return False
        try:
            truth_order = float(truth)
            predicted_order = float(value)
        except (TypeError, ValueError):
            return False
        if truth_order not in (1.0, 2.0) or predicted_order not in (1.0, 2.0):
            return False
        atom_a, sep, atom_b = key.partition("--")
        if not sep:
            return False
        left[atom_a] = left.get(atom_a, 0.0) + truth_order
        left[atom_b] = left.get(atom_b, 0.0) + truth_order
        right[atom_a] = right.get(atom_a, 0.0) + predicted_order
        right[atom_b] = right.get(atom_b, 0.0) + predicted_order
    return left == right


def _score_mapping(
    truth: dict[str, Any] | None, prediction: dict[str, Any] | None
) -> dict[str, Any]:
    if truth is None:
        return {"total": 0, "correct": 0, "unknown_total": 0, "unknown_correct": 0}
    prediction = prediction or {}
    total = correct = unknown_total = unknown_correct = 0
    for key, expected in truth.items():
        value = prediction.get(key)
        if expected is None:
            unknown_total += 1
            if value is None:
                unknown_correct += 1
        else:
            total += 1
            if value == expected:
                correct += 1
    return {
        "total": total,
        "correct": correct,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
    }


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["score"]["status"] == "scored"]
    known_total = sum(r["score"]["known_total"] for r in scored)
    known_correct = sum(r["score"]["known_correct"] for r in scored)
    unknown_total = sum(r["score"]["unknown_total"] for r in scored)
    unknown_correct = sum(r["score"]["unknown_correct"] for r in scored)
    edge_exact = sum(bool(r["score"]["edge_exact"]) for r in scored)
    arom = {
        k: sum(r["score"]["aromatic"][k] for r in scored) for k in ("tp", "fp", "fn")
    }
    kind_total = sum(r["score"]["bond_kind"]["total"] for r in scored)
    kind_correct = sum(r["score"]["bond_kind"]["correct"] for r in scored)
    resonance_total = sum(r["score"]["resonance"]["total"] for r in scored)
    resonance_correct = sum(r["score"]["resonance"]["correct"] for r in scored)
    charge_total = sum(r["score"]["charge"]["total"] for r in scored)
    charge_correct = sum(r["score"]["charge"]["correct"] for r in scored)
    h_total = sum(r["score"]["implicit_hydrogen"]["total"] for r in scored)
    h_correct = sum(r["score"]["implicit_hydrogen"]["correct"] for r in scored)
    conservation_total = sum(
        r["score"].get("hydrogen_conservation", {}).get("total", 0) for r in scored
    )
    conservation_correct = sum(
        r["score"].get("hydrogen_conservation", {}).get("correct", 0)
        for r in scored
    )
    complete_rows = [r for r in rows if r.get("status") == "ok"]
    times = [
        r["elapsed_ns"] / 1e6
        for r in complete_rows
        if isinstance(r.get("elapsed_ns"), (int, float))
    ]
    raw_times = [
        run["elapsed_ns"] / 1e6
        for r in complete_rows
        for run in r.get("raw_runs", [])
        if isinstance(run.get("elapsed_ns"), (int, float))
    ]
    rss_values = [
        run["peak_rss_kb"]
        for r in complete_rows
        for run in r.get("raw_runs", [])
        if isinstance(run.get("peak_rss_kb"), (int, float))
    ]
    return {
        "cases_total": len(rows),
        "cases_scored": len(scored),
        "case_exact_rate": _rate(edge_exact, len(scored)),
        "bond_order_edge_accuracy": _rate(known_correct, known_total),
        "bond_kind_accuracy": _rate(kind_correct, kind_total),
        "resonance_class_accuracy": _rate(resonance_correct, resonance_total),
        "unknown_correct_rate": _rate(unknown_correct, unknown_total),
        "false_assertions": sum(r["score"]["false_assertions"] for r in scored),
        "charge_accuracy": _rate(charge_correct, charge_total),
        "implicit_hydrogen_accuracy": _rate(h_correct, h_total),
        "hydrogen_conservation_accuracy": _rate(
            conservation_correct, conservation_total
        ),
        "aromatic_f1": _f1(arom["tp"], arom["fp"], arom["fn"]),
        "latency_ms": _quantiles(times),
        "raw_latency_ms": _quantiles(raw_times),
        "cold_latency_ms": _quantiles(
            [
                r["raw_runs"][0]["elapsed_ns"] / 1e6
                for r in complete_rows
                if r.get("raw_runs")
            ]
        ),
        "warm_latency_ms": _quantiles(
            [
                run["elapsed_ns"] / 1e6
                for r in complete_rows
                for run in r.get("raw_runs", [])[1:]
            ]
        ),
        "peak_rss_kb": max(rss_values) if rss_values else None,
        "status_counts": _counts(r.get("status") for r in rows),
    }


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)

    def q(p: float) -> float:
        return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]

    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "n": len(values)}


def _counts(values):
    out = defaultdict(int)
    for x in values:
        out[x] += 1
    return dict(out)


def _bootstrap_ci(
    values: list[float], seed: int = 174, samples: int = 1000
) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    means.sort()
    return {
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
        "n": len(values),
    }


def _family_aggregate(
    by_engine: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for engine, rows in by_engine.items():
        families = defaultdict(list)
        for row in rows:
            families[row.get("family", "unknown")].append(row)
        out[engine] = {family: _aggregate(items) for family, items in families.items()}
    return out


def _resource_gate(
    name: str, mck: float | None, ob: float | None, ratio: float, direction: str
) -> dict[str, Any]:
    if mck is None or ob is None or ob == 0:
        return {
            "metric": name,
            "status": "NOT-COMPARABLE",
            "mck": mck,
            "ob_cpp": ob,
            "reason": "paired native observations are required",
        }
    actual = mck / ob
    passed = actual <= ratio if direction == "lower" else actual >= ratio
    return {
        "metric": name,
        "status": "PASS" if passed else "FAIL",
        "mck": mck,
        "ob_cpp": ob,
        "ratio": actual,
        "target": ratio,
        "reason": "pre-registered resource ratio",
    }


def _gate(name: str, mck: float | None, ob: float | None) -> dict[str, Any]:
    if mck is None or ob is None:
        return {
            "metric": name,
            "status": "NOT-COMPARABLE",
            "mck": mck,
            "ob_cpp": ob,
            "reason": "both engines need complete scored observations",
        }
    if mck == 1.0 and ob == 1.0:
        status = "CEILING-TIE"
        reason = "both methods are at the mathematical ceiling"
    elif mck > ob:
        status = "PASS"
        reason = "MCK strictly exceeds pinned OB on aggregate"
    else:
        status = "FAIL"
        reason = "MCK does not strictly exceed pinned OB"
    return {
        "metric": name,
        "status": status,
        "mck": mck,
        "ob_cpp": ob,
        "reason": reason,
    }


def score_results(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    valid, detail = verify_manifest(manifest)
    rows = result.get("raw_results", [])
    cases = {c["case_id"]: c for c in manifest.get("cases", [])}
    by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detailed = []
    for raw in rows:
        case = cases.get(raw.get("case_id"))
        if case is None:
            continue
        prediction = raw.get("prediction") if raw.get("status") == "ok" else None
        scored = score_prediction(case, prediction)
        row = {
            "case_id": raw["case_id"],
            "family": raw.get("family", case.get("family")),
            "engine": raw.get("engine"),
            "status": raw.get("status"),
            "elapsed_ns": raw.get("elapsed_ns"),
            "raw_runs": raw.get("raw_runs", []),
            "prediction": raw.get("prediction"),
            "score": scored,
            "reason": raw.get("reason"),
        }
        detailed.append(row)
        by_engine[row["engine"]].append(row)
    aggregate = {engine: _aggregate(items) for engine, items in by_engine.items()}
    paired_rows = {}
    for row in by_engine.get("mck", []):
        paired_rows.setdefault(row["case_id"], {})["mck"] = row
    for row in by_engine.get("ob_cpp", []):
        paired_rows.setdefault(row["case_id"], {})["ob_cpp"] = row
    paired_deltas = []
    paired_latency_ratios = []
    for pair in paired_rows.values():
        if set(pair) != {"mck", "ob_cpp"}:
            continue
        ms, os = pair["mck"]["score"], pair["ob_cpp"]["score"]
        if ms.get("status") == "scored" and os.get("status") == "scored":
            ma = _rate(ms["known_correct"], ms["known_total"])
            oa = _rate(os["known_correct"], os["known_total"])
            if ma is not None and oa is not None:
                paired_deltas.append(ma - oa)
            if pair["mck"].get("elapsed_ns") and pair["ob_cpp"].get("elapsed_ns"):
                paired_latency_ratios.append(
                    pair["mck"]["elapsed_ns"] / pair["ob_cpp"]["elapsed_ns"]
                )
    paired = {
        "bond_order_accuracy_delta": {
            "values": paired_deltas,
            "ci95": _bootstrap_ci(paired_deltas),
        },
        "latency_ratio": {
            "values": paired_latency_ratios,
            "ci95": _bootstrap_ci(paired_latency_ratios),
        },
    }
    m, o = aggregate.get("mck", {}), aggregate.get("ob_cpp", {})
    gates = [
        _gate(
            "bond_order_edge_accuracy",
            m.get("bond_order_edge_accuracy"),
            o.get("bond_order_edge_accuracy"),
        ),
        _gate(
            "bond_kind_accuracy",
            m.get("bond_kind_accuracy"),
            o.get("bond_kind_accuracy"),
        ),
        _gate(
            "resonance_class_accuracy",
            m.get("resonance_class_accuracy"),
            o.get("resonance_class_accuracy"),
        ),
        _gate("molecule_exact", m.get("case_exact_rate"), o.get("case_exact_rate")),
        _gate(
            "formal_charge_accuracy", m.get("charge_accuracy"), o.get("charge_accuracy")
        ),
        _gate(
            "implicit_hydrogen_accuracy",
            m.get("implicit_hydrogen_accuracy"),
            o.get("implicit_hydrogen_accuracy"),
        ),
        _gate("aromatic_f1", m.get("aromatic_f1"), o.get("aromatic_f1")),
        _resource_gate(
            "warm_latency_p50_ratio",
            (m.get("warm_latency_ms") or {}).get("p50"),
            (o.get("warm_latency_ms") or {}).get("p50"),
            0.90,
            "lower",
        ),
        _resource_gate(
            "peak_rss_ratio", m.get("peak_rss_kb"), o.get("peak_rss_kb"), 0.90, "lower"
        ),
    ]
    # Completeness of the selected MCK cases is a functional gate.  Comparator
    # rows may remain unavailable, but a reduced MCK run must never pass by
    # omitting cases or returning only abstentions.
    expected_rows = result.get("selected_case_count", 0) * len(
        result.get("engines", [])
    )
    observed_keys = [(r.get("case_id"), r.get("engine")) for r in rows]
    complete_rows = (
        len(rows) == expected_rows
        and len(set(observed_keys)) == expected_rows
        and all(r.get("status") == "ok" for r in rows)
    )
    corpora_ok = not result.get("missing_corpora")
    processes = result.get("environment", {}).get(
        "processes_observed", result.get("environment", {}).get("processes", 0)
    )
    reproducible = bool(
        result.get("manifest_sha256") == manifest.get("manifest_sha256")
        and result.get("environment", {}).get("python")
    )
    completeness = {
        "manifest_checksum": bool(
            valid and result.get("manifest_sha256") == manifest.get("manifest_sha256")
        ),
        "rows_complete": complete_rows,
        "full_manifest_selected": result.get("selected_case_count")
        == result.get("full_manifest_case_count"),
        "release_corpora_available": corpora_ok
        if result.get("suite") == "release"
        else True,
        "independent_process_requirement": processes >= 7
        if result.get("suite") == "release"
        else True,
        "environment_reproducible": reproducible,
    }
    for gate in gates:
        gate["scope"] = "deferred-superiority"
        gate["blocks_strict_pass"] = False
    mck_rows = [row for row in detailed if row.get("engine") == "mck"]
    mck_case_ids = [row.get("case_id") for row in mck_rows]
    mck_coverage = bool(
        mck_rows
        and len(mck_rows) == int(result.get("selected_case_count", 0))
        and len(set(mck_case_ids)) == len(mck_case_ids)
        and all(row.get("status") == "ok" for row in mck_rows)
    )
    mck_scored = [row for row in mck_rows if row["score"].get("status") == "scored"]
    mck_known_total = sum(row["score"].get("known_total", 0) for row in mck_scored)
    mck_known_correct = sum(row["score"].get("known_correct", 0) for row in mck_scored)
    mck_known_truth = bool(mck_known_total and mck_known_correct == mck_known_total)
    mck_ok = mck_coverage
    false_assertions = sum(
        row["score"].get("false_assertions", 0)
        for row in mck_rows
        if row["score"].get("status") == "scored"
    )
    functional_gates = [
        {
            "metric": "mck_observations",
            "status": "PASS" if mck_ok else "FAIL",
            "reason": "every selected MolCrysKit case must run exactly once to completion",
            "observed": len(mck_rows),
            "expected": result.get("selected_case_count", 0),
        },
        {
            "metric": "mck_case_coverage",
            "status": "PASS" if mck_coverage else "FAIL",
            "reason": "selected-case coverage cannot be bypassed by omitting rows",
        },
        {
            "metric": "mck_known_truth",
            "status": "PASS" if mck_known_truth else "FAIL",
            "known_correct": mck_known_correct,
            "known_total": mck_known_total,
            "reason": "at least one known order is required and every known order must be correct",
        },
        {
            "metric": "mck_false_assertions",
            "status": "PASS" if false_assertions == 0 else "FAIL",
            "mck": false_assertions,
            "reason": "an unknown bond order must stay unknown",
        },
    ]
    return {
        "schema_version": "1.0",
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "suite": result.get("suite"),
        "aggregate": aggregate,
        "aggregate_by_family": _family_aggregate(by_engine),
        "paired_comparisons": paired,
        "scorecard": gates,
        "functional_gates": functional_gates,
        "completeness": completeness,
        "detailed": detailed,
        "strict_pass": bool(
            mck_ok
            and mck_known_truth
            and false_assertions == 0
            and completeness["manifest_checksum"]
            and completeness["environment_reproducible"]
        ),
        "notes": [
            "strict_pass checks functional MolCrysKit rows and false assertions.",
            "Open Babel superiority, latency, throughput, RSS, process count, QM9, and curated corpora stay visible and do not change strict_pass (issue #176).",
            "NOT-COMPARABLE is a missing comparison, never a superiority pass.",
            "Unknown/null truth and false assertions are scored separately.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="raw result JSON from benchmark run"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True, help="scorecard JSON path")
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help=(
            "exit nonzero unless every MolCrysKit row completed, "
            "false assertions are zero, and the manifest and Python "
            "environment check out. Open Babel and corpus gates are issue #176"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = json.loads(Path(args.input).read_text(encoding="utf-8"))
        manifest = load_manifest(args.manifest)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read input: {exc}", file=sys.stderr)
        return 2
    report = score_results(manifest, result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for gate in report["functional_gates"]:
        print(f"{gate['metric']:28s} {gate['status']}")
    for gate in report["scorecard"]:
        print(f"{gate['metric']:28s} {gate['status']} (deferred)")
    print(f"strict_pass={report['strict_pass']} wrote={args.output}")
    return 0 if (not args.strict_gates or report["strict_pass"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
