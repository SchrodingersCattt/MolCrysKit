"""Contract tests for the auditable #174 benchmark package."""
from benchmarks.bond_order.common import load_manifest, verify_manifest
from benchmarks.bond_order.score import score_prediction, score_results


def test_frozen_manifest_verifies_and_has_required_named_cases():
    manifest = load_manifest()
    ok, _ = verify_manifest(manifest)
    assert ok
    ids = {case["case_id"] for case in manifest["cases"]}
    assert {"c13h28_long_chain", "first_64_premature_cut", "input_order_invariance", "missing_charge_and_h", "shifted_pbc_cc"} <= ids
    assert manifest["openbabel_baseline"]["commit"] == "0e94434fa75c9f61095023e3c12e0d5f2ac035ff"


def test_unknown_truth_is_not_counted_as_correct_order():
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["case_id"] == "missing_charge_and_h")
    result = score_prediction(case, {"bond_orders": {"C1--N1": 1, "N1--H1": 1}})
    assert result["unknown_total"] == 0
    assert result["false_assertions"] == 0
    # The fixture's unknown fields are charge/H metadata, not bond order.
    assert result["edge_exact"] is True


def test_incomplete_comparison_cannot_pass_strict_scorecard():
    manifest = load_manifest()
    result = {
        "suite": "ci", "manifest_sha256": manifest["manifest_sha256"], "engines": ["mck", "ob_cpp"],
        "selected_case_count": 1, "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {"case_id": "ethane_c2h6", "family": "aliphatic", "engine": "mck", "status": "unavailable", "reason": "missing"},
            {"case_id": "ethane_c2h6", "family": "aliphatic", "engine": "ob_cpp", "status": "incomplete", "reason": "mapping"},
        ],
    }
    report = score_results(manifest, result)
    assert report["strict_pass"] is False
    assert all(row["status"] == "NOT-COMPARABLE" for row in report["scorecard"])
