"""Contract tests for the auditable #174 benchmark package."""

from benchmarks.bond_order.common import (
    _edge_key,
    call_mck,
    load_manifest,
    normalise_prediction,
    verify_manifest,
)
from benchmarks.bond_order.score import score_prediction, score_results


def test_frozen_manifest_verifies_and_has_required_named_cases():
    manifest = load_manifest()
    ok, _ = verify_manifest(manifest)
    assert ok
    ids = {case["case_id"] for case in manifest["cases"]}
    assert {
        "c13h28_long_chain",
        "first_64_premature_cut",
        "input_order_invariance",
        "missing_charge_and_h",
        "shifted_pbc_cc",
    } <= ids
    assert (
        manifest["openbabel_baseline"]["commit"]
        == "0e94434fa75c9f61095023e3c12e0d5f2ac035ff"
    )


def test_mck_adapter_scores_entity_bonds_not_provenance():
    """``bond_evidence`` is provenance and must not hide predicted orders."""
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["case_id"] == "ethane_c2h6")
    result = call_mck(case)
    assert result["status"] == "ok"
    assert result["prediction"]["bond_orders"]["C1--C2"] == 1.0
    assert result["prediction"]["bond_orders"]["C1--H1"] == 1.0


def test_declared_kekule_forms_are_edge_equivalent():
    manifest = load_manifest()
    case = next(
        c for c in manifest["cases"] if c["case_id"] == "benzene_aromatic_resonance"
    )
    flipped = {
        key: 3 - int(value) for key, value in case["expected"]["bond_orders"].items()
    }
    prediction = {
        "bond_orders": flipped,
        "aromatic_bonds": list(case["expected"]["aromatic_bonds"]),
        "formal_charges": dict(case["expected"]["formal_charges"]),
        "implicit_hydrogens": dict(case["expected"]["implicit_hydrogens"]),
    }
    result = score_prediction(case, prediction)
    assert result["edge_exact"] is True
    assert result["known_correct"] == result["known_total"]
    ethane = next(c for c in manifest["cases"] if c["case_id"] == "ethane_c2h6")
    wrong = score_prediction(
        ethane,
        {
            "bond_orders": {
                "C1--C2": 2,
                "C1--H1": 1,
                "C1--H2": 1,
                "C1--H3": 1,
                "C2--H4": 1,
                "C2--H5": 1,
                "C2--H6": 1,
            }
        },
    )
    assert wrong["edge_exact"] is False


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
        "suite": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "engines": ["mck", "ob_cpp"],
        "selected_case_count": 1,
        "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {
                "case_id": "ethane_c2h6",
                "family": "aliphatic",
                "engine": "mck",
                "status": "unavailable",
                "reason": "missing",
            },
            {
                "case_id": "ethane_c2h6",
                "family": "aliphatic",
                "engine": "ob_cpp",
                "status": "incomplete",
                "reason": "mapping",
            },
        ],
    }
    report = score_results(manifest, result)
    assert report["strict_pass"] is False
    assert all(row["status"] == "NOT-COMPARABLE" for row in report["scorecard"])
    assert all(row["blocks_strict_pass"] is False for row in report["scorecard"])


def test_functional_strict_pass_does_not_require_open_babel():
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["case_id"] == "ethane_c2h6")
    result = {
        "suite": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "engines": ["mck", "ob_cpp"],
        "selected_case_count": 1,
        "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {
                "case_id": "ethane_c2h6",
                "family": "aliphatic",
                "engine": "mck",
                "status": "ok",
                "prediction": {
                    "bond_orders": dict(case["expected"]["bond_orders"]),
                },
            },
            {
                "case_id": "ethane_c2h6",
                "family": "aliphatic",
                "engine": "ob_cpp",
                "status": "incomplete",
                "reason": "mapping",
            },
        ],
    }
    report = score_results(manifest, result)
    assert report["strict_pass"] is True
    assert all(gate["status"] == "PASS" for gate in report["functional_gates"])
    assert all(row["status"] == "NOT-COMPARABLE" for row in report["scorecard"])
    assert all(row["scope"] == "deferred-superiority" for row in report["scorecard"])


def test_false_assertion_fails_functional_strict_gate():
    manifest = load_manifest()
    result = {
        "suite": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "engines": ["mck"],
        "selected_case_count": 1,
        "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {
                "case_id": "zinc_nitrogen_coordination",
                "family": "metal_semantics",
                "engine": "mck",
                "status": "ok",
                "prediction": {"bond_orders": {"Zn1--N1": 1}},
            }
        ],
    }
    report = score_results(manifest, result)
    gates = {gate["metric"]: gate["status"] for gate in report["functional_gates"]}
    assert gates["mck_observations"] == "PASS"
    assert gates["mck_false_assertions"] == "FAIL"
    assert report["strict_pass"] is False


def test_strict_gate_rejects_missing_selected_mck_cases():
    manifest = load_manifest()
    result = {
        "suite": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "engines": ["mck"],
        "selected_case_count": 2,
        "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {
                "case_id": "ethane_c2h6",
                "engine": "mck",
                "status": "ok",
                "prediction": {
                    "bond_orders": dict(
                        next(c for c in manifest["cases"] if c["case_id"] == "ethane_c2h6")["expected"]["bond_orders"]
                    )
                },
            }
        ],
    }
    report = score_results(manifest, result)
    gates = {gate["metric"]: gate["status"] for gate in report["functional_gates"]}
    assert gates["mck_case_coverage"] == "FAIL"
    assert report["strict_pass"] is False


def test_strict_gate_rejects_all_abstention_known_truth_run():
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["case_id"] == "ethane_c2h6")
    result = {
        "suite": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "engines": ["mck"],
        "selected_case_count": 1,
        "environment": {"processes": 1, "python": "test"},
        "raw_results": [
            {
                "case_id": case["case_id"],
                "engine": "mck",
                "status": "ok",
                "prediction": {"bond_orders": {key: None for key in case["expected"]["bond_orders"]}},
            }
        ],
    }
    report = score_results(manifest, result)
    gates = {gate["metric"]: gate["status"] for gate in report["functional_gates"]}
    assert gates["mck_known_truth"] == "FAIL"
    assert report["strict_pass"] is False


def test_periodic_multiedges_keep_distinct_prediction_keys():
    bonds = [
        {"atom1_id": "C1", "atom2_id": "C2", "order": 1, "atom2_image_shift": (0, 0, 0)},
        {"atom1_id": "C1", "atom2_id": "C2", "order": 2, "atom2_image_shift": (1, 0, 0)},
    ]
    value = type("Result", (), {"entity": type("Entity", (), {"bonds": bonds, "atoms": ()})(), "status": "inferred"})()
    prediction = normalise_prediction(value, {"bonds": []})
    assert set(prediction["bond_orders"]) == {
        _edge_key("C1", "C2"),
        _edge_key("C1", "C2", (1, 0, 0)),
    }


def test_periodic_multiedges_are_scored_independently():
    zero = _edge_key("C1", "C2")
    image = _edge_key("C1", "C2", (1, 0, 0))
    case = {
        "bonds": [
            {"a": "C1", "b": "C2", "image_shift": [0, 0, 0]},
            {"a": "C1", "b": "C2", "image_shift": [1, 0, 0]},
        ],
        "expected": {"bond_orders": {zero: 1, image: 2}},
    }
    result = score_prediction(case, {"bond_orders": {zero: 1, image: 2}})
    assert result["known_total"] == 2
    assert result["known_correct"] == 2
    assert result["edge_exact"] is True
