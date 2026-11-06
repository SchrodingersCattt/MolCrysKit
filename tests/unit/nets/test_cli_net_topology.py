"""CLI JSON for periodic-net topology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from molcrys_kit.cli import main

REPO_ROOT = Path(__file__).resolve().parents[3]
CU = REPO_ROOT / "examples" / "Cu.cif"
NACL = REPO_ROOT / "examples" / "NaCl.cif"


def _square_graph() -> dict[str, object]:
    return {
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            ["ab", "a", "b", [0, 0, 0]],
            ["bc", "b", "c", [0, 0, 0]],
            ["cd", "c", "d", [0, 0, 0]],
            ["da", "d", "a", [0, 0, 0]],
        ],
        "pbc": [True, True, True],
    }


def test_graph_json_reports_cycle_rank_separately_from_periodic_rank(tmp_path: Path):
    path = tmp_path / "square.json"
    path.write_text(json.dumps(_square_graph()), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["analyze", "net-topology", "--graph", str(path), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "mck.net_topology/v1"
    candidate = next(
        item
        for proposal in payload["proposals"]
        if proposal["proposal_id"] == payload["primary_proposal_id"]
        for item in proposal["candidates"]
    )
    assert candidate["periodic_rank"] == 0
    assert candidate["quotient_cycle_rank"] == 1
    assert candidate["topology"]["status"] == "identified"
    assert candidate["topology"]["proof"] == "explicit_isomorphism"
    forbidden = ("sql", "dia", "pcu", "hcb", "srs")
    assert all(name not in result.output for name in forbidden)


def test_crystal_text_output_has_no_prototype_name():
    result = CliRunner().invoke(main, ["analyze", "net-topology", str(CU)])
    assert result.exit_code == 0, result.output
    assert "periodic_rank=" in result.output
    assert "topology=" in result.output
    for name in ("sql", "dia", "pcu", "hcb", "srs"):
        assert name not in result.output


def test_nacl_contacts_are_a_second_public_crystal_path():
    result = CliRunner().invoke(
        main,
        [
            "analyze",
            "net-topology",
            str(NACL),
            "--nodes",
            "atom_sites",
            "--edges",
            "contacts",
            "--cutoff",
            "3.5",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "mck.net_topology/v1"
    candidate = next(
        item
        for proposal in payload["proposals"]
        if proposal["proposal_id"] == payload["primary_proposal_id"]
        for item in proposal["candidates"]
    )
    assert candidate["edge_ids"]
    assert candidate["periodic_rank"] == 3
    assert isinstance(candidate["quotient_cycle_rank"], int)
    assert candidate["topology"]["status"] in {"identified", "undetermined"}
    for name in ("sql", "dia", "pcu", "hcb", "srs"):
        assert name not in result.output


def test_policy_and_crystal_flags_override_and_reject_conflicts(tmp_path: Path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"nodes": {"kind": "molecular_units"}, "edges": {"cutoff_A": 4.0}}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main,
        [
            "analyze",
            "net-topology",
            str(CU),
            "--policy",
            str(policy),
            "--nodes",
            "atom_sites",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["settings"]["site_graph"]["node_policy"]["kind"] == "atom_sites"
    assert payload["settings"]["site_graph"]["edge_policy"]["cutoff_A"] == 4.0
    both = CliRunner().invoke(
        main,
        ["analyze", "net-topology", str(CU), "--graph", str(policy)],
    )
    assert both.exit_code != 0
    missing = CliRunner().invoke(main, ["analyze", "net-topology"])
    assert missing.exit_code != 0
    graph = tmp_path / "square.json"
    graph.write_text(json.dumps(_square_graph()), encoding="utf-8")
    rejected = CliRunner().invoke(
        main,
        ["analyze", "net-topology", "--graph", str(graph), "--nodes", "atom_sites"],
    )
    assert rejected.exit_code != 0


@pytest.mark.parametrize(
    "edge",
    [
        {"source": "a", "target": "a", "image_shift": [0.9, 0, 0]},
        {"source": "a", "target": "a", "image_shift": [True, 0, 0]},
        {"edge_id": "loop", "source": "a", "target": "a", "image_shift": [0.9, 0, 0]},
        ["a", "a", [0.9, 0, 0]],
        ["a", "a", [True, 0, 0]],
        ["loop", "a", "a", [0.9, 0, 0]],
        ["loop", "a", "a", [True, 0, 0]],
    ],
)
def test_graph_json_rejects_fractional_and_boolean_shifts(tmp_path: Path, edge):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"nodes": ["a"], "edges": [edge]}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main, ["analyze", "net-topology", "--graph", str(path), "--json"]
    )
    assert result.exit_code != 0
    assert "exact integer" in result.output


def test_yaml_policy_is_refused_when_pyyaml_is_absent(tmp_path: Path, monkeypatch):
    import builtins

    policy = tmp_path / "policy.yaml"
    policy.write_text("nodes:\n  kind: atom_sites\n", encoding="utf-8")
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yaml":
            raise ImportError("yaml blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = CliRunner().invoke(
        main,
        ["analyze", "net-topology", str(CU), "--policy", str(policy)],
    )
    assert result.exit_code != 0
    assert "PyYAML" in result.output


def test_package_exports_match_the_root_helpers():
    import molcrys_kit as mck
    from molcrys_kit import nets

    assert mck.build_periodic_site_graph is nets.build_periodic_site_graph
    assert mck.decompose_periodic_nets is nets.decompose_periodic_nets
    assert mck.analyze_periodic_net_system is nets.analyze_periodic_net_system
    assert mck.nets is nets
