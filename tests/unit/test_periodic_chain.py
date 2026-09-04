from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.analysis.periodic_validation import validate_periodic_bundle
from molcrys_kit.cli.periodic_chain_cmd import _rule as load_rule
from molcrys_kit.cli.periodic_chain_cmd import _template as load_template
from molcrys_kit.io.periodic_bundle import read_periodic_bundle, write_periodic_bundle
from molcrys_kit.operations.periodic_chain import build_periodic_chains
from molcrys_kit.structures.periodic_geometry import BoundaryPort, ChainSpec, ConnectionRule, FragmentTemplate, ScrewSpec


def _template() -> FragmentTemplate:
    return FragmentTemplate("atom", ("C",), ((0.0, 0.0, 0.0),), (BoundaryPort("join", (0.0, 0.0, 0.0), faces=("x0", "x1")),))


def _rule() -> ConnectionRule:
    return ConnectionRule("join", "atom", "join", "atom", "join", allowed_image_shifts=((0, 0, 0), (1, 0, 0)), distance_range=(0.0, 6.0))


def _valid_repeat() -> tuple[FragmentTemplate, ConnectionRule]:
    template = FragmentTemplate(
        "repeat",
        ("C", "C"),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (BoundaryPort("in", (0.0, 0.0, 0.0)), BoundaryPort("out", (1.0, 0.0, 0.0))),
        ((0, 1),),
    )
    rule = ConnectionRule(
        "repeat-close", "repeat", "out", "repeat", "in",
        allowed_image_shifts=((0, 0, 0), (1, 0, 0)), distance_range=(0.5, 1.5),
    )
    return template, rule


def test_zero_and_nonzero_winding_from_graph():
    cell = np.diag([10.0, 10.0, 10.0])
    zero = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, True), ChainSpec(("atom", "atom"), instance_centers=((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), target_winding=(0, 0, 0)))
    nonzero = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, True), ChainSpec(("atom", "atom"), instance_centers=((0.0, 0.5, 0.5), (0.5, 0.5, 0.5)), target_winding=(1, 0, 0)))
    assert zero.graph.winding == (0, 0, 0)
    assert nonzero.graph.winding == (1, 0, 0)
    assert zero.graph.cycle_rank == nonzero.graph.cycle_rank == 1


def test_triclinic_partial_pbc_and_round_trip(tmp_path):
    cell = np.array([[8.0, 0.0, 0.0], [1.2, 7.5, 0.0], [0.1, 0.3, 9.0]])
    template, rule = _valid_repeat()
    bundle = build_periodic_chains({"repeat": template}, (rule,), cell, (True, True, False), ChainSpec(("repeat",), instance_centers=((0.3, 0.4, 0.2),), min_distance=0.5))
    validate_periodic_bundle(bundle)
    structure, sidecar = write_periodic_bundle(bundle, tmp_path / "bundle")
    assert structure.name == "structure.cif"
    atoms, metadata = read_periodic_bundle(structure, sidecar)
    assert len(atoms) == 2
    validate_periodic_bundle(atoms, metadata)
    assert metadata["periodic_graph"]["winding_cycles"] == [[0, 0, 0]]
    assert json.loads(sidecar.read_text())["files"]["structure_sha256"]


@pytest.mark.parametrize("format_name", ("cif", "poscar", "xyz", "extxyz"))
def test_supported_structure_formats(tmp_path, format_name):
    cell = np.diag([10.0, 10.0, 10.0])
    template, rule = _valid_repeat()
    bundle = build_periodic_chains({"repeat": template}, (rule,), cell, (True, True, True), ChainSpec(("repeat",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5))
    structure, sidecar = write_periodic_bundle(bundle, tmp_path / format_name, format=format_name)
    atoms, metadata = read_periodic_bundle(structure, sidecar)
    assert len(atoms) == 2
    validate_periodic_bundle(atoms, metadata)
    assert metadata["files"]["format"] == format_name


def test_zero_winding_same_port_self_loop_is_rejected():
    with pytest.raises(ValueError, match="degenerate one-instance"):
        build_periodic_chains(
            {"atom": _template()}, (_rule(),), np.diag([10.0, 10.0, 10.0]),
            (True, True, True),
            ChainSpec(("atom",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5),
        )


@pytest.mark.parametrize(
    ("fixture", "formula", "atom_count", "winding"),
    (
        ("synthetic_nonzero_winding", "C2", 2, (1, 0, 0)),
        ("red_phosphorus_local_chain", "P9", 9, (1, 0, 0)),
        ("polyethylene_like_chain", "C2H4", 6, (1, 0, 0)),
        ("alpha_se_local_chain", "Se3", 3, (0, 0, 1)),
    ),
)
def test_material_fixture_requests_are_non_degenerate_and_round_trip(
    fixture, formula, atom_count, winding, tmp_path
):
    request = json.loads((Path("examples/periodic_chains") / f"{fixture}.json").read_text())
    templates = {item["template_id"]: load_template(item) for item in request["templates"]}
    rules = tuple(load_rule(item) for item in request.get("rules", ()))
    raw = request["spec"]
    screw = ScrewSpec(**raw["screw"]) if raw.get("screw") else None
    spec = ChainSpec(
        tuple(raw["sequence"]), raw.get("chain_count", 1), raw.get("closure", "translation"),
        tuple(raw["target_winding"]) if raw.get("target_winding") is not None else None,
        tuple(tuple(item) for item in raw["instance_centers"]) if raw.get("instance_centers") is not None else None,
        screw, raw.get("seed", 0), raw.get("max_backtracks", 64), raw.get("min_distance", 0.8), raw.get("tolerance", 1e-6),
    )
    bundle = build_periodic_chains(templates, rules, request["cell"], request.get("pbc", (True, True, True)), spec)
    report = validate_periodic_bundle(bundle)
    assert bundle.atoms.get_chemical_formula() == formula
    assert report["atom_count"] == atom_count
    assert tuple(report["winding_cycles"][0]) == winding
    assert all(edge.left_port is not None and edge.right_port is not None for edge in bundle.graph.edges)

    structure, sidecar = write_periodic_bundle(bundle, tmp_path / fixture)
    restored, metadata = read_periodic_bundle(structure, sidecar)
    validate_periodic_bundle(restored, metadata)


def test_screw_compatibility_and_no_implicit_expansion():
    cell = np.diag([10.0, 10.0, 10.0])
    template = FragmentTemplate("atom", ("C",), ((2.0, 0.0, 0.0),), (BoundaryPort("join", (2.0, 0.0, 0.0)),))
    compatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0)
    bundle = build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=compatible, min_distance=0.5))
    assert bundle.graph.winding == (0, 0, 0)
    incompatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0, translation=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=incompatible))
