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


def _periodic_position(atoms, index: int, image_shift=(0, 0, 0)) -> np.ndarray:
    return np.asarray(atoms.positions[index]) + np.asarray(image_shift, dtype=float) @ atoms.cell.array


def _periodic_distance(atoms, left: int, right: int, right_image_shift=(0, 0, 0)) -> float:
    return float(np.linalg.norm(_periodic_position(atoms, right, right_image_shift) - _periodic_position(atoms, left)))


def _periodic_angle(
    atoms,
    left: int,
    center: int,
    right: int,
    left_image_shift=(0, 0, 0),
    right_image_shift=(0, 0, 0),
) -> float:
    first = _periodic_position(atoms, left, left_image_shift) - _periodic_position(atoms, center)
    second = _periodic_position(atoms, right, right_image_shift) - _periodic_position(atoms, center)
    cosine = np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


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


def test_sidecar_rejects_reordered_symbols_and_metadata_mismatch(tmp_path):
    cell = np.diag([10.0, 10.0, 10.0])
    template, rule = _valid_repeat()
    bundle = build_periodic_chains({"repeat": template}, (rule,), cell, (True, True, True), ChainSpec(("repeat",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5))
    structure, sidecar = write_periodic_bundle(bundle, tmp_path / "bundle", format="extxyz")
    payload = json.loads(sidecar.read_text())
    payload["atom_records"][0]["symbol"] = "O"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="symbols"):
        read_periodic_bundle(structure, sidecar)

    payload["atom_records"][0]["symbol"] = "C"
    payload["atom_records"][0]["chain_id"] = 99
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="array 'chain_id'"):
        read_periodic_bundle(structure, sidecar)


def test_zero_winding_same_port_self_loop_is_rejected():
    with pytest.raises(ValueError, match="degenerate one-instance"):
        build_periodic_chains(
            {"atom": _template()}, (_rule(),), np.diag([10.0, 10.0, 10.0]),
            (True, True, True),
            ChainSpec(("atom",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5),
        )


def test_multiple_chains_use_explicit_centers_and_are_recorded():
    cell = np.diag([12.0, 10.0, 10.0])
    template, rule = _valid_repeat()
    bundle = build_periodic_chains(
        {"repeat": template},
        (rule,),
        cell,
        (True, True, True),
        ChainSpec(
            ("repeat",),
            chain_count=2,
            target_winding=(0, 0, 0),
            instance_centers=((0.0, 0.0, 0.0),),
            chain_centers=((0.1, 0.2, 0.5), (0.1, 0.7, 0.5)),
            min_distance=0.5,
        ),
    )
    assert len(bundle.atoms) == 4
    assert set(bundle.atoms.arrays["chain_id"].tolist()) == {0, 1}
    assert bundle.graph.cycle_rank == 2
    assert bundle.metadata["chain_count"] == 2


def test_skew_cell_collision_is_not_hidden_by_fractional_binning():
    cell = np.array([[10.0, 0.0, 0.0], [9.9, 1.0, 0.0], [0.0, 0.0, 10.0]])
    first_fractional = np.array([0.1, 0.1, 0.0])
    second_fractional = np.array([0.5, 0.7, 0.0])
    first = first_fractional @ cell
    second = second_fractional @ cell
    assert np.linalg.norm((second - first) - np.array([0.0, 1.0, 0.0]) @ cell) < 0.5
    template = FragmentTemplate(
        "skew",
        ("C", "C"),
        (tuple(first), tuple(second)),
        (BoundaryPort("in", tuple(first)), BoundaryPort("out", tuple(second))),
    )
    rule = ConnectionRule(
        "close",
        "skew",
        "out",
        "skew",
        "in",
        allowed_image_shifts=((0, 0, 0),),
        distance_range=(0.0, 20.0),
    )
    with pytest.raises(ValueError, match="periodic collision"):
        build_periodic_chains(
            {"skew": template},
            (rule,),
            cell,
            (True, True, True),
            ChainSpec(
                ("skew",),
                instance_centers=((0.0, 0.0, 0.0),),
                min_distance=1.0,
            ),
        )


@pytest.mark.parametrize(
    ("fixture", "formula", "atom_count", "winding"),
    (
        ("periodic_winding_regression", "C2", 2, (1, 0, 0)),
        ("red_phosphorus_local_chain", "P16", 16, (1, 0, 0)),
        ("polyethylene_like_chain", "C4H8", 12, (1, 0, 0)),
        ("trigonal_se_chain", "Se6", 6, (0, 0, 1)),
    ),
)
def test_periodic_fixture_requests_are_non_degenerate_and_round_trip(
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


def _build_fixture(fixture: str):
    request = json.loads((Path("examples/periodic_chains") / f"{fixture}.json").read_text(encoding="utf-8"))
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
    return request, build_periodic_chains(templates, rules, request["cell"], request.get("pbc", (True, True, True)), spec).atoms


def test_polyethylene_fixture_has_tetrahedral_local_geometry():
    _, atoms = _build_fixture("polyethylene_like_chain")
    assert _periodic_distance(atoms, 0, 1) == pytest.approx(1.5415, abs=1e-4)
    assert _periodic_distance(atoms, 1, 0, (1, 0, 0)) == pytest.approx(1.5415, abs=1e-4)
    assert _periodic_angle(atoms, 1, 0, 1, (-1, 0, 0)) == pytest.approx(112.0, abs=0.05)
    assert _periodic_angle(atoms, 0, 1, 0, (1, 0, 0)) == pytest.approx(112.0, abs=0.05)
    for carbon, hydrogens in ((0, (2, 3)), (1, (4, 5))):
        for hydrogen in hydrogens:
            assert _periodic_distance(atoms, carbon, hydrogen) == pytest.approx(1.09, abs=1e-4)
        carbon_neighbors = ((1, (0, 0, 0)), (1, (-1, 0, 0))) if carbon == 0 else ((0, (0, 0, 0)), (0, (1, 0, 0)))
        for hydrogen in hydrogens:
            for neighbor, image in carbon_neighbors:
                angle = _periodic_angle(atoms, hydrogen, carbon, neighbor, (0, 0, 0), image)
                assert 105.0 <= angle <= 114.0


def test_red_phosphorus_fixture_has_closed_p_p_zigzag_geometry():
    _, atoms = _build_fixture("red_phosphorus_local_chain")
    for index in range(8):
        next_index = (index + 1) % 8
        next_shift = (1, 0, 0) if index == 7 else (0, 0, 0)
        assert _periodic_distance(atoms, index, next_index, next_shift) == pytest.approx(2.21, abs=1e-4)
        previous_index = (index - 1) % 8
        previous_shift = (-1, 0, 0) if index == 0 else (0, 0, 0)
        angle = _periodic_angle(atoms, previous_index, index, next_index, previous_shift, next_shift)
        assert angle == pytest.approx(100.0, abs=0.05)


def test_trigonal_selenium_fixture_has_helical_geometry():
    _, atoms = _build_fixture("trigonal_se_chain")
    for index in range(3):
        next_index = (index + 1) % 3
        next_shift = (0, 0, 1) if index == 2 else (0, 0, 0)
        assert _periodic_distance(atoms, index, next_index, next_shift) == pytest.approx(2.3672, abs=1e-3)
    for index in range(3):
        previous_index = (index - 1) % 3
        next_index = (index + 1) % 3
        previous_shift = (0, 0, -1) if index == 0 else (0, 0, 0)
        next_shift = (0, 0, 1) if index == 2 else (0, 0, 0)
        assert _periodic_angle(atoms, previous_index, index, next_index, previous_shift, next_shift) == pytest.approx(103.224, abs=0.1)


def test_screw_compatibility_and_no_implicit_expansion():
    cell = np.diag([10.0, 10.0, 10.0])
    template = FragmentTemplate("atom", ("C",), ((2.0, 0.0, 0.0),), (BoundaryPort("join", (2.0, 0.0, 0.0)),))
    compatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0)
    bundle = build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=compatible, min_distance=0.5))
    assert bundle.graph.winding == (0, 0, 0)
    incompatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0, translation=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=incompatible))
