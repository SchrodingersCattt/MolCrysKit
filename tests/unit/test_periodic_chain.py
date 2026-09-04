from __future__ import annotations

import json

import numpy as np
import pytest

from molcrys_kit.analysis.periodic_validation import validate_periodic_bundle
from molcrys_kit.io.periodic_bundle import read_periodic_bundle, write_periodic_bundle
from molcrys_kit.operations.periodic_chain import build_periodic_chains
from molcrys_kit.structures.periodic_geometry import BoundaryPort, ChainSpec, ConnectionRule, FragmentTemplate, ScrewSpec


def _template() -> FragmentTemplate:
    return FragmentTemplate("atom", ("C",), ((0.0, 0.0, 0.0),), (BoundaryPort("join", (0.0, 0.0, 0.0), faces=("x0", "x1")),))


def _rule() -> ConnectionRule:
    return ConnectionRule("join", "atom", "join", "atom", "join", allowed_image_shifts=((0, 0, 0), (1, 0, 0)), distance_range=(0.0, 6.0))


def test_zero_and_nonzero_winding_from_graph():
    cell = np.diag([10.0, 10.0, 10.0])
    zero = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, True), ChainSpec(("atom", "atom"), instance_centers=((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), target_winding=(0, 0, 0)))
    nonzero = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, True), ChainSpec(("atom", "atom"), instance_centers=((0.0, 0.5, 0.5), (0.5, 0.5, 0.5)), target_winding=(1, 0, 0)))
    assert zero.graph.winding == (0, 0, 0)
    assert nonzero.graph.winding == (1, 0, 0)
    assert zero.graph.cycle_rank == nonzero.graph.cycle_rank == 1


def test_triclinic_partial_pbc_and_round_trip(tmp_path):
    cell = np.array([[8.0, 0.0, 0.0], [1.2, 7.5, 0.0], [0.1, 0.3, 9.0]])
    bundle = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, False), ChainSpec(("atom",), instance_centers=((0.3, 0.4, 0.2),), min_distance=0.5))
    validate_periodic_bundle(bundle)
    structure, sidecar = write_periodic_bundle(bundle, tmp_path / "bundle")
    assert structure.name == "structure.cif"
    atoms, metadata = read_periodic_bundle(structure, sidecar)
    assert len(atoms) == 1
    validate_periodic_bundle(atoms, metadata)
    assert metadata["periodic_graph"]["winding_cycles"] == [[0, 0, 0]]
    assert json.loads(sidecar.read_text())["files"]["structure_sha256"]


@pytest.mark.parametrize("format_name", ("cif", "poscar", "xyz", "extxyz"))
def test_supported_structure_formats(tmp_path, format_name):
    cell = np.diag([10.0, 10.0, 10.0])
    bundle = build_periodic_chains({"atom": _template()}, (_rule(),), cell, (True, True, True), ChainSpec(("atom",), instance_centers=((0.3, 0.4, 0.5),), min_distance=0.5))
    structure, sidecar = write_periodic_bundle(bundle, tmp_path / format_name, format=format_name)
    atoms, metadata = read_periodic_bundle(structure, sidecar)
    assert len(atoms) == 1
    validate_periodic_bundle(atoms, metadata)
    assert metadata["files"]["format"] == format_name


def test_screw_compatibility_and_no_implicit_expansion():
    cell = np.diag([10.0, 10.0, 10.0])
    template = FragmentTemplate("atom", ("C",), ((2.0, 0.0, 0.0),), (BoundaryPort("join", (2.0, 0.0, 0.0)),))
    compatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0)
    bundle = build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=compatible, min_distance=0.5))
    assert bundle.graph.winding == (0, 0, 0)
    incompatible = ScrewSpec(4, (0.0, 0.0, 1.0), 90.0, translation=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        build_periodic_chains({"atom": template}, (), cell, (True, True, True), ChainSpec(("atom",) * 4, closure="screw", screw=incompatible))
