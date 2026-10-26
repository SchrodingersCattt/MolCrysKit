"""Synthetic truth only: this suite never imports CCDC or downloads CSD data."""

from copy import deepcopy
from types import SimpleNamespace

from ase import Atoms
import networkx as nx
import numpy as np
import pytest

from benchmarks.csd_roundtrip.common import digest, public_result, verify_manifest
from benchmarks.csd_roundtrip.match import (
    compare_molecules,
    molecule_mapping,
    parse_operation,
)
from benchmarks.csd_roundtrip.report import summarize
from benchmarks.csd_roundtrip.run import compatible_resume
from benchmarks.csd_roundtrip.worker import CheckpointStages, snapshot_mck


def synthetic_molecule(
    symbols=("C", "O", "N"),
    positions=((9.7, 0.0, 0.0), (10.9, 0.0, 0.0), (9.7, 1.4, 0.0)),
):
    return {
        "atoms": [
            {
                "symbol": s,
                "xyz": list(p),
                "occupancy": 1.0,
                "formal_charge": 0,
                "isotope": None,
                "stereo": None,
            }
            for s, p in zip(symbols, positions)
        ],
        "bonds": [[0, 1, "Single"], [0, 2, "Single"]],
    }


def reorder(molecule, order):
    inverse = {old: new for new, old in enumerate(order)}
    return {
        "atoms": [deepcopy(molecule["atoms"][i]) for i in order],
        "bonds": [[inverse[a], inverse[b], kind] for a, b, kind in molecule["bonds"]],
    }


def test_atom_and_molecule_permutations_and_coherent_boundary_shift():
    cell = np.diag([10.0, 11.0, 12.0])
    original = synthetic_molecule()
    candidate = reorder(original, [2, 0, 1])
    for atom in candidate["atoms"]:
        atom["xyz"] = (np.array(atom["xyz"]) + cell[0] - cell[1]).tolist()
    ion = {
        "atoms": [{"symbol": "Cl", "xyz": [3.0, 3.0, 3.0], "occupancy": 1.0}],
        "bonds": [],
    }
    assert (
        compare_molecules([original, ion], [ion, candidate], cell)["status"] == "pass"
    )


def test_independent_atom_wrapping_is_not_whole_molecule_match():
    original = synthetic_molecule()
    candidate = deepcopy(original)
    candidate["atoms"][1]["xyz"][0] -= 10.0
    assert molecule_mapping(original, candidate, np.eye(3) * 10) is None


def test_nonorthogonal_cell_integer_translation():
    lattice = np.array([[10.0, 0.0, 0.0], [9.8, 2.0, 0.0], [2.0, 1.0, 12.0]])
    original = synthetic_molecule()
    candidate = deepcopy(original)
    for atom in candidate["atoms"]:
        atom["xyz"] = (
            np.array(atom["xyz"]) + 3 * lattice[0] - 4 * lattice[1] + lattice[2]
        ).tolist()
    assert molecule_mapping(original, candidate, lattice) is not None


@pytest.mark.parametrize(
    "defect",
    ["missing_ion", "graph", "stereo", "isotope", "charge", "occupancy", "bond_order"],
)
def test_adversarial_changes_are_rejected(defect):
    original = synthetic_molecule()
    candidate = deepcopy(original)
    if defect == "missing_ion":
        assert (
            compare_molecules([original, original], [candidate], np.eye(3) * 20)[
                "status"
            ]
            == "mismatch"
        )
        return
    if defect == "graph":
        candidate["bonds"] = [[0, 1, "Single"], [1, 2, "Single"]]
    elif defect == "bond_order":
        candidate["bonds"][0][2] = "Double"
    elif defect == "occupancy":
        candidate["atoms"][0]["occupancy"] = 0.5
    else:
        key, value = {
            "stereo": ("stereo", "R"),
            "isotope": ("isotope", 13),
            "charge": ("formal_charge", 1),
        }[defect]
        candidate["atoms"][0][key] = value
    assert molecule_mapping(original, candidate, np.eye(3) * 20, chemistry=True) is None


def test_parser_cartesian_basis_normalization_is_not_fitting():
    atoms = Atoms("CO", positions=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    atoms._graph = nx.Graph([(0, 1)])
    atoms.get_graph = lambda: atoms._graph
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    lattice = np.diag([10.0, 11.0, 12.0])
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation
    rotated._graph = atoms._graph
    rotated.get_graph = lambda: rotated._graph
    observed = snapshot_mck(
        SimpleNamespace(lattice=lattice @ rotation, molecules=[rotated]), lattice
    )
    assert np.allclose([a["xyz"] for a in observed[0]["atoms"]], atoms.positions)


def test_operation_parser_handles_fractional_translations():
    rotation, shift = parse_operation("-x,1/2+y,1/2-z")
    assert np.array_equal(rotation, np.diag([-1.0, 1.0, -1.0]))
    assert np.allclose(shift, [0.0, 0.5, 0.5])


def test_public_projection_drops_coordinates_logs_and_arbitrary_diagnostics():
    raw = {
        "refcode": "ABCDEF",
        "status": "error",
        "coordinates": [[1, 2, 3]],
        "traceback": "private",
        "diagnostics": {"coordinates": {"xyz": [1, 2, 3]}},
        "stages": {
            "memory": {
                "status": "error",
                "code": "reader_error",
                "coordinates": [[1, 2, 3]],
                "max_A": [1, 2, 3],
            }
        },
    }
    public = public_result(raw)
    assert "coordinates" not in str(public)
    assert "private" not in str(public)
    assert "max_A" not in public["stages"]["memory"]


def test_resume_refuses_different_harness_or_manifest():
    meta = {"manifest_sha256": "a", "harness_sha256": "b"}
    assert compatible_resume(meta, meta)
    assert not compatible_resume({**meta, "harness_sha256": "c"}, meta)
    assert not compatible_resume({**meta, "manifest_sha256": "c"}, meta)


def test_completed_stages_are_checkpointed_before_next_stage():
    checkpoints = []
    stages = CheckpointStages(lambda: checkpoints.append(deepcopy(dict(stages))))
    stages["memory"] = {"status": "pass", "code": "complete"}
    stages["readback"] = {"status": "error", "code": "native_crash"}
    assert checkpoints[0] == {"memory": {"status": "pass", "code": "complete"}}
    assert checkpoints[1]["memory"]["status"] == "pass"


def test_summary_rejects_mixed_harness_versions():
    with pytest.raises(ValueError, match="mixed harness"):
        summarize({"cases": []}, [{"harness_sha256": "a"}, {"harness_sha256": "b"}])


def test_manifest_checksum_is_checked_before_loading_cases():
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest({"sha256": "wrong", "cases": []})


def test_failed_and_missing_cases_stay_in_denominator_and_duplicates_rejected():
    manifest = {
        "sha256": "a",
        "cases": [
            {"refcode": "AAAAAA", "stratum": "multi_ion", "targeted": False},
            {"refcode": "BBBBBB", "stratum": "multi_ion", "targeted": False},
        ],
    }
    row = {
        **manifest["cases"][0],
        "manifest_sha256": "a",
        "status": "error",
        "stages": {},
    }
    completed, summary = summarize(manifest, [row])
    assert summary["groups"]["all"]["total"] == 2
    assert summary["groups"]["all"]["status"] == {"error": 1, "indeterminate": 1}
    assert completed[1]["stages"]["execution"]["code"] == "not_executed"
    with pytest.raises(ValueError, match="duplicate"):
        summarize(manifest, [row, row])
    assert digest(manifest) == digest(deepcopy(manifest))
