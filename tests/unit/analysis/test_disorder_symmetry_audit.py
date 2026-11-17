from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.analysis.disorder.symmetry_audit import (
    audit_parent_symmetry,
    replica_respects_parent_symmetry,
)
from molcrys_kit.cli import main
from molcrys_kit.io.cif import DisorderInfo
from molcrys_kit.structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
)


CIF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cif")
)
SYNTHETIC = Path(CIF_DATA_DIR) / "synthetic_parent_symmetry_disorder.cif"


def _pm_symmetry() -> CrystalSymmetry:
    identity = FractionalAffineOperation(
        np.eye(3), np.zeros(3), xyz="x,y,z", index=0, source="test"
    )
    mirror = FractionalAffineOperation(
        np.diag([1.0, -1.0, 1.0]),
        np.zeros(3),
        xyz="x,-y,z",
        index=1,
        source="test",
    )
    return CrystalSymmetry(
        operations=(identity, mirror),
        space_group_number=6,
        space_group_symbol="P m",
        source="test",
    )


def _hand_built_info(*, keep_consistent: bool) -> tuple[DisorderInfo, list[int]]:
    # Two ASU alternatives expanded by Pm. Same-group copies close under the
    # mirror; mixed-group selections do not.
    labels = ["C1A", "C1A", "C1B", "C1B"]
    symbols = ["C", "C", "C", "C"]
    frac = np.array(
        [
            [0.20, 0.30, 0.40],
            [0.20, 0.70, 0.40],
            [0.20, 0.35, 0.40],
            [0.20, 0.65, 0.40],
        ],
        dtype=float,
    )
    info = DisorderInfo(
        labels=labels,
        symbols=symbols,
        frac_coords=frac,
        occupancies=[0.6, 0.6, 0.4, 0.4],
        disorder_groups=[1, 1, 2, 2],
        assemblies=["A", "A", "A", "A"],
        sym_op_indices=[0, 1, 0, 1],
        asym_id=[0, 0, 0, 0],
        crystal_symmetry=_pm_symmetry(),
    )
    kept = [0, 1] if keep_consistent else [0, 3]
    return info, kept


def test_audit_marks_consistent_same_component_selection():
    info, kept = _hand_built_info(keep_consistent=True)
    lattice = np.eye(3) * 10.0
    audit = audit_parent_symmetry(info, kept, lattice)

    assert audit.parent_space_group_number == 6
    assert audit.respects_parent_group is True
    assert all(operation.closes for operation in audit.operations)
    assert replica_respects_parent_symmetry(info, kept, lattice)


def test_audit_marks_inconsistent_mixed_component_selection():
    info, kept = _hand_built_info(keep_consistent=False)
    lattice = np.eye(3) * 10.0
    audit = audit_parent_symmetry(info, kept, lattice)

    assert audit.respects_parent_group is False
    assert any(not operation.closes for operation in audit.operations)
    assert not replica_respects_parent_symmetry(info, kept, lattice)


def test_provenance_includes_symmetry_audit_by_default():
    crystals = generate_ordered_replicas_from_disordered_sites(
        str(SYNTHETIC),
        method="enumerate",
        generate_count=4,
        coupled=False,
    )
    assert crystals
    provenance = crystals[0].disorder_provenance
    assert provenance is not None
    assert provenance.symmetry_audit is not None
    assert provenance.parent_space_group_number == 6
    assert provenance.respects_parent_group is not None
    payload = provenance.to_dict()
    assert "symmetry_audit" in payload
    assert "operations" in payload["symmetry_audit"]


def test_symmetry_consistent_filters_to_parent_closed_replicas():
    all_replicas = generate_ordered_replicas_from_disordered_sites(
        str(SYNTHETIC),
        method="enumerate",
        generate_count=0,
        coupled=False,
    )
    consistent = generate_ordered_replicas_from_disordered_sites(
        str(SYNTHETIC),
        method="enumerate",
        generate_count=0,
        coupled=False,
        symmetry_consistent=True,
    )
    assert consistent
    assert len(consistent) <= len(all_replicas)
    assert all(
        crystal.disorder_provenance.respects_parent_group for crystal in consistent
    )


def test_optimal_provenance_notes_possible_subgroup():
    crystal = generate_ordered_replicas_from_disordered_sites(
        str(SYNTHETIC),
        method="optimal",
    )[0]
    notes = crystal.disorder_provenance.notes
    assert any("optimal" in note for note in notes)


def test_operate_disorder_report_writes_json(tmp_path: Path):
    output = tmp_path / "ordered.cif"
    report = tmp_path / "audit.json"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "disorder",
            str(SYNTHETIC),
            "-o",
            str(output),
            "--method",
            "enumerate",
            "--count",
            "4",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.exists()
    rows = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows
    assert "symmetry_audit" in rows[0]
    assert "respects_parent_group" in rows[0]


def test_operate_disorder_symmetry_consistent_option(tmp_path: Path):
    output = tmp_path / "consistent.cif"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "disorder",
            str(SYNTHETIC),
            "-o",
            str(output),
            "--method",
            "enumerate",
            "--symmetry-consistent",
        ],
    )
    assert result.exit_code == 0, result.output


def test_skew_cell_uses_the_short_periodic_image():
    identity = FractionalAffineOperation(
        np.eye(3),
        np.array([0.49, 0.49, 0.0]),
        xyz="x+0.49,y+0.49,z",
        index=0,
        source="test",
    )
    info = DisorderInfo(
        labels=["C1"],
        symbols=["C"],
        frac_coords=np.zeros((1, 3)),
        occupancies=[1.0],
        disorder_groups=[0],
        assemblies=[""],
        crystal_symmetry=CrystalSymmetry(
            operations=(identity,),
            space_group_number=1,
            space_group_symbol="P 1",
            source="test",
        ),
    )
    lattice = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    audit = audit_parent_symmetry(info, [0], lattice, tolerance_angstrom=0.2)
    assert audit.operations[0].closes
    assert audit.operations[0].residual_angstrom_max < 0.1
    assert audit.respects_parent_group is True


def test_mapping_must_be_bijective():
    # A crystallographic isometry can put two kept atoms inside the tolerance
    # ball of one target. Nearest neighbours then collide and the third atom
    # is missed. Closure requires a one-to-one match.
    cell = 10.0
    identity = FractionalAffineOperation(
        np.eye(3), np.zeros(3), xyz="x,y,z", index=0, source="test"
    )
    rotation = np.array([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    folded = FractionalAffineOperation(
        rotation,
        np.array([0.075, 0.1, 0.0]),
        xyz="x-z+0.075,-x+y+0.1,z",
        index=1,
        source="test",
    )
    frac = np.array([[1.0, 1.0, 1.0], [1.3, 1.0, 1.0], [1.0, 4.0, 1.0]]) / cell
    info = DisorderInfo(
        labels=["C1", "C2", "C3"],
        symbols=["C", "C", "C"],
        frac_coords=frac,
        occupancies=[1.0, 1.0, 1.0],
        disorder_groups=[0, 0, 0],
        assemblies=["", "", ""],
        crystal_symmetry=CrystalSymmetry(
            operations=(identity, folded),
            space_group_number=1,
            space_group_symbol="P 1",
            source="test",
        ),
    )
    audit = audit_parent_symmetry(
        info, [0, 1, 2], np.eye(3) * cell, tolerance_angstrom=0.35
    )
    assert audit.operations[0].closes
    assert audit.operations[1].closes is False
    assert audit.respects_parent_group is False


def test_occupancy_sum_of_parts_does_not_alias_another_component():
    info, _kept = _hand_built_info(keep_consistent=True)
    info.occupancies = [0.5, 0.5, 0.3, 0.3, 0.2, 0.2]
    info.disorder_groups = [1, 1, 2, 2, 3, 3]
    info.labels = ["C1A", "C1A", "C1B", "C1B", "C1C", "C1C"]
    info.symbols = ["C"] * 6
    info.assemblies = ["A"] * 6
    info.asym_id = [0] * 6
    info.frac_coords = np.vstack([info.frac_coords, info.frac_coords[:2]])
    info.sym_op_indices = [0, 1, 0, 1, 0, 1]
    audit = audit_parent_symmetry(info, [2, 3, 4, 5], np.eye(3) * 10.0)
    assert audit.occupancy_closure
    assert audit.occupancy_closure[0].closes is False
    assert audit.respects_parent_group is False


def test_nonidentity_at_index_zero_is_still_noted():
    mirror = FractionalAffineOperation(
        np.diag([1.0, -1.0, 1.0]),
        np.zeros(3),
        xyz="x,-y,z",
        index=0,
        source="test",
    )
    identity = FractionalAffineOperation(
        np.eye(3), np.zeros(3), xyz="x,y,z", index=1, source="test"
    )
    info, kept = _hand_built_info(keep_consistent=False)
    info.crystal_symmetry = CrystalSymmetry(
        operations=(mirror, identity),
        space_group_number=6,
        space_group_symbol="P m",
        source="test",
    )
    audit = audit_parent_symmetry(info, kept, np.eye(3) * 10.0)
    assert any("subgroup" in note for note in audit.notes)
