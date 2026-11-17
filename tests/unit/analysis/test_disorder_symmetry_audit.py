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
    ParentSymmetryNotAssessed,
    _exact_shift_half_width,
    _pairwise_min_image_distances,
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


def _lattice_from_lengths_angles(lengths, angles_deg) -> np.ndarray:
    a_len, b_len, c_len = (float(value) for value in lengths)
    alpha, beta, gamma = np.radians(np.asarray(angles_deg, dtype=float))
    cos_alpha, cos_beta, cos_gamma = np.cos([alpha, beta, gamma])
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1e-8:
        raise ValueError("gamma is degenerate")
    vector_c_x = c_len * cos_beta
    vector_c_y = c_len * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    height_sq = c_len**2 - vector_c_x**2 - vector_c_y**2
    if height_sq <= 0.0:
        raise ValueError("cell angles are inconsistent")
    return np.array(
        [
            [a_len, 0.0, 0.0],
            [b_len * np.cos(gamma), b_len * sin_gamma, 0.0],
            [vector_c_x, vector_c_y, np.sqrt(height_sq)],
        ]
    )


def _brute_image_distance(frac_delta, lattice, half_width: int) -> float:
    rounded = np.asarray(frac_delta, dtype=float) - np.round(frac_delta)
    best = np.inf
    for shift in np.ndindex(2 * half_width + 1, 2 * half_width + 1, 2 * half_width + 1):
        offset = rounded + np.asarray(shift, dtype=float) - half_width
        best = min(best, float(np.linalg.norm(offset @ lattice)))
    return best


def _fixed_window_distance(frac_delta, lattice) -> float:
    return _brute_image_distance(frac_delta, lattice, 1)


def test_flat_metric_uses_the_image_outside_a_27_window():
    lattice = _lattice_from_lengths_angles(
        [5.108, 2.895, 4.291], [73.19, 79.94, 153.04]
    )
    frac_delta = np.array([0.161, 1.476, 1.494])
    window = _fixed_window_distance(frac_delta, lattice)
    brute = _brute_image_distance(frac_delta, lattice, 8)
    assert window > 0.35
    assert brute <= 0.35
    translation = FractionalAffineOperation(
        np.eye(3), frac_delta, xyz="x,y,z", index=1, source="test"
    )
    identity = FractionalAffineOperation(
        np.eye(3), np.zeros(3), xyz="x,y,z", index=0, source="test"
    )
    info = DisorderInfo(
        labels=["C1"],
        symbols=["C"],
        frac_coords=np.zeros((1, 3)),
        occupancies=[1.0],
        disorder_groups=[0],
        assemblies=[""],
        crystal_symmetry=CrystalSymmetry(
            operations=(identity, translation),
            space_group_number=1,
            space_group_symbol="P 1",
            source="test",
        ),
    )
    audit = audit_parent_symmetry(info, [0], lattice, tolerance_angstrom=0.35)
    assert audit.respects_parent_group is True
    assert audit.operations[1].closes
    assert abs(audit.operations[1].residual_angstrom_max - brute) < 1e-6
    assert audit.operations[1].residual_angstrom_max < window


def test_more_skew_cells_match_an_independent_search():
    generator = np.random.default_rng(20261117)
    compared = 0
    for _draw in range(4):
        basis = generator.normal(size=(3, 3))
        basis[0] *= 4.0
        basis[1] *= 2.5
        basis[2] *= 3.0
        if abs(float(np.linalg.det(basis))) < 0.5:
            continue
        frac_delta = generator.uniform(-2.0, 2.0, size=3)
        try:
            half_width = _exact_shift_half_width(basis, 1.0)
        except ParentSymmetryNotAssessed:
            continue
        brute = _brute_image_distance(frac_delta, basis, half_width + 1)
        translation = FractionalAffineOperation(
            np.eye(3), frac_delta, xyz="x,y,z", index=0, source="test"
        )
        info = DisorderInfo(
            labels=["C1"],
            symbols=["C"],
            frac_coords=np.zeros((1, 3)),
            occupancies=[1.0],
            disorder_groups=[0],
            assemblies=[""],
            crystal_symmetry=CrystalSymmetry(
                operations=(translation,),
                space_group_number=1,
                space_group_symbol="P 1",
                source="test",
            ),
        )
        audit = audit_parent_symmetry(info, [0], basis, tolerance_angstrom=1.0)
        if brute <= 1.0:
            assert audit.respects_parent_group is True
            assert abs(audit.operations[0].residual_angstrom_max - brute) < 1e-6
            compared += 1
        elif audit.respects_parent_group is not None:
            assert audit.operations[0].residual_angstrom_max > 1.0
            compared += 1
    assert compared >= 3


def test_degenerate_metric_is_not_a_false_parent_closure():
    lattice = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    identity = FractionalAffineOperation(
        np.eye(3), np.array([0.2, 0.2, 0.0]), xyz="x,y,z", index=0, source="test"
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
    audit = audit_parent_symmetry(info, [0], lattice, tolerance_angstrom=0.35)
    assert audit.respects_parent_group is None
    assert any("not assessed" in note for note in audit.notes)
    assert audit.operations == ()


def test_exact_image_search_stays_bounded_on_an_orthogonal_cell():
    import time

    generator = np.random.default_rng(1)
    frac = generator.random((60, 3))
    lattice = np.diag([8.0, 9.0, 11.0])
    started = time.perf_counter()
    for _repeat in range(4):
        distances = _pairwise_min_image_distances(
            frac, frac, lattice, (True, True, True), 0.35
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 1.5
    assert np.max(np.abs(np.diag(distances))) < 1e-8


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
