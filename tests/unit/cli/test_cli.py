from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from molcrys_kit.cli import main
from molcrys_kit.__main__ import main as module_main


DATA = Path(__file__).resolve().parents[2] / "data" / "cif"
DAP4 = DATA / "DAP-4.cif"
CAFFEINE = DATA / "anhydrousCaffeine_CGD_2007_7_1406.cif"
PETN = DATA / "PETN_PERYTN10.cif"
ACETAMINOPHEN = DATA / "Acetaminophen_HXACAN.cif"
ONE_HTP = DATA / "1-HTP.cif"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_carve_cluster_module():
    script_path = REPO_ROOT / "scripts" / "carve_cluster.py"
    spec = importlib.util.spec_from_file_location("carve_cluster_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_root_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "io" in result.output
    assert "operate" in result.output
    assert "analyze" in result.output


def test_io_info() -> None:
    result = CliRunner().invoke(main, ["io", "info", str(DAP4)])
    assert result.exit_code == 0
    assert "MolecularCrystal" in result.output
    assert "Total atoms" in result.output


def test_io_info_shows_disorder_for_disordered_cif() -> None:
    """``mck io info`` should print disorder stats for a disordered CIF."""
    result = CliRunner().invoke(main, ["io", "info", str(DAP4)])
    assert result.exit_code == 0, result.output
    assert "Disorder:" in result.output
    assert "Atoms with occupancy < 1.0:" in result.output
    # DAP-4 has special-position disorder (occupancy < 1.0)
    assert "none detected" not in result.output


def test_io_info_shows_disorder_assemblies() -> None:
    """``mck io info`` should show assemblies when present (caffeine CIF)."""
    result = CliRunner().invoke(main, ["io", "info", str(CAFFEINE)])
    assert result.exit_code == 0, result.output
    assert "Disorder:" in result.output
    assert "Assemblies:" in result.output


def test_io_info_no_disorder_for_ordered_cif() -> None:
    """``mck io info`` should report 'none detected' for ordered CIFs."""
    result = CliRunner().invoke(main, ["io", "info", str(PETN)])
    assert result.exit_code == 0, result.output
    assert "Disorder: none detected" in result.output


def test_io_info_resolve_disorder_flag() -> None:
    """--resolve-disorder should produce clean molecules without partial-occ fragments."""
    result = CliRunner().invoke(
        main, ["io", "info", str(CAFFEINE), "--resolve-disorder"]
    )
    assert result.exit_code == 0, result.output
    # After disorder resolution, molecule list should be clean —
    # no tiny single-atom or 2-atom fragments from disorder remnants
    assert "CH2" not in result.output
    assert "MolecularCrystal" in result.output


def test_io_info_bond_scale_changes_output() -> None:
    """--bond-scale < 1 should produce different molecule counts."""
    default = CliRunner().invoke(main, ["io", "info", str(DAP4)])
    scaled = CliRunner().invoke(
        main, ["io", "info", str(DAP4), "--bond-scale", "0.5"]
    )
    assert default.exit_code == 0
    assert scaled.exit_code == 0
    # With tighter thresholds, output should differ
    assert default.output != scaled.output


def test_io_convert_cif(tmp_path: Path) -> None:
    output = tmp_path / "converted.cif"
    result = CliRunner().invoke(main, ["io", "convert", str(DAP4), "-o", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("data_")


def test_io_convert_rejects_unknown_output_format(tmp_path: Path) -> None:
    output = tmp_path / "converted.unknown"
    result = CliRunner().invoke(main, ["io", "convert", str(DAP4), "-o", str(output)])
    assert result.exit_code != 0
    assert "Unsupported output format" in result.output


def test_io_molecules_text() -> None:
    result = CliRunner().invoke(main, ["io", "molecules", str(DAP4)])
    assert result.exit_code == 0
    assert "index" in result.output
    assert "formula" in result.output
    assert "species_id" in result.output


def test_io_molecules_json() -> None:
    result = CliRunner().invoke(main, ["io", "molecules", str(DAP4), "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert rows
    assert rows[0]["index"] == 0
    assert rows[0]["formula"]
    assert rows[0]["atom_count"] > 0
    assert len(rows[0]["centroid"]) == 3
    assert rows[0]["species_id"]


def test_analyze_sanity_check_accepts_cif() -> None:
    result = CliRunner().invoke(
        main,
        ["analyze", "sanity-check", str(PETN), "--json"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n_frames"] == 1
    assert len(report["frames"][0]["results"]) == 6


def test_analyze_sanity_check_reads_all_extxyz_frames(tmp_path: Path) -> None:
    from molcrys_kit.io import read_mol_crystal, write_extxyz

    crystal = read_mol_crystal(str(PETN))
    path = tmp_path / "frames.extxyz"
    write_extxyz([crystal, crystal], str(path))

    result = CliRunner().invoke(
        main,
        ["analyze", "sanity-check", str(path), "--json"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n_frames"] == 2


def test_analyze_summary_json() -> None:
    result = CliRunner().invoke(main, ["analyze", "summary", str(PETN), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n_atoms"] == sum(report["species_counts"].values())
    assert report["formula"]
    assert report["pbc"] == [True, True, True]
    assert report["cell"]["volume_A3"] > 0
    assert report["symmetry"]["status"] == "ok"
    assert report["symmetry"]["space_group_number"] > 0
    assert report["symmetry"]["wyckoff_sites"]


def test_analyze_summary_reports_disorder() -> None:
    result = CliRunner().invoke(main, ["analyze", "summary", str(DAP4), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["disorder"]["has_disorder"] is True
    assert report["disorder"]["partial_occupancy_sites"] > 0


def test_analyze_summary_text() -> None:
    result = CliRunner().invoke(main, ["analyze", "summary", str(PETN)])
    assert result.exit_code == 0, result.output
    assert "Structure summary:" in result.output
    assert "Formula:" in result.output
    assert "Symmetry:" in result.output
    assert "Wyckoff:" in result.output


def test_analyze_summary_rejects_non_positive_symprec() -> None:
    result = CliRunner().invoke(
        main, ["analyze", "summary", str(PETN), "--symprec", "0"]
    )
    assert result.exit_code != 0
    assert "--symprec must be positive." in result.output


def test_io_extract_molecule_by_index(tmp_path: Path) -> None:
    output = tmp_path / "mol.xyz"
    result = CliRunner().invoke(
        main,
        ["io", "extract-molecule", str(DAP4), "-o", str(output), "--index", "0"],
    )
    assert result.exit_code == 0
    assert output.exists()
    lines = output.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) > 0


def test_io_extract_molecule_by_formula(tmp_path: Path) -> None:
    rows_result = CliRunner().invoke(main, ["io", "molecules", str(DAP4), "--json"])
    assert rows_result.exit_code == 0
    formula = json.loads(rows_result.output)[0]["formula"]
    output = tmp_path / "mol.xyz"
    result = CliRunner().invoke(
        main,
        ["io", "extract-molecule", str(DAP4), "-o", str(output), "--formula", formula],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_io_extract_molecule_all(tmp_path: Path) -> None:
    output = tmp_path / "mol.xyz"
    result = CliRunner().invoke(
        main,
        ["io", "extract-molecule", str(DAP4), "-o", str(output), "--all"],
    )
    assert result.exit_code == 0
    assert list(tmp_path.glob("mol_*.xyz"))


def test_io_extract_molecule_json_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "mol.xyz"
    sidecar = tmp_path / "mol.json"
    result = CliRunner().invoke(
        main,
        [
            "io",
            "extract-molecule",
            str(DAP4),
            "-o",
            str(output),
            "--index",
            "0",
            "--json-sidecar",
            str(sidecar),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["index"] == 0
    assert payload["formula"]
    assert len(payload["centroid"]) == 3
    assert payload["output"] == str(output)


def test_io_extract_molecule_center_vacuum_cif(tmp_path: Path) -> None:
    output = tmp_path / "mol.cif"
    result = CliRunner().invoke(
        main,
        [
            "io",
            "extract-molecule",
            str(DAP4),
            "-o",
            str(output),
            "--index",
            "0",
            "--center-vacuum",
            "10",
            "--pbc",
            "true",
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("data_")


def test_io_extract_molecule_rejects_multiple_selectors_before_load(tmp_path: Path) -> None:
    output = tmp_path / "mol.xyz"
    result = CliRunner().invoke(
        main,
        [
            "io",
            "extract-molecule",
            str(DAP4),
            "-o",
            str(output),
            "--index",
            "0",
            "--largest",
        ],
    )
    assert result.exit_code != 0
    assert "Use only one molecule selector" in result.output


def test_operate_supercell(tmp_path: Path) -> None:
    output = tmp_path / "super.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "supercell", str(DAP4), "-o", str(output), "--scale", "1", "1", "1"],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_operate_supercell_rejects_zero_scale(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "supercell", str(DAP4), "-o", str(tmp_path / "super.cif"), "--scale", "0", "1", "1"],
    )
    assert result.exit_code != 0
    assert "--scale factors must each be >= 1." in result.output


def test_operate_disorder_supercell(tmp_path: Path) -> None:
    from molcrys_kit.io import read_extxyz

    output = tmp_path / "disorder_supercell.extxyz"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "disorder-supercell",
            str(ONE_HTP),
            "-o",
            str(output),
            "--scale",
            "2",
            "1",
            "1",
            "--method",
            "enumerate",
            "--replica-index",
            "0",
            "--replica-index",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    crystal = read_extxyz(str(output))
    assert crystal.metadata["supercell"]["scaling_factors"] == [2, 1, 1]
    assert [
        cell["replica_index"] for cell in crystal.metadata["replica_supercell"]["cells"]
    ] == [0, 1]


def test_operate_disorder_supercell_rejects_mapping_length(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "disorder-supercell",
            str(DAP4),
            "-o",
            str(tmp_path / "disorder_supercell.cif"),
            "--scale",
            "2",
            "1",
            "1",
            "--replica-index",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "exactly 2 --replica-index values" in result.output




def test_operate_nanocluster_fixed_unit_cell_count(tmp_path: Path) -> None:
    output = tmp_path / "nanocluster.extxyz"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(output),
            "--shape",
            "sphere",
            "--radius",
            "15",
            "--topology-unit",
            "unit_cell",
            "--target-units",
            "1",
            "--vacuum",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "units: 1" in result.output
    assert "atoms: 368" in result.output


def test_operate_nanocluster_requires_shape_dimensions(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(tmp_path / "nanocluster.extxyz"),
            "--shape",
            "sphere",
        ],
    )
    assert result.exit_code != 0
    assert "--radius is required for --shape sphere" in result.output


def test_operate_nanocluster_bfdh_requires_max_dimension(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(tmp_path / "bfdh.extxyz"),
            "--shape",
            "bfdh",
        ],
    )

    assert result.exit_code != 0
    assert "--max-dimension is required for --shape bfdh" in result.output


def test_operate_nanocluster_bfdh_explicit_millers(tmp_path: Path) -> None:
    input_path = tmp_path / "p1.cif"
    input_path.write_text(
        """data_p1
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
He1 He 0.5 0.5 0.5
""",
        encoding="utf-8",
    )
    output = tmp_path / "bfdh.extxyz"
    sidecar = tmp_path / "bfdh.json"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(input_path),
            "-o",
            str(output),
            "--shape",
            "bfdh",
            "--max-dimension",
            "10",
            "--max-index",
            "0",
            "--miller",
            "1",
            "0",
            "0",
            "--miller",
            "0",
            "1",
            "0",
            "--miller",
            "0",
            "0",
            "1",
            "--topology-unit",
            "unit_cell",
            "--json-sidecar",
            str(sidecar),
        ],
    )

    assert result.exit_code == 0, result.output
    stats = json.loads(sidecar.read_text(encoding="utf-8"))
    parameters = stats["shape_parameters"]
    assert parameters["max_dimension_A"] == 10.0
    assert parameters["miller_indices"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert parameters["symmetry"]["kind"] == "explicit_parent"
    assert parameters["symmetry"]["space_group_number"] == 1
    distances = {
        tuple(plane["miller_index"]): plane["distance_A"]
        for plane in parameters["planes"]
    }
    assert distances[(1, 0, 0)] / distances[(0, 1, 0)] == pytest.approx(6.0 / 5.0)


def test_operate_nanocluster_bfdh_can_disable_parent_extinctions(
    tmp_path: Path,
) -> None:
    common_args = [
        "operate",
        "nanocluster",
        str(DAP4),
        "--shape",
        "bfdh",
        "--max-dimension",
        "60",
        "--max-index",
        "0",
        "--miller",
        "1",
        "0",
        "0",
        "--topology-unit",
        "unit_cell",
        "--target-units",
        "1",
    ]
    filtered = CliRunner().invoke(
        main,
        [*common_args, "-o", str(tmp_path / "filtered.extxyz")],
    )

    assert filtered.exit_code != 0
    assert "BFDH enumeration produced no allowed facets" in filtered.output

    sidecar = tmp_path / "unfiltered.json"
    unfiltered = CliRunner().invoke(
        main,
        [
            *common_args,
            "-o",
            str(tmp_path / "unfiltered.extxyz"),
            "--no-extinction-filter",
            "--json-sidecar",
            str(sidecar),
        ],
    )

    assert unfiltered.exit_code == 0, unfiltered.output
    parameters = json.loads(sidecar.read_text(encoding="utf-8"))["shape_parameters"]
    assert parameters["extinction_filter"] is False
    assert parameters["miller_indices"] == [[1, 0, 0]]
    assert len(parameters["planes"]) == 6


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--max-index", "0"], "max_index must be >= 1"),
        (["--miller", "0", "0", "0"], "cannot all be zero"),
    ],
)
def test_operate_nanocluster_bfdh_rejects_invalid_facet_options(
    tmp_path: Path,
    extra_args: list[str],
    message: str,
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(tmp_path / "invalid_bfdh.extxyz"),
            "--shape",
            "bfdh",
            "--max-dimension",
            "20",
            *extra_args,
        ],
    )

    assert result.exit_code != 0
    assert message in result.output


def test_operate_nanocluster_bfdh_uses_disordered_cif_parent_symmetry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ordered_bfdh.extxyz"
    sidecar = tmp_path / "ordered_bfdh.json"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(output),
            "--shape",
            "bfdh",
            "--max-dimension",
            "80",
            "--topology-unit",
            "unit_cell",
            "--target-units",
            "1",
            "--resolve-disorder",
            "--json-sidecar",
            str(sidecar),
        ],
    )

    assert result.exit_code == 0, result.output
    stats = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stats["input_disorder"]["all_atom_ordered"] is True
    assert stats["shape_parameters"]["symmetry"]["kind"] == "explicit_parent"
    assert stats["shape_parameters"]["symmetry"]["space_group_number"] is not None


def test_operate_nanocluster_resolves_disorder_and_writes_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ordered_nanocluster.extxyz"
    sidecar = tmp_path / "ordered_nanocluster.json"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "nanocluster",
            str(DAP4),
            "-o",
            str(output),
            "--shape",
            "cylinder",
            "--radius",
            "12",
            "--height",
            "20",
            "--axis-vector",
            "1",
            "1",
            "0.5",
            "--center-frac",
            "0.5",
            "0.5",
            "0.5",
            "--topology-unit",
            "unit_cell",
            "--target-units",
            "1",
            "--resolve-disorder",
            "--json-sidecar",
            str(sidecar),
        ],
    )
    assert result.exit_code == 0, result.output
    stats = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stats["selected_atom_count"] == 336
    assert stats["input_disorder"]["all_atom_ordered"] is True


def test_operate_void_fixed_stoichiometry_charge_and_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "void.extxyz"
    sidecar = tmp_path / "void.json"
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "void",
            str(DAP4),
            "-o",
            str(output),
            "--shape",
            "sphere",
            "--radius",
            "5",
            "--center-frac",
            "0.5",
            "0.5",
            "0.5",
            "--target-units",
            "1",
            "--species-charge",
            "C6H14N2_1",
            "2",
            "--species-charge",
            "ClO4_1",
            "-1",
            "--species-charge",
            "H4N_1",
            "1",
            "--resolve-disorder",
            "--json-sidecar",
            str(sidecar),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    stats = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stats["removed_species_counts"] == {
        "C6H14N2_1": 1,
        "ClO4_1": 3,
        "H4N_1": 1,
    }
    assert stats["removed_atom_count"] == 42
    assert stats["remaining_atom_count"] == 294
    assert stats["charge_verified"] is True
    assert stats["removed_net_charge_e"] == 0.0
    assert stats["input_disorder"]["all_atom_ordered"] is True


def test_operate_void_cover_nonperiodic_and_species_validation(tmp_path: Path) -> None:
    import numpy as np
    from ase import Atoms

    from molcrys_kit.io import write_extxyz
    from molcrys_kit.structures import MolecularCrystal

    cell = 10.0 * np.eye(3)
    entries = [
        ("H", (5.0, 5.0, 5.0)),
        ("H", (5.9, 5.0, 5.0)),
        ("H", (8.0, 5.0, 5.0)),
        ("He", (5.2, 5.0, 5.0)),
        ("He", (6.2, 5.0, 5.0)),
        ("He", (8.5, 5.0, 5.0)),
    ]
    crystal = MolecularCrystal(
        cell,
        [
            Atoms(symbol, positions=[position], cell=cell, pbc=False)
            for symbol, position in entries
        ],
        pbc=(False, False, False),
    )
    input_path = tmp_path / "cover_input.extxyz"
    output = tmp_path / "cover_void.extxyz"
    sidecar = tmp_path / "cover_void.json"
    write_extxyz(crystal, str(input_path))

    result = CliRunner().invoke(
        main,
        [
            "operate",
            "void",
            str(input_path),
            "-o",
            str(output),
            "--shape",
            "sphere",
            "--radius",
            "1",
            "--center",
            "5",
            "5",
            "5",
            "--boundary-policy",
            "cover",
            "--no-periodic-images",
            "--species",
            "H_1",
            "1",
            "--species",
            "He_1",
            "1",
            "--json-sidecar",
            str(sidecar),
        ],
    )
    assert result.exit_code == 0, result.output
    stats = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stats["boundary_policy"] == "cover"
    assert stats["periodic_images"] is False
    assert stats["removed_species_counts"] == {"H_1": 2, "He_1": 2}

    invalid = CliRunner().invoke(
        main,
        [
            "operate",
            "void",
            str(input_path),
            "-o",
            str(output),
            "--shape",
            "sphere",
            "--radius",
            "1",
            "--species",
            "H_1",
            "not-an-integer",
        ],
    )
    assert invalid.exit_code != 0
    assert "--species COUNT must be an integer" in invalid.output


def test_stats_sidecar_reports_filesystem_errors(tmp_path: Path) -> None:
    import click
    import pytest

    from molcrys_kit.cli.operate_cmd import _write_stats_sidecar

    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(click.ClickException, match="Could not write JSON sidecar"):
        _write_stats_sidecar(blocking_file / "stats.json", {"count": 1})


def test_operate_void_through_cylinder_requires_hkl(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "void",
            str(DAP4),
            "-o",
            str(tmp_path / "void.extxyz"),
            "--shape",
            "through-cylinder",
            "--radius",
            "3",
        ],
    )
    assert result.exit_code != 0
    assert "--direction-hkl H K L" in result.output


def test_slab_requires_layers_or_thickness(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "slab", str(DAP4), "-o", str(tmp_path / "slab.cif"), "--miller", "1", "1", "0"],
    )
    assert result.exit_code != 0
    assert "Specify --layers N or --min-thickness T" in result.output


def test_slab_rejects_all_zero_miller(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "slab", str(DAP4), "-o", str(tmp_path / "slab.cif"), "--miller", "0", "0", "0", "--layers", "1"],
    )
    assert result.exit_code != 0
    assert "Miller indices cannot all be zero." in result.output


def test_slab_rejects_invalid_terminations(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "slab", str(DAP4), "-o", str(tmp_path / "slab.cif"), "--miller", "1", "1", "0", "--layers", "1", "--terminations", "garbage"],
    )
    assert result.exit_code != 0
    assert "--terminations must be one of" in result.output


def test_vacancy_rejects_bad_species_count(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "vacancy", str(DAP4), "-o", str(tmp_path / "vacancy.cif"), "--species", "foo", "bar"],
    )
    assert result.exit_code != 0
    assert "--species COUNT must be an integer" in result.output


def test_vacancy_rejects_negative_seed_index(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "vacancy", str(DAP4), "-o", str(tmp_path / "vacancy.cif"), "--seed-index", "-1"],
    )
    assert result.exit_code != 0
    assert "--seed-index must be non-negative." in result.output


def test_interpolate_rejects_zero_images(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "interpolate", str(DAP4), str(DAP4), "-o", str(tmp_path / "traj.extxyz"), "--n-images", "0"],
    )
    assert result.exit_code != 0
    assert "--n-images must be >= 1." in result.output


def test_analyze_bfdh() -> None:
    result = CliRunner().invoke(main, ["analyze", "bfdh", str(DAP4), "--top-n", "1"])
    assert result.exit_code == 0
    assert "miller" in result.output
    assert "d_hkl" in result.output


def test_analyze_bfdh_json() -> None:
    result = CliRunner().invoke(main, ["analyze", "bfdh", str(DAP4), "--top-n", "1", "--json"])
    assert result.exit_code == 0
    assert '"miller_index"' in result.output


def test_bfdh_rejects_max_index_zero() -> None:
    result = CliRunner().invoke(main, ["analyze", "bfdh", str(DAP4), "--max-index", "0"])
    assert result.exit_code != 0
    assert "--max-index must be >= 1." in result.output


def test_polyhedra_rejects_non_positive_cutoff() -> None:
    result = CliRunner().invoke(
        main,
        ["analyze", "polyhedra", str(DAP4), "--central", "Zn", "--ligand", "N", "--cutoff", "0"],
    )
    assert result.exit_code != 0
    assert "--cutoff must be positive." in result.output


def test_cluster_seed_options_are_mutually_exclusive(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operate",
            "cluster",
            str(DAP4),
            "-o",
            str(tmp_path / "cluster"),
            "--seed-index",
            "0",
            "--seed-element",
            "Zn",
        ],
    )
    assert result.exit_code != 0
    assert "Specify --seed-element OR --seed-index" in result.output


def test_cluster_requires_seed(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["operate", "cluster", str(DAP4), "-o", str(tmp_path / "cluster")],
    )
    assert result.exit_code != 0
    assert "Specify a seed" in result.output


def test_carve_cluster_legacy_arg_translation_space_form() -> None:
    _translate_legacy_args = _load_carve_cluster_module()._translate_legacy_args
    assert _translate_legacy_args(["--cif", "bulk.cif", "--out", "cluster", "--seed-index", "1"]) == [
        "bulk.cif",
        "--output",
        "cluster",
        "--seed-index",
        "1",
    ]


def test_carve_cluster_legacy_arg_translation_equals_form() -> None:
    _translate_legacy_args = _load_carve_cluster_module()._translate_legacy_args
    assert _translate_legacy_args(["--cif=bulk.cif", "--out=cluster", "--seed-index", "1"]) == [
        "bulk.cif",
        "--output=cluster",
        "--seed-index",
        "1",
    ]


def test_module_entrypoint_imports() -> None:
    assert module_main is main


# ---------------------------------------------------------------------------
# operate reorient
# ---------------------------------------------------------------------------

def test_reorient_basic(tmp_path: Path) -> None:
    """Smoke test: mck operate reorient should produce an output file."""
    out = tmp_path / "reoriented.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "reorient", str(ACETAMINOPHEN), "-o", str(out), "--direction", "1", "1", "0"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "d-spacing" in result.output
    assert "supercell factor" in result.output


def test_reorient_target_axis_x(tmp_path: Path) -> None:
    """Reorient with --target-axis x."""
    out = tmp_path / "reoriented.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "reorient", str(ACETAMINOPHEN), "-o", str(out), "--direction", "1", "0", "0", "--target-axis", "x"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_reorient_rejects_zero_direction(tmp_path: Path) -> None:
    """(0,0,0) direction should fail."""
    out = tmp_path / "reoriented.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "reorient", str(ACETAMINOPHEN), "-o", str(out), "--direction", "0", "0", "0"],
    )
    assert result.exit_code != 0
    assert "cannot be (0, 0, 0)" in result.output


# =====================================================================
# operate add-h
# =====================================================================


def test_add_h_basic(tmp_path: Path) -> None:
    """Smoke test: mck operate add-h produces output."""
    out = tmp_path / "hydrogenated.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "add-h", str(PETN), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Wrote" in result.output


def test_add_h_bond_scale(tmp_path: Path) -> None:
    """add-h with --bond-scale should succeed and produce output."""
    out = tmp_path / "hydrogenated.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "add-h", str(PETN), "-o", str(out), "--bond-scale", "0.95"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_add_h_rule(tmp_path: Path) -> None:
    """add-h with --rule should pass the rule through to add_hydrogens."""
    out = tmp_path / "hydrogenated.cif"
    result = CliRunner().invoke(
        main,
        [
            "operate", "add-h", str(PETN), "-o", str(out),
            "--rule", "N:target_coordination=3,geometry=trigonal_planar",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_add_h_rule_multiple(tmp_path: Path) -> None:
    """Multiple --rule options should all be parsed."""
    out = tmp_path / "hydrogenated.cif"
    result = CliRunner().invoke(
        main,
        [
            "operate", "add-h", str(PETN), "-o", str(out),
            "--rule", "N:target_coordination=3",
            "--rule", "O:neighbors=C+S,target_coordination=1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_add_h_rule_parse_error() -> None:
    """Malformed --rule should produce an error."""
    from molcrys_kit.cli.operate_cmd import _parse_rule
    import pytest

    # Missing element symbol
    with pytest.raises(Exception):
        _parse_rule(":target_coordination=3")

    # Invalid key
    with pytest.raises(Exception):
        _parse_rule("N:bogus_key=3")

    # Non-integer target_coordination
    with pytest.raises(Exception):
        _parse_rule("N:target_coordination=abc")


def test_parse_rule_valid() -> None:
    """_parse_rule should correctly parse valid rule strings."""
    from molcrys_kit.cli.operate_cmd import _parse_rule

    # Simple case
    r = _parse_rule("N:target_coordination=3")
    assert r == {"symbol": "N", "target_coordination": 3}

    # With geometry
    r = _parse_rule("N:target_coordination=3,geometry=trigonal_planar")
    assert r == {"symbol": "N", "target_coordination": 3, "geometry": "trigonal_planar"}

    # With neighbors
    r = _parse_rule("O:neighbors=C+Cl,target_coordination=1")
    assert r == {"symbol": "O", "neighbors": ["C", "Cl"], "target_coordination": 1}

    # Symbol only (no overrides)
    r = _parse_rule("N")
    assert r == {"symbol": "N"}
