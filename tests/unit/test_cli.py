from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from click.testing import CliRunner

from molcrys_kit.cli import main
from molcrys_kit.__main__ import main as module_main


DATA = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA / "DAP-4.cif"
CAFFEINE = DATA / "anhydrousCaffeine_CGD_2007_7_1406.cif"
PETN = DATA / "PETN_PERYTN10.cif"
ACETAMINOPHEN = DATA / "Acetaminophen_HXACAN.cif"
REPO_ROOT = Path(__file__).resolve().parents[2]


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
