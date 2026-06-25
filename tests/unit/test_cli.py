from __future__ import annotations

import importlib.util
from pathlib import Path

from click.testing import CliRunner

from molcrys_kit.cli import main
from molcrys_kit.__main__ import main as module_main


DATA = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA / "DAP-4.cif"
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


def test_operate_supercell(tmp_path: Path) -> None:
    output = tmp_path / "super.cif"
    result = CliRunner().invoke(
        main,
        ["operate", "supercell", str(DAP4), "-o", str(output), "--scale", "1", "1", "1"],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_analyze_bfdh() -> None:
    result = CliRunner().invoke(main, ["analyze", "bfdh", str(DAP4), "--top-n", "1"])
    assert result.exit_code == 0
    assert "miller" in result.output
    assert "d_hkl" in result.output


def test_analyze_bfdh_json() -> None:
    result = CliRunner().invoke(main, ["analyze", "bfdh", str(DAP4), "--top-n", "1", "--json"])
    assert result.exit_code == 0
    assert '"miller_index"' in result.output


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
