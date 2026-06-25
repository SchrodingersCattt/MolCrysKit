from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from molcrys_kit.cli import main


DATA = Path(__file__).resolve().parents[1] / "data" / "cif"
DAP4 = DATA / "DAP-4.cif"


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
