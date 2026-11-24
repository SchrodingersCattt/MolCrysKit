"""The lightweight install is a resolvable package, not a --no-deps workaround."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_CORE = {"numpy", "networkx", "click"}
_DESKTOP = {"scipy", "ase", "pymatgen", "spglib", "defusedxml"}


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=\[]", requirement, maxsplit=1)[0].strip().lower()


def test_core_and_desktop_dependencies_are_disjoint():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    required = {_distribution_name(item) for item in project["dependencies"]}
    desktop = {
        _distribution_name(item) for item in project["optional-dependencies"]["desktop"]
    }
    assert required == _CORE
    assert desktop == _DESKTOP
    assert required.isdisjoint(desktop)


_CORE_CLI = """
import importlib.abc
import sys

class BlockDesktop(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"ase", "scipy", "pymatgen", "spglib", "defusedxml"}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockDesktop())
from molcrys_kit.cli import main
main(prog_name="mck")
"""



def test_core_cli_help_and_desktop_command_errors(tmp_path):
    # Run in a fresh process: the normal test environment has the desktop
    # packages installed, and other tests may already have imported ASE.
    source = tmp_path / "unused.json"
    source.write_text("{}")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-c", _CORE_CLI, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    for args in ((), ("build", "chain"), ("validate-periodic-bundle",), ("io",)):
        result = run(*args, "--help")
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout

    for args in (
        ("build", "chain", source, "-o", tmp_path / "bundle"),
        ("validate-periodic-bundle", source),
    ):
        result = run(*args)
        assert result.returncode == 1
        assert 'python -m pip install "molcrys-kit[desktop]"' in result.stderr
        assert "Traceback" not in result.stderr


def test_core_cli_can_convert_cif_without_ase(tmp_path):
    source = tmp_path / "neon.cif"
    output = tmp_path / "roundtrip.cif"
    source.write_text(
        """data_neon
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ne1 Ne 0.1 0.2 0.3
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", _CORE_CLI, "io", "convert", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Ne" in output.read_text(encoding="utf-8")

    poscar = tmp_path / "POSCAR"
    result = subprocess.run(
        [sys.executable, "-c", _CORE_CLI, "io", "convert", str(source), "-o", str(poscar)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert 'python -m pip install "molcrys-kit[desktop]"' in result.stderr
    assert "Traceback" not in result.stderr
    assert not poscar.exists()
