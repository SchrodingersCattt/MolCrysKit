"""The lightweight install is a resolvable package, not a --no-deps workaround."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_CORE = {"numpy", "networkx", "click"}
_DESKTOP = {"scipy", "ase", "pymatgen", "spglib"}


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
