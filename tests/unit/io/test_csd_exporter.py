"""Regression tests for the CCDC export boundary without a CCDC install."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("skills/query-csd-structures/scripts/demo_csd_disorder_workflow.py")
SPEC = importlib.util.spec_from_file_location("csd_exporter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Coords:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


class _Atom:
    def __init__(self, label: str, symbol: str, coords: _Coords | None, occupancy: float = 1.0):
        self.label = label
        self.atomic_symbol = symbol
        self.fractional_coordinates = coords
        self.occupancy = occupancy
        self.displacement_parameters = None


class _Bond:
    def __init__(self, a1: _Atom, a2: _Atom, broken: bool = False):
        self.atoms = (a1, a2)
        self._broken = broken
        self.bond_type = "Single"

    @property
    def length(self):
        if self._broken:
            raise RuntimeError("null TSite")
        return 1.4


class _Group:
    def __init__(self, group_id: int, atoms: list[_Atom]):
        self.id = group_id
        self.atoms = atoms


class _Assembly:
    def __init__(self, assembly_id: int, groups: list[_Group]):
        self.id = assembly_id
        self.groups = groups


class _Disorder:
    def __init__(self, assemblies: list[_Assembly]):
        self.assemblies = assemblies


class _Molecule:
    def __init__(self, atoms: list[_Atom], bonds: list[_Bond]):
        self.atoms = atoms
        self.bonds = bonds


class _Crystal:
    # BENZEN02-like path: no disordered molecule, but a regular molecule exists.
    disordered_molecule = None
    cell_lengths = (7.46, 9.66, 7.03)
    cell_angles = (90.0, 90.0, 90.0)
    cell_volume = 506.61
    z_value = 4
    spacegroup_symbol = "Pbca"
    spacegroup_number_and_setting = (61, 1)
    symmetry_operators = ("x,y,z", "-x,-y,-z")

    def __init__(self, molecule: _Molecule, disorder: _Disorder | None = None):
        self.molecule = molecule
        self.disorder = disorder


class _Entry:
    chemical_name = "private name"
    formula = "C2 H1"
    r_factor = 2.0


class _Reader:
    def __init__(self, crystal: _Crystal):
        self.crystal_value = crystal

    def entry(self, _refcode: str):
        return _Entry()

    def crystal(self, _refcode: str):
        return self.crystal_value


class _MemoryPath:
    """Path-like sink so this test does not depend on pytest temp ACLs."""

    def __init__(self):
        self.text = ""
        self.parent = self
        self.name = "memory.cif"

    def mkdir(self, **_kwargs):
        return None

    def write_text(self, text: str, **_kwargs):
        self.text = text
        return len(text)

    def __str__(self):
        return self.name


def test_exporter_falls_back_from_null_disordered_molecule_and_skips_null_bonds():
    c = _Atom("C1", "C", _Coords(0.1, 0.2, 0.3))
    h = _Atom("H1", "H", None, occupancy=1.0)
    molecule = _Molecule([c, h], [_Bond(c, h, broken=True)])
    output = _MemoryPath()
    result = MODULE.export_full_cif_from_csd(_Reader(_Crystal(molecule)), "BENZEN02", output)
    assert result["status"] == "ok"
    assert result["n_source_atoms"] == 2
    assert result["n_written_atoms"] == 1
    assert "_space_group_IT_number 61" in output.text
    assert "H1" not in output.text
    assert any("omitted because fractional coordinates are unavailable" in warning for warning in result["warnings"])
    assert any("Bond export skipped" in warning for warning in result["warnings"])


def test_exporter_preserves_disorder_when_ccdc_groups_start_at_zero():
    a = _Atom("A1", "C", _Coords(0.1, 0.2, 0.3), occupancy=0.5)
    b = _Atom("B1", "N", _Coords(0.2, 0.2, 0.3), occupancy=0.5)
    disorder = _Disorder([_Assembly(0, [_Group(0, [a]), _Group(1, [b])])])
    crystal = _Crystal(_Molecule([a, b], [_Bond(a, b)]), disorder)
    output = _MemoryPath()
    result = MODULE.export_full_cif_from_csd(_Reader(crystal), "IQUHUY", output)
    assert result["n_partial_occupancy"] == 2
    assert "_atom_site_disorder_assembly" in output.text
    assert "_atom_site_disorder_group" in output.text
    assert result["n_disorder_sites"] == 2
