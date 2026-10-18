"""
Unit tests for molcrys_kit.io.output (write_xyz, write_cif, write_vesta, write_molecule_summary).
"""

import os

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.constants.config import KEY_ATOM_ID
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.io.output import (
    _cif_quote,
    write_xyz,
    write_cif,
    write_vesta,
    write_molecule_summary,
)
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal


class TestWriteXyz:
    """write_xyz to string and file."""

    def test_to_string(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        result = write_xyz(mol)
        lines = result.strip().split("\n")
        assert lines[0] == "3"
        assert lines[1] == ""
        assert "O" in lines[2]
        assert "H" in lines[3]
        assert "H" in lines[4]

    def test_to_file(self, water_atoms, tmp_path):
        mol = CrystalMolecule(water_atoms)
        out_file = str(tmp_path / "test.xyz")
        write_xyz(mol, filename=out_file)
        assert os.path.exists(out_file)
        with open(out_file) as f:
            first_line = f.readline().strip()
        assert first_line == "3"

    def test_single_atom(self):
        mol = CrystalMolecule(Atoms("Ar", positions=[[1.0, 2.0, 3.0]]))
        result = write_xyz(mol)
        lines = result.strip().split("\n")
        assert lines[0] == "1"
        assert "Ar" in lines[2]
        assert "1.000000" in lines[2]


class TestWriteMoleculeSummary:
    """write_molecule_summary."""

    def test_basic_summary(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        summary = write_molecule_summary(mol)
        assert "Molecule:" in summary
        assert "H2O" in summary
        assert "Number of atoms: 3" in summary
        assert "Center of mass" in summary
        assert "Ellipsoid radii" in summary

    def test_summary_with_crystal_reference(self, cubic_lattice_10, water_atoms):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(water_atoms)])
        mol = crystal.molecules[0]
        summary = write_molecule_summary(mol)
        assert "Centroid (Cartesian)" in summary
        assert "Centroid (Fractional)" in summary


class TestWriteCif:
    """write_cif to string and file."""

    def test_to_string(self, crystal_single_water):
        cif = write_cif(crystal_single_water)
        assert "data_crystal" in cif
        assert "MolCrysKit" in cif
        assert "_cell_length_a" in cif
        assert "_atom_site_label" in cif

    def test_to_file(self, crystal_single_water, tmp_path):
        out_file = str(tmp_path / "test.cif")
        write_cif(crystal_single_water, filename=out_file)
        assert os.path.exists(out_file)
        with open(out_file) as f:
            content = f.read()
        assert "data_crystal" in content

    def test_cell_parameters_present(self, crystal_single_water):
        cif = write_cif(crystal_single_water)
        assert "10.000000" in cif
        assert "90.000000" in cif

    def test_no_metadata_no_molcrys_fields(self, crystal_single_water):
        cif = write_cif(crystal_single_water)
        assert "_molcrys_termination" not in cif

    def test_atom_sites_present(self, crystal_single_water):
        cif = write_cif(crystal_single_water)
        assert "_atom_site_fract_x" in cif
        assert "_atom_site_type_symbol" in cif

    @pytest.mark.parametrize(
        "value",
        ("Type'I", "it's ok", "Smiths' method", "trailing'"),
    )
    def test_termination_text_round_trip_preserves_apostrophes(
        self, crystal_single_water, value
    ):
        cif = write_cif(
            crystal_single_water,
            metadata={
                "termination_info": {
                    "tasker_type": value,
                    "charge_source": value,
                }
            },
        )
        restored = read_mol_crystal(cif_text=cif, attach_chemistry=False)
        termination = restored.metadata["termination_info"]
        assert termination["tasker_type"] == value
        assert termination["charge_source"] == value

    def test_termination_text_falls_back_to_semicolon_field(
        self, crystal_single_water
    ):
        value = 'both\' bad" value'
        cif = write_cif(
            crystal_single_water,
            metadata={
                "termination_info": {
                    "tasker_type": value,
                    "charge_source": value,
                }
            },
        )
        assert ";\nboth' bad\" value\n;" in cif
        restored = read_mol_crystal(cif_text=cif, attach_chemistry=False)
        termination = restored.metadata["termination_info"]
        assert termination["tasker_type"] == value
        assert termination["charge_source"] == value

    @pytest.mark.parametrize(
        "atom_ids",
        (
            ("Type'I", "it's ok"),
            ("Smiths' atom", "trailing'"),
            ('both\' bad" value', "plain"),
        ),
    )
    def test_atom_id_text_round_trip_preserves_quotes_and_whitespace(
        self, cubic_lattice_10, atom_ids
    ):
        atoms = Atoms("CC", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        atoms.set_array(KEY_ATOM_ID, np.asarray(atom_ids, dtype=str))
        molecule = CrystalMolecule(atoms, check_pbc=False)
        crystal = MolecularCrystal(cubic_lattice_10, [molecule])

        restored = read_mol_crystal(
            cif_text=write_cif(crystal), attach_chemistry=False
        )
        actual = [
            str(atom_id)
            for molecule in restored.molecules
            for atom_id in molecule.arrays[KEY_ATOM_ID]
        ]
        assert actual == list(atom_ids)

    @pytest.mark.parametrize(
        ("value", "expected"),
        (
            ("Type'I", "'Type'I'"),
            ("it's ok", "'it's ok'"),
            ("Smiths' method", '"Smiths\' method"'),
            ("trailing'", '"trailing\'"'),
            ('both\' bad" value', ';\nboth\' bad" value\n;'),
        ),
    )
    def test_cif_quote_uses_cif_safe_form(self, value, expected):
        assert _cif_quote(value) == expected


class TestWriteVesta:
    """write_vesta to string and file."""

    def test_to_string(self, crystal_single_water):
        vesta = write_vesta(crystal_single_water)
        assert "#VESTA_FORMAT_VERSION" in vesta
        assert "CRYSTAL" in vesta
        assert "CELLP" in vesta
        assert "STRUC" in vesta

    def test_to_file(self, crystal_single_water, tmp_path):
        out_file = str(tmp_path / "test.vesta")
        result = write_vesta(crystal_single_water, filename=out_file)
        assert os.path.exists(out_file)
        assert result is None

    def test_atom_entries(self, crystal_single_water):
        vesta = write_vesta(crystal_single_water)
        assert "O" in vesta
        assert "H" in vesta
