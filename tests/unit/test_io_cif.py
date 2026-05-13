"""
Unit tests for molcrys_kit.io.cif (read_mol_crystal, parse_cif_advanced, identify_molecules).
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

# Suppress pymatgen/CIF parsing warnings in tests (test data may have occupancy quirks)
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

from molcrys_kit.constants.config import KEY_DISORDER_GROUP
from molcrys_kit.io.cif import (
    identify_molecules,
    parse_cif_advanced,
    read_mol_crystal,
    scan_cif_disorder,
)
from molcrys_kit.structures.molecule import CrystalMolecule


class TestReadMolCrystal:
    """read_mol_crystal from CIF file."""

    def test_parse_test_cif(self, test_cif_path):
        crystal = read_mol_crystal(test_cif_path)
        assert crystal is not None
        assert len(crystal.molecules) >= 1
        assert all(isinstance(m, CrystalMolecule) for m in crystal.molecules)

    def test_question_mark_attached_hydrogens_is_tolerated(self, tmp_path):
        cif = tmp_path / "attached_hydrogens_unknown.cif"
        cif.write_text(
            """data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_attached_hydrogens
C1 C 0 0 0 1 ?
""",
            encoding="utf-8",
        )

        info = scan_cif_disorder(str(cif))
        assert info.labels == ["C1"]
        assert info.occupancies == [1.0]

    def test_molecules_have_graph_and_ase_api(self, test_cif_path):
        crystal = read_mol_crystal(test_cif_path)
        for mol in crystal.molecules:
            assert hasattr(mol, "graph")
            assert hasattr(mol, "get_chemical_symbols")
            assert hasattr(mol, "get_positions")
            assert hasattr(mol, "get_chemical_formula")


class TestParseCifAdvancedDeprecated:
    """parse_cif_advanced is deprecated; behavior and warning."""

    def test_deprecation_warning(self, test_cif_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            crystal = parse_cif_advanced(test_cif_path)
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
        assert all(isinstance(m, CrystalMolecule) for m in crystal.molecules)

    def test_returns_crystal_molecules(self, test_cif_path):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            crystal = parse_cif_advanced(test_cif_path)
        for mol in crystal.molecules:
            assert hasattr(mol, "get_chemical_symbols")
            assert hasattr(mol, "get_positions")


class TestIdentifyMoleculesFromAtoms:
    """identify_molecules(Atoms) -> list of CrystalMolecule (bond-based)."""

    def test_simple_ho_and_isolated_cl(self):
        atoms = Atoms(
            symbols=["H", "O", "Cl"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
        )
        molecules = identify_molecules(atoms)
        assert len(molecules) == 2
        by_size = sorted(molecules, key=len)
        assert len(by_size[0]) == 1
        assert by_size[0].get_chemical_symbols()[0] == "Cl"
        assert len(by_size[1]) == 2
        assert set(by_size[1].get_chemical_symbols()) == {"H", "O"}
        assert len(by_size[1].graph.nodes()) == 2
        assert len(by_size[1].graph.edges()) == 1
        assert len(by_size[0].graph.edges()) == 0

    def test_exclude_indices_skips_bonding_but_preserves_indices(self):
        atoms = Atoms(
            symbols=["H", "O", "Cl"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
        )
        molecules = identify_molecules(atoms, exclude_indices={1})

        assert sorted(len(mol) for mol in molecules) == [1, 1, 1]
        assert sorted(mol.info["atom_indices"] for mol in molecules) == [[0], [1], [2]]
        assert all(mol.info["bond_pairs"] == [] for mol in molecules)

    def test_water_ammonia_neon(self):
        atoms = Atoms(
            symbols=["O", "H", "H", "N", "H", "H", "H", "Ne"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.95, 0.0, 0.0],
                [0.0, 0.95, 0.0],
                [5.0, 5.0, 5.0],
                [5.95, 5.0, 5.0],
                [5.0, 5.95, 5.0],
                [5.0, 5.0, 5.95],
                [10.0, 10.0, 10.0],
            ],
        )
        molecules = identify_molecules(atoms)
        assert len(molecules) == 3
        by_size = sorted(molecules, key=len)
        assert [len(m) for m in by_size] == [1, 3, 4]
        assert by_size[0].get_chemical_symbols()[0] == "Ne"
        assert len(by_size[1].graph.edges()) == 2
        assert len(by_size[2].graph.edges()) == 3

    def test_disorder_groups_skip_cross_part_bonds(self):
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
        )
        atoms.set_array(KEY_DISORDER_GROUP, np.array([1, -1], dtype=int))

        molecules = identify_molecules(atoms)

        assert len(molecules) == 2
        assert sorted(mol.info["atom_indices"] for mol in molecules) == [[0], [1]]
        assert all(mol.info["bond_pairs"] == [] for mol in molecules)

    @pytest.mark.parametrize("groups", ([1, 1], [0, -1]))
    def test_disorder_group_compatible_atoms_can_still_bond(self, groups):
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
        )
        atoms.set_array(KEY_DISORDER_GROUP, np.array(groups, dtype=int))

        molecules = identify_molecules(atoms)

        assert len(molecules) == 1
        assert molecules[0].info["atom_indices"] == [0, 1]
        assert molecules[0].info["bond_pairs"] == [(0, 1)]
