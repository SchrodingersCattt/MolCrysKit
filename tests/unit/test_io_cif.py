"""
Unit tests for molcrys_kit.io.cif (read_mol_crystal, parse_cif_advanced, identify_molecules).
"""

import warnings

import pytest
from ase import Atoms

# Suppress pymatgen/CIF parsing warnings in tests (test data may have occupancy quirks)
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

from molcrys_kit.io.cif import read_mol_crystal, parse_cif_advanced, identify_molecules
from molcrys_kit.structures.molecule import CrystalMolecule


class TestReadMolCrystal:
    """read_mol_crystal from CIF file."""

    def test_parse_test_cif(self, test_cif_path):
        crystal = read_mol_crystal(test_cif_path)
        assert crystal is not None
        assert len(crystal.molecules) >= 1
        assert all(isinstance(m, CrystalMolecule) for m in crystal.molecules)

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
