"""
Unit tests for molcrys_kit.analysis (species, interactions, stoichiometry).
"""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.species import identify_molecules, assign_atoms_to_molecules
from molcrys_kit.analysis.interactions import (
    HydrogenBond,
    find_hydrogen_bonds,
    get_bonding_threshold,
)
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule


# ----- Species -----


class TestSpeciesIdentify:
    """identify_molecules(crystal) and assign_atoms_to_molecules."""

    def test_identify_two_waters(self, crystal_single_water, cubic_lattice_10):
        h2o2 = Atoms(
            symbols=["O", "H", "H"],
            positions=[[3.0, 0.0, 0.0], [3.757, 0.586, 0.0], [2.243, 0.586, 0.0]],
        )
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [crystal_single_water.molecules[0], CrystalMolecule(h2o2)],
        )
        molecules = identify_molecules(crystal)
        assert len(molecules) == 2
        assert isinstance(molecules[0], CrystalMolecule)
        assert isinstance(molecules[1], CrystalMolecule)

    def test_assign_atoms_same_crystal(self, crystal_single_water):
        new_crystal = assign_atoms_to_molecules(crystal_single_water)
        np.testing.assert_allclose(new_crystal.lattice, crystal_single_water.lattice)
        assert len(new_crystal.molecules) == len(crystal_single_water.molecules)
        assert new_crystal is crystal_single_water

    def test_identify_empty_crystal(self, empty_crystal):
        molecules = identify_molecules(empty_crystal)
        assert len(molecules) == 0

    def test_assign_empty_crystal(self, empty_crystal):
        new_crystal = assign_atoms_to_molecules(empty_crystal)
        assert new_crystal is empty_crystal
        assert len(new_crystal.molecules) == 0


# ----- Interactions -----


class TestHydrogenBond:
    """HydrogenBond and find_hydrogen_bonds."""

    def test_hydrogen_bond_initialization(self):
        mol1 = Atoms(
            symbols=["N", "H", "H", "H"],
            positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        )
        mol2 = Atoms(symbols=["O", "H"], positions=[[3, 0, 0], [3.5, 0, 0]])
        cm1 = CrystalMolecule(mol1)
        cm2 = CrystalMolecule(mol2)
        hb = HydrogenBond(
            donor=cm1,
            acceptor=cm2,
            distance=2.0,
            donor_atom_index=0,
            hydrogen_index=1,
            acceptor_atom_index=0,
        )
        assert hb.donor is cm1
        assert hb.acceptor is cm2
        assert hb.distance == 2.0
        assert hb.donor_atom_index == 0
        assert hb.hydrogen_index == 1
        assert hb.acceptor_atom_index == 0
        r = repr(hb)
        assert "HydrogenBond" in r and "2.000" in r

    def test_find_hydrogen_bonds_returns_list(self, crystal_single_water, cubic_lattice_10):
        water2 = Atoms(
            symbols=["O", "H", "H"],
            positions=[[2.7, 0, 0], [3.2, 0, 0], [2.7, 0.8, 0]],
        )
        molecules = [
            crystal_single_water.molecules[0],
            CrystalMolecule(water2),
        ]
        hbonds = find_hydrogen_bonds(molecules, max_distance=3.0)
        assert isinstance(hbonds, list)

    def test_no_hydrogen_bonds_when_far(self, cubic_lattice_10):
        w1 = CrystalMolecule(
            Atoms(
                symbols=["O", "H", "H"],
                positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
            )
        )
        w2 = CrystalMolecule(
            Atoms(
                symbols=["O", "H", "H"],
                positions=[[20, 0, 0], [20.757, 0.586, 0], [19.243, 0.586, 0]],
            )
        )
        hbonds = find_hydrogen_bonds([w1, w2], max_distance=3.0)
        assert len(hbonds) == 0


class TestGetBondingThreshold:
    """get_bonding_threshold for coverage."""

    def test_metal_metal(self):
        t = get_bonding_threshold(1.0, 1.0, True, True)
        assert t == 1.0  # (1+1)*0.5

    def test_nonmetal_nonmetal(self):
        t = get_bonding_threshold(1.0, 1.0, False, False)
        assert abs(t - 2.5) < 1e-10  # (1+1)*1.25

    def test_metal_nonmetal(self):
        t = get_bonding_threshold(1.0, 1.0, True, False)
        assert t > 0
        assert t == (1.0 + 1.0) * (0.5 + 1.25) / 2


# ----- Stoichiometry -----


class TestStoichiometryAnalyzer:
    """StoichiometryAnalyzer species map and simplest unit."""

    def test_identification(self, simple_crystal):
        analyzer = StoichiometryAnalyzer(simple_crystal)
        assert len(analyzer.species_map) == 2
        assert len(analyzer.species_map["CO_1"]) == 2
        assert len(analyzer.species_map["N2_1"]) == 4

    def test_simplest_unit(self, simple_crystal):
        analyzer = StoichiometryAnalyzer(simple_crystal)
        unit = analyzer.get_simplest_unit()
        assert unit == {"CO_1": 1, "N2_1": 2}

    def test_isomer_distinction(self):
        lattice = np.array([[20.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        butane = Atoms(
            "CCCC",
            positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]],
        )
        isobutane = Atoms(
            "CCCC",
            positions=[[0, 2, 0], [3, 2, 0], [1.5, 2, 0], [1.5, 3.5, 0]],
        )
        crystal = MolecularCrystal(
            lattice,
            [CrystalMolecule(butane), CrystalMolecule(isobutane)],
        )
        analyzer = StoichiometryAnalyzer(crystal)
        assert len(analyzer.species_map) == 2
        ids = list(analyzer.species_map.keys())
        formulas = [sid.split("_")[0] for sid in ids]
        assert len(set(formulas)) == 1
        assert len(set(ids)) == 2
