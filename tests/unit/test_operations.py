"""
Unit tests for molcrys_kit.operations (defects, hydrogen_completion, rotation).
"""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations.defects import VacancyGenerator
from molcrys_kit.operations.hydrogen_completion import (
    HydrogenCompleter,
    add_hydrogens,
)
from molcrys_kit.operations.rotation import (
    rotate_molecule_at_center,
    rotate_molecule_at_com,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.analysis.chemical_env import ChemicalEnvironment


# ----- Defects -----


class TestVacancyGenerator:
    """VacancyGenerator and generate_vacancy."""

    def test_target_spec(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        new_crystal = gen.generate_vacancy(target_spec={"CO_1": 1, "N2_1": 2})
        assert len(new_crystal.molecules) == 3
        orig_atoms = sum(len(m) for m in simple_crystal.molecules)
        new_atoms = sum(len(m) for m in new_crystal.molecules)
        assert orig_atoms - new_atoms == 6

    def test_with_seed_index(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        indices = gen.find_removable_cluster_indices({"CO_1": 1}, 0)
        assert 0 in indices
        assert len(indices) == 1

    def test_find_removable_cluster_indices(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        indices = gen.find_removable_cluster_indices({"CO_1": 1, "N2_1": 1})
        assert len(indices) == 2
        for idx in indices:
            assert 0 <= idx < len(simple_crystal.molecules)

    def test_no_target_spec_uses_simplest_unit(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        new_crystal = gen.generate_vacancy()
        assert len(new_crystal.molecules) == len(simple_crystal.molecules) - 3

    def test_invalid_target_spec_raises(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        with pytest.raises(ValueError):
            gen.generate_vacancy(target_spec={"CO_1": 10})

    def test_invalid_seed_index_raises(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        with pytest.raises(ValueError):
            gen.generate_vacancy(
                target_spec={"N2_1": 1},
                seed_index=0,
            )

    def test_return_removed_cluster(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        new_crystal, removed = gen.generate_vacancy(
            target_spec={"CO_1": 1, "N2_1": 2},
            return_removed_cluster=True,
        )
        assert len(new_crystal.molecules) == 3
        assert len(removed.molecules) == 3
        assert len(new_crystal.molecules) + len(removed.molecules) == len(
            simple_crystal.molecules
        )
        formulas = [m.get_chemical_formula() for m in removed.molecules]
        assert formulas.count("CO") == 1
        assert formulas.count("N2") == 2
        np.testing.assert_array_equal(removed.lattice, simple_crystal.lattice)
        assert removed.pbc == simple_crystal.pbc

    def test_atom_count_conserved(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        new_crystal, removed = gen.generate_vacancy(
            target_spec={"CO_1": 1, "N2_1": 1},
            return_removed_cluster=True,
        )
        orig = sum(len(m) for m in simple_crystal.molecules)
        new_sum = sum(len(m) for m in new_crystal.molecules)
        rem_sum = sum(len(m) for m in removed.molecules)
        assert orig == new_sum + rem_sum
        assert rem_sum == 4


# ----- Hydrogen completion -----


class TestHydrogenCompleter:
    """HydrogenCompleter and add_hydrogens."""

    def test_initialization(self, crystal_single_water):
        hc = HydrogenCompleter(crystal_single_water)
        assert hc.crystal is crystal_single_water
        assert hc.default_rules["C"]["target_coordination"] == 4
        assert hc.default_rules["N"]["target_coordination"] == 3
        assert hc.default_rules["O"]["target_coordination"] == 2
        assert hc.default_bond_lengths["O-H"] == 0.96

    def test_add_hydrogens_to_ch3(self, cubic_lattice_10):
        ch3 = Atoms(
            symbols=["C", "H", "H", "H"],
            positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        )
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(ch3)])
        new_crystal = add_hydrogens(crystal)
        symbols = new_crystal.molecules[0].get_chemical_symbols()
        assert symbols[0] == "C"
        assert len(symbols) >= 4

    def test_custom_rules(self, cubic_lattice_10):
        nh2 = Atoms(
            symbols=["N", "H", "H"],
            positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        )
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(nh2)])
        hc = HydrogenCompleter(crystal)
        custom = [{"symbol": "N", "target_coordination": 3, "geometry": "trigonal_pyramidal"}]
        new_crystal = hc.add_hydrogens(rules=custom)
        symbols = new_crystal.molecules[0].get_chemical_symbols()
        assert symbols[0] == "N"
        assert len(symbols) >= 3

    def test_find_matching_rule(self, ch2_atoms, cubic_lattice_10):
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [CrystalMolecule(ch2_atoms)],
        )
        hc = HydrogenCompleter(crystal)
        chem_env = ChemicalEnvironment(crystal.molecules[0])
        # No match: C with O neighbor
        rule = hc._find_matching_rule(
            chem_env, 0, "C",
            [{"symbol": "C", "neighbors": ["O"], "target_coordination": 4}],
            {},
        )
        assert rule is None
        # Match: C with H neighbor
        rule = hc._find_matching_rule(
            chem_env, 0, "C",
            [{"symbol": "C", "neighbors": ["H"], "target_coordination": 4, "geometry": "tetrahedral"}],
            {"C": {"target_coordination": 3, "geometry": "trigonal_planar"}},
        )
        assert rule is not None
        assert rule["target_coordination"] == 4
        assert rule["geometry"] == "tetrahedral"
        # Fallback general rule
        rule = hc._find_matching_rule(
            chem_env, 0, "C",
            [{"symbol": "N", "neighbors": ["H"], "target_coordination": 3}],
            {"C": {"target_coordination": 3, "geometry": "trigonal_planar"}},
        )
        assert rule is not None
        assert rule["target_coordination"] == 3

    def test_custom_bond_lengths(self, cubic_lattice_10):
        oh = Atoms(symbols=["O", "H"], positions=[[0, 0, 0], [0, 0, 1]])
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(oh)])
        hc = HydrogenCompleter(crystal)
        new_crystal = hc.add_hydrogens(bond_lengths={"O-H": 1.1})
        assert len(new_crystal.molecules) == 1


# ----- Rotation -----


class TestRotation:
    """rotate_molecule_at_center and rotate_molecule_at_com."""

    @pytest.fixture
    def water_mol(self, water_atoms):
        return CrystalMolecule(water_atoms)

    def test_rotate_at_center_keeps_centroid(self, water_mol):
        orig_pos = water_mol.get_positions().copy()
        orig_centroid = water_mol.get_centroid().copy()
        axis = np.array([0.0, 0.0, 1.0])
        rotate_molecule_at_center(water_mol, axis, 90.0)
        np.testing.assert_allclose(
            water_mol.get_centroid(), orig_centroid, rtol=1e-7, atol=1e-10
        )
        assert not np.allclose(water_mol.get_positions(), orig_pos)

    def test_rotate_at_com_keeps_com(self, water_mol):
        orig_pos = water_mol.get_positions().copy()
        orig_com = water_mol.get_center_of_mass().copy()
        axis = np.array([0.0, 0.0, 1.0])
        rotate_molecule_at_com(water_mol, axis, 90.0)
        np.testing.assert_allclose(
            water_mol.get_center_of_mass(), orig_com, rtol=1e-7, atol=1e-10
        )
        assert not np.allclose(water_mol.get_positions(), orig_pos)

    def test_rotate_180_changes_positions(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        rotate_molecule_at_center(mol, np.array([0, 0, 1]), 180.0)
        new = mol.get_positions()
        assert not np.allclose(new[1], orig[1])
        assert not np.allclose(new[2], orig[2])

    def test_rotate_different_axis(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        rotate_molecule_at_com(mol, np.array([1, 0, 0]), 90.0)
        assert not np.allclose(mol.get_positions(), orig)
