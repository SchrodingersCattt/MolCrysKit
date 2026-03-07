"""
Unit tests for molcrys_kit.analysis (species, interactions, stoichiometry, chemical_env).
"""

import numpy as np
import pytest
import networkx as nx
from ase import Atoms

from molcrys_kit.analysis.species import identify_molecules, assign_atoms_to_molecules
from molcrys_kit.analysis.interactions import (
    HydrogenBond,
    find_hydrogen_bonds,
    get_bonding_threshold,
)
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.analysis.chemical_env import (
    ChemicalEnvironment,
    CarbonSite,
    NitrogenSite,
    GenericSite,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule


# =====================================================================
# Species
# =====================================================================


class TestSpeciesIdentify:
    """identify_molecules(crystal) and assign_atoms_to_molecules."""

    def test_identify_two_waters(self, crystal_single_water, cubic_lattice_10):
        h2o2 = Atoms(
            symbols=["O", "H", "H"],
            positions=[[3.0, 0, 0], [3.757, 0.586, 0], [2.243, 0.586, 0]],
        )
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [crystal_single_water.molecules[0], CrystalMolecule(h2o2)],
        )
        molecules = identify_molecules(crystal)
        assert len(molecules) == 2
        assert all(isinstance(m, CrystalMolecule) for m in molecules)

    def test_assign_atoms_same_crystal(self, crystal_single_water):
        new_crystal = assign_atoms_to_molecules(crystal_single_water)
        np.testing.assert_allclose(new_crystal.lattice, crystal_single_water.lattice)
        assert len(new_crystal.molecules) == len(crystal_single_water.molecules)
        assert new_crystal is crystal_single_water

    def test_identify_empty_crystal(self, empty_crystal):
        assert len(identify_molecules(empty_crystal)) == 0

    def test_assign_empty_crystal(self, empty_crystal):
        new_crystal = assign_atoms_to_molecules(empty_crystal)
        assert new_crystal is empty_crystal
        assert len(new_crystal.molecules) == 0


# =====================================================================
# Interactions
# =====================================================================


class TestHydrogenBond:
    """HydrogenBond and find_hydrogen_bonds."""

    def test_hydrogen_bond_initialization(self):
        mol1 = Atoms(
            symbols=["N", "H", "H", "H"],
            positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        )
        mol2 = Atoms(symbols=["O", "H"], positions=[[3, 0, 0], [3.5, 0, 0]])
        cm1, cm2 = CrystalMolecule(mol1), CrystalMolecule(mol2)
        hb = HydrogenBond(
            donor=cm1,
            acceptor=cm2,
            distance=2.0,
            donor_atom_index=0,
            hydrogen_index=1,
            acceptor_atom_index=0,
        )
        assert hb.donor is cm1
        assert hb.distance == 2.0
        r = repr(hb)
        assert "HydrogenBond" in r and "2.000" in r

    def test_find_hydrogen_bonds_returns_list(self, crystal_single_water, cubic_lattice_10):
        water2 = Atoms(
            symbols=["O", "H", "H"],
            positions=[[2.7, 0, 0], [3.2, 0, 0], [2.7, 0.8, 0]],
        )
        molecules = [crystal_single_water.molecules[0], CrystalMolecule(water2)]
        hbonds = find_hydrogen_bonds(molecules, max_distance=3.0)
        assert isinstance(hbonds, list)

    def test_no_hydrogen_bonds_when_far(self):
        w1 = CrystalMolecule(
            Atoms("OHH", positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        )
        w2 = CrystalMolecule(
            Atoms("OHH", positions=[[20, 0, 0], [20.757, 0.586, 0], [19.243, 0.586, 0]])
        )
        assert len(find_hydrogen_bonds([w1, w2], max_distance=3.0)) == 0


class TestGetBondingThreshold:
    """get_bonding_threshold for coverage."""

    def test_metal_metal(self):
        assert get_bonding_threshold(1.0, 1.0, True, True) == 1.0

    def test_nonmetal_nonmetal(self):
        assert abs(get_bonding_threshold(1.0, 1.0, False, False) - 2.5) < 1e-10

    def test_metal_nonmetal(self):
        t = get_bonding_threshold(1.0, 1.0, True, False)
        assert t == (1.0 + 1.0) * (0.5 + 1.25) / 2


# =====================================================================
# Stoichiometry
# =====================================================================


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
        butane = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
        isobutane = Atoms("CCCC", positions=[[0, 2, 0], [3, 2, 0], [1.5, 2, 0], [1.5, 3.5, 0]])
        crystal = MolecularCrystal(
            lattice, [CrystalMolecule(butane), CrystalMolecule(isobutane)]
        )
        analyzer = StoichiometryAnalyzer(crystal)
        assert len(analyzer.species_map) == 2
        ids = list(analyzer.species_map.keys())
        assert len(set(ids)) == 2

    def test_single_species(self, cubic_lattice_10):
        mol = CrystalMolecule(Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]))
        crystal = MolecularCrystal(cubic_lattice_10, [mol])
        analyzer = StoichiometryAnalyzer(crystal)
        assert len(analyzer.species_map) == 1
        unit = analyzer.get_simplest_unit()
        assert list(unit.values()) == [1]


# =====================================================================
# Chemical Environment
# =====================================================================


def _make_env(symbols, edges, positions):
    """Helper: create a ChemicalEnvironment from explicit graph data."""
    pos = np.array(positions, dtype=float)
    g = nx.Graph()
    for i, sym in enumerate(symbols):
        g.add_node(i, symbol=sym)
    for u, v in edges:
        d = np.linalg.norm(pos[v] - pos[u])
        g.add_edge(u, v, distance=d)
    return ChemicalEnvironment((g, pos))


class TestChemicalEnvironment:
    """ChemicalEnvironment geometry and ring analysis."""

    def test_geometry_stats_isolated_atom(self):
        env = _make_env(["C"], [], [[0, 0, 0]])
        stats = env.get_local_geometry_stats(0)
        assert stats["coordination_number"] == 0
        assert stats["average_bond_length"] == 0.0

    def test_geometry_stats_two_neighbors(self):
        positions = [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]]
        env = _make_env(["C", "H", "H"], [(0, 1), (0, 2)], positions)
        stats = env.get_local_geometry_stats(0)
        assert stats["coordination_number"] == 2
        assert stats["average_bond_length"] == pytest.approx(1.5, abs=0.01)
        assert stats["bond_angle_single"] == pytest.approx(90.0, abs=0.5)

    def test_ring_info_no_ring(self):
        positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        env = _make_env(["C", "C", "C"], [(0, 1), (1, 2)], positions)
        info = env.detect_ring_info(0)
        assert info["in_ring"] is False
        assert info["ring_sizes"] == []

    def test_ring_info_triangle(self):
        positions = [[0, 0, 0], [1, 0, 0], [0.5, 0.87, 0]]
        env = _make_env(["C", "C", "C"], [(0, 1), (1, 2), (2, 0)], positions)
        info = env.detect_ring_info(0)
        assert info["in_ring"] is True
        assert 3 in info["ring_sizes"]

    def test_ring_planarity(self):
        r = 1.4
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        positions = [[r * np.cos(a), r * np.sin(a), 0] for a in angles]
        edges = [(i, (i + 1) % 6) for i in range(6)]
        env = _make_env(["C"] * 6, edges, positions)
        info = env.detect_ring_info(0)
        assert info["in_ring"] is True
        assert info["is_ring_planar"] == True

    def test_get_site_carbon(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        env = _make_env(["C", "H"], [(0, 1)], positions)
        site = env.get_site(0)
        assert isinstance(site, CarbonSite)

    def test_get_site_nitrogen(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        env = _make_env(["N", "H"], [(0, 1)], positions)
        site = env.get_site(0)
        assert isinstance(site, NitrogenSite)

    def test_get_site_generic(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        env = _make_env(["O", "H"], [(0, 1)], positions)
        site = env.get_site(0)
        assert isinstance(site, GenericSite)

    def test_from_crystal_molecule(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        env = ChemicalEnvironment(mol)
        stats = env.get_local_geometry_stats(0)
        assert stats["coordination_number"] == 2


class TestCarbonSite:
    """CarbonSite hydrogen completion strategy."""

    def test_sp2_coord3_planar(self):
        s3 = np.sqrt(3) / 2
        positions = [
            [0, 0, 0],
            [1.4, 0, 0],
            [-0.7, 1.4 * s3, 0],
            [-0.7, -1.4 * s3, 0],
        ]
        env = _make_env(["C", "C", "C", "C"], [(0, 1), (0, 2), (0, 3)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 0
        assert strat["geometry"] == "trigonal_planar"

    def test_sp3_coord3_pyramidal(self):
        s = 1.0 / np.sqrt(3)
        positions = [
            [0, 0, 0],
            [s, s, s],
            [s, -s, -s],
            [-s, s, -s],
        ]
        env = _make_env(["C", "H", "H", "H"], [(0, 1), (0, 2), (0, 3)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 1
        assert strat["geometry"] == "tetrahedral"

    def test_coord2_sp3_long_bonds(self):
        angle = np.radians(109.5)
        positions = [
            [0, 0, 0],
            [1.52, 0, 0],
            [1.52 * np.cos(angle), 1.52 * np.sin(angle), 0],
        ]
        env = _make_env(["C", "C", "C"], [(0, 1), (0, 2)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 2
        assert strat["geometry"] == "tetrahedral"

    def test_coord1_single_bond_methyl(self):
        positions = [[0, 0, 0], [1.54, 0, 0]]
        env = _make_env(["C", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 3
        assert strat["geometry"] == "tetrahedral"

    def test_coord1_double_bond(self):
        positions = [[0, 0, 0], [1.34, 0, 0]]
        env = _make_env(["C", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 2
        assert strat["geometry"] == "trigonal_planar"

    def test_coord1_triple_bond(self):
        positions = [[0, 0, 0], [1.2, 0, 0]]
        env = _make_env(["C", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 1
        assert strat["geometry"] == "linear"


class TestNitrogenSite:
    """NitrogenSite hydrogen completion strategy."""

    def test_coord2_amine(self):
        positions = [[0, 0, 0], [1.47, 0, 0], [0, 1.47, 0]]
        env = _make_env(["N", "C", "C"], [(0, 1), (0, 2)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 1
        assert strat["geometry"] == "tetrahedral"

    def test_coord2_pyridine_like(self):
        r = 1.4
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        positions = [[r * np.cos(a), r * np.sin(a), 0] for a in angles]
        symbols = ["N", "C", "C", "C", "C", "C"]
        edges = [(i, (i + 1) % 6) for i in range(6)]
        env = _make_env(symbols, edges, positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 0

    def test_coord1_primary_amine(self):
        positions = [[0, 0, 0], [1.47, 0, 0]]
        env = _make_env(["N", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 2
        assert strat["geometry"] == "tetrahedral"


class TestGenericSite:
    """GenericSite hydrogen completion strategy (O, S, others)."""

    def test_oxygen_double_bond(self):
        positions = [[0, 0, 0], [1.2, 0, 0]]
        env = _make_env(["O", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 0

    def test_oxygen_single_bond(self):
        positions = [[0, 0, 0], [1.5, 0, 0]]
        env = _make_env(["O", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 1
        assert strat["geometry"] == "bent"

    def test_sulfur_coord1(self):
        positions = [[0, 0, 0], [1.8, 0, 0]]
        env = _make_env(["S", "C"], [(0, 1)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 1
        assert strat["geometry"] == "bent"

    def test_generic_no_completion(self):
        positions = [[0, 0, 0], [2.0, 0, 0], [0, 2.0, 0]]
        env = _make_env(["P", "C", "C"], [(0, 1), (0, 2)], positions)
        site = env.get_site(0)
        strat = site.get_hydrogen_completion_strategy()
        assert strat["num_h"] == 0
