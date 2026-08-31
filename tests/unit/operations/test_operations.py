"""
Unit tests for molcrys_kit.operations (defects, hydrogen_completion, rotation,
perturbation, desolvation, builders).
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
from molcrys_kit.operations.perturbation import (
    apply_gaussian_displacement_molecule,
    apply_gaussian_displacement_crystal,
    apply_directional_displacement,
    apply_random_rotation,
)
from molcrys_kit.operations.desolvation import Desolvator, remove_solvents
from molcrys_kit.operations.builders import (
    assemble_replica_supercell,
    create_supercell,
    create_supercell_matrix,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.analysis.chemical_env import ChemicalEnvironment
from molcrys_kit.analysis.disorder import DisorderProvenance


# =====================================================================
# Defects
# =====================================================================


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

    def test_unknown_species_id_raises_with_available_species(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        with pytest.raises(
            ValueError, match=r"Species 'BAD_1' not found in crystal"
        ) as exc_info:
            gen.generate_vacancy(target_spec={"BAD_1": 1})
        assert "Available species:" in str(exc_info.value)

    def test_invalid_seed_index_raises(self, simple_crystal):
        gen = VacancyGenerator(simple_crystal)
        with pytest.raises(ValueError):
            gen.generate_vacancy(target_spec={"N2_1": 1}, seed_index=0)

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

    def test_random_seed_reproducible(self, simple_crystal):
        """Same random_seed must always select the same seed molecule."""
        gen = VacancyGenerator(simple_crystal)
        idx_a = gen.find_removable_cluster_indices({"CO_1": 1}, random_seed=0)
        idx_b = gen.find_removable_cluster_indices({"CO_1": 1}, random_seed=0)
        assert idx_a == idx_b

    def test_random_seed_different_seeds_may_differ(self, simple_crystal):
        """Different seeds should be able to produce different selections
        when more than one candidate exists."""
        gen = VacancyGenerator(simple_crystal)
        results = {
            tuple(gen.find_removable_cluster_indices({"CO_1": 1}, random_seed=s))
            for s in range(20)
        }
        # simple_crystal has 2 CO molecules; at least 2 distinct selections possible
        assert len(results) >= 2  # simple_crystal has 2 CO molecules


# =====================================================================
# Hydrogen completion
# =====================================================================


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
        # Simulate CIF-loaded structure with occupancy metadata
        ch3.set_array("occupancy", np.array([1.0, 1.0, 1.0, 1.0]))
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(ch3)])
        new_crystal = add_hydrogens(crystal)
        mol = new_crystal.molecules[0]
        symbols = mol.get_chemical_symbols()
        assert symbols[0] == "C"
        assert len(symbols) >= 4
        # Regression: added H atoms must have occupancy 1.0, not 0.0
        assert "occupancy" in mol.arrays
        assert np.all(mol.arrays["occupancy"] == 1.0), (
            f"H atoms have occupancy != 1.0: {mol.arrays['occupancy']}"
        )

    def test_custom_rules(self, cubic_lattice_10):
        nh2 = Atoms(
            symbols=["N", "H", "H"],
            positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        )
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(nh2)])
        hc = HydrogenCompleter(crystal)
        custom = [
            {"symbol": "N", "target_coordination": 3, "geometry": "trigonal_pyramidal"}
        ]
        new_crystal = hc.add_hydrogens(rules=custom)
        symbols = new_crystal.molecules[0].get_chemical_symbols()
        assert symbols[0] == "N"
        assert len(symbols) >= 3

    def test_find_matching_rule(self, ch2_atoms, cubic_lattice_10):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(ch2_atoms)])
        hc = HydrogenCompleter(crystal)
        chem_env = ChemicalEnvironment(crystal.molecules[0])
        rule = hc._find_matching_rule(
            chem_env,
            0,
            "C",
            [{"symbol": "C", "neighbors": ["O"], "target_coordination": 4}],
            {},
        )
        assert rule is None
        rule = hc._find_matching_rule(
            chem_env,
            0,
            "C",
            [
                {
                    "symbol": "C",
                    "neighbors": ["H"],
                    "target_coordination": 4,
                    "geometry": "tetrahedral",
                }
            ],
            {"C": {"target_coordination": 3, "geometry": "trigonal_planar"}},
        )
        assert rule is not None
        assert rule["target_coordination"] == 4

    def test_custom_bond_lengths(self, cubic_lattice_10):
        oh = Atoms(symbols=["O", "H"], positions=[[0, 0, 0], [0, 0, 1]])
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(oh)])
        hc = HydrogenCompleter(crystal)
        new_crystal = hc.add_hydrogens(bond_lengths={"O-H": 1.1})
        assert len(new_crystal.molecules) == 1

    def test_formula_constraint_skips_unmatched_inorganic_fragment(
        self, cubic_lattice_10
    ):
        organic = Atoms(symbols=["C"], positions=[[0, 0, 0]])
        inorganic = Atoms(symbols=["S"], positions=[[5, 0, 0]])
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [CrystalMolecule(organic), CrystalMolecule(inorganic)],
            formula_moiety="C H4",
        )

        with pytest.warns(
            RuntimeWarning, match="No .*fragment matches|Skipping H addition"
        ):
            completed = add_hydrogens(crystal, use_formula_moiety=True)

        assert completed.molecules[0].get_chemical_formula() == "CH4"
        assert completed.molecules[1].get_chemical_formula() == "S"

    def test_formula_constraint_completes_organic_fragment_without_hydrogenating_framework(
        self, cubic_lattice_10
    ):
        """A sanitized organic/inorganic pair mirrors a split molecular salt.

        The formula constraint must complete the organic CNO fragment to
        C2H7NO while leaving the matched-external inorganic framework H-free.
        No CSD identifiers, coordinates, or source chemistry are used.
        """
        organic = Atoms(
            symbols=["C", "C", "N", "O"],
            positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [0, 1.4, 0]],
        )
        framework = Atoms(
            symbols=["Cd", "S", "C", "N"],
            positions=[[5, 0, 0], [6.5, 0, 0], [8.0, 0, 0], [9.2, 0, 0]],
        )
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [CrystalMolecule(organic), CrystalMolecule(framework)],
            formula_moiety="C2 H7 N1 O1",
        )

        with pytest.warns(
            RuntimeWarning, match="No .*fragment matches|Skipping H addition"
        ):
            completed = add_hydrogens(crystal, use_formula_moiety=True)

        formulas = sorted(
            molecule.get_chemical_formula() for molecule in completed.molecules
        )
        assert formulas == ["C2H7NO", "CCdNS"]

    def test_placement_vector_shortfall_raises(self, cubic_lattice_10, monkeypatch):
        carbon = Atoms(symbols=["C"], positions=[[0, 0, 0]])
        crystal = MolecularCrystal(
            cubic_lattice_10,
            [CrystalMolecule(carbon)],
            formula_moiety="C H4",
        )
        monkeypatch.setattr(
            "molcrys_kit.operations.hydrogen_completion.get_missing_vectors",
            lambda *args, **kwargs: [np.array([1.0, 0.0, 0.0])],
        )

        with pytest.raises(ValueError, match="fewer vectors than requested"):
            add_hydrogens(crystal, use_formula_moiety=True)

    def test_unconstrained_vector_shortfall_warns_and_clips(
        self, cubic_lattice_10, monkeypatch
    ):
        carbon = Atoms(symbols=["C"], positions=[[0, 0, 0]])
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(carbon)])
        monkeypatch.setattr(
            "molcrys_kit.operations.hydrogen_completion.get_missing_vectors",
            lambda *args, **kwargs: [np.array([1.0, 0.0, 0.0])],
        )

        with pytest.warns(RuntimeWarning, match="placing only the available"):
            completed = add_hydrogens(crystal, use_formula_moiety=False)

        assert completed.molecules[0].get_chemical_formula() == "CH"


# =====================================================================
# Rotation
# =====================================================================


class TestRotation:
    """rotate_molecule_at_center and rotate_molecule_at_com."""

    @pytest.fixture
    def water_mol(self, water_atoms):
        return CrystalMolecule(water_atoms)

    def test_rotate_at_center_keeps_centroid(self, water_mol):
        orig_centroid = water_mol.get_centroid().copy()
        rotate_molecule_at_center(water_mol, np.array([0, 0, 1.0]), 90.0)
        np.testing.assert_allclose(water_mol.get_centroid(), orig_centroid, atol=1e-10)

    def test_rotate_at_com_keeps_com(self, water_mol):
        orig_com = water_mol.get_center_of_mass().copy()
        rotate_molecule_at_com(water_mol, np.array([0, 0, 1.0]), 90.0)
        np.testing.assert_allclose(water_mol.get_center_of_mass(), orig_com, atol=1e-10)

    def test_rotate_180_changes_positions(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        rotate_molecule_at_center(mol, np.array([0, 0, 1]), 180.0)
        assert not np.allclose(mol.get_positions()[1], orig[1])

    def test_rotate_different_axis(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        rotate_molecule_at_com(mol, np.array([1, 0, 0]), 90.0)
        assert not np.allclose(mol.get_positions(), orig)


# =====================================================================
# Perturbation
# =====================================================================


class TestPerturbation:
    """Gaussian/directional displacement and random rotation."""

    def test_gaussian_displacement_molecule(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        np.random.seed(42)
        apply_gaussian_displacement_molecule(mol, sigma=0.01)
        assert not np.allclose(mol.get_positions(), orig)
        np.testing.assert_allclose(mol.get_positions(), orig, atol=0.2)

    def test_gaussian_displacement_crystal(self, crystal_single_water):
        orig_pos = crystal_single_water.molecules[0].get_positions().copy()
        np.random.seed(42)
        apply_gaussian_displacement_crystal(crystal_single_water, sigma=0.01)
        assert not np.allclose(
            crystal_single_water.molecules[0].get_positions(), orig_pos
        )

    def test_directional_displacement(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig = mol.get_positions().copy()
        apply_directional_displacement(mol, np.array([1.0, 0, 0]), 0.5)
        new_pos = mol.get_positions()
        np.testing.assert_allclose(new_pos[:, 0], orig[:, 0] + 0.5, atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 1], orig[:, 1], atol=1e-10)

    def test_random_rotation_preserves_centroid(self, water_atoms):
        mol = CrystalMolecule(water_atoms)
        orig_centroid = mol.get_centroid().copy()
        np.random.seed(42)
        apply_random_rotation(mol, max_angle=10.0)
        np.testing.assert_allclose(mol.get_centroid(), orig_centroid, atol=1e-10)


# =====================================================================
# Desolvation
# =====================================================================


class TestDesolvation:
    """Desolvator and remove_solvents."""

    @pytest.fixture
    def crystal_with_water(self, cubic_lattice_10):
        water = CrystalMolecule(
            Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        )
        co = CrystalMolecule(Atoms("CO", positions=[[5, 5, 5], [6.2, 5, 5]]))
        return MolecularCrystal(cubic_lattice_10, [co, water])

    def test_remove_by_name(self, crystal_with_water):
        result = Desolvator.remove_solvents(crystal_with_water, ["Water"])
        assert len(result.molecules) < len(crystal_with_water.molecules)

    def test_remove_by_formula(self, crystal_with_water):
        result = remove_solvents(crystal_with_water, ["H2O"])
        assert len(result.molecules) < len(crystal_with_water.molecules)

    def test_empty_result_raises(self, cubic_lattice_10):
        water = CrystalMolecule(
            Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        )
        crystal = MolecularCrystal(cubic_lattice_10, [water])
        with pytest.raises(ValueError, match="empty"):
            remove_solvents(crystal, ["Water"])

    def test_no_match_unchanged(self, crystal_with_water):
        result = remove_solvents(crystal_with_water, ["Benzene"])
        assert len(result.molecules) == len(crystal_with_water.molecules)


# =====================================================================
# Builders
# =====================================================================


class TestBuilders:
    """create_supercell."""

    def test_create_supercell(self, cubic_lattice_10, co_molecule):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(co_molecule)])
        supercell = create_supercell(crystal, (2, 1, 1))
        assert isinstance(supercell, MolecularCrystal)
        total_atoms = sum(len(m) for m in supercell.molecules)
        assert total_atoms == 4

    @pytest.mark.parametrize(
        ("matrix", "determinant"),
        [
            ([[3, 0, 0], [0, 2, 2], [0, -2, 2]], 24),
            ([[3, 2, 0], [-3, 2, 0], [0, 0, 2]], 24),
            ([[3, 1, 0], [-3, 1, 0], [0, 0, 4]], 24),
            ([[2, 0, 0], [0, 3, 0], [0, 0, 2]], 12),
        ],
    )
    def test_create_supercell_matrix_counts_and_lattice(
        self, cubic_lattice_10, co_molecule, matrix, determinant
    ):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(co_molecule)])
        supercell = create_supercell_matrix(crystal, matrix)
        np.testing.assert_allclose(
            supercell.lattice, np.asarray(matrix) @ crystal.lattice
        )
        assert len(supercell.molecules) == determinant
        assert sum(len(molecule) for molecule in supercell.molecules) == 2 * determinant
        assert supercell.metadata["supercell"]["determinant"] == determinant

    def test_create_supercell_matrix_keeps_cross_boundary_molecule_intact(
        self, cubic_lattice_10
    ):
        molecule = CrystalMolecule(
            Atoms("HH", positions=[[9.8, 0.0, 0.0], [10.2, 0.0, 0.0]])
        )
        crystal = MolecularCrystal(cubic_lattice_10, [molecule])
        supercell = create_supercell_matrix(crystal, [[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        assert len(supercell.molecules) == 2
        for replica in supercell.molecules:
            assert len(replica) == 2
            assert replica.get_distance(0, 1) == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
    )
    def test_create_supercell_matrix_rejects_non_right_handed(
        self, cubic_lattice_10, co_molecule, matrix
    ):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(co_molecule)])
        with pytest.raises(ValueError, match="right-handed"):
            create_supercell_matrix(crystal, matrix)

    def test_create_supercell_allows_incomplete_periodic_topology(
        self, cubic_lattice_10, co_molecule
    ):
        molecule = CrystalMolecule(co_molecule)
        molecule.info["unwrap_completed"] = False
        crystal = MolecularCrystal(cubic_lattice_10, [molecule])
        supercell = create_supercell(crystal, (2, 1, 1))
        assert len(supercell.molecules) == 2

    def test_repeated_supercell_preserves_build_history(
        self, cubic_lattice_10, co_molecule
    ):
        crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(co_molecule)])
        first = create_supercell(crystal, (2, 1, 1))
        second = create_supercell(first, (1, 3, 1))

        assert first.metadata["supercell_history"] == [first.metadata["supercell"]]
        assert [
            item["scaling_factors"] for item in second.metadata["supercell_history"]
        ] == [[2, 1, 1], [1, 3, 1]]
        assert second.metadata["supercell_history"][0]["source_atom_count"] == 2
        assert second.metadata["supercell_history"][1]["source_atom_count"] == 4


class TestReplicaSupercell:
    """assemble_replica_supercell."""

    @staticmethod
    def _replicas(lattice):
        replicas = []
        for index in range(4):
            molecule = Atoms(
                "CO",
                positions=[
                    [float(index), 0.0, 0.0],
                    [float(index) + 1.2, 0.0, 0.0],
                ],
            )
            replicas.append(
                MolecularCrystal(
                    lattice,
                    [molecule],
                    disorder_provenance=DisorderProvenance(
                        kept_indices=[index],
                        dropped_indices=[],
                        method="enumerate",
                        coupled=False,
                    ),
                    extra_arrays={
                        "source_rank": np.asarray([index * 10, index * 10 + 1])
                    },
                )
            )
        return replicas

    def test_assembles_2x2x1_mapping_and_provenance(self, cubic_lattice_10):
        replicas = self._replicas(cubic_lattice_10)

        result = assemble_replica_supercell(
            replicas,
            scaling_factors=(2, 2, 1),
            replica_indices=[0, 1, 2, 3],
        )

        np.testing.assert_allclose(result.lattice, np.diag([20.0, 20.0, 10.0]))
        np.testing.assert_allclose(
            [molecule.positions[0] for molecule in result.molecules],
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 10.0, 0.0],
            ],
        )
        assert [
            (cell["translation"], cell["replica_index"])
            for cell in result.metadata["replica_supercell"]["cells"]
        ] == [
            ([0, 0, 0], 0),
            ([0, 1, 0], 1),
            ([1, 0, 0], 2),
            ([1, 1, 0], 3),
        ]
        assert result.metadata["replica_supercell"]["cells"][1]["disorder_provenance"][
            "kept_indices"
        ] == [1]
        np.testing.assert_array_equal(
            result.extra_arrays["source_rank"],
            [0, 1, 10, 11, 20, 21, 30, 31],
        )
        assert result.disorder_provenance["kind"] == "replica_supercell"
        np.testing.assert_allclose(replicas[1].molecules[0].positions[0], [1, 0, 0])

    def test_repeated_indices_and_k_fastest_order(self, cubic_lattice_10):
        replicas = self._replicas(cubic_lattice_10)

        result = assemble_replica_supercell(replicas, (1, 2, 2), [0, 1, 0, 1])

        cells = result.metadata["replica_supercell"]["cells"]
        assert [cell["translation"] for cell in cells] == [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ]
        assert [cell["replica_index"] for cell in cells] == [0, 1, 0, 1]

    def test_boundary_spanning_molecule_uses_contiguous_image_shifts(
        self, cubic_lattice_10
    ):
        molecule = Atoms(
            "CO",
            positions=[[9.6, 0.0, 0.0], [10.8, 0.0, 0.0]],
        )
        molecule.set_array(
            "image_shift",
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int),
        )
        replica = MolecularCrystal(cubic_lattice_10, [molecule])

        result = assemble_replica_supercell([replica], (2, 1, 1), [0, 0])

        assert {site.image_shift for site in result.get_site_records()} == {(0, 0, 0)}
        assert {bond.right_image_shift for bond in result.get_bond_records()} == {
            (0, 0, 0)
        }
        np.testing.assert_allclose(
            [bond.vector_A for bond in result.get_bond_records()],
            [[1.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        )

    def test_rejects_invalid_mapping_length_and_index(self, cubic_lattice_10):
        replicas = self._replicas(cubic_lattice_10)

        with pytest.raises(ValueError, match="has length 1; expected 2"):
            assemble_replica_supercell(replicas, (2, 1, 1), [0])
        with pytest.raises(ValueError, match="index 4.*out of range"):
            assemble_replica_supercell(replicas, (1, 1, 1), [4])

    def test_rejects_incompatible_replicas(self, cubic_lattice_10):
        replicas = self._replicas(cubic_lattice_10)
        incompatible_lattice = self._replicas(np.diag([11.0, 10.0, 10.0]))[0]

        with pytest.raises(ValueError, match="lattice incompatible"):
            assemble_replica_supercell(
                [replicas[0], incompatible_lattice], (2, 1, 1), [0, 1]
            )

        incompatible_schema = MolecularCrystal(
            cubic_lattice_10,
            [Atoms("NO", positions=[[0, 0, 0], [1.2, 0, 0]])],
            extra_arrays={"source_rank": np.asarray([0, 1])},
        )
        with pytest.raises(ValueError, match="atom/molecule schema incompatible"):
            assemble_replica_supercell(
                [replicas[0], incompatible_schema], (2, 1, 1), [0, 1]
            )
