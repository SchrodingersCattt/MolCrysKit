"""
Unit tests for molcrys_kit.io.cif (read_mol_crystal, parse_cif_advanced, identify_molecules).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import molcrys_kit.io.cif as cif_module
from molcrys_kit.constants import DEFAULT_NEIGHBOR_CUTOFF
from molcrys_kit.constants.config import KEY_DISORDER_GROUP, KEY_SYM_OP_INDEX
from molcrys_kit.io.cif import (
    SymmetryAutoExpandedWarning,
    _parse_symmetry_operations,
    identify_molecules,
    parse_cif_advanced,
    read_mol_crystal,
    scan_cif_disorder,
    # Internal parser imported directly for focused CIF numeric/SU verification.
    _parse_cif_number_with_su,
)
from molcrys_kit.structures.molecule import CrystalMolecule

# Suppress pymatgen/CIF parsing warnings in tests (test data may have occupancy quirks)
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestSymmetryExpansion:
    def test_parse_symmetry_operations_expands_non_p1_with_identity_only(self):
        block = {
            "_space_group_IT_number": "14",
            "_space_group_name_H-M_alt": "P 21/c",
            "_symmetry_equiv_pos_as_xyz": ["x,y,z"],
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ops = _parse_symmetry_operations(block)

        assert len(ops) == 4
        assert any(
            issubclass(item.category, SymmetryAutoExpandedWarning) for item in caught
        )

    def test_parse_symmetry_operations_respects_explicit_p1(self):
        block = {
            "_space_group_IT_number": "1",
            "_space_group_name_H-M_alt": "P 1",
            "_symmetry_equiv_pos_as_xyz": ["x,y,z"],
        }

        ops = _parse_symmetry_operations(block)

        assert len(ops) == 1

    def test_parse_symmetry_operations_expand_symmetry_false_opt_out(self):
        block = {
            "_space_group_IT_number": "14",
            "_space_group_name_H-M_alt": "P 21/c",
            "_symmetry_equiv_pos_as_xyz": ["x,y,z"],
        }

        ops = _parse_symmetry_operations(block, expand_symmetry=False)

        assert len(ops) == 1

    def test_scan_cif_disorder_auto_expands_identity_only_non_p1(self, tmp_path):
        cif = tmp_path / "identity_only_p21c.cif"
        cif.write_text(
            """data_test
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 21/c'
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 100
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
C1 C 0.11 0.22 0.33 1
""",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expanded = scan_cif_disorder(str(cif))
        verbatim = scan_cif_disorder(str(cif), expand_symmetry=False)

        assert len(expanded.labels) == 4
        assert len(verbatim.labels) == 1
        assert any(
            issubclass(item.category, SymmetryAutoExpandedWarning) for item in caught
        )


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

    def test_read_attaches_chemistry_and_preserves_names(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "cif"
            / "Acetaminophen_HXACAN.cif"
        )
        crystal = read_mol_crystal(str(path))

        assert crystal.chemistry is not None
        assert all(
            molecule.chemical_entity is not None for molecule in crystal.molecules
        )
        metadata = crystal.metadata["cif_chemistry"]
        assert metadata["chemical_name_common"] == "Acetaminophen"
        assert metadata["chemical_name_systematic"] == "N-(4-Hydroxyphenyl)acetamide"
        symmetry = crystal.metadata["crystal_symmetry"]
        assert symmetry.operations
        assert symmetry.space_group_number is not None

    def test_read_can_skip_chemistry_attachment(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "cif"
            / "Acetaminophen_HXACAN.cif"
        )

        crystal = read_mol_crystal(str(path), attach_chemistry=False)

        assert crystal.chemistry is None
        assert all(
            molecule.chemical_entity is None for molecule in crystal.molecules
        )

    def test_flack_value_retains_standard_uncertainty(self):
        path = Path(__file__).resolve().parents[2] / "data" / "cif" / "NOKGIH01.cif"
        crystal = read_mol_crystal(str(path))

        flack = crystal.metadata["cif_chemistry"]["absolute_structure"]["flack"]
        assert flack["raw"] == "0.06(3)"
        assert flack["value"] == pytest.approx(0.06)
        assert flack["standard_uncertainty"] == pytest.approx(0.03)

    @pytest.mark.parametrize(
        "raw,value,su",
        [("-0.65(5)", -0.65, 0.05), ("1.234(12)", 1.234, 0.012), ("2.0", 2.0, None)],
    )
    def test_cif_number_parser_preserves_su(self, raw, value, su):
        parsed = _parse_cif_number_with_su(raw)
        assert parsed["value"] == pytest.approx(value)
        if su is None:
            assert parsed["standard_uncertainty"] is None
        else:
            assert parsed["standard_uncertainty"] == pytest.approx(su)


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

    @pytest.mark.parametrize(
        "atoms, bond_scale",
        [
            pytest.param(Atoms(), 1.0, id="empty"),
            pytest.param(
                Atoms("CH", positions=[[0, 0, 0], [1, 0, 0]]),
                0.0,
                id="zero-scale",
            ),
            pytest.param(
                Atoms("CH", positions=[[0, 0, 0], [1, 0, 0]]),
                -1.0,
                id="negative-scale",
            ),
        ],
    )
    def test_empty_or_nonpositive_scale_skips_candidate_generation(
        self, atoms, bond_scale, monkeypatch
    ):
        def unexpected_neighbor_list(*args, **kwargs):
            pytest.fail("neighbor_list should not run without a positive bond cutoff")

        monkeypatch.setattr(cif_module, "neighbor_list", unexpected_neighbor_list)

        graph = cif_module._build_molecule_graph(atoms, bond_scale=bond_scale)

        assert graph.number_of_nodes() == len(atoms)
        assert graph.number_of_edges() == 0

    def test_pair_cutoffs_include_explicit_directed_thresholds_and_scale(
        self, monkeypatch
    ):
        atoms = Atoms(
            symbols=["C", "H"],
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )
        observed = {"cutoffs": []}
        real_neighbor_list = cif_module.neighbor_list

        def recording_neighbor_list(quantities, candidate_atoms, cutoff):
            observed["cutoffs"].append(cutoff)
            return real_neighbor_list(quantities, candidate_atoms, cutoff)

        monkeypatch.setattr(cif_module, "neighbor_list", recording_neighbor_list)

        graph = cif_module._build_molecule_graph(
            atoms,
            bond_thresholds={("C", "H"): 1.0, ("H", "C"): 2.0},
            bond_scale=0.8,
        )

        assert observed["cutoffs"][0][("C", "H")] == pytest.approx(1.6)
        assert graph.has_edge(0, 1)

        negative_graph = cif_module._build_molecule_graph(
            atoms,
            bond_thresholds={("C", "H"): -1.0},
        )

        assert observed["cutoffs"][1][("C", "H")] == pytest.approx(-1.0)
        assert negative_graph.number_of_edges() == 0

    def test_dense_caffeine_disorder_has_bounded_bond_candidates(self, monkeypatch):
        cif_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "cif"
            / "anhydrousCaffeine2_CGD_2007_7_1406.cif"
        )
        disorder_info = scan_cif_disorder(str(cif_path))
        atoms = Atoms(
            symbols=disorder_info.symbols,
            positions=disorder_info.frac_coords @ disorder_info.lattice_matrix,
            cell=disorder_info.lattice_matrix,
            pbc=True,
        )
        atoms.set_array(
            KEY_DISORDER_GROUP,
            np.asarray(disorder_info.disorder_groups, dtype=int),
        )
        atoms.set_array(
            KEY_SYM_OP_INDEX,
            np.asarray(disorder_info.sym_op_indices, dtype=int),
        )
        observed = {"calls": []}
        real_neighbor_list = cif_module.neighbor_list

        def recording_neighbor_list(quantities, candidate_atoms, cutoff):
            result = real_neighbor_list(quantities, candidate_atoms, cutoff)
            observed["calls"].append((cutoff, len(result[0])))
            return result

        monkeypatch.setattr(cif_module, "neighbor_list", recording_neighbor_list)

        graph = cif_module._build_molecule_graph(atoms)

        assert len(atoms) == 936
        cutoff, candidate_count = observed["calls"][0]
        assert max(cutoff.values()) == pytest.approx(1.9)
        # Current ASE yields 14,976 directed candidates. Keep version tolerance
        # while tripping the old global 3.5 Å path (110,700 candidates).
        assert candidate_count < 20_000
        assert graph.number_of_edges() == 1_854

        def global_cutoff_neighbor_list(quantities, candidate_atoms, cutoff):
            return real_neighbor_list(
                quantities,
                candidate_atoms,
                cutoff=DEFAULT_NEIGHBOR_CUTOFF,
            )

        monkeypatch.setattr(
            cif_module,
            "neighbor_list",
            global_cutoff_neighbor_list,
        )
        reference_graph = cif_module._build_molecule_graph(atoms)

        def edge_provenance(candidate_graph):
            return sorted(
                (
                    left,
                    right,
                    tuple(data["vector"]),
                    tuple(data["image_shift"]),
                )
                for left, right, data in candidate_graph.edges(data=True)
            )

        assert edge_provenance(graph) == edge_provenance(reference_graph)

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

    def test_same_part_atoms_from_different_symmetry_ops_do_not_bond(self):
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
        )
        atoms.set_array(KEY_DISORDER_GROUP, np.array([-1, -1], dtype=int))
        atoms.set_array(KEY_SYM_OP_INDEX, np.array([0, 1], dtype=int))

        molecules = identify_molecules(atoms)

        assert len(molecules) == 2
        assert sorted(mol.info["atom_indices"] for mol in molecules) == [[0], [1]]
        assert all(mol.info["bond_pairs"] == [] for mol in molecules)

    def test_same_part_atoms_from_same_symmetry_op_can_bond(self):
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
        )
        atoms.set_array(KEY_DISORDER_GROUP, np.array([-1, -1], dtype=int))
        atoms.set_array(KEY_SYM_OP_INDEX, np.array([0, 0], dtype=int))

        molecules = identify_molecules(atoms)

        assert len(molecules) == 1
        assert molecules[0].info["atom_indices"] == [0, 1]
        assert molecules[0].info["bond_pairs"] == [(0, 1)]

    @pytest.mark.parametrize(
        "cell, positions, expected_shift",
        [
            (
                np.diag([10.0, 10.0, 10.0]),
                [[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]],
                [1, 0, 0],
            ),
            (
                np.array([[8.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.0, 10.0]]),
                None,
                [1, 0, 0],
            ),
        ],
    )
    def test_periodic_bond_records_preserve_signed_image_shift(
        self, cell, positions, expected_shift
    ):
        if positions is None:
            frac = np.array([[0.98, 0.5, 0.5], [0.02, 0.5, 0.5]])
            positions = frac @ cell
        atoms = Atoms(symbols=["C", "C"], positions=positions, cell=cell, pbc=True)

        molecules = identify_molecules(atoms)

        assert len(molecules) == 1
        molecule = molecules[0]
        assert molecule.info["bond_pairs"] == [(0, 1)]
        record = molecule.info["bond_records"]
        assert len(record) == 1
        assert record[0]["left"] == 0
        assert record[0]["right"] == 1
        assert record[0]["right_image_shift"] == expected_shift
        assert record[0]["vector"] == pytest.approx(
            (
                np.asarray(positions[1])
                + np.asarray(expected_shift) @ cell
                - np.asarray(positions[0])
            ).tolist()
        )
