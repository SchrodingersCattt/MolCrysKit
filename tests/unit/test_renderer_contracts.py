"""Renderer-facing structural contracts and crystallographic ADP handling."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from ase import Atoms

from molcrys_kit.analysis.interactions import LocalGeometry
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.constants.config import KEY_ASYM_ID, KEY_LABEL, KEY_U_CART, KEY_UISO
from molcrys_kit.io import read_extxyz, read_mol_crystal, write_cif, write_extxyz
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


TRICLINIC_ADP_CIF = """\
data_adp
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 70
_cell_angle_beta 80
_cell_angle_gamma 75
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
_atom_site_U_iso_or_equiv
C1 C 0.1 0.2 0.3 1.0 0.025
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
C1 0.010 0.020 0.030 0.002 0.003 0.004
"""


ROTATED_ADP_CIF = """\
data_rotated
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
'-y,x,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.1 0.2 0.3 1.0
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
C1 0.010 0.020 0.030 0 0 0
"""


SPECIAL_POSITION_ADP_CIF = """\
data_special_position
_symmetry_space_group_name_H-M 'P -1'
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_site_symmetry_order
C1 C 0 0 0 0.025 2
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
C1 0.010 0.020 0.030 0 0 0
"""


B_ADP_CIF = """\
data_b_adp
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 8
_cell_length_b 8
_cell_length_c 8
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
_atom_site_B_iso_or_equiv
C1 C 0.2 0.3 0.4 3.158273408
loop_
_atom_site_aniso_label
_atom_site_aniso_B_11
_atom_site_aniso_B_22
_atom_site_aniso_B_33
_atom_site_aniso_B_12
_atom_site_aniso_B_13
_atom_site_aniso_B_23
C1 0.789568352 1.579136704 2.368705056 0 0 0
"""


COINCIDENT_LABELS_CIF = """\
data_coincident
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
C1 C 0.25 0.25 0.25 0.5
C2 C 0.25 0.25 0.25 0.5
"""


SYMMETRY_RELATED_LABELS_CIF = """\
data_symmetry_related_labels
_symmetry_space_group_name_H-M 'P -1'
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
C2 C 0.9 0.8 0.7
"""


PERIODIC_BOND_CIF = """\
data_periodic
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
C1 C 0.98 0.5 0.5
C2 C 0.02 0.5 0.5
"""


TRICLINIC_U_CART_REFERENCE = np.array(
    [
        [0.012365052736, 0.006461842321, 0.008564732704],
        [0.006461842321, 0.020000000000, 0.012091254693],
        [0.008564732704, 0.012091254693, 0.041994794029],
    ]
)


def test_site_and_bond_records_expose_indices_and_periodic_image():
    atoms = Atoms(
        "CC",
        positions=[[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.set_array(KEY_LABEL, np.array(["C1", "C2"]))
    atoms.set_array(KEY_ASYM_ID, np.array([7, 8]))

    crystal = MolecularCrystal.from_ase(atoms)
    sites = crystal.get_site_records()
    bonds = crystal.get_bond_records()

    assert [(site.global_index, site.molecule_index, site.local_index) for site in sites] == [
        (0, 0, 0),
        (1, 0, 1),
    ]
    assert [site.asym_index for site in sites] == [7, 8]
    assert sites[1].image_shift == (1, 0, 0)
    assert len(bonds) == 1
    assert bonds[0].right_image_shift == (1, 0, 0)
    assert bonds[0].vector_A == pytest.approx((0.4, 0.0, 0.0))


def test_missing_adp_is_reported_as_none():
    crystal = MolecularCrystal.from_ase(
        Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    )
    site = crystal.get_site_records()[0]
    assert site.uiso_A2 is None
    assert site.u_cart_A2 is None


def test_added_hydrogens_keep_missing_adp_as_nan():
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10, pbc=True)
    atoms.set_array(KEY_UISO, np.array([0.02]))
    atoms.set_array(KEY_U_CART, np.diag([0.01, 0.02, 0.03]).reshape(1, 9))
    crystal = MolecularCrystal.from_ase(atoms)

    completed = add_hydrogens(
        crystal,
        target_elements=["C"],
        use_formula_moiety=False,
    )
    records = completed.get_site_records()

    carbon = next(record for record in records if record.symbol == "C")
    hydrogens = [record for record in records if record.symbol == "H"]
    assert np.asarray(carbon.u_cart_A2) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )
    assert hydrogens
    assert all(record.uiso_A2 is None for record in hydrogens)
    assert all(record.u_cart_A2 is None for record in hydrogens)


def test_coincident_distinct_asu_labels_are_not_deduplicated(tmp_path):
    path = tmp_path / "coincident_labels.cif"
    path.write_text(COINCIDENT_LABELS_CIF, encoding="utf-8")
    records = MolecularCrystal.from_cif(str(path)).get_site_records()

    assert [record.label for record in records] == ["C1", "C2"]
    assert [record.asym_index for record in records] == [0, 1]
    assert records[0].cartesian_position_A == records[1].cartesian_position_A


def test_symmetry_related_duplicate_asu_rows_share_one_canonical_orbit():
    records = read_mol_crystal(
        cif_text=SYMMETRY_RELATED_LABELS_CIF
    ).get_site_records()

    assert len(records) == 2
    assert {record.sym_op_index for record in records} == {0, 1}


def test_public_bond_contract_reconstructs_without_private_info(tmp_path):
    crystal = read_mol_crystal(cif_text=PERIODIC_BOND_CIF)
    assert [site.image_shift for site in crystal.get_site_records()] == [
        (0, 0, 0),
        (1, 0, 0),
    ]
    assert crystal.get_bond_records()[0].right_image_shift == (1, 0, 0)

    for molecule in crystal.molecules:
        molecule.info.pop("bond_records", None)
        molecule.info.pop("bond_pairs", None)
    assert crystal.get_bond_records()[0].right_image_shift == (1, 0, 0)

    path = tmp_path / "periodic.extxyz"
    write_extxyz(crystal, str(path))
    loaded = read_extxyz(str(path))
    assert loaded.get_bond_records()[0].right_image_shift == (1, 0, 0)


def test_triclinic_cif_adp_is_converted_to_cartesian():
    crystal = read_mol_crystal(cif_text=TRICLINIC_ADP_CIF)
    site = crystal.get_site_records()[0]

    assert site.uiso_A2 == pytest.approx(0.025)
    assert np.asarray(site.u_cart_A2) == pytest.approx(TRICLINIC_U_CART_REFERENCE)


def test_symmetry_rotation_rotates_cartesian_adp():
    crystal = read_mol_crystal(cif_text=ROTATED_ADP_CIF)
    by_symop = {site.sym_op_index: site for site in crystal.get_site_records()}

    assert np.asarray(by_symop[0].u_cart_A2) == pytest.approx(np.diag([0.01, 0.02, 0.03]))
    assert np.asarray(by_symop[1].u_cart_A2) == pytest.approx(np.diag([0.02, 0.01, 0.03]))


def test_asu_first_rotates_cartesian_adp_without_standard_fallback(
    tmp_path, monkeypatch
):
    path = tmp_path / "rotated_adp.cif"
    path.write_text(ROTATED_ADP_CIF, encoding="utf-8")

    def fail_standard_path(*args, **kwargs):
        raise AssertionError("ASU-first unexpectedly used the standard CIF path")

    monkeypatch.setattr("molcrys_kit.io.cif.read_mol_crystal", fail_standard_path)
    crystal = MolecularCrystal.from_cif(str(path), use_asu_first=True)
    by_symop = {site.sym_op_index: site for site in crystal.get_site_records()}

    assert np.asarray(by_symop[0].u_cart_A2) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )
    assert np.asarray(by_symop[1].u_cart_A2) == pytest.approx(
        np.diag([0.02, 0.01, 0.03])
    )


def test_special_position_adp_is_retained_once():
    sites = read_mol_crystal(cif_text=SPECIAL_POSITION_ADP_CIF).get_site_records()

    assert len(sites) == 1
    assert sites[0].site_symmetry_order == 2
    assert sites[0].uiso_A2 == pytest.approx(0.025)
    assert np.asarray(sites[0].u_cart_A2) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )


def test_cif_b_factors_are_converted_to_u():
    site = read_mol_crystal(cif_text=B_ADP_CIF).get_site_records()[0]

    assert site.uiso_A2 == pytest.approx(0.04)
    assert np.asarray(site.u_cart_A2) == pytest.approx(np.diag([0.01, 0.02, 0.03]))


def test_adp_survives_ase_extxyz_supercell_and_cif_round_trips(tmp_path):
    crystal = read_mol_crystal(cif_text=TRICLINIC_ADP_CIF)
    expected = np.asarray(crystal.get_site_records()[0].u_cart_A2)

    flat = crystal.to_ase()
    assert flat.arrays[KEY_UISO][0] == pytest.approx(0.025)
    assert flat.arrays[KEY_U_CART][0].reshape(3, 3) == pytest.approx(expected)
    from_ase = MolecularCrystal.from_ase_atoms(flat)
    assert np.asarray(from_ase.get_site_records()[0].u_cart_A2) == pytest.approx(expected)

    crystal_copy = crystal.copy()
    assert np.asarray(crystal_copy.get_site_records()[0].u_cart_A2) == pytest.approx(
        expected
    )
    crystal_copy.molecules[0].arrays[KEY_U_CART][0, 0] = -1.0
    assert crystal.molecules[0].arrays[KEY_U_CART][0, 0] != -1.0

    extxyz_path = tmp_path / "adp.extxyz"
    write_extxyz(crystal, str(extxyz_path))
    from_extxyz = read_extxyz(str(extxyz_path))
    assert np.asarray(from_extxyz.get_site_records()[0].u_cart_A2) == pytest.approx(expected)

    supercell = crystal.get_supercell(2, 1, 1)
    assert all(
        np.asarray(site.u_cart_A2) == pytest.approx(expected)
        for site in supercell.get_site_records()
    )

    from_cif = read_mol_crystal(cif_text=write_cif(crystal))
    assert np.asarray(from_cif.get_site_records()[0].u_cart_A2) == pytest.approx(
        expected, abs=2e-8
    )


def _ring_molecule(graph: nx.Graph, positions=None) -> CrystalMolecule:
    if positions is None:
        angles = np.linspace(0.0, 2.0 * np.pi, len(graph), endpoint=False)
        positions = np.column_stack(
            (1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(len(graph)))
        )
    atoms = Atoms(
        "C" * len(graph),
        positions=positions,
    )
    molecule = CrystalMolecule(atoms, check_pbc=False)
    molecule._graph = graph.copy()
    nx.set_node_attributes(molecule._graph, "C", "symbol")
    return molecule


def test_ring_geometry_preserves_deterministic_edge_order():
    graph = nx.Graph([(4, 1), (1, 5), (5, 0), (0, 3), (3, 2), (2, 4)])
    edge_order = (0, 3, 2, 4, 1, 5)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    positions = np.zeros((6, 3))
    for atom_index, angle in zip(edge_order, angles):
        positions[atom_index] = (1.4 * np.cos(angle), 1.4 * np.sin(angle), 0.0)
    ring = LocalGeometry(_ring_molecule(graph, positions)).rings()[0]

    assert ring.atom_indices == (0, 1, 2, 3, 4, 5)
    assert ring.is_aromatic is True
    assert ring.cycle_atom_indices[0] == 0
    assert all(
        graph.has_edge(ring.cycle_atom_indices[index], ring.cycle_atom_indices[(index + 1) % 6])
        for index in range(6)
    )
    assert ring.cycle_atom_indices == LocalGeometry(
        _ring_molecule(graph, positions)
    ).rings()[0].cycle_atom_indices


def test_fused_ring_cycles_each_remain_edge_connected():
    graph = nx.Graph(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
         (2, 6), (6, 7), (7, 8), (8, 9), (9, 3)]
    )
    rings = LocalGeometry(_ring_molecule(graph)).rings()

    assert len(rings) == 2
    for ring in rings:
        cycle = ring.cycle_atom_indices
        assert all(graph.has_edge(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle)))


def test_formula_unit_selection_is_stoichiometric_compact_and_deterministic(simple_crystal):
    analyzer = StoichiometryAnalyzer(simple_crystal)
    first = analyzer.select_formula_unit()
    second = analyzer.select_formula_unit()

    assert first == second
    assert first.counts() == {"CO_1": 1, "N2_1": 2}
    assert len(first.members) == 3
    assert [member.species_id for member in first.members].count("CO_1") == 1
    assert [member.species_id for member in first.members].count("N2_1") == 2

    lattice = np.asarray(simple_crystal.lattice)
    shifted_centroids = [
        simple_crystal.molecules[member.molecule_index].get_centroid()
        + np.asarray(member.image_shift) @ lattice
        for member in first.members
    ]
    assert max(
        np.linalg.norm(left - right)
        for left in shifted_centroids
        for right in shifted_centroids
    ) < 6.0


def test_formula_unit_selection_uses_nearest_image_in_triclinic_cell():
    lattice = np.array([[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [1.0, 1.5, 9.0]])

    def molecule(symbols: str, centroid_frac) -> CrystalMolecule:
        centroid = np.asarray(centroid_frac) @ lattice
        offsets = np.zeros((len(symbols), 3))
        if len(symbols) == 2:
            offsets = np.array([[-0.6, 0, 0], [0.6, 0, 0]])
        return CrystalMolecule(
            Atoms(symbols, positions=centroid + offsets), check_pbc=False
        )

    crystal = MolecularCrystal(
        lattice,
        [
            molecule("CC", (0.95, 0.5, 0.5)),
            molecule("N", (0.05, 0.5, 0.5)),
            molecule("N", (0.90, 0.65, 0.5)),
        ],
    )
    selection = StoichiometryAnalyzer(crystal).select_formula_unit()

    nitrogen_members = [member for member in selection.members if member.species_id.startswith("N_")]
    assert len(nitrogen_members) == 2
    assert any(member.image_shift == (1, 0, 0) for member in nitrogen_members)


def test_formula_unit_selection_breaks_equidistant_ties_deterministically():
    lattice = np.eye(3) * 10.0

    def molecule(symbols: str, centroid) -> CrystalMolecule:
        positions = np.tile(np.asarray(centroid, dtype=float), (len(symbols), 1))
        if len(symbols) == 2:
            positions += np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
        return CrystalMolecule(Atoms(symbols, positions=positions), check_pbc=False)

    crystal = MolecularCrystal(
        lattice,
        [
            molecule("CC", (0.0, 0.0, 0.0)),
            molecule("CC", (2.0, 2.0, 2.0)),
            molecule("N", (5.0, 0.0, 0.0)),
            molecule("N", (-5.0, 0.0, 0.0)),
        ],
    )

    selection = StoichiometryAnalyzer(crystal).select_formula_unit()
    nitrogen = next(
        member for member in selection.members if member.species_id.startswith("N_")
    )

    assert nitrogen.molecule_index == 2
    assert nitrogen.image_shift == (-1, 0, 0)
