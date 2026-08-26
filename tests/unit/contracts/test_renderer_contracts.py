"""Renderer-facing structural contracts and crystallographic ADP handling."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from ase import Atoms

from molcrys_kit.analysis.interactions import LocalGeometry
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.constants.config import (
    KEY_ASYM_ID,
    KEY_LABEL,
    KEY_SYM_OP_INDEX,
    KEY_U_CART,
    KEY_UISO,
)
from molcrys_kit.io import read_extxyz, read_mol_crystal, write_cif, write_extxyz
from molcrys_kit.io.cif import identify_molecules
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


CASE_INSENSITIVE_U_ADP_CIF = TRICLINIC_ADP_CIF.replace(
    "_atom_site_U_iso_or_equiv", "_atom_site_u_ISO_or_equiv"
).replace("_atom_site_aniso_U_", "_atom_site_ANISO_u_")


CASE_INSENSITIVE_B_ADP_CIF = B_ADP_CIF.replace(
    "_atom_site_B_iso_or_equiv", "_atom_site_b_ISO_or_equiv"
).replace("_atom_site_aniso_B_", "_atom_site_ANISO_b_")


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


def test_supercell_rebases_periodic_site_and_bond_images():
    atoms = Atoms(
        "CC",
        positions=[[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.set_array(KEY_ASYM_ID, np.array([7, 8]))
    atoms.set_array(KEY_SYM_OP_INDEX, np.array([3, 4]))
    supercell = MolecularCrystal.from_ase(atoms).get_supercell(2, 1, 1)

    site_records = supercell.get_site_records()
    sites_by_molecule = {
        molecule_index: [
            site.image_shift
            for site in site_records
            if site.molecule_index == molecule_index
        ]
        for molecule_index in range(2)
    }
    bonds = supercell.get_bond_records()

    assert sites_by_molecule == {
        0: [(0, 0, 0), (0, 0, 0)],
        1: [(0, 0, 0), (1, 0, 0)],
    }
    assert [bond.right_image_shift for bond in bonds] == [
        (0, 0, 0),
        (1, 0, 0),
    ]
    assert all(bond.vector_A == pytest.approx((0.4, 0.0, 0.0)) for bond in bonds)
    assert all("bond_records" not in molecule.info for molecule in supercell.molecules)
    assert [
        [(site.asym_index, site.sym_op_index) for site in site_records if site.molecule_index == i]
        for i in range(2)
    ] == [[(7, 3), (8, 4)], [(7, 3), (8, 4)]]


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


def test_bond_records_preserve_periodic_framework_cycle_edges():
    cell = np.array(
        [
            [10.0585, 0.0, 0.0],
            [-5.02925, 8.71091652, 0.0],
            [0.0, 0.0, 6.795],
        ]
    )
    atoms = Atoms(
        symbols=["Cd", "Cd", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl"],
        scaled_positions=[
            [0.0, 0.0, 0.25182],
            [0.0, 0.0, 0.75182],
            [0.2332, 0.1166, 0.001141],
            [0.1166, 0.2332, 0.501141],
            [0.8834, 0.1166, 0.001141],
            [0.7668, 0.8834, 0.501141],
            [0.8834, 0.7668, 0.001141],
            [0.1166, 0.8834, 0.501141],
        ],
        cell=cell,
        pbc=True,
    )
    crystal = MolecularCrystal(cell, identify_molecules(atoms))

    record = next(
        record
        for record in crystal.get_bond_records()
        if (record.left_global_index, record.right_global_index) == (1, 2)
    )

    assert record.right_image_shift == (0, 0, 1)
    assert record.distance_A == pytest.approx(2.6451135)


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


def test_cif_adp_tag_lookup_is_case_insensitive():
    u_site = read_mol_crystal(
        cif_text=CASE_INSENSITIVE_U_ADP_CIF
    ).get_site_records()[0]
    b_site = read_mol_crystal(
        cif_text=CASE_INSENSITIVE_B_ADP_CIF
    ).get_site_records()[0]

    assert u_site.uiso_A2 == pytest.approx(0.025)
    assert np.asarray(u_site.u_cart_A2) == pytest.approx(TRICLINIC_U_CART_REFERENCE)
    assert b_site.uiso_A2 == pytest.approx(0.04)
    assert np.asarray(b_site.u_cart_A2) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )


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


def test_partial_molecule_adp_survives_ase_and_extxyz_round_trips(tmp_path):
    with_adp = Atoms("C", positions=[[1.0, 1.0, 1.0]])
    with_adp.set_array(KEY_UISO, np.array([0.02]))
    with_adp.set_array(KEY_U_CART, np.diag([0.01, 0.02, 0.03]).reshape(1, 9))
    without_adp = Atoms("N", positions=[[5.0, 5.0, 5.0]])
    crystal = MolecularCrystal(np.eye(3) * 10.0, [with_adp, without_adp])

    flat = crystal.to_ase()
    assert flat.arrays[KEY_UISO][0] == pytest.approx(0.02)
    assert np.isnan(flat.arrays[KEY_UISO][1])
    assert flat.arrays[KEY_U_CART][0].reshape(3, 3) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )
    assert np.all(np.isnan(flat.arrays[KEY_U_CART][1]))

    from_ase = MolecularCrystal.from_ase_atoms(flat)
    ase_records = {record.symbol: record for record in from_ase.get_site_records()}
    assert ase_records["C"].uiso_A2 == pytest.approx(0.02)
    assert ase_records["N"].uiso_A2 is None
    assert ase_records["N"].u_cart_A2 is None

    path = tmp_path / "partial_adp.extxyz"
    write_extxyz(crystal, str(path))
    from_extxyz = read_extxyz(str(path))
    extxyz_records = {
        record.symbol: record for record in from_extxyz.get_site_records()
    }
    assert extxyz_records["C"].uiso_A2 == pytest.approx(0.02)
    assert np.asarray(extxyz_records["C"].u_cart_A2) == pytest.approx(
        np.diag([0.01, 0.02, 0.03])
    )
    assert extxyz_records["N"].uiso_A2 is None
    assert extxyz_records["N"].u_cart_A2 is None


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
            molecule("CC", (0.0, 0.0, 0.0)),
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


def test_formula_unit_selection_scores_all_equivalent_anchors():
    lattice = np.eye(3) * 100.0

    def molecule(symbols: str, centroid_x: float) -> CrystalMolecule:
        positions = np.zeros((len(symbols), 3), dtype=float)
        positions[:, 0] = centroid_x
        if len(symbols) == 2:
            positions[:, 0] += np.array([-0.6, 0.6])
        return CrystalMolecule(Atoms(symbols, positions=positions), check_pbc=False)

    crystal = MolecularCrystal(
        lattice,
        [
            molecule("CC", 0.0),
            molecule("CC", 40.0),
            molecule("N", 20.0),
            molecule("N", 41.0),
        ],
    )

    selection = StoichiometryAnalyzer(crystal).select_formula_unit()
    shifted_centroids = [
        crystal.molecules[member.molecule_index].get_centroid()
        + np.asarray(member.image_shift) @ lattice
        for member in selection.members
    ]

    assert selection.molecule_indices == (1, 3)
    assert np.linalg.norm(shifted_centroids[0] - shifted_centroids[1]) == pytest.approx(
        1.0
    )


def test_formula_unit_selection_uses_global_mic_in_highly_skew_cell():
    lattice = np.array(
        [[1.0, 0.0, 0.0], [10.1, 0.1, 0.0], [0.0, 0.0, 10.0]]
    )

    def molecule(symbols: str, centroid) -> CrystalMolecule:
        positions = np.tile(np.asarray(centroid, dtype=float), (len(symbols), 1))
        if len(symbols) == 2:
            positions += np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
        return CrystalMolecule(Atoms(symbols, positions=positions), check_pbc=False)

    crystal = MolecularCrystal(
        lattice,
        [
            molecule("CC", (0.0, 0.0, 0.0)),
            molecule("CC", (0.0, 0.0, 0.0)),
            molecule("N", (-0.06, -0.05, 0.0)),
            molecule("N", (0.4, 0.0, 0.0)),
        ],
    )

    selection = StoichiometryAnalyzer(crystal).select_formula_unit()
    nitrogen = next(
        member for member in selection.members if member.species_id.startswith("N_")
    )
    shifted = (
        crystal.molecules[nitrogen.molecule_index].get_centroid()
        + np.asarray(nitrogen.image_shift) @ lattice
    )

    assert nitrogen.molecule_index == 2
    assert nitrogen.image_shift == (-10, 1, 0)
    assert shifted == pytest.approx((0.04, 0.05, 0.0))
