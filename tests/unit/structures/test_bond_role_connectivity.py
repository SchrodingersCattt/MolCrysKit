"""Renderer-facing bond roles and connectivity regime.

Extended solids must be distinguishable from molecular crystals without the
caller inspecting element symbols.  Edge role selects stick-eligible contacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.io import read_extxyz, read_mol_crystal, write_extxyz
from molcrys_kit.structures import (
    BOND_ROLE_COORDINATION,
    BOND_ROLE_COVALENT,
    CONNECTIVITY_REGIME_EXTENDED,
    CONNECTIVITY_REGIME_MOLECULAR,
    BondRecord,
    classify_bond_role,
)
from molcrys_kit.structures.connectivity import covalent_metal_nonmetal_limit
from molcrys_kit.structures.crystal import MolecularCrystal


NACL_A = 5.64

NACL_CIF = """\
data_nacl
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
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
Na1 Na 0.0 0.0 0.0
Na2 Na 0.5 0.5 0.0
Na3 Na 0.5 0.0 0.5
Na4 Na 0.0 0.5 0.5
Cl1 Cl 0.5 0.0 0.0
Cl2 Cl 0.0 0.5 0.0
Cl3 Cl 0.0 0.0 0.5
Cl4 Cl 0.5 0.5 0.5
"""

ORGANIC_CIF = """\
data_formaldehyde
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 12
_cell_length_b 12
_cell_length_c 12
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
C1 C 0.40 0.50 0.50
O1 O 0.50 0.50 0.50
H1 H 0.36 0.57 0.50
"""


def _rocksalt_nacl() -> Atoms:
    return Atoms(
        symbols=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ],
        cell=np.eye(3) * NACL_A,
        pbc=True,
    )


def _formaldehyde() -> Atoms:
    return Atoms(
        "COH",
        positions=[[0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [-0.55, 0.95, 0.0]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )


def _minimal_bond_record(**overrides) -> BondRecord:
    payload = dict(
        molecule_index=0,
        left_local_index=0,
        right_local_index=1,
        left_global_index=0,
        right_global_index=1,
        left_asym_index=None,
        right_asym_index=None,
        right_image_shift=(0, 0, 0),
        vector_A=(1.0, 0.0, 0.0),
        distance_A=1.0,
    )
    payload.update(overrides)
    return BondRecord(**payload)


def test_classify_bond_role_uses_metal_flags_and_covalent_ceiling():
    assert covalent_metal_nonmetal_limit("Na", "Cl") == pytest.approx(2.05)
    assert classify_bond_role("C", "C", 1.40) == BOND_ROLE_COVALENT
    assert classify_bond_role("C", "H", 1.09) == BOND_ROLE_COVALENT
    assert classify_bond_role("Na", "Cl", 2.82) == BOND_ROLE_COORDINATION
    assert classify_bond_role("Na", "Cl", 2.05) == BOND_ROLE_COVALENT
    assert classify_bond_role("Fe", "C", 1.95) == BOND_ROLE_COVALENT
    assert classify_bond_role("Fe", "C", 2.30) == BOND_ROLE_COORDINATION
    assert classify_bond_role("Fe", "Fe", 1.20) == BOND_ROLE_COORDINATION


def test_bond_record_role_defaults_and_rejects_unknown_labels():
    assert _minimal_bond_record().role == BOND_ROLE_COVALENT
    with pytest.raises(ValueError, match="role"):
        _minimal_bond_record(role="ionic")


def test_ase_rocksalt_is_extended_with_coordination_edges():
    crystal = MolecularCrystal.from_ase(_rocksalt_nacl())
    bonds = crystal.get_bond_records()
    report = crystal.get_connectivity_report()

    assert bonds
    assert {bond.role for bond in bonds} == {BOND_ROLE_COORDINATION}
    assert report.regime == CONNECTIVITY_REGIME_EXTENDED
    assert report.max_rank == 3
    assert [component.regime for component in report.components] == [
        CONNECTIVITY_REGIME_EXTENDED
    ]
    assert report.components[0].rank == 3
    stored_roles = {
        record["role"]
        for molecule in crystal.molecules
        for record in molecule.info["bond_records"]
    }
    assert stored_roles == {BOND_ROLE_COORDINATION}


def test_cif_rocksalt_matches_ase_regime_and_roles():
    loaded = read_mol_crystal(cif_text=NACL_CIF)
    report = loaded.get_connectivity_report()
    bonds = loaded.get_bond_records()

    assert report.regime == CONNECTIVITY_REGIME_EXTENDED
    assert report.max_rank == 3
    assert bonds
    assert {bond.role for bond in bonds} == {BOND_ROLE_COORDINATION}


def test_ase_and_cif_organic_molecule_is_covalent_and_molecular():
    ase_crystal = MolecularCrystal.from_ase(_formaldehyde())
    cif_crystal = read_mol_crystal(cif_text=ORGANIC_CIF)

    for crystal in (ase_crystal, cif_crystal):
        bonds = crystal.get_bond_records()
        report = crystal.get_connectivity_report()
        assert bonds
        assert {bond.role for bond in bonds} == {BOND_ROLE_COVALENT}
        assert report.regime == CONNECTIVITY_REGIME_MOLECULAR
        assert report.max_rank == 0
        assert all(component.rank == 0 for component in report.components)


def test_finite_ion_pair_stays_molecular_with_coordination_role():
    atoms = Atoms(
        "NaCl",
        positions=[[0.0, 0.0, 0.0], [2.82, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    bonds = crystal.get_bond_records()

    assert len(bonds) == 1
    assert bonds[0].role == BOND_ROLE_COORDINATION
    assert crystal.get_connectivity_report().regime == CONNECTIVITY_REGIME_MOLECULAR


def test_short_metal_carbon_contact_is_a_covalent_stick():
    atoms = Atoms(
        "FeC",
        positions=[[0.0, 0.0, 0.0], [1.95, 0.0, 0.0]],
        cell=np.eye(3) * 15.0,
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    bonds = crystal.get_bond_records()

    assert len(bonds) == 1
    assert bonds[0].role == BOND_ROLE_COVALENT
    assert crystal.get_connectivity_report().regime == CONNECTIVITY_REGIME_MOLECULAR


def test_periodic_carbon_chain_is_extended_and_covalent():
    atoms = Atoms(
        "C",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.40, 8.0, 8.0]),
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    bonds = crystal.get_bond_records()
    report = crystal.get_connectivity_report()

    assert len(bonds) == 1
    assert bonds[0].left_global_index == bonds[0].right_global_index
    assert bonds[0].role == BOND_ROLE_COVALENT
    assert report.regime == CONNECTIVITY_REGIME_EXTENDED
    assert report.max_rank == 1
    assert report.components[0].translation_generators == ((1, 0, 0),)


def test_mixed_molecule_and_chain_is_extended_with_per_component_regimes():
    atoms = Atoms(
        "CCCO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [1.4, 6.0, 6.0],
            [1.4, 7.2, 6.0],
        ],
        cell=np.diag([2.8, 12.0, 12.0]),
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    report = crystal.get_connectivity_report()

    assert report.regime == CONNECTIVITY_REGIME_EXTENDED
    assert report.max_rank == 1
    assert {component.regime for component in report.components} == {
        CONNECTIVITY_REGIME_MOLECULAR,
        CONNECTIVITY_REGIME_EXTENDED,
    }
    assert all(bond.role == BOND_ROLE_COVALENT for bond in crystal.get_bond_records())


def test_missing_stored_role_is_classified_on_emission():
    crystal = MolecularCrystal.from_ase(_formaldehyde())
    for molecule in crystal.molecules:
        for record in molecule.info["bond_records"]:
            record.pop("role", None)

    bonds = crystal.get_bond_records()
    assert bonds
    assert {bond.role for bond in bonds} == {BOND_ROLE_COVALENT}


def test_extxyz_round_trip_keeps_coordination_roles(tmp_path):
    crystal = MolecularCrystal.from_ase(_rocksalt_nacl())
    path = tmp_path / "nacl.extxyz"
    write_extxyz(crystal, str(path))
    loaded = read_extxyz(str(path))
    bonds = loaded.get_bond_records()

    assert bonds
    assert {bond.role for bond in bonds} == {BOND_ROLE_COORDINATION}
    assert loaded.get_connectivity_report().regime == CONNECTIVITY_REGIME_EXTENDED
    assert loaded.get_connectivity_report().max_rank == 3
