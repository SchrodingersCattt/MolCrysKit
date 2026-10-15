"""User-confirmed values outrank CIF bond types; contradictions stay unlocked."""

import numpy as np
from ase import Atoms

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    infer_bond_orders,
)
from molcrys_kit.chemistry.annotation import annotate_chemistry
from molcrys_kit.chemistry.source import apply_source_precedence, parse_bond_type
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def test_cif_tokens_map_to_orders_or_aromatic_flags():
    assert parse_bond_type("D").order == 2
    assert parse_bond_type("single").order == 1
    assert parse_bond_type("triple").order == 3
    aromatic = parse_bond_type("delocalised")
    assert aromatic is not None and aromatic.aromatic is True and aromatic.order is None
    assert parse_bond_type(".") is None
    assert parse_bond_type("weird") is None


def test_user_confirmed_order_outranks_cif_double():
    bond = ChemicalBond(
        "C1",
        "C2",
        order=1.0,
        kind=BondKind.COVALENT,
        evidence=(
            Evidence(EvidenceSource.USER_CONFIRMED, "user", "order:1"),
            Evidence(EvidenceSource.EXPLICIT_CIF, "cif_geom_bond_type", "bond_type:D"),
        ),
    )
    resolved = apply_source_precedence((bond,))
    assert resolved.bonds[0].order == 1.0
    assert resolved.locked_orders == 1
    assert resolved.contradictions == 1
    assert any("not applied" in note for note in resolved.warnings)


def test_contradictory_cif_types_do_not_lock_an_order():
    bond = ChemicalBond(
        "C1",
        "C2",
        kind=BondKind.COVALENT,
        evidence=(
            Evidence(EvidenceSource.EXPLICIT_CIF, "cif_geom_bond_type", "bond_type:S"),
            Evidence(EvidenceSource.EXPLICIT_CIF, "cif_geom_bond_type", "bond_type:D"),
        ),
    )
    resolved = apply_source_precedence((bond,))
    assert resolved.bonds[0].order is None
    assert resolved.locked_orders == 0
    assert resolved.contradictions == 1
    atoms = (
        ChemicalAtom("C1", "C", explicit_hydrogens=3),
        ChemicalAtom("C2", "C", explicit_hydrogens=2),
        ChemicalAtom("O", "O"),
    )
    # The unlocked C-C bond is still free to satisfy valence; the warning remains.
    entity = FiniteChemicalEntity(
        "contradiction",
        atoms,
        (
            resolved.bonds[0],
            ChemicalBond("C2", "O", kind=BondKind.COVALENT),
        ),
    )
    result = infer_bond_orders(entity)
    assert result.diagnostics["source_contradictions"] == 1
    assert any(
        "contradictory" in note for note in result.diagnostics["source_warnings"]
    )


def test_published_cif_double_bond_is_fixed_on_annotation():
    ase_atoms = Atoms(
        symbols=["C", "C", "H", "H", "H", "H"],
        positions=[
            [-0.67, 0, 0],
            [0.67, 0, 0],
            [-1.2, 0.93, 0],
            [-1.2, -0.93, 0],
            [1.2, 0.93, 0],
            [1.2, -0.93, 0],
        ],
    )
    ase_atoms.set_array(
        "label",
        np.array(["C1", "C2", "H1", "H2", "H3", "H4"]),
    )
    crystal = MolecularCrystal(
        np.eye(3) * 20.0,
        [CrystalMolecule(ase_atoms, check_pbc=False)],
        pbc=(False, False, False),
    )
    crystal.metadata["cif_chemistry"] = {
        "published_bonds": (
            {"atom1_label": "C1", "atom2_label": "C2", "bond_type": "D"},
            {"atom1_label": "C2", "atom2_label": "C1", "bond_type": "S"},
        )
    }
    annotated = annotate_chemistry(crystal)
    carbon = next(
        bond
        for bond in annotated.components[0].bonds
        if {bond.atom1_id, bond.atom2_id}
        == {
            annotated.components[0].atoms[0].atom_id,
            annotated.components[0].atoms[1].atom_id,
        }
    )
    assert carbon.order is None
    assert any("contradictory" in note for note in annotated.warnings)

    crystal.metadata["cif_chemistry"] = {
        "published_bonds": (
            {"atom1_label": "C1", "atom2_label": "C2", "bond_type": "D"},
        )
    }
    annotated = annotate_chemistry(crystal)
    carbon = next(
        bond
        for bond in annotated.components[0].bonds
        if bond.evidence[-1].method == "cif_geom_bond_type"
    )
    assert carbon.order == 2.0
    result = infer_bond_orders(annotated.components[0])
    perceived = next(
        bond
        for bond in result.entity.bonds
        if {bond.atom1_id, bond.atom2_id} == {carbon.atom1_id, carbon.atom2_id}
    )
    assert perceived.order == 2.0
    assert result.diagnostics["source_locked_orders"] >= 1
