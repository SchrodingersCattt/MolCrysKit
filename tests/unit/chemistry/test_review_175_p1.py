"""Counterexamples for the eight merge-blocking findings on b6558df."""

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    PeriodicChemicalEntity,
    from_line_notation,
    infer_bond_orders,
    infer_bond_orders_array,
)
from molcrys_kit.chemistry.source import apply_source_precedence


def test_default_array_connectivity_does_not_crash():
    result = infer_bond_orders_array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], [1, 1])
    assert result.entity.bonds
    assert result.entity.bonds[0].order == 1.0
    assert result.status is not InferenceStatus.INDETERMINATE


def test_stored_implicit_hydrogens_satisfy_methane_and_ethane():
    methane = FiniteChemicalEntity(
        "methane",
        (ChemicalAtom("C", "C", implicit_hydrogens=4),),
        (),
    )
    methane_result = infer_bond_orders(methane)
    assert methane_result.status is InferenceStatus.INFERRED
    assert methane_result.entity.atoms[0].formal_charge == 0
    assert methane_result.entity.net_charge == 0

    ethane = FiniteChemicalEntity(
        "ethane",
        (
            ChemicalAtom("C1", "C", implicit_hydrogens=3),
            ChemicalAtom("C2", "C", implicit_hydrogens=3),
        ),
        (ChemicalBond("C1", "C2", kind=BondKind.COVALENT),),
    )
    ethane_result = infer_bond_orders(ethane)
    assert ethane_result.status is InferenceStatus.INFERRED
    assert ethane_result.entity.bonds[0].order == 1.0
    assert ethane_result.entity.net_charge == 0


def test_parser_implicit_hydrogens_cover_methane_ethane_and_cyclohexane():
    methane = infer_bond_orders(from_line_notation("[CH4]"))
    assert methane.status is InferenceStatus.INFERRED
    assert methane.entity.net_charge == 0

    ethane = infer_bond_orders(from_line_notation("[CH3][CH3]"))
    assert ethane.status is InferenceStatus.INFERRED
    assert ethane.entity.bonds[0].order == 1.0
    assert ethane.entity.net_charge == 0

    cyclohexane = infer_bond_orders(
        from_line_notation("[CH2]1[CH2][CH2][CH2][CH2][CH2]1")
    )
    assert cyclohexane.status is InferenceStatus.INFERRED
    assert cyclohexane.diagnostics["aromatic_rings"] == 0
    assert all(bond.order == 1.0 for bond in cyclohexane.entity.bonds)
    assert all(not bond.aromatic for bond in cyclohexane.entity.bonds)


def test_opensmiles_benzene_keeps_a_kekule_assignment():
    parsed = from_line_notation("c1ccccc1")
    assert all(bond.order == 1.5 and bond.aromatic for bond in parsed.bonds)
    result = infer_bond_orders(parsed)
    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)
    assert all(bond.order in {1.0, 2.0} for bond in result.entity.bonds)
    assert all(atom.formal_charge == 0 for atom in result.entity.atoms)
    assert all(atom.implicit_hydrogens == 1 for atom in result.entity.atoms)
    assert result.entity.net_charge == 0


def test_periodic_self_edge_charge_is_counted_twice():
    entity = PeriodicChemicalEntity(
        "nn-image",
        (ChemicalAtom("N1", "N"),),
        (
            ChemicalBond(
                "N1",
                "N1",
                order=1.0,
                kind=BondKind.COVALENT,
                atom2_image_shift=(1, 0, 0),
            ),
        ),
        periodic_rank=1,
        translation_generators=((1, 0, 0),),
    )
    result = infer_bond_orders(entity, total_charge=-1)
    assert result.status is not InferenceStatus.INDETERMINATE
    assert result.diagnostics["total_charge_unsatisfied"] is False
    assert result.entity.atoms[0].formal_charge == -1
    assert result.entity.net_charge_per_repeat == -1
    assert result.entity.bonds[0].order == 1.0
    assert result.entity.bonds[0].atom2_image_shift == (1, 0, 0)


def test_equal_rank_contradiction_clears_a_prepopulated_order():
    bond = ChemicalBond(
        "C1",
        "C2",
        order=1.0,
        kind=BondKind.COVALENT,
        evidence=(
            Evidence(EvidenceSource.EXPLICIT_CIF, "cif", "bond_type:S"),
            Evidence(EvidenceSource.EXPLICIT_CIF, "cif", "bond_type:D"),
        ),
    )
    resolved = apply_source_precedence((bond,))
    assert resolved.bonds[0].order is None
    assert resolved.locked_orders == 0
    assert resolved.contradictions == 1
    entity = FiniteChemicalEntity(
        "prepopulated",
        (
            ChemicalAtom("C1", "C", explicit_hydrogens=2),
            ChemicalAtom("C2", "C", explicit_hydrogens=2),
        ),
        resolved.bonds,
    )
    result = infer_bond_orders(entity)
    assert result.entity.bonds[0].order == 2.0
    assert result.status is not InferenceStatus.INDETERMINATE


def test_pyridinium_stays_aromatic_and_pyrrole_donates_a_lone_pair():
    pyridinium = infer_bond_orders(from_line_notation("[nH+]1ccccc1"))
    assert pyridinium.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in pyridinium.entity.bonds)
    nitrogen = next(atom for atom in pyridinium.entity.atoms if atom.element == "N")
    assert nitrogen.formal_charge == 1
    assert pyridinium.entity.net_charge == 1
    assert pyridinium.status is InferenceStatus.INFERRED

    atoms = (
        ChemicalAtom("N", "N", explicit_hydrogens=1, formal_charge=1),
        *(ChemicalAtom(f"C{i}", "C") for i in range(5)),
    )
    bonds = []
    order = ("N", "C0", "C1", "C2", "C3", "C4")
    for index, left in enumerate(order):
        right = order[(index + 1) % len(order)]
        bonds.append(
            ChemicalBond(
                left,
                right,
                order=2.0 if index % 2 == 0 else 1.0,
                kind=BondKind.COVALENT,
            )
        )
    kekule = infer_bond_orders(FiniteChemicalEntity("pyridinium", atoms, tuple(bonds)))
    assert kekule.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in kekule.entity.bonds)

    pyrrole = infer_bond_orders(from_line_notation("[nH]1cccc1"))
    assert pyrrole.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in pyrrole.entity.bonds)
    pyrrole_nitrogen = next(
        atom for atom in pyrrole.entity.atoms if atom.element == "N"
    )
    assert pyrrole_nitrogen.formal_charge == 0


def test_array_adapter_preserves_typed_bond_semantics():
    bond = ChemicalBond(
        "Zn",
        "C",
        order=1.0,
        kind=BondKind.COVALENT,
        evidence=(Evidence(EvidenceSource.USER_CONFIRMED, "user", "order:1"),),
    )
    entity = FiniteChemicalEntity(
        "zinc-carbon",
        (
            ChemicalAtom("Zn", "Zn"),
            ChemicalAtom("C", "C", explicit_hydrogens=3),
        ),
        (bond,),
    )
    direct = infer_bond_orders(entity)
    adapted = infer_bond_orders_array(
        [[0.0, 0.0, 0.0], [1.95, 0.0, 0.0]],
        ["Zn", "C"],
        bonds=(bond,),
        atom_ids=("Zn", "C"),
    )
    assert direct.entity.bonds[0].kind is BondKind.COVALENT
    assert adapted.entity.bonds[0].kind is BondKind.COVALENT
    assert adapted.entity.bonds[0].order == direct.entity.bonds[0].order == 1.0
    assert any(
        item.source is EvidenceSource.USER_CONFIRMED
        for item in adapted.entity.bonds[0].evidence
    )
    assert adapted.entity.bonds[0].kind is direct.entity.bonds[0].kind


def test_disconnected_charge_coupling_stays_inside_the_search_budget():
    count = 18
    atoms = []
    bonds = []
    for index in range(count):
        left, right = f"N{index}a", f"N{index}b"
        atoms.extend(
            (ChemicalAtom(left, "N"), ChemicalAtom(right, "N")),
        )
        bonds.append(ChemicalBond(left, right, kind=BondKind.COVALENT))
    result = infer_bond_orders(
        FiniteChemicalEntity("nn-pairs", tuple(atoms), tuple(bonds)),
        total_charge=-2,
    )
    assert result.status is not InferenceStatus.INDETERMINATE
    assert result.diagnostics["search_complete"] is True
    assert result.diagnostics["solver"] == "charge_convolution"
    assert result.diagnostics["search_states"] < 5_000
    orders = [bond.order for bond in result.entity.bonds]
    assert orders.count(2.0) == 1
    assert orders.count(3.0) == count - 1
    assert result.entity.net_charge == -2
