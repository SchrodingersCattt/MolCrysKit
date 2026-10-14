"""Counterexamples for the eight merge-blocking findings on b6558df."""

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
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
import math


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


def test_geometry_disambiguates_an_h_omitted_saturated_ring():
    side = 1.54
    radius = side / (2 * math.sin(math.pi / 6))
    atoms = tuple(ChemicalAtom(f"C{i}", "C") for i in range(6))
    bonds = tuple(
        ChemicalBond(f"C{i}", f"C{(i + 1) % 6}", kind=BondKind.COVALENT)
        for i in range(6)
    )
    embedding = Embedding(
        tuple(
            (
                f"C{i}",
                (
                    radius * math.cos(i * math.pi / 3),
                    radius * math.sin(i * math.pi / 3),
                    0.0,
                ),
            )
            for i in range(6)
        )
    )
    result = infer_bond_orders(FiniteChemicalEntity("cyclohexane-unknown", atoms, bonds, embedding=embedding))
    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(f"C{i}" for i in range(6))
    assert result.diagnostics["aromatic_rings"] == 0
    assert all(bond.order is None and not bond.aromatic for bond in result.entity.bonds)


def test_uncertain_saturated_ring_does_not_abort_an_aromatic_fragment():
    def ring(prefix, count, side, z):
        radius = side / (2 * math.sin(math.pi / count))
        atoms = tuple(ChemicalAtom(f"{prefix}{i}", "C") for i in range(count))
        bonds = tuple(
            ChemicalBond(
                f"{prefix}{i}",
                f"{prefix}{(i + 1) % count}",
                kind=BondKind.COVALENT,
            )
            for i in range(count)
        )
        coordinates = tuple(
            (
                f"{prefix}{i}",
                (
                    radius * math.cos(i * 2 * math.pi / count),
                    radius * math.sin(i * 2 * math.pi / count),
                    z,
                ),
            )
            for i in range(count)
        )
        return atoms, bonds, coordinates

    aromatic_atoms, aromatic_bonds, aromatic_coordinates = ring("A", 6, 1.40, 0.0)
    saturated_atoms, saturated_bonds, saturated_coordinates = ring(
        "S", 6, 1.54, 4.0
    )
    connecting = ChemicalBond("A0", "S0", kind=BondKind.COVALENT)
    result = infer_bond_orders(
        FiniteChemicalEntity(
            "phenyl-cyclohexane-boundary",
            aromatic_atoms + saturated_atoms,
            aromatic_bonds + saturated_bonds + (connecting,),
            embedding=Embedding(aromatic_coordinates + saturated_coordinates),
        )
    )

    assert result.status is InferenceStatus.PROVISIONAL
    assert result.diagnostics["aromatic_rings"] == 1
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"S{i}" for i in range(6)
    )
    aromatic = result.entity.bonds[:6]
    saturated = result.entity.bonds[6:12]
    assert all(bond.aromatic and bond.order in {1.0, 2.0} for bond in aromatic)
    assert all(not bond.aromatic and bond.order is None for bond in saturated)
    assert result.entity.bonds[-1].order == 1.0


def test_plain_eight_to_ten_membered_carbon_cycles_are_not_aromatic():
    for count in (8, 9, 10):
        atoms = tuple(ChemicalAtom(f"C{count}_{i}", "C") for i in range(count))
        bonds = tuple(
            ChemicalBond(
                f"C{count}_{i}",
                f"C{count}_{(i + 1) % count}",
                kind=BondKind.COVALENT,
            )
            for i in range(count)
        )
        result = infer_bond_orders(
            FiniteChemicalEntity(f"plain-c{count}", atoms, bonds)
        )
        assert result.diagnostics["aromatic_rings"] == 0
        assert all(not bond.aromatic for bond in result.entity.bonds)


def test_periodic_hydrogen_boundary_uses_the_supplied_cell_fallback():
    class PositionOnlyEmbedding:
        def __init__(self, positions):
            self._positions = dict(positions)

        def position(self, atom_id):
            return self._positions[atom_id]

    count = 6
    atoms = tuple(ChemicalAtom(f"P{i}", "C") for i in range(count))
    bonds = tuple(
        ChemicalBond(
            f"P{i}",
            f"P{(i + 1) % count}",
            kind=BondKind.COVALENT,
            atom2_image_shift=(1, 0, 0) if i == count - 1 else (0, 0, 0),
        )
        for i in range(count)
    )
    embedding = PositionOnlyEmbedding(
        [(f"P{i}", (1.54 * i, 0.0, 0.0)) for i in range(count)]
    )
    result = infer_bond_orders(
        PeriodicChemicalEntity(
            "periodic-unknown-cyclohexane",
            atoms,
            bonds,
            periodic_rank=1,
            translation_generators=((1, 0, 0),),
            embedding=embedding,
        ),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
    )
    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"P{i}" for i in range(count)
    )


def test_periodic_winding_cycle_is_not_a_finite_aromatic_ring():
    atoms = tuple(ChemicalAtom(f"C{i}", "C") for i in range(6))
    bonds = tuple(
        ChemicalBond(
            f"C{i}",
            f"C{(i + 1) % 6}",
            kind=BondKind.COVALENT,
            atom2_image_shift=(1, 0, 0) if i == 5 else (0, 0, 0),
        )
        for i in range(6)
    )
    result = infer_bond_orders(FiniteChemicalEntity("winding-chain", atoms, bonds))
    assert result.diagnostics["aromatic_rings"] == 0
    assert all(not bond.aromatic for bond in result.entity.bonds)


def test_zero_winding_periodic_ring_can_be_aromatic():
    atoms = tuple(ChemicalAtom(f"C{i}", "C") for i in range(6))
    bonds = tuple(
        ChemicalBond(f"C{i}", f"C{(i + 1) % 6}", kind=BondKind.COVALENT)
        for i in range(6)
    )
    result = infer_bond_orders(FiniteChemicalEntity("finite-ring", atoms, bonds))
    assert result.diagnostics["aromatic_rings"] == 1


def test_confirmed_kekule_order_does_not_block_aromaticity():
    atoms = tuple(ChemicalAtom(f"C{i}", "C") for i in range(6))
    bonds = []
    for i in range(6):
        evidence = (
            (Evidence(EvidenceSource.USER_CONFIRMED, "user", "order:1"),)
            if i == 0
            else ()
        )
        bonds.append(
            ChemicalBond(
                f"C{i}",
                f"C{(i + 1) % 6}",
                order=1.0 if i == 0 else None,
                kind=BondKind.COVALENT,
                evidence=evidence,
            )
        )
    result = infer_bond_orders(FiniteChemicalEntity("kekule-authority", atoms, tuple(bonds)))
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)


def test_azulene_perimeter_is_a_ten_pi_aromatic_circuit():
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (1, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    ]
    atoms = tuple(ChemicalAtom(f"C{i}", "C") for i in range(10))
    bonds = tuple(
        ChemicalBond(f"C{left}", f"C{right}", kind=BondKind.COVALENT)
        for left, right in pairs
    )
    result = infer_bond_orders(FiniteChemicalEntity("azulene", atoms, bonds))
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)


def test_radical_electron_count_is_not_collapsed_to_a_boolean():
    oxygen = infer_bond_orders(
        FiniteChemicalEntity(
            "triplet-oxygen",
            (ChemicalAtom("O", "O", radical_electrons=2),),
            (),
        )
    )
    methylene = infer_bond_orders(
        FiniteChemicalEntity(
            "triplet-methylene",
            (ChemicalAtom("C", "C", explicit_hydrogens=2, radical_electrons=2),),
            (),
        )
    )
    assert oxygen.status is InferenceStatus.INFERRED
    assert oxygen.entity.atoms[0].formal_charge == 0
    assert methylene.status is InferenceStatus.INFERRED
    assert methylene.entity.atoms[0].formal_charge == 0


def test_tree_dp_retains_equal_scoring_carboxylate_resonance_forms():
    atoms = (
        ChemicalAtom("C0", "C", explicit_hydrogens=3),
        ChemicalAtom("C1", "C"),
        ChemicalAtom("O1", "O"),
        ChemicalAtom("O2", "O"),
    )
    bonds = (
        ChemicalBond("C0", "C1", kind=BondKind.COVALENT),
        ChemicalBond("C1", "O1", kind=BondKind.COVALENT),
        ChemicalBond("C1", "O2", kind=BondKind.COVALENT),
    )
    result = infer_bond_orders(FiniteChemicalEntity("acetate", atoms, bonds))
    assert result.status is InferenceStatus.INFERRED
    assert len(result.alternatives) == 1
    assert result.resonance_classes == ((0, 1),)
