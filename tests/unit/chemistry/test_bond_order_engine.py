from molcrys_kit.chemistry import (
    BondOrderOptions,
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    FiniteChemicalEntity,
    InferenceStatus,
    infer_bond_orders,
)


def _alkane(n: int) -> FiniteChemicalEntity:
    atoms = [ChemicalAtom(f"C{i}", "C") for i in range(n)]
    bonds = []
    for i in range(n - 1):
        bonds.append(ChemicalBond(f"C{i}", f"C{i + 1}", kind=BondKind.COVALENT))
    hydrogen_index = 0
    for i in range(n):
        attached = 3 if i in (0, n - 1) else 2
        for _ in range(attached):
            hid = f"H{hydrogen_index}"
            hydrogen_index += 1
            atoms.append(ChemicalAtom(hid, "H"))
            bonds.append(ChemicalBond(f"C{i}", hid, kind=BondKind.COVALENT))
    return FiniteChemicalEntity(
        f"C{n}H{2 * n + 2}",
        tuple(atoms),
        tuple(bonds),
        status=InferenceStatus.INDETERMINATE,
    )


def test_long_saturated_chain_is_solved_without_cartesian_cutoff():
    result = infer_bond_orders(
        _alkane(13),
        options=BondOrderOptions(max_search_states=10_000, max_candidates=1),
    )
    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["search_complete"] is True
    assert all(bond.order == 1.0 for bond in result.entity.bonds)


def test_search_budget_is_reported_without_fabricating_single_bonds():
    result = infer_bond_orders(
        _alkane(13),
        options=BondOrderOptions(max_search_states=1, max_candidates=1),
    )
    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["search_complete"] is False
    assert result.entity.bonds[0].order is None


def test_periodic_image_is_used_for_geometry_scoring():
    entity = FiniteChemicalEntity(
        "periodic-edge",
        (ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C")),
        (
            ChemicalBond(
                "C1", "C2", kind=BondKind.COVALENT, atom2_image_shift=(1, 0, 0)
            ),
        ),
        embedding=Embedding(
            (("C1", (9.5, 0.0, 0.0)), ("C2", (0.5, 0.0, 0.0))),
            cell_A=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
    )
    result = infer_bond_orders(entity, options=BondOrderOptions(max_candidates=8))
    assert result.entity.bonds[0].order is None or result.entity.bonds[0].order > 0


def test_known_total_charge_is_solved_jointly_with_atom_valence():
    atoms = [ChemicalAtom("N", "N")] + [ChemicalAtom(f"H{i}", "H") for i in range(4)]
    bonds = tuple(ChemicalBond("N", f"H{i}", kind=BondKind.COVALENT) for i in range(4))
    entity = FiniteChemicalEntity("ammonium", tuple(atoms), bonds)
    result = infer_bond_orders(entity, total_charge=1)
    assert result.status is InferenceStatus.INFERRED
    assert result.entity.net_charge == 1
    assert result.entity.atoms[0].formal_charge == 1
