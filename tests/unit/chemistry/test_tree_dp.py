"""Forests use tree DP; cycles stay on the bounded search."""

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    infer_bond_orders,
)
from molcrys_kit.chemistry.tree_dp import decompose_variable_graph


def _chain(n_carbons: int, *, prefix: str = "C") -> FiniteChemicalEntity:
    atoms = []
    for index in range(n_carbons):
        hydrogens = 3 if index in {0, n_carbons - 1} else 2
        atoms.append(
            ChemicalAtom(f"{prefix}{index}", "C", explicit_hydrogens=hydrogens)
        )
    bonds = [
        ChemicalBond(f"{prefix}{index}", f"{prefix}{index + 1}", kind=BondKind.COVALENT)
        for index in range(n_carbons - 1)
    ]
    return FiniteChemicalEntity(
        f"chain-{prefix}-{n_carbons}", tuple(atoms), tuple(bonds)
    )


def test_disconnected_components_are_split_and_trees_are_exact():
    left = _chain(2, prefix="L")
    right_atoms = [
        ChemicalAtom("A", "C", explicit_hydrogens=3),
        ChemicalAtom("B", "C", explicit_hydrogens=3),
    ]
    entity = FiniteChemicalEntity(
        "two",
        tuple(left.atoms) + tuple(right_atoms),
        left.bonds + (ChemicalBond("A", "B", kind=BondKind.COVALENT),),
    )
    components = decompose_variable_graph(entity.bonds, list(range(len(entity.bonds))))
    assert len(components) == 2
    assert all(not component.cyclic for component in components)
    result = infer_bond_orders(entity)
    assert result.diagnostics["solver"] == "tree_dp"
    assert result.diagnostics["bond_components"] == 2
    assert all(bond.order == 1.0 for bond in result.entity.bonds)
    assert all(atom.formal_charge == 0 for atom in result.entity.atoms)
    assert result.status is InferenceStatus.INFERRED


def test_benzene_component_stays_on_the_search():
    atoms = [ChemicalAtom(f"C{i}", "C") for i in range(6)]
    bonds = [
        ChemicalBond(f"C{i}", f"C{(i + 1) % 6}", kind=BondKind.COVALENT)
        for i in range(6)
    ]
    entity = FiniteChemicalEntity("benzene", tuple(atoms), tuple(bonds))
    components = decompose_variable_graph(entity.bonds, list(range(6)))
    assert len(components) == 1 and components[0].cyclic
    result = infer_bond_orders(entity)
    assert result.diagnostics["solver"] == "search"
    assert result.diagnostics["cyclic_components"] == 1


def test_long_alkane_tree_dp_is_linear_in_the_chain():
    short = infer_bond_orders(_chain(32))
    long = infer_bond_orders(_chain(96))
    assert short.diagnostics["solver"] == "tree_dp"
    assert long.diagnostics["solver"] == "tree_dp"
    assert long.diagnostics["search_states"] < 20 * 96
    assert long.diagnostics["search_states"] < short.diagnostics["search_states"] * 4
    assert all(bond.order == 1.0 for bond in long.entity.bonds)
    assert long.status is InferenceStatus.INFERRED


def test_geometry_picks_the_short_carbonyl_and_not_a_user_lock():
    atoms = (
        ChemicalAtom("C", "C", explicit_hydrogens=3),
        ChemicalAtom("K", "C"),
        ChemicalAtom("O1", "O"),
        ChemicalAtom("O2", "O"),
    )
    bonds = (
        ChemicalBond("C", "K", kind=BondKind.COVALENT),
        ChemicalBond("K", "O1", kind=BondKind.COVALENT),
        ChemicalBond("K", "O2", kind=BondKind.COVALENT),
    )
    embedding = Embedding(
        (
            ("C", (-1.54, 0.0, 0.0)),
            ("K", (0.0, 0.0, 0.0)),
            ("O1", (1.22, 0.0, 0.0)),
            ("O2", (0.0, 1.36, 0.0)),
        )
    )
    result = infer_bond_orders(
        FiniteChemicalEntity("acetate", atoms, bonds, embedding=embedding)
    )
    orders = {
        (bond.atom1_id, bond.atom2_id): bond.order for bond in result.entity.bonds
    }
    assert orders[("K", "O1")] == 2.0
    assert orders[("K", "O2")] == 1.0
    assert orders[("C", "K")] == 1.0
    charges = {atom.atom_id: atom.formal_charge for atom in result.entity.atoms}
    assert charges["O1"] == 0 and charges["O2"] == -1

    locked_bonds = (
        bonds[0],
        ChemicalBond(
            "K",
            "O1",
            order=1.0,
            kind=BondKind.COVALENT,
            evidence=(Evidence(EvidenceSource.USER_CONFIRMED, "user", "order:1"),),
        ),
        bonds[2],
    )
    locked = infer_bond_orders(
        FiniteChemicalEntity("acetate-locked", atoms, locked_bonds, embedding=embedding)
    )
    locked_orders = {
        (bond.atom1_id, bond.atom2_id): bond.order for bond in locked.entity.bonds
    }
    assert locked_orders[("K", "O1")] == 1.0
    assert locked_orders[("K", "O2")] == 2.0
    assert locked.diagnostics["source_locked_orders"] >= 1
