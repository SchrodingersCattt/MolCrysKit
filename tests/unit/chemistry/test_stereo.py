"""Human-specified tetrahedral stereochemistry golden cases."""

from dataclasses import replace

import numpy as np

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    FiniteChemicalEntity,
    InferenceStatus,
    assign_stereochemistry,
)


def _entity(atoms, bonds, coordinates, entity_id="golden"):
    return FiniteChemicalEntity(
        entity_id=entity_id,
        atoms=tuple(atoms),
        bonds=tuple(
            ChemicalBond(left, right, order=order, kind=BondKind.COVALENT)
            for left, right, order in bonds
        ),
        embedding=Embedding(tuple((atom_id, tuple(position)) for atom_id, position in coordinates.items())),
    )


def _halomethane():
    atoms = [ChemicalAtom("C*", "C")]
    atoms.extend(ChemicalAtom(symbol, symbol) for symbol in ("Br", "Cl", "F", "H"))
    bonds = [("C*", symbol, 1.0) for symbol in ("Br", "Cl", "F", "H")]
    coordinates = {
        "C*": (0.0, 0.0, 0.0),
        "Br": (0.0, 1.0, 0.0),
        "Cl": (1.0, -1.0, 0.0),
        "F": (-1.0, -1.0, 0.0),
        "H": (0.0, 0.0, -1.0),
    }
    return _entity(atoms, bonds, coordinates)


def test_direct_atomic_number_golden_and_mirror_flip() -> None:
    entity = _halomethane()
    assigned = assign_stereochemistry(entity).for_atom("C*")
    assert assigned is not None
    assert assigned.cip_order == ("Br", "Cl", "F", "H")
    assert assigned.descriptor == "R"

    mirrored_coordinates = tuple(
        (atom_id, (-position[0], position[1], position[2]))
        for atom_id, position in entity.embedding.coordinates_A
    )
    mirrored = assign_stereochemistry(entity, Embedding(mirrored_coordinates)).for_atom("C*")
    assert mirrored is not None
    assert mirrored.descriptor == "S"


def test_rotation_and_translation_do_not_change_descriptor() -> None:
    entity = _halomethane()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = Embedding(
        tuple(
            (atom_id, tuple(np.asarray(position) @ rotation.T + np.array([4.0, -3.0, 2.0])))
            for atom_id, position in entity.embedding.coordinates_A
        )
    )
    assert assign_stereochemistry(entity, transformed).for_atom("C*").descriptor == "R"

    reordered = replace(
        entity,
        atoms=tuple(reversed(entity.atoms)),
        bonds=tuple(reversed(entity.bonds)),
        embedding=Embedding(tuple(reversed(entity.embedding.coordinates_A))),
    )
    assert assign_stereochemistry(reordered).for_atom("C*").descriptor == "R"


def test_recursive_ligand_comparison_ranks_ch2oh_over_methyl() -> None:
    atoms = [
        ChemicalAtom("C*", "C"), ChemicalAtom("Cl", "Cl"), ChemicalAtom("C1", "C"),
        ChemicalAtom("C2", "C"), ChemicalAtom("H*", "H"), ChemicalAtom("O", "O"),
        ChemicalAtom("H1a", "H"), ChemicalAtom("H1b", "H"),
        ChemicalAtom("H2a", "H"), ChemicalAtom("H2b", "H"), ChemicalAtom("H2c", "H"),
        ChemicalAtom("HO", "H"),
    ]
    bonds = [
        ("C*", "Cl", 1), ("C*", "C1", 1), ("C*", "C2", 1), ("C*", "H*", 1),
        ("C1", "O", 1), ("C1", "H1a", 1), ("C1", "H1b", 1), ("O", "HO", 1),
        ("C2", "H2a", 1), ("C2", "H2b", 1), ("C2", "H2c", 1),
    ]
    coordinates = {atom.atom_id: (float(index), 0.0, 0.0) for index, atom in enumerate(atoms)}
    coordinates.update({"C*": (0, 0, 0), "Cl": (0, 1, 0), "C1": (1, -1, 0), "C2": (-1, -1, 0), "H*": (0, 0, -1)})
    assigned = assign_stereochemistry(_entity(atoms, bonds, coordinates)).for_atom("C*")
    assert assigned.cip_order == ("Cl", "C1", "C2", "H*")


def test_isotope_mass_is_applied_only_after_atomic_number() -> None:
    atoms = [
        ChemicalAtom("C*", "C"), ChemicalAtom("O", "O"), ChemicalAtom("C", "C"),
        ChemicalAtom("D", "H", isotope=2), ChemicalAtom("H", "H"),
    ]
    bonds = [("C*", atom.atom_id, 1) for atom in atoms[1:]]
    coordinates = {"C*": (0, 0, 0), "O": (0, 1, 0), "C": (1, -1, 0), "D": (-1, -1, 0), "H": (0, 0, -1)}
    assigned = assign_stereochemistry(_entity(atoms, bonds, coordinates)).for_atom("C*")
    assert assigned.cip_order == ("O", "C", "D", "H")
    assert assigned.descriptor == "R"


def test_multiple_bond_duplicate_nodes_affect_recursive_priority() -> None:
    atoms = [
        ChemicalAtom("C*", "C"), ChemicalAtom("O*", "O"), ChemicalAtom("Cal", "C"),
        ChemicalAtom("Cvi", "C"), ChemicalAtom("H*", "H"), ChemicalAtom("Oal", "O"),
        ChemicalAtom("Hvi", "H"), ChemicalAtom("Cend", "C"),
    ]
    bonds = [
        ("C*", "O*", 1), ("C*", "Cal", 1), ("C*", "Cvi", 1), ("C*", "H*", 1),
        ("Cal", "Oal", 2), ("Cvi", "Cend", 2), ("Cvi", "Hvi", 1),
    ]
    coordinates = {atom.atom_id: (float(index), 0.0, 0.0) for index, atom in enumerate(atoms)}
    coordinates.update({"C*": (0, 0, 0), "O*": (0, 1, 0), "Cal": (1, -1, 0), "Cvi": (-1, -1, 0), "H*": (0, 0, -1)})
    assigned = assign_stereochemistry(_entity(atoms, bonds, coordinates)).for_atom("C*")
    assert assigned.cip_order == ("O*", "Cal", "Cvi", "H*")


def test_saturated_ring_duplicate_nodes_rank_cyclobutyl_over_cyclopropyl() -> None:
    atoms = [
        ChemicalAtom("C*", "C"),
        ChemicalAtom("O", "O"),
        ChemicalAtom("H", "H"),
        ChemicalAtom("c4a", "C", implicit_hydrogens=1),
        ChemicalAtom("c4b", "C", implicit_hydrogens=2),
        ChemicalAtom("c4c", "C", implicit_hydrogens=2),
        ChemicalAtom("c4d", "C", implicit_hydrogens=2),
        ChemicalAtom("c3a", "C", implicit_hydrogens=1),
        ChemicalAtom("c3b", "C", implicit_hydrogens=2),
        ChemicalAtom("c3c", "C", implicit_hydrogens=2),
    ]
    bonds = [
        ("C*", "O", 1),
        ("C*", "c4a", 1),
        ("C*", "c3a", 1),
        ("C*", "H", 1),
        ("c4a", "c4b", 1),
        ("c4b", "c4c", 1),
        ("c4c", "c4d", 1),
        ("c4d", "c4a", 1),
        ("c3a", "c3b", 1),
        ("c3b", "c3c", 1),
        ("c3c", "c3a", 1),
    ]
    coordinates = {
        atom.atom_id: (float(index), 0.0, 0.0) for index, atom in enumerate(atoms)
    }
    coordinates.update(
        {
            "C*": (0, 0, 0),
            "O": (0, 1, 0),
            "c4a": (1, -1, 0),
            "c3a": (-1, -1, 0),
            "H": (0, 0, -1),
        }
    )
    assigned = assign_stereochemistry(_entity(atoms, bonds, coordinates)).for_atom("C*")
    assert assigned.cip_order == ("O", "c4a", "c3a", "H")


def test_tied_ligands_and_planar_coordinates_are_indeterminate() -> None:
    tied = _halomethane()
    tied_atoms = tuple(
        replace(atom, element="H") if atom.atom_id == "F" else atom for atom in tied.atoms
    )
    tied_report = assign_stereochemistry(replace(tied, atoms=tied_atoms)).for_atom("C*")
    assert tied_report.descriptor is None
    assert tied_report.status is InferenceStatus.INDETERMINATE

    planar = Embedding(
        tuple(
            (atom_id, (position[0], position[1], 0.0))
            for atom_id, position in tied.embedding.coordinates_A
        )
    )
    planar_report = assign_stereochemistry(tied, planar).for_atom("C*")
    assert planar_report.descriptor is None
    assert "planar" in planar_report.reason


def _but_2_ene(*, together: bool) -> FiniteChemicalEntity:
    atoms = [
        ChemicalAtom("C1", "C"),
        ChemicalAtom("C2", "C"),
        ChemicalAtom("Me1", "C", implicit_hydrogens=3),
        ChemicalAtom("H1", "H"),
        ChemicalAtom("Me2", "C", implicit_hydrogens=3),
        ChemicalAtom("H2", "H"),
    ]
    bonds = [
        ("C1", "C2", 2),
        ("C1", "Me1", 1),
        ("C1", "H1", 1),
        ("C2", "Me2", 1),
        ("C2", "H2", 1),
    ]
    right_sign = 1.0 if together else -1.0
    coordinates = {
        "C1": (-0.5, 0.0, 0.0),
        "C2": (0.5, 0.0, 0.0),
        "Me1": (-1.5, 1.0, 0.0),
        "H1": (-1.5, -1.0, 0.0),
        "Me2": (1.5, right_sign, 0.0),
        "H2": (1.5, -right_sign, 0.0),
    }
    return _entity(atoms, bonds, coordinates, entity_id="but-2-ene")


def test_double_bond_golden_assigns_z_and_e_from_cip_sides() -> None:
    together = assign_stereochemistry(_but_2_ene(together=True)).for_atom("C1")
    opposite = assign_stereochemistry(_but_2_ene(together=False)).for_atom("C1")

    assert together.kind.value == "double_bond"
    assert together.descriptor == "Z"
    assert together.cip_order == ("Me1", "H1", "Me2", "H2")
    assert opposite.descriptor == "E"


def test_double_bond_descriptor_is_rotation_and_atom_order_invariant() -> None:
    entity = _but_2_ene(together=False)
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = Embedding(
        tuple(
            (atom_id, tuple(np.asarray(position) @ rotation.T + np.array([2, 3, 4])))
            for atom_id, position in entity.embedding.coordinates_A
        )
    )
    reordered = replace(
        entity,
        atoms=tuple(reversed(entity.atoms)),
        bonds=tuple(reversed(entity.bonds)),
        embedding=transformed,
    )

    assert assign_stereochemistry(reordered).for_atom("C1").descriptor == "E"


def test_double_bond_with_equivalent_ligands_is_indeterminate() -> None:
    entity = _but_2_ene(together=True)
    atoms = tuple(
        replace(atom, element="H", implicit_hydrogens=None)
        if atom.atom_id == "Me1"
        else atom
        for atom in entity.atoms
    )

    descriptor = assign_stereochemistry(replace(entity, atoms=atoms)).for_atom("C1")

    assert descriptor.descriptor is None
    assert descriptor.status is InferenceStatus.INDETERMINATE
    assert "CIP-equivalent" in descriptor.reason
