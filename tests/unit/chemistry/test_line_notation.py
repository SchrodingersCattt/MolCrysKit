from __future__ import annotations

from dataclasses import replace

import pytest

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    FiniteChemicalEntity,
    InferenceStatus,
    LineNotationError,
    MulticomponentEntity,
    PeriodicChemicalEntity,
    PolymerChemicalEntity,
    from_line_notation,
    to_line_notation,
)


def _finite(order=("a", "b", "c")) -> FiniteChemicalEntity:
    atoms = {
        "a": ChemicalAtom("a", "C", implicit_hydrogens=3, formal_charge=0),
        "b": ChemicalAtom("b", "C", implicit_hydrogens=2, formal_charge=0),
        "c": ChemicalAtom("c", "O", implicit_hydrogens=1, formal_charge=0),
    }
    return FiniteChemicalEntity(
        entity_id="ethanol",
        atoms=tuple(atoms[value] for value in order),
        bonds=(
            ChemicalBond("a", "b", 1.0, BondKind.COVALENT),
            ChemicalBond("b", "c", 1.0, BondKind.COVALENT),
        ),
        net_charge=0,
        status=InferenceStatus.EXPLICIT,
    )


def _graph_signature(entity):
    atom_by_id = {atom.atom_id: atom for atom in entity.atoms}
    atoms = sorted(
        (
            atom.element,
            atom.isotope,
            atom.formal_charge,
            atom.radical_electrons,
            atom.explicit_hydrogens,
            atom.implicit_hydrogens,
            atom.oxidation_state,
            atom.stereochemistry,
        )
        for atom in entity.atoms
    )
    bonds = sorted(
        (
            tuple(sorted((atom_by_id[bond.atom1_id].element, atom_by_id[bond.atom2_id].element))),
            bond.order,
            bond.kind,
            bond.aromatic,
            bond.atom2_image_shift,
            bond.stereochemistry,
        )
        for bond in entity.bonds
    )
    return atoms, bonds


def test_opensmiles_generation_is_stable_under_atom_reordering() -> None:
    first = to_line_notation(_finite(("a", "b", "c")))
    second = to_line_notation(_finite(("c", "a", "b")))

    assert first.dialect == "OpenSMILES"
    assert first.version == "1.0"
    assert first.lossless
    assert first.value == second.value


@pytest.mark.parametrize(
    "text",
    (
        "CCO",
        "C(C)(F)Cl",
        "C1CCCCC1",
        "c1ccccc1",
        "[13CH3][NH3+]",
        "C.C",
    ),
)
def test_opensmiles_supported_subset_round_trips(text: str) -> None:
    entity = from_line_notation(text)
    generated = to_line_notation(entity)
    reparsed = from_line_notation(generated.value)

    assert generated.dialect == "OpenSMILES"
    assert _graph_signature(reparsed) == _graph_signature(entity)


def test_opensmiles_parser_retains_stereo_tokens_then_uses_lossless_extension() -> None:
    entity = from_line_notation("N[C@@H](C)C(=O)O")

    center = next(atom for atom in entity.atoms if atom.stereochemistry)
    assert center.stereochemistry == "@@"
    generated = to_line_notation(entity)
    assert generated.dialect == "MCK-LN"
    assert from_line_notation(generated.value).atoms[1].stereochemistry == "@@"


def test_mck_ln_round_trip_preserves_full_finite_graph_and_embedding() -> None:
    entity = FiniteChemicalEntity(
        entity_id="complex id/1",
        atoms=(
            ChemicalAtom(
                "site:C1",
                "C",
                label="C one",
                isotope=13,
                formal_charge=-1,
                radical_electrons=1,
                explicit_hydrogens=0,
                implicit_hydrogens=1,
                oxidation_state=-2,
                stereochemistry="R",
            ),
            ChemicalAtom("site:Fe1", "Fe", formal_charge=2, oxidation_state=2),
        ),
        bonds=(
            ChemicalBond(
                "site:C1",
                "site:Fe1",
                order=None,
                kind=BondKind.COORDINATION,
                stereochemistry="trans-reference",
            ),
        ),
        embedding=Embedding(
            (("site:C1", (0.1, 0.2, 0.3)), ("site:Fe1", (1.0, 2.0, 3.0)))
        ),
        net_charge=1,
        status=InferenceStatus.CONFIRMED,
    )

    notation = to_line_notation(entity)
    rebuilt = from_line_notation(notation.value)

    assert notation.dialect == "MCK-LN"
    assert notation.lossless
    assert rebuilt.entity_id == entity.entity_id
    assert rebuilt.atoms == entity.atoms
    assert rebuilt.bonds == entity.bonds
    assert rebuilt.embedding.coordinates_A == entity.embedding.coordinates_A
    assert rebuilt.net_charge == 1
    assert rebuilt.status is InferenceStatus.CONFIRMED


def test_mck_ln_round_trip_preserves_periodic_edges() -> None:
    entity = PeriodicChemicalEntity(
        entity_id="chain",
        atoms=(ChemicalAtom("Cu1", "Cu", oxidation_state=2), ChemicalAtom("N1", "N")),
        bonds=(
            ChemicalBond(
                "Cu1",
                "N1",
                kind=BondKind.COORDINATION,
                atom2_image_shift=(1, 0, 0),
            ),
        ),
        periodic_rank=1,
        translation_generators=((1, 0, 0),),
        net_charge_per_repeat=2,
        status=InferenceStatus.EXPLICIT,
    )

    rebuilt = from_line_notation(to_line_notation(entity).value)

    assert isinstance(rebuilt, PeriodicChemicalEntity)
    assert rebuilt.periodic_rank == 1
    assert rebuilt.translation_generators == ((1, 0, 0),)
    assert rebuilt.bonds[0].atom2_image_shift == (1, 0, 0)


def test_mck_ln_round_trip_preserves_polymer_and_multicomponent_nesting() -> None:
    repeat = replace(_finite(), entity_id="repeat")
    polymer = PolymerChemicalEntity(
        entity_id="polymer",
        repeat_units=(repeat,),
        connections=("a->c",),
        status=InferenceStatus.EXPLICIT,
    )
    mixture = MulticomponentEntity(
        entity_id="mixture",
        components=((polymer, 2), (repeat, 1)),
        status=InferenceStatus.CONFIRMED,
    )

    rebuilt = from_line_notation(to_line_notation(mixture).value)

    assert isinstance(rebuilt, MulticomponentEntity)
    assert rebuilt.components[0][1] == 2
    assert isinstance(rebuilt.components[0][0], PolymerChemicalEntity)
    assert rebuilt.components[0][0].connections == ("a->c",)


def test_explicit_opensmiles_rejects_semantics_it_would_discard() -> None:
    periodic_semantics = replace(
        _finite(),
        atoms=(replace(_finite().atoms[0], oxidation_state=-4), *_finite().atoms[1:]),
    )

    with pytest.raises(LineNotationError, match="oxidation states"):
        to_line_notation(periodic_semantics, dialect="opensmiles")


@pytest.mark.parametrize("text", ("", "C(", "C1CC", "MCK-LN1|type=wat"))
def test_invalid_notation_fails_loudly(text: str) -> None:
    with pytest.raises(LineNotationError):
        from_line_notation(text)
