from __future__ import annotations

from dataclasses import replace

import pytest

from molcrys_kit import read_mol_crystal
from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    CrystalChemistry,
    InferenceStatus,
    MulticomponentEntity,
    NamingIndeterminateError,
    NamingKind,
    PeriodicChemicalEntity,
    PolymerChemicalEntity,
    from_line_notation,
    name_crystal,
    name_entity,
)


@pytest.mark.parametrize(
    "notation, expected",
    (
        ("[CH4]", "methane"),
        ("[CH3][CH3]", "ethane"),
        ("[CH3][CH2][OH]", "ethanol"),
        ("[CH3][CH]([OH])[CH3]", "propan-2-ol"),
        ("[CH3][C](=[O])[OH]", "ethanoic acid"),
        ("c1ccccc1", "benzene"),
        ("c1cc(Cl)ccc1[CH3]", "1-chloro-4-methylbenzene"),
    ),
)
def test_blue_book_organic_golden_examples(notation: str, expected: str) -> None:
    result = name_entity(from_line_notation(notation))

    assert result.name == expected
    assert result.kind is NamingKind.PREFERRED_IUPAC_NAME
    assert result.standard == "Blue Book"
    assert result.version == "2013"
    assert result.status is InferenceStatus.EXPLICIT
    assert result.rule_trace


def test_acetaminophen_is_named_from_graph_not_cif_name() -> None:
    crystal = read_mol_crystal("tests/data/cif/Acetaminophen_HXACAN.cif")
    crystal.metadata["cif_chemistry"]["chemical_name_systematic"] = "wrong source name"

    result = name_entity(crystal.chemistry.components[0])

    assert result.name == "N-(4-hydroxyphenyl)acetamide"
    assert result.preferred is True
    assert result.status is InferenceStatus.PROVISIONAL
    assert "carboxamide" in result.rule_trace[0]
    assert "wrong source name" not in result.name


def test_unsupported_finite_entity_returns_honest_composition_description() -> None:
    entity = from_line_notation("C#N")

    result = name_entity(entity)

    assert result.name == "molecular entity CN"
    assert result.kind is NamingKind.IUPAC_COMPOSITION_DESCRIPTION
    assert result.status is InferenceStatus.INDETERMINATE
    assert "unique IUPAC name" in result.warnings[0]
    with pytest.raises(NamingIndeterminateError):
        name_entity(entity, strict=True)


def test_periodic_entity_reports_dimension_without_inventing_network_name() -> None:
    entity = PeriodicChemicalEntity(
        entity_id="chain",
        atoms=(ChemicalAtom("Cu1", "Cu"), ChemicalAtom("N1", "N")),
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
    )

    result = name_entity(entity)

    assert result.name == "1-dimensional periodic entity CuN"
    assert result.standard == "Red Book"
    assert result.status is InferenceStatus.INDETERMINATE


def test_polymer_and_multicomponent_results_keep_their_nomenclature_scope() -> None:
    repeat = replace(from_line_notation("[CH3][CH3]"), entity_id="repeat")
    polymer = PolymerChemicalEntity(
        entity_id="poly",
        repeat_units=(repeat,),
        connections=("a->b",),
    )
    mixture = MulticomponentEntity(
        entity_id="mix",
        components=((polymer, 2), (repeat, 1)),
    )

    polymer_name = name_entity(polymer)
    mixture_name = name_entity(mixture)

    assert polymer_name.name == "poly(ethane)"
    assert polymer_name.standard == "Purple Book"
    assert mixture_name.name == "2(poly(ethane)) · ethane"
    assert mixture_name.kind is NamingKind.IUPAC_COMPOSITION_DESCRIPTION


def test_crystal_collapses_equivalent_component_names_and_retains_status() -> None:
    ethane = from_line_notation("[CH3][CH3]")
    chemistry = CrystalChemistry(
        components=(ethane, replace(ethane, entity_id="line:1")),
        atom_ids_by_global_index=tuple(atom.atom_id for atom in ethane.atoms),
        status=InferenceStatus.EXPLICIT,
    )

    result = name_crystal(chemistry)

    assert result.name == "ethane"
    assert result.kind is NamingKind.PREFERRED_IUPAC_NAME
    assert result.status is InferenceStatus.EXPLICIT
    assert "collapsed" in result.rule_trace[-1]
