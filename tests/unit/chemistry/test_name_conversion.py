from __future__ import annotations

import pytest

from molcrys_kit import (
    NamingIndeterminateError,
    NamingParseError,
    from_iupac_name,
    iupac_to_smiles,
    smiles_to_iupac,
)
from molcrys_kit.chemistry import (
    EvidenceSource,
    InferenceStatus,
    from_line_notation,
    notations_equivalent,
)
from molcrys_kit.chemistry.name_conversion import complete_open_smiles_hydrogens


@pytest.mark.parametrize(
    ("smiles", "name"),
    (
        ("[CH4]", "methane"),
        ("[CH3][CH3]", "ethane"),
        ("[CH3][CH2][OH]", "ethanol"),
        ("[CH3][CH]([OH])[CH3]", "propan-2-ol"),
        ("[CH3][C](=[O])[OH]", "ethanoic acid"),
        ("c1ccccc1", "benzene"),
        ("c1cc(Cl)ccc1[CH3]", "1-chloro-4-methylbenzene"),
        (
            "[CH3][C](=[O])[NH]c1ccc([OH])cc1",
            "N-(4-hydroxyphenyl)acetamide",
        ),
    ),
)
def test_smiles_and_iupac_round_trip(smiles: str, name: str) -> None:
    result = smiles_to_iupac(smiles)
    assert result.name == name

    entity = from_iupac_name(name)
    assert entity.status is InferenceStatus.EXPLICIT
    assert entity.evidence[0].source is EvidenceSource.IUPAC_NAME

    notation = iupac_to_smiles(name)
    assert notation.dialect == "OpenSMILES"
    assert notation.lossless is True
    assert notations_equivalent(smiles, notation.value) is True


def test_name_parser_is_case_insensitive_but_returns_canonical_graph() -> None:
    entity = from_iupac_name("  EthAnOl ")
    assert entity.entity_id == "iupac:ethanol"
    assert iupac_to_smiles("ETHANOL").value


@pytest.mark.parametrize(
    "name",
    (
        "propan-3-ol",
        "isopropyl alcohol",
        "(2r)-butan-2-ol",
        "molecular entity CN",
        "2(poly(ethane)) · water",
    ),
)
def test_unsupported_iupac_names_fail_closed(name: str) -> None:
    with pytest.raises(NamingParseError):
        from_iupac_name(name)


@pytest.mark.parametrize(
    "smiles",
    (
        "C#N",
        "C.C",
        "N[C@@H](C)C(=O)O",
        "[NH4+]",
    ),
)
def test_strict_smiles_conversion_rejects_nonreversible_semantics(smiles: str) -> None:
    with pytest.raises(NamingIndeterminateError):
        smiles_to_iupac(smiles)


def test_non_strict_smiles_conversion_keeps_existing_fallback() -> None:
    result = smiles_to_iupac("C#N", strict=False)
    assert result.name == "molecular entity CN"
    assert result.status is InferenceStatus.INDETERMINATE


@pytest.mark.parametrize(
    ("smiles", "name"),
    (("C", "methane"), ("CC", "ethane"), ("CCO", "ethanol"), ("C(=O)O", "methanoic acid")),
)
def test_unbracketed_open_smiles_default_hydrogens_are_named(
    smiles: str, name: str
) -> None:
    assert smiles_to_iupac(smiles).name == name
    assert notations_equivalent(smiles, iupac_to_smiles(name).value) is True


def test_bracket_hydrogen_and_unbracketed_open_smiles_are_equivalent() -> None:
    assert notations_equivalent("[CH3][CH2][OH]", "CCO") is True
    assert notations_equivalent("[CH3][CH2][OH]", "[CH3][CH2][OH]") is True


def test_bracket_atom_without_hydrogen_does_not_gain_default_hydrogens() -> None:
    with pytest.raises(NamingIndeterminateError):
        smiles_to_iupac("[C]")


def test_open_smiles_hydrogen_completion_handles_bare_bracket_and_mixed_atoms() -> None:
    bare = complete_open_smiles_hydrogens(from_line_notation("C"))
    assert bare.atoms[0].implicit_hydrogens == 4

    bracket = from_line_notation("[C]")
    assert complete_open_smiles_hydrogens(bracket) == bracket

    mixed = complete_open_smiles_hydrogens(from_line_notation("C[O]"))
    assert mixed.atoms[0].implicit_hydrogens == 3
    assert mixed.atoms[1].implicit_hydrogens is None
