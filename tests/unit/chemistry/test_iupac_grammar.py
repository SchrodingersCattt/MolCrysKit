"""Bidirectional grammar checks for the substitutive naming rules.

Expected strings are Blue Book or literature spellings chosen independently
of a generator snapshot: retained suffixes (``benzene-1,2-diol``), standard
stems (``hexan-2-one``, ``hepta-2,4-diene``), and published glucuronide and
sulfate names. They are general IUPAC names, not a preferred-name certificate.
A round trip must preserve constitution, charge, and radical count.
"""

from __future__ import annotations

import pytest

from molcrys_kit import (
    NamingIndeterminateError,
    NamingParseError,
    from_iupac_name,
    iupac_to_smiles,
    smiles_to_iupac,
)
from molcrys_kit.chemistry import from_line_notation, notations_equivalent
from molcrys_kit.chemistry.equivalence import constitution_equivalent
from molcrys_kit.chemistry.representation import normalize_kekule_aromatic
from molcrys_kit.chemistry.name_conversion import complete_open_smiles_hydrogens
from molcrys_kit.chemistry.naming import NamingKind


# (smiles, canonical general name). Kekulé and aromatic benzene are one row
# family; both inputs are listed so representation invariance is explicit.
_EXACT = (
    ("CCCCCC", "hexane"),
    ("CCCCCCO", "hexan-1-ol"),
    ("CCC(O)CCC", "hexan-3-ol"),
    ("CC(O)CCCC", "hexan-2-ol"),
    ("CC(O)CCC(O)C", "hexane-2,5-diol"),
    ("CC(=O)CCCC", "hexan-2-one"),
    ("CC(=O)CCC(O)C", "5-hydroxyhexan-2-one"),
    ("CC(=O)CCC(=O)C", "hexane-2,5-dione"),
    ("c1ccccc1", "benzene"),
    ("C1=CC=CC=C1", "benzene"),
    ("C1=CC=CC2OC12", "7-oxabicyclo[4.1.0]hepta-2,4-diene"),
    ("O1C=CC=CC=C1", "oxepine"),
    ("O1C=CC=CC2OC12", "2,8-dioxabicyclo[5.1.0]octa-3,5-diene"),
    ("O=C(O)/C=C/C=C/CO", "(2E,4E)-6-hydroxyhexa-2,4-dienoic acid"),
    ("O=C(O)/C=C/C=C/C(=O)O", "(2E,4E)-hexa-2,4-dienedioic acid"),
    ("CC(=O)N[C@@H](CSc1ccccc1)C(=O)O", "(2R)-2-acetamido-3-(phenylsulfanyl)propanoic acid"),
    (
        "CC(=O)N[C@@H](CSC1C=CC=CC1)C(=O)O",
        "(2R)-2-acetamido-3-(cyclohexa-2,4-dien-1-ylsulfanyl)propanoic acid",
    ),
    ("OC1C(O)C=CC=C1", "cyclohexa-3,5-diene-1,2-diol"),
    ("Oc1ccccc1O", "benzene-1,2-diol"),
    ("C1=C(O)C(O)=CC=C1", "benzene-1,2-diol"),
    ("O=C1C=CC=CC1=O", "cyclohexa-3,5-diene-1,2-dione"),
    ("Oc1ccccc1", "phenol"),
    ("C1=CC=C(O)C=C1", "phenol"),
    ("Oc1ccc(O)cc1", "benzene-1,4-diol"),
    ("O=C1C=CC(=O)C=C1", "cyclohexa-2,5-diene-1,4-dione"),
    ("Oc1ccc(O)c(O)c1", "benzene-1,2,4-triol"),
    ("[O]c1ccc([O-])cc1", "4-oxidophenoxyl"),
    ("[O-]c1ccc([O-])cc1", "benzene-1,4-diolate"),
    ("[O]c1ccc(O)cc1", "4-hydroxyphenoxyl"),
    ("[O-]c1ccc(O)cc1", "4-hydroxyphenolate"),
    ("[O-]S(=O)(=O)Oc2ccccc2", "phenyl sulfate"),
    ("[O-]S(=O)(=O)Oc2c(O)cccc2", "2-hydroxyphenyl sulfate"),
    ("[O-]S(=O)(=O)Oc2ccc(O)cc2", "4-hydroxyphenyl sulfate"),
    ("ClC(Cl)(Cl)Cl", "tetrachloromethane"),
    ("[C](Cl)(Cl)Cl", "trichloromethyl"),
    ("ClC(Cl)Cl", "trichloromethane"),
    ("[O]OC(Cl)(Cl)Cl", "(trichloromethyl)peroxyl"),
    ("OOC(Cl)(Cl)Cl", "trichloromethyl hydroperoxide"),
    ("OC(Cl)(Cl)Cl", "trichloromethanol"),
    ("[CH](Cl)Cl", "dichloromethyl"),
    ("ClCCl", "dichloromethane"),
    ("C1CCOC1", "oxolane"),
    ("OC1CCCO1", "oxolan-2-ol"),
    ("O=C1CCCO1", "oxolan-2-one"),
    ("OCCCC(=O)O", "4-hydroxybutanoic acid"),
    ("O=CCCC(=O)O", "4-oxobutanoic acid"),
    ("O=C(O)CCC(=O)O", "butanedioic acid"),
    ("OCCCC=O", "4-hydroxybutanal"),
    ("OCCCCO", "butane-1,4-diol"),
    ("O=CCCC=O", "butanedial"),
    ("CN(C)C=O", "N,N-dimethylformamide"),
    ("CN(CO)C=O", "N-(hydroxymethyl)-N-methylformamide"),
    ("CNC=O", "N-methylformamide"),
    ("O=CNCO", "N-(hydroxymethyl)formamide"),
    ("NC=O", "formamide"),
    ("CN=C=O", "methyl isocyanate"),
    (
        "CC(=O)N[C@@H](CSC(=O)N(C)O)C(=O)O",
        "(2R)-2-acetamido-3-[(N-hydroxy-N-methylcarbamoyl)sulfanyl]propanoic acid",
    ),
    ("CS(C)=O", "(methanesulfinyl)methane"),
    ("CS(C)(=O)=O", "(methanesulfonyl)methane"),
    ("CSC", "(methylsulfanyl)methane"),
    (
        "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",
        "(2S)-2-amino-5-[[(2R)-1-(carboxymethylamino)-1-oxo-3-sulfanylpropan-2-yl]amino]-5-oxopentanoic acid",
    ),
    ("[CH3]", "methyl"),
    ("CO[O]", "methylperoxyl"),
    ("COO", "methyl hydroperoxide"),
    ("C", "methane"),
    ("C=O", "methanal"),
    ("O=C=O", "carbon dioxide"),
    ("O", "water"),
    ("Cl", "hydrogen chloride"),
    ("O=O", "dioxygen"),
    ("[O-][O]", "superoxide"),
    ("[OH]", "hydroxyl"),
    ("[H+]", "hydron"),
    # Held-out family members, not a closed allow-list.
    ("CCC(=O)C", "butan-2-one"),
    ("CCCC(=O)C", "pentan-2-one"),
    ("CC(O)CO", "propane-1,2-diol"),
    ("OCCO", "ethane-1,2-diol"),
    ("CC(O)C=O", "2-hydroxypropanal"),
    ("CC(=O)CO", "1-hydroxypropan-2-one"),
    ("O=C(O)C=CC", "but-2-enoic acid"),
    ("O=C(O)C(=O)O", "ethanedioic acid"),
    ("ClC", "chloromethane"),
    ("FC(F)F", "trifluoromethane"),
    ("BrCBr", "dibromomethane"),
    ("CCNC=O", "N-ethylformamide"),
    ("CCN(CC)C=O", "N,N-diethylformamide"),
    ("CCSCC", "(ethylsulfanyl)ethane"),
    ("CCS(=O)CC", "(ethanesulfinyl)ethane"),
    ("CCS(=O)(=O)CC", "(ethanesulfonyl)ethane"),
    ("CCOO", "ethyl hydroperoxide"),
    ("O=C1CCCCC1", "cyclohexanone"),
    ("OC1CCCC1", "cyclopentanol"),
    ("OC1CCCCC1", "cyclohexanol"),
    ("[CH2]C", "ethyl"),
    ("CCCl", "chloroethane"),
    ("C=CC=C", "buta-1,3-diene"),
    ("CC=CC=CC", "hexa-2,4-diene"),
    ("CC=CC=CCC", "hepta-2,4-diene"),
    ("NCCO", "2-aminoethan-1-ol"),
    ("c1ccncc1", "pyridine"),
    ("C1=CC=NC=C1", "pyridine"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccoc1", "furan"),
    ("C1=COC=C1", "furan"),
    ("c1ccsc1", "thiophene"),
    ("[nH]1cccc1", "pyrrole"),
    ("[13CH4]", "(13C)methane"),
    ("[2H]C([2H])([2H])[2H]", "(2H4)methane"),
    (
        "O[C@@H]1[C@@H](O)[C@H](Oc2ccccc2)O[C@@H]([C@H]1O)C(=O)O",
        "(2S,3S,4S,5R,6S)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid",
    ),
    ("[O-]S(=O)(=O)Oc2c(O)cc(O)cc2", "2,4-dihydroxyphenyl sulfate"),
    ("[O-]S(=O)(=O)Oc2c(C)cc(C)cc2", "2,4-dimethylphenyl sulfate"),
)


_ALIASES = (
    ("1,2-dihydroxybenzene", "Oc1ccccc1O"),
    ("1,4-dihydroxybenzene", "Oc1ccc(O)cc1"),
    ("1,2,4-trihydroxybenzene", "Oc1ccc(O)c(O)c1"),
    ("1-chloroethane", "CCCl"),
    ("2-aminoethanol", "NCCO"),
    ("bicyclo[4.4.0]deca-1,3,5,7,9-pentaene", "c1ccc2ccccc2c1"),
    ("oxolane", "C1CCOC1"),
    ("1-oxacyclopentane", "C1CCOC1"),
    ("methylsulfanylmethane", "CSC"),
    ("methanesulfinylmethane", "CS(C)=O"),
    ("bicyclo[2.2.1]hept-2,5-diene", "C1=CC2C=CC1C2"),
    ("1-aza-4-azabicyclo[2.2.2]octane", "C1CN2CCN1CC2"),
)


def _state(smiles: str) -> tuple[int, int]:
    entity = complete_open_smiles_hydrogens(from_line_notation(smiles))
    charge = sum(atom.formal_charge or 0 for atom in entity.atoms)
    radicals = sum(atom.radical_electrons for atom in entity.atoms)
    return charge, radicals


@pytest.mark.parametrize(("smiles", "name"), _EXACT)
def test_grammar_name_round_trip(smiles: str, name: str) -> None:
    result = smiles_to_iupac(smiles, strict=True)
    assert result.name == name
    assert result.kind in {
        NamingKind.GENERAL_IUPAC_NAME,
        NamingKind.PREFERRED_IUPAC_NAME,
    }
    assert "molecule-" not in result.name
    parsed = normalize_kekule_aromatic(
        complete_open_smiles_hydrogens(from_iupac_name(result.name))
    )
    original = normalize_kekule_aromatic(
        complete_open_smiles_hydrogens(from_line_notation(smiles))
    )
    # OpenSMILES keeps aromatic and Kekulé edges distinct. Naming normalizes
    # a proven aromatic sextet before comparing constitutions.
    assert constitution_equivalent(original, parsed) is True
    assert (
        sum(atom.formal_charge or 0 for atom in parsed.atoms),
        sum(atom.radical_electrons for atom in parsed.atoms),
    ) == _state(smiles)


@pytest.mark.parametrize(("alias", "smiles"), _ALIASES)
def test_documented_aliases_parse_to_the_same_graph(alias: str, smiles: str) -> None:
    rebuilt = iupac_to_smiles(alias)
    assert notations_equivalent(smiles, rebuilt.value) is True


@pytest.mark.parametrize("name", ("hex-2-one", "but-2-one", "pent-2-one"))
def test_malformed_ketone_stems_are_not_canonical(name: str) -> None:
    with pytest.raises(NamingParseError):
        from_iupac_name(name)


def test_succinic_acid_is_not_renamed_as_a_hydroxy_oxo_acid() -> None:
    result = smiles_to_iupac("O=C(O)CCC(=O)O", strict=True)
    assert result.name == "butanedioic acid"
    assert "hydroxy" not in result.name


def test_multiplied_unsaturation_keeps_the_linking_vowel() -> None:
    assert (
        smiles_to_iupac("C1=CC=CC2OC12", strict=True).name
        == "7-oxabicyclo[4.1.0]hepta-2,4-diene"
    )
    assert "hept-2,4-diene" not in smiles_to_iupac("C1=CC=CC2OC12").name


def test_repeated_oxa_prefixes_use_the_lowest_heteroatom_locants() -> None:
    first = smiles_to_iupac("O1C=CC=CC2OC12", strict=True).name
    second = smiles_to_iupac("C1=CC=COC2OC12", strict=True).name
    assert first == second == "2,8-dioxabicyclo[5.1.0]octa-3,5-diene"
    assert notations_equivalent("O1C=CC=CC2OC12", "C1=CC=COC2OC12") is True


def test_bracket_radicals_are_stored_on_the_atom() -> None:
    entity = from_line_notation("[C](Cl)(Cl)Cl")
    carbon = next(atom for atom in entity.atoms if atom.element == "C")
    assert carbon.radical_electrons == 1
    hydroxyl = from_line_notation("[OH]").atoms[0]
    assert hydroxyl.radical_electrons == 1
    peroxyl = from_line_notation("CO[O]")
    assert sum(atom.radical_electrons for atom in peroxyl.atoms) == 1
    superoxide = from_line_notation("[O-][O]")
    assert sum(atom.formal_charge or 0 for atom in superoxide.atoms) == -1
    assert sum(atom.radical_electrons for atom in superoxide.atoms) == 1
    phenolate = from_line_notation("[O-]c1ccc(O)cc1")
    assert sum(atom.radical_electrons for atom in phenolate.atoms) == 0
    assert sum(atom.formal_charge or 0 for atom in phenolate.atoms) == -1
    assert from_line_notation("[C]").atoms[0].radical_electrons == 4
    assert from_line_notation("[B]").atoms[0].radical_electrons == 3
    assert from_line_notation("[P]").atoms[0].radical_electrons == 3
    assert from_line_notation("[Si]").atoms[0].radical_electrons == 4


def test_hydroperoxide_and_peroxyl_remain_distinct() -> None:
    assert smiles_to_iupac("COO", strict=True).name == "methyl hydroperoxide"
    assert smiles_to_iupac("CO[O]", strict=True).name == "methylperoxyl"
    assert smiles_to_iupac("[CH3]", strict=True).name == "methyl"
    assert smiles_to_iupac("C", strict=True).name == "methane"


def test_equivalent_ketone_traversals_share_one_name() -> None:
    assert smiles_to_iupac("CC(=O)CCC", strict=True).name == smiles_to_iupac(
        "CCCC(=O)C", strict=True
    ).name


def test_compound_prefix_keeps_its_multiplier_when_alphabetized() -> None:
    dimethyl = smiles_to_iupac(
        "O[C@@H]1[C@@H](O)[C@H](Oc2c(C)cc(C)cc2)O[C@@H]([C@H]1O)C(=O)O",
        strict=True,
    ).name
    dihydroxy = smiles_to_iupac(
        "O[C@@H]1[C@@H](O)[C@H](Oc2c(O)cc(O)cc2)O[C@@H]([C@H]1O)C(=O)O",
        strict=True,
    ).name
    assert dimethyl.startswith("(2S,3S,4S,5R,6S)-6-(2,4-dimethylphenoxy)-3,4,5-trihydroxy")
    assert dihydroxy.startswith("(2S,3S,4S,5R,6S)-6-(2,4-dihydroxyphenoxy)-3,4,5-trihydroxy")
    assert iupac_to_smiles(dimethyl).lossless is True
    assert iupac_to_smiles(dihydroxy).lossless is True


def test_polyene_configuration_uses_double_bond_locants() -> None:
    ee = smiles_to_iupac("C/C=C/C=C/C", strict=True)
    assert ee.name == "(2E,4E)-hexa-2,4-diene"
    assert notations_equivalent("C/C=C/C=C/C", iupac_to_smiles(ee.name).value) is True
    ez = smiles_to_iupac(r"C/C=C/C=C\C", strict=True)
    assert ez.name == "(2E,4Z)-hexa-2,4-diene"
    assert smiles_to_iupac(r"C/C=C\C=C\C", strict=True).name == ez.name
    assert notations_equivalent(r"C/C=C/C=C\C", iupac_to_smiles(ez.name).value) is True
    zz = smiles_to_iupac(r"C/C=C\C=C/C", strict=True)
    assert zz.name == "(2Z,4Z)-hexa-2,4-diene"
    assert notations_equivalent(r"C/C=C\C=C/C", iupac_to_smiles(zz.name).value) is True
    assert smiles_to_iupac("C/C=C/C", strict=True).name == "(E)-but-2-ene"


def test_repeated_nuclide_is_one_grouped_label() -> None:
    assert smiles_to_iupac("[13CH3][13CH3]", strict=True).name == "(1,2-13C2)ethane"
    assert iupac_to_smiles("(13C1)methane").lossless is True
    assert from_iupac_name("(13C1)methane").atoms[0].isotope == 13


def test_quinoline_is_the_retained_benzopyridine() -> None:
    assert smiles_to_iupac("c1ccc2ncccc2c1", strict=True).name == "quinoline"
    assert smiles_to_iupac("n1cccc2ccccc12", strict=True).name == "quinoline"
    isoquinoline = smiles_to_iupac("c1ccc2cnccc2c1", strict=True)
    assert isoquinoline.name != "quinoline"
    assert "aza" in isoquinoline.name
    indole = smiles_to_iupac("c1ccc2[nH]ccc2c1", strict=False)
    assert indole.kind is NamingKind.IUPAC_COMPOSITION_DESCRIPTION


def test_alkene_configuration_round_trip_checks_stereo() -> None:
    """E and Z are different stereoisomers, not only different constitutions."""
    pairs = (
        ("C/C=C/C(=O)O", "(2E)-but-2-enoic acid", "(2Z)-but-2-enoic acid"),
        ("C/C=C\\C(=O)O", "(2Z)-but-2-enoic acid", "(2E)-but-2-enoic acid"),
        ("C/C=C/C", "(E)-but-2-ene", "(Z)-but-2-ene"),
    )
    for smiles, name, other in pairs:
        result = smiles_to_iupac(smiles, strict=True)
        assert result.name == name
        assert notations_equivalent(smiles, iupac_to_smiles(name).value) is True
        assert notations_equivalent(smiles, iupac_to_smiles(other).value) is False
    assert notations_equivalent("C/C=C/C(=O)O", "OC(=O)/C=C/C") is True


def test_bis_glutathione_carbonyl_stays_outside_the_grammar() -> None:
    smiles = (
        "O=C(SC[C@H](NC(=O)CC[C@H](N)C(=O)O)C(=O)NCC(=O)O)"
        "SC[C@H](NC(=O)CC[C@H](N)C(=O)O)C(=O)NCC(=O)O"
    )
    result = smiles_to_iupac(smiles, strict=False)
    assert result.kind is NamingKind.IUPAC_COMPOSITION_DESCRIPTION
    with pytest.raises(NamingIndeterminateError):
        smiles_to_iupac(smiles, strict=True)
