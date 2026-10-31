"""Literature and Blue Book names that are not generator snapshots.

Provenance is the spelling itself: Blue Book construction (P-14, P-31, P-44,
P-63, P-82) or the published phenyl glucuronide and aryl sulfate names from
the bidirectional-naming acceptance set. A row is general IUPAC unless it is
one of the reviewed preferred parents. Parsing the published glucuronide name
must rebuild that anomer, not the other one.
"""

from __future__ import annotations

import pytest

from molcrys_kit import from_iupac_name, smiles_to_iupac
from molcrys_kit.chemistry import from_line_notation
from molcrys_kit.chemistry.crystal_stereo import (
    EntityRelationship,
    classify_entity_relationship,
)
from molcrys_kit.chemistry.name_conversion import complete_open_smiles_hydrogens
from molcrys_kit.chemistry.naming import NamingKind


# family, smiles, canonical name, provenance
_GOLD = (
    (
        "ketone",
        "CC(=O)CCCC",
        "hexan-2-one",
        "Blue Book P-44 saturated suffix keeps the 'an' ending",
    ),
    (
        "polyol",
        "Oc1ccccc1O",
        "benzene-1,2-diol",
        "Blue Book P-63 hydroxy is the principal characteristic group",
    ),
    (
        "polyol",
        "Oc1ccc(O)cc1",
        "benzene-1,4-diol",
        "Blue Book P-63",
    ),
    (
        "polyol",
        "Oc1ccc(O)c(O)c1",
        "benzene-1,2,4-triol",
        "Blue Book P-63",
    ),
    (
        "polyene",
        "C=CC=C",
        "buta-1,3-diene",
        "Blue Book P-31.1 linking vowel",
    ),
    (
        "polyene",
        "CC=CC=CCC",
        "hepta-2,4-diene",
        "Blue Book P-31.1; lowest unsaturation locants",
    ),
    (
        "haloalkane",
        "CCCl",
        "chloroethane",
        "Blue Book P-14.3 locant 1 is omitted on monosubstituted ethane",
    ),
    (
        "aminoalcohol",
        "NCCO",
        "2-aminoethan-1-ol",
        "Blue Book substitutive suffix -ol with an amino prefix",
    ),
    (
        "retained-ring",
        "c1ccncc1",
        "pyridine",
        "Blue Book P-22 retained parent; skeletal replacement is not used",
    ),
    (
        "retained-ring",
        "c1ccc2ccccc2c1",
        "naphthalene",
        "Blue Book retained fused parent",
    ),
    (
        "retained-ring",
        "c1ccoc1",
        "furan",
        "Blue Book retained parent",
    ),
    (
        "isotope",
        "[13CH4]",
        "(13C)methane",
        "Blue Book P-82 nuclide prefix without a redundant locant",
    ),
    (
        "isotope",
        "[2H]C([2H])([2H])[2H]",
        "(2H4)methane",
        "Blue Book P-82 hydrogen nuclide count",
    ),
    (
        "glucuronide",
        "O[C@@H]1[C@@H](O)[C@H](Oc2ccccc2)O[C@@H]([C@H]1O)C(=O)O",
        "(2S,3S,4S,5R,6S)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid",
        "phenyl beta-D-glucuronide; anomeric descriptor 6S",
    ),
    (
        "glucuronide",
        "O[C@@H]1[C@@H](O)[C@@H](Oc2ccccc2)O[C@@H]([C@H]1O)C(=O)O",
        "(2S,3S,4S,5R,6R)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid",
        "phenyl alpha anomer; the beta literature name must not be reused",
    ),
    (
        "glucuronide",
        "O[C@@H]1[C@@H](O)[C@H](Oc2c(O)cc(O)cc2)O[C@@H]([C@H]1O)C(=O)O",
        "(2S,3S,4S,5R,6S)-3,4,5-trihydroxy-6-(2,4-dihydroxyphenoxy)oxane-2-carboxylic acid",
        "2,4-dihydroxyphenyl glucuronide",
    ),
    (
        "sulfate",
        "[O-]S(=O)(=O)Oc2c(O)cc(O)cc2",
        "2,4-dihydroxyphenyl sulfate",
        "aryl sulfate monoester, 2,4-dihydroxy substitution",
    ),
    (
        "sulfate",
        "[O-]S(=O)(=O)Oc2c(C)cc(C)cc2",
        "2,4-dimethylphenyl sulfate",
        "aryl sulfate monoester, 2,4-dimethyl substitution",
    ),
)

_BETA = "O[C@@H]1[C@@H](O)[C@H](Oc2ccccc2)O[C@@H]([C@H]1O)C(=O)O"
_ALPHA = "O[C@@H]1[C@@H](O)[C@@H](Oc2ccccc2)O[C@@H]([C@H]1O)C(=O)O"
_BETA_NAME = "(2S,3S,4S,5R,6S)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid"
_ALPHA_NAME = "(2S,3S,4S,5R,6R)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid"


def _relationship(smiles: str, name: str) -> EntityRelationship:
    left = complete_open_smiles_hydrogens(from_line_notation(smiles))
    right = complete_open_smiles_hydrogens(from_iupac_name(name))
    return classify_entity_relationship(left, right)


@pytest.mark.parametrize(("family", "smiles", "name", "provenance"), _GOLD)
def test_literature_name_is_canonical(family: str, smiles: str, name: str, provenance: str) -> None:
    assert family
    assert provenance
    result = smiles_to_iupac(smiles, strict=True)
    assert result.name == name
    assert result.kind in {
        NamingKind.GENERAL_IUPAC_NAME,
        NamingKind.PREFERRED_IUPAC_NAME,
    }
    assert result.preferred is False or name in {"benzene", "phenol", "methane"}
    assert "not published as an R/S descriptor" not in " ".join(result.warnings)


@pytest.mark.parametrize(
    ("left", "right", "name"),
    (
        ("Oc1ccccc1O", "c1ccc(O)c(O)c1", "benzene-1,2-diol"),
        ("C1=C(O)C(O)=CC=C1", "Oc1ccccc1O", "benzene-1,2-diol"),
        ("CC=CC=CCC", "CCC=CC=CC", "hepta-2,4-diene"),
        ("c1ccncc1", "C1=CC=NC=C1", "pyridine"),
        ("c1ccoc1", "C1=COC=C1", "furan"),
    ),
)
def test_equivalent_inputs_share_one_literature_name(left: str, right: str, name: str) -> None:
    assert smiles_to_iupac(left, strict=True).name == name
    assert smiles_to_iupac(right, strict=True).name == name


def test_published_glucuronide_names_rebuild_the_matching_anomer() -> None:
    assert smiles_to_iupac(_BETA, strict=True).name == _BETA_NAME
    assert smiles_to_iupac(_ALPHA, strict=True).name == _ALPHA_NAME
    assert _relationship(_BETA, _BETA_NAME) is EntityRelationship.SAME_STEREOISOMER
    assert _relationship(_ALPHA, _ALPHA_NAME) is EntityRelationship.SAME_STEREOISOMER
    assert _relationship(_ALPHA, _BETA_NAME) is EntityRelationship.STEREOISOMER
    assert _relationship(_BETA, _ALPHA_NAME) is EntityRelationship.STEREOISOMER


def test_family_grid_covers_each_required_family() -> None:
    families = {family for family, *_rest in _GOLD}
    assert families >= {
        "ketone",
        "polyol",
        "polyene",
        "haloalkane",
        "aminoalcohol",
        "retained-ring",
        "isotope",
        "glucuronide",
        "sulfate",
    }
