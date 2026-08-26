import pytest

from molcrys_kit.analysis.formula_moiety import (
    Fragment,
    heavy_signature,
    match_molecule_to_fragment,
    parse_moiety_string,
)


def test_parse_single_neutral_fragment():
    fragments = parse_moiety_string("'C8 H9 N1 O2'")

    assert fragments == [
        Fragment(
            multiplier=1.0,
            composition={"C": 8, "H": 9, "N": 1, "O": 2},
            charge=0,
            raw="C8 H9 N1 O2",
        )
    ]


def test_parse_ion_pair_semicolon_value():
    fragments = parse_moiety_string(";\nC1 H6 N1 1+,Cl1 O4 1-\n;")

    assert fragments is not None
    assert fragments[0].composition == {"C": 1, "H": 6, "N": 1}
    assert fragments[0].charge == 1
    assert fragments[1].composition == {"Cl": 1, "O": 4}
    assert fragments[1].charge == -1


def test_parse_fractional_solvent_fragment():
    fragments = parse_moiety_string(
        "C32 H41 Br1 N5 O5 1+,C1 H3 O3 S1 1-,0.5(C3 H8 O1)"
    )

    assert fragments is not None
    assert len(fragments) == 3
    assert fragments[0].composition["H"] == 41
    assert fragments[0].charge == 1
    assert fragments[1].composition == {"C": 1, "H": 3, "O": 3, "S": 1}
    assert fragments[1].charge == -1
    assert fragments[2].multiplier == 0.5
    assert fragments[2].composition == {"C": 3, "H": 8, "O": 1}


@pytest.mark.parametrize("value", [None, "", "?", "'?'"])
def test_parse_absent_or_unknown_returns_none(value):
    assert parse_moiety_string(value) is None


def test_parse_malformed_returns_none_with_warning():
    with pytest.warns(RuntimeWarning):
        assert parse_moiety_string("C8 H9 @") is None


def test_heavy_signature_ignores_hydrogen():
    assert heavy_signature({"C": 3, "H": 8, "O": 1}) == (("C", 3), ("O", 1))


def test_match_molecule_to_unique_fragment():
    fragments = parse_moiety_string(
        "C32 H41 Br1 N5 O5 1+,C1 H3 O3 S1 1-,0.5(C3 H8 O1)"
    )

    assert fragments is not None
    cation_symbols = ["C"] * 32 + ["Br"] + ["N"] * 5 + ["O"] * 5
    mesylate_symbols = ["C"] + ["H"] * 3 + ["O"] * 3 + ["S"]

    assert match_molecule_to_fragment(cation_symbols, fragments) == fragments[0]
    assert match_molecule_to_fragment(mesylate_symbols, fragments) == fragments[1]


def test_match_ambiguous_fragment_returns_none_with_warning():
    fragments = parse_moiety_string("H2 O1,H3 O1 1+")

    assert fragments is not None
    with pytest.warns(RuntimeWarning, match="Ambiguous"):
        assert match_molecule_to_fragment(["O"], fragments) is None
