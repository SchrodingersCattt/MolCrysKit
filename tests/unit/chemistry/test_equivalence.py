"""Tests for the notation equivalence convenience function."""

from __future__ import annotations

from molcrys_kit.chemistry import notations_equivalent


def test_identical_smiles_are_equivalent() -> None:
    assert notations_equivalent("CCO", "CCO") is True


def test_reordered_smiles_are_equivalent() -> None:
    assert notations_equivalent("CCO", "OCC") is True


def test_different_molecules_are_not_equivalent() -> None:
    assert notations_equivalent("CCO", "CCN") is False


def test_stereo_without_coordinates_is_indeterminate() -> None:
    # OpenSMILES stereo tokens without 3D coordinates cannot be resolved.
    assert notations_equivalent("N[C@@H](C)C(=O)O", "N[C@@H](C)C(=O)O") is None


def test_stereo_mirror_without_coordinates_is_indeterminate() -> None:
    assert notations_equivalent("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O") is None


def test_branched_ethanol_representations() -> None:
    assert notations_equivalent("C(O)C", "CCO") is True


def test_cyclohexane_equivalent() -> None:
    assert notations_equivalent("C1CCCCC1", "C1CCCCC1") is True


def test_aromatic_and_kekule_are_distinct_graphs() -> None:
    # MCK treats aromatic-flag bonds and explicit double bonds as different
    # graph edges; this is correct OpenSMILES semantics.
    assert notations_equivalent("c1ccccc1", "C1=CC=CC=C1") is False


def test_same_aromatic_notation_is_equivalent() -> None:
    assert notations_equivalent("c1ccccc1", "c1ccccc1") is True


def test_invalid_notation_returns_none() -> None:
    assert notations_equivalent("not_a_molecule", "CCO") is None


def test_empty_notation_returns_none() -> None:
    assert notations_equivalent("", "CCO") is None


def test_malformed_mck_ln_returns_none_instead_of_crashing() -> None:
    # Invalid BondKind inside MCK-LN raises bare ValueError, not LineNotationError.
    bogus = (
        "MCK-LN1|type=finite|id=x"
        "|atoms=x~C~_~_~_~_~_~_~_~_"
        "|bonds=0~0~1.0~BOGUS~0~0,0,0~_"
        "|coord=_|xyz=_|charge=_|status=explicit"
    )
    assert notations_equivalent(bogus, "CCO") is None


def test_malformed_mck_ln_bad_float_returns_none() -> None:
    bad_float = (
        "MCK-LN1|type=finite|id=x"
        "|atoms=x~C~_~_~_~_~_~_~_~_"
        "|bonds=0~0~not_a_number~covalent~0~0,0,0~_"
        "|coord=_|xyz=_|charge=_|status=explicit"
    )
    assert notations_equivalent(bad_float, "CCO") is None


def test_disconnected_fragments() -> None:
    assert notations_equivalent("C.C", "C.C") is True
    assert notations_equivalent("C.C", "C.N") is False
