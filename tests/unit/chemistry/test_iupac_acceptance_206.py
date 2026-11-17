"""Acceptance contract for bidirectional IUPAC naming.

Gold spellings in ``iupac_acceptance_206.json`` are Blue Book or literature
names, not a snapshot of the generator. Rows 87 and 171 stay provisional:
a graph round trip is recorded separately and is not a certified IUPAC name.
The 389 probes are six seeded RDKit traversals per diagram input, deduplicated,
plus the Kekulé form. Graph identity uses constitution and explicit stereo
labels, not raw SMILES text.

Unsupported strict refusals stay visible. ``strict=False`` composition text
is not treated as a name.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pytest

from molcrys_kit import (
    NamingParseError,
    from_iupac_name,
    iupac_to_smiles,
    smiles_to_iupac,
)
from molcrys_kit.chemistry import from_line_notation
from molcrys_kit.chemistry.crystal_stereo import classify_entity_relationship
from molcrys_kit.chemistry.equivalence import constitution_equivalent
from molcrys_kit.chemistry.name_conversion import complete_open_smiles_hydrogens
from molcrys_kit.chemistry.naming import NamingKind, _formula
from molcrys_kit.chemistry.representation import normalize_kekule_aromatic
from molcrys_kit.chemistry.stereo import assign_stereochemistry

_DATA = Path(__file__).resolve().parents[2] / "data" / "chemistry_golden"
_MANIFEST = json.loads((_DATA / "iupac_acceptance_206.json").read_text(encoding="utf-8"))
_SWEEP = json.loads((_DATA / "iupac_invariance_206.json").read_text(encoding="utf-8"))
_REPORT_PATH = _DATA / "iupac_coverage_206.json"
_TIMEOUT = float(_MANIFEST["timeout_seconds"])
_ROWS = _MANIFEST["rows"]
_BY_ID = {row["id"]: row for row in _ROWS}


def _entity(smiles: str):
    return normalize_kekule_aromatic(
        complete_open_smiles_hydrogens(from_line_notation(smiles))
    )


def _labels(entity) -> tuple[tuple[str, str], ...]:
    report = assign_stereochemistry(entity)
    return tuple(
        sorted(
            (item.kind.value, item.descriptor)
            for item in report.descriptors
            if item.descriptor
        )
    )


def _atom_state(entity) -> tuple:
    return tuple(
        sorted(
            (
                atom.element,
                atom.isotope,
                atom.formal_charge or 0,
                atom.radical_electrons,
                atom.implicit_hydrogens or 0,
                atom.explicit_hydrogens or 0,
            )
            for atom in entity.atoms
        )
    )


def _bond_state(entity) -> tuple:
    return tuple(
        sorted((bond.order, bond.aromatic, bond.kind.value) for bond in entity.bonds)
    )


def _same_state(left, right) -> bool:
    """Connectivity, bond order, H, isotope, charge, radicals, and explicit stereo."""
    return (
        constitution_equivalent(left, right)
        and _atom_state(left) == _atom_state(right)
        and _bond_state(left) == _bond_state(right)
        and _labels(left) == _labels(right)
    )


def _timed(function):
    start = perf_counter()
    try:
        value = function()
    except Exception as exc:
        return None, exc, perf_counter() - start
    return value, None, perf_counter() - start


def _evaluate(row: dict) -> dict:
    smiles = row["smiles"]
    source = _entity(smiles)
    forward, forward_error, forward_seconds = _timed(
        lambda: smiles_to_iupac(smiles, strict=True)
    )
    loose, loose_error, _loose_seconds = _timed(
        lambda: smiles_to_iupac(smiles, strict=False)
    )
    inverse_match = None
    inverse_dialect = None
    inverse_error = None
    inverse_seconds = 0.0
    if row["name"]:
        inverse, inverse_error, inverse_seconds = _timed(
            lambda: iupac_to_smiles(row["name"])
        )
        if inverse is not None:
            inverse_dialect = inverse.dialect
            rebuilt = normalize_kekule_aromatic(
                complete_open_smiles_hydrogens(from_iupac_name(row["name"]))
            )
            inverse_match = _same_state(source, rebuilt)
    emitted_match = None
    emitted_dialect = None
    emitted_seconds = 0.0
    emitted_name = forward.name if forward is not None else None
    if emitted_name:
        emitted, emitted_error, emitted_seconds = _timed(
            lambda: iupac_to_smiles(emitted_name)
        )
        if emitted is not None and emitted_error is None:
            emitted_dialect = emitted.dialect
            rebuilt = normalize_kekule_aromatic(
                complete_open_smiles_hydrogens(from_iupac_name(emitted_name))
            )
            emitted_match = _same_state(source, rebuilt)
        else:
            emitted_match = False
    return {
        "id": row["id"],
        "family": row["family"],
        "role": row["role"],
        "support": row["support"],
        "formula": _formula(source),
        "charge": sum(atom.formal_charge or 0 for atom in source.atoms),
        "radical_electrons": sum(atom.radical_electrons for atom in source.atoms),
        "isotopes": sorted(atom.isotope for atom in source.atoms if atom.isotope),
        "forward_name": emitted_name,
        "forward_error": type(forward_error).__name__ if forward_error else None,
        "forward_kind": forward.kind.value if forward is not None else None,
        "preferred": forward.preferred if forward is not None else None,
        "forward_seconds": forward_seconds,
        "nonstrict_kind": loose.kind.value if loose is not None else None,
        "nonstrict_error": type(loose_error).__name__ if loose_error else None,
        "inverse_match": inverse_match,
        "inverse_dialect": inverse_dialect,
        "inverse_error": type(inverse_error).__name__ if inverse_error else None,
        "inverse_seconds": inverse_seconds,
        "emitted_match": emitted_match,
        "emitted_dialect": emitted_dialect,
        "emitted_seconds": emitted_seconds,
    }


def _bucket(result: dict) -> str:
    if result["support"] == "unsupported":
        if result["forward_error"] == "NamingIndeterminateError":
            return "unsupported"
        if result["forward_name"]:
            return "wrong"
        return "unsupported"
    if result["support"] == "uncertified_round_trip":
        if result["emitted_match"] is True and result["forward_kind"] == "general_iupac_name":
            return "uncertified_round_trip"
        return "wrong"
    if result["forward_error"]:
        return "unsupported"
    if result["forward_name"] != _BY_ID[result["id"]]["name"]:
        return "wrong"
    if result["inverse_match"] is not True or result["emitted_match"] is not True:
        return "wrong"
    return "certified"


def _coverage(results: dict[str, dict], invariance: dict) -> dict:
    diagram = [item for item in results.values() if item["role"] == "diagram"]
    buckets = ["certified", "uncertified_round_trip", "unsupported", "wrong"]

    def counts(items):
        found = {name: 0 for name in buckets}
        for item in items:
            found[_bucket(item)] += 1
        return found

    by_family: dict[str, dict[str, int]] = {}
    for item in diagram:
        family = by_family.setdefault(item["family"], {name: 0 for name in buckets})
        family[_bucket(item)] += 1
    directions = {
        "forward_strict": {name: 0 for name in buckets},
        "inverse_of_reference_name": {"matched_graph": 0, "refused": 0, "not_applicable": 0},
        "emitted_name_round_trip": {
            "opensmiles": 0,
            "mck_ln": 0,
            "failed": 0,
            "no_strict_name": 0,
        },
    }
    for item in diagram:
        bucket = _bucket(item)
        directions["forward_strict"][bucket] += 1
        if item["support"] != "certified":
            directions["inverse_of_reference_name"]["not_applicable"] += 1
        elif item["inverse_match"] is True:
            directions["inverse_of_reference_name"]["matched_graph"] += 1
        else:
            directions["inverse_of_reference_name"]["refused"] += 1
        if not item["forward_name"]:
            directions["emitted_name_round_trip"]["no_strict_name"] += 1
        elif item["emitted_match"] is True and item["emitted_dialect"] == "OpenSMILES":
            directions["emitted_name_round_trip"]["opensmiles"] += 1
        elif item["emitted_match"] is True and item["emitted_dialect"] == "MCK-LN":
            directions["emitted_name_round_trip"]["mck_ln"] += 1
        else:
            directions["emitted_name_round_trip"]["failed"] += 1
    timed = [
        item["forward_seconds"]
        for item in results.values()
    ] + [
        item["inverse_seconds"]
        for item in results.values()
        if item["inverse_seconds"]
    ] + [
        item["emitted_seconds"]
        for item in results.values()
        if item["emitted_seconds"]
    ]
    return {
        "timeout_seconds": _TIMEOUT,
        "timeouts": sum(1 for value in timed if value > _TIMEOUT),
        "max_seconds_cap_note": "Wall-clock samples are not frozen. Every timed call must finish within timeout_seconds.",
        "diagram_rows": len(diagram),
        "diagram": counts(diagram),
        "seed_rows": counts(item for item in results.values() if item["role"] == "seed"),
        "held_out_rows": counts(item for item in results.values() if item["role"] == "held_out"),
        "by_family": by_family,
        "by_direction": directions,
        "aliases_accepted": len(_MANIFEST["aliases"]),
        "malformed_spellings_rejected": len(_MANIFEST["rejected"]),
        "invariance_probes": invariance["probes"],
        "invariance_identity_failures": invariance["identity_failures"],
        "invariance_support_changes": invariance["support_changes"],
        "invariance_name_changes": invariance["name_changes"],
        "note": "Certified counts are standard-name matches against the independent manifest. Emitted-name round trips, including row 171, are not certified IUPAC names.",
    }


@pytest.fixture(scope="module")
def evaluated() -> dict[str, dict]:
    return {row["id"]: _evaluate(row) for row in _ROWS}


@pytest.fixture(scope="module")
def invariance(evaluated) -> dict:
    failures = {"identity_failures": 0, "support_changes": 0, "name_changes": 0}
    probes = 0
    parents = {row["id"]: row for row in _ROWS if row["role"] == "diagram"}
    for parent in _SWEEP["parents"]:
        row = parents[parent["id"]]
        base = evaluated[row["id"]]
        source = _entity(row["smiles"])
        for variant in parent["variants"]:
            probes += 1
            try:
                parsed = _entity(variant)
            except Exception:
                failures["identity_failures"] += 1
                continue
            if not _same_state(source, parsed):
                failures["identity_failures"] += 1
                continue
            result, error, elapsed = _timed(lambda variant=variant: smiles_to_iupac(variant, strict=True))
            assert elapsed <= _TIMEOUT, parent["id"]
            got_name = result.name if result is not None else None
            got_error = type(error).__name__ if error else None
            if got_error != base["forward_error"]:
                failures["support_changes"] += 1
            elif got_name != base["forward_name"]:
                failures["name_changes"] += 1
    failures["probes"] = probes
    return failures


def test_sweep_file_is_the_seeded_389_probe_set() -> None:
    assert _SWEEP["seed"] == 20261031
    assert _SWEEP["traversals"] == 6
    assert _SWEEP["probe_count"] == 389
    assert sum(len(parent["variants"]) for parent in _SWEEP["parents"]) == 389
    assert len(_SWEEP["parents"]) == 80


def test_frozen_sweep_matches_rdkit_when_available() -> None:
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem

    assert rdkit.__version__ == "2026.09.1"
    fresh = []
    for row in _ROWS:
        if row["role"] != "diagram":
            continue
        molecule = Chem.MolFromSmiles(row["smiles"])
        assert molecule is not None
        variants = set(Chem.MolToRandomSmilesVect(molecule, 6, randomSeed=20261031))
        Chem.Kekulize(molecule)
        variants.add(Chem.MolToSmiles(molecule, kekuleSmiles=True))
        parent = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))
        for variant in variants:
            parsed = Chem.MolFromSmiles(variant)
            assert parsed is not None
            assert Chem.MolToSmiles(parsed) == parent
        fresh.append({"id": row["id"], "variants": sorted(variants)})
    assert fresh == _SWEEP["parents"]


@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row["id"])
def test_manifest_state_and_directions(row: dict, evaluated: dict[str, dict]) -> None:
    result = evaluated[row["id"]]
    assert result["forward_seconds"] <= _TIMEOUT
    assert result["inverse_seconds"] <= _TIMEOUT
    assert result["emitted_seconds"] <= _TIMEOUT
    assert result["formula"] == row["formula"]
    assert result["charge"] == row["charge"]
    assert result["radical_electrons"] == row["radical_electrons"]
    assert result["isotopes"] == row["isotopes"]
    if row["support"] == "unsupported":
        assert result["forward_error"] == "NamingIndeterminateError"
        assert result["forward_name"] is None
        assert result["nonstrict_kind"] == NamingKind.IUPAC_COMPOSITION_DESCRIPTION.value
        assert result["nonstrict_kind"] != NamingKind.GENERAL_IUPAC_NAME.value
        return
    if row["support"] == "uncertified_round_trip":
        assert result["forward_kind"] == NamingKind.GENERAL_IUPAC_NAME.value
        assert result["preferred"] is False
        assert result["emitted_match"] is True
        assert row["name"] is None
        return
    assert result["forward_name"] == row["name"]
    assert result["inverse_match"] is True
    assert result["emitted_match"] is True
    if row["name_status"] == "preferred_retained":
        assert result["preferred"] is True
        assert result["forward_kind"] == NamingKind.PREFERRED_IUPAC_NAME.value
    else:
        assert result["preferred"] is False
        assert result["forward_kind"] == NamingKind.GENERAL_IUPAC_NAME.value


@pytest.mark.parametrize("alias", _MANIFEST["aliases"], ids=lambda alias: alias["alias"])
def test_alias_is_accepted_and_not_canonical(alias: dict) -> None:
    rebuilt = iupac_to_smiles(alias["alias"])
    assert _same_state(_entity(alias["smiles"]), _entity(rebuilt.value))
    assert smiles_to_iupac(alias["smiles"], strict=True).name == alias["canonical"]
    assert alias["alias"] != alias["canonical"]


@pytest.mark.parametrize("case", _MANIFEST["rejected"], ids=lambda case: case["name"])
def test_malformed_spelling_is_not_a_standard_name(case: dict) -> None:
    with pytest.raises(NamingParseError, match="not canonical"):
        from_iupac_name(case["name"])


@pytest.mark.parametrize("case", _MANIFEST["distinct"], ids=lambda case: case["label"])
def test_distinct_chemical_states_do_not_collapse(case: dict) -> None:
    left = complete_open_smiles_hydrogens(from_line_notation(case["left"]))
    right = complete_open_smiles_hydrogens(from_line_notation(case["right"]))
    relationship = classify_entity_relationship(left, right)
    assert relationship.value == case["relationship"]
    assert smiles_to_iupac(case["left"], strict=True).name != smiles_to_iupac(
        case["right"], strict=True
    ).name


def test_invariance_sweep_has_no_support_or_name_changes(invariance: dict) -> None:
    assert invariance["probes"] == 389
    assert invariance["identity_failures"] == 0
    assert invariance["support_changes"] == 0
    assert invariance["name_changes"] == 0


def test_published_coverage_report(evaluated: dict[str, dict], invariance: dict) -> None:
    report = _coverage(evaluated, invariance)
    frozen = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    assert report == frozen
    assert report["diagram"]["wrong"] == 0
    assert report["diagram"]["unsupported"] == 1
    assert report["diagram"]["uncertified_round_trip"] == 1
    assert report["diagram"]["certified"] == 78
    assert report["timeouts"] == 0
    assert report["by_direction"]["emitted_name_round_trip"]["mck_ln"] > 0
    assert report["by_direction"]["emitted_name_round_trip"]["opensmiles"] > 0
    assert report["seed_rows"]["wrong"] == 0
    assert report["held_out_rows"]["wrong"] == 0
    assert report["held_out_rows"]["certified"] > 20
    handbook = (
        Path(__file__).resolve().parents[3] / "docs" / "chemistry_coverage.md"
    ).read_text(encoding="utf-8")
    assert f"{report['diagram']['certified']} match that manifest" in handbook
    assert f"{report['invariance_probes']} equivalent inputs" in handbook
    assert "within 5 seconds" in handbook
