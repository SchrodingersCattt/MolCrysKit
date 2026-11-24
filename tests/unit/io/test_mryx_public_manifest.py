"""Offline contracts for public Mercury fixture metadata (no network)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "mryx" / "public_manifest.json"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _fixture_paths():
    paths_file = MANIFEST.parent / "fixture_paths.py"
    spec = importlib.util.spec_from_file_location("mryx_fixture_paths", paths_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _derive_module():
    derive_file = MANIFEST.parent / "derive_fixture.py"
    spec = importlib.util.spec_from_file_location("mryx_derive_fixture", derive_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_mryx_manifest_lists_bundled_fixtures_with_hashes():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "molcrys-kit.mryx-public-fixtures.v2"
    fixtures = data["fixtures"]
    assert fixtures, "at least one public fixture entry is required"
    for entry in fixtures:
        assert _SHA256.match(entry["derived_sha256"])
        assert entry["bundled_path"]
        upstream = entry["upstream"]
        assert _SHA256.match(upstream["sha256"])
        assert upstream["source_url"].startswith("https://")
        assert upstream["license"]
        assert entry["anonymization"]["byte_identical_to_upstream"] is False
        derivation = entry["derivation"]
        assert derivation["gzip_mtime"] == 0
        assert derivation["gzip_compresslevel"] == 9
        tier = entry["tier_a_expectations"]
        assert tier["n_atoms"] > 0
        assert len(tier["cellpar_deg"]) == 6
        assert tier["structure_label"] == entry["anonymization"]["replaces_structure_label"]


def test_public_mryx_manifest_documents_blocked_sources():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    blocked = data.get("investigated_but_blocked", [])
    assert blocked, "record publishers or hosts that blocked automated download"
    for item in blocked:
        assert item.get("blocker")


@pytest.mark.parametrize(
    "entry",
    json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"],
    ids=lambda item: item["id"],
)
def test_bundled_fixture_present_and_matches_derived_sha256(entry):
    path = _fixture_paths().resolve_bundled_fixture(entry)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_resolve_bundled_fixture_fails_closed_on_checksum_mismatch():
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"][0]
    _fixture_paths().resolve_bundled_fixture(entry)
    with pytest.raises(ValueError, match="derived sha256 mismatch"):
        _fixture_paths().resolve_bundled_fixture({**entry, "derived_sha256": "0" * 64})


def test_stable_gzip_derivation_is_independent_of_wall_clock(monkeypatch):
    derive = _derive_module()
    payload = b"<?xml version='1.0'?><All><label>mck_tier_a_triclinic_anon</label></All>"
    monkeypatch.setattr("time.time", lambda: 1_000_000.0)
    first = derive.stable_gzip_compress(payload)
    monkeypatch.setattr("time.time", lambda: 9_999_999_999.0)
    second = derive.stable_gzip_compress(payload)
    assert first == second


def test_bundled_fixture_matches_deterministic_gzip_payload():
    import gzip

    derive = _derive_module()
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"][0]
    bundled = _fixture_paths().resolve_bundled_fixture(entry)
    payload = gzip.decompress(bundled.read_bytes())
    recompressed = derive.stable_gzip_compress(payload)
    assert recompressed == bundled.read_bytes()


def test_anonymize_upstream_twice_yields_identical_bytes(monkeypatch):
    derive = _derive_module()
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"][0]
    bundled = _fixture_paths().resolve_bundled_fixture(entry)
    # Round-trip: bundled → replace label back → anonymize again → same bundled bytes.
    import gzip

    xml = gzip.decompress(bundled.read_bytes()).decode("utf-8")
    anon = entry["anonymization"]["replaces_structure_label"]
    raw = entry["anonymization"]["removed_identifiers"][0]
    restored = xml.replace(anon, raw)
    upstream_like = derive.stable_gzip_compress(restored.encode("utf-8"))
    monkeypatch.setattr("time.time", lambda: 100.0)
    first = derive.anonymize_upstream_bytes(
        upstream_like,
        entry["anonymization"]["removed_identifiers"][0],
        entry["anonymization"]["replaces_structure_label"],
    )
    monkeypatch.setattr("time.time", lambda: 9_999_999_999.0)
    second = derive.anonymize_upstream_bytes(
        upstream_like,
        entry["anonymization"]["removed_identifiers"][0],
        entry["anonymization"]["replaces_structure_label"],
    )
    assert first == second == bundled.read_bytes()


@pytest.mark.parametrize(
    "entry",
    json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"],
    ids=lambda item: item["id"],
)
def test_bundled_fixture_anonymization_preserves_site_count(entry):
    import warnings

    from molcrys_kit.io import read_mryx_atoms

    path = _fixture_paths().resolve_bundled_fixture(entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        atoms = read_mryx_atoms(path)
    tier = entry["tier_a_expectations"]
    assert len(atoms) == tier["n_atoms"]
    assert atoms.info["mryx"].get("label") == tier["structure_label"]