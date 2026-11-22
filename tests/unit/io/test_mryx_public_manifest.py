"""Offline contracts for public Mercury fixture metadata (no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "mryx" / "public_manifest.json"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def test_public_mryx_manifest_lists_traceable_fixtures_with_hashes():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "molcrys-kit.mryx-public-fixtures.v1"
    fixtures = data["fixtures"]
    assert fixtures, "at least one public fixture entry is required"
    for entry in fixtures:
        assert _SHA256.match(entry["sha256"])
        assert entry["url"].startswith("https://")
        assert entry["license"]
        tier = entry["tier_a_expectations"]
        assert tier["n_atoms"] > 0
        assert len(tier["cellpar_deg"]) == 6


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
def test_offline_public_fixture_when_cache_populated(entry, tmp_path_factory):
    """If MOLCRYSKIT_MRYX_CACHE is set, run tier-(A) without network."""
    import os
    import warnings

    cache_root = os.environ.get("MOLCRYSKIT_MRYX_CACHE")
    if not cache_root:
        pytest.skip("offline: set MOLCRYSKIT_MRYX_CACHE to a pre-downloaded fixture directory")

    from molcrys_kit.io import read_mryx_atoms

    filename = entry["filename"].replace(",", "_")
    path = Path(cache_root) / filename
    if not path.is_file():
        pytest.fail(f"cache missing fixture {filename}; run scripts/validate_mryx_public_roundtrip.py")

    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        pytest.fail(f"cache sha256 mismatch for {entry['id']}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        atoms = read_mryx_atoms(path)
    assert len(atoms) == entry["tier_a_expectations"]["n_atoms"]