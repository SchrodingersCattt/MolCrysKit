"""Resolve git-tracked Mercury ``.mryx`` fixtures (offline; no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MRYX_DATA = Path(__file__).resolve().parent
MANIFEST = MRYX_DATA / "public_manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bundled_fixture(entry: dict) -> Path:
    """Return the vendored fixture path or raise if missing or corrupt."""
    rel = entry["bundled_path"]
    path = MRYX_DATA / rel
    if not path.is_file():
        raise FileNotFoundError(f"bundled fixture missing: {rel}")
    digest = sha256_file(path)
    expected = entry["derived_sha256"]
    if digest != expected:
        raise ValueError(
            f"derived sha256 mismatch for {entry['id']}: got {digest}, expected {expected}"
        )
    return path