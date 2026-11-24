"""Deterministic derivation of anonymized tier-(A) Mercury ``.mryx`` fixtures."""

from __future__ import annotations

import gzip

# Contract pinned in public_manifest.json ``derivation`` (offline CI + maintainer tools).
GZIP_COMPRESSLEVEL = 9
GZIP_MTIME = 0


def stable_gzip_compress(payload: bytes) -> bytes:
    """Return reproducible gzip bytes for a fixed XML payload.

    CPython routes ``gzip.compress(..., mtime=0)`` through ``zlib.compress(...,
    wbits=31)``, so wall-clock timestamps do not enter the gzip header.
    """
    return gzip.compress(payload, compresslevel=GZIP_COMPRESSLEVEL, mtime=GZIP_MTIME)


def anonymize_upstream_bytes(data: bytes, old_label: str, new_label: str) -> bytes:
    """Replace a single structure display label and re-encode with stable gzip."""
    text = gzip.decompress(data).decode("utf-8")
    if old_label not in text:
        raise ValueError(f"upstream missing expected structure label {old_label!r}")
    if text.count(old_label) != 1:
        raise ValueError(f"expected exactly one occurrence of {old_label!r}")
    text = text.replace(old_label, new_label)
    return stable_gzip_compress(text.encode("utf-8"))