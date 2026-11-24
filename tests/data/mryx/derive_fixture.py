"""Deterministic derivation of anonymized tier-(A) Mercury ``.mryx`` fixtures."""

from __future__ import annotations

import struct
import zlib

# Contract pinned in public_manifest.json ``derivation`` (offline CI + maintainer tools).
GZIP_COMPRESSLEVEL = 9
GZIP_MTIME = 0
GZIP_OS_UNKNOWN = 255
GZIP_XFL_BEST_COMPRESSION = 0x02


def stable_gzip_compress(payload: bytes) -> bytes:
    """Return reproducible gzip bytes with a fixed header on CPython 3.10–3.12+.

    stdlib ``gzip.compress(..., mtime=0)`` is **not** byte-identical across
    CPython minor releases: recent 3.11/3.12 builds set header byte 9 (OS) to 3
    (Unix), while 3.10 ``GzipFile`` uses 255 (unknown). Payload deflate and
    trailer still match; only the header OS octet differs. This helper pins
    mtime, XFL, and OS and uses raw deflate (``zlib.compress(..., wbits=-15)``).
    """
    level = GZIP_COMPRESSLEVEL
    xfl = GZIP_XFL_BEST_COMPRESSION if level == 9 else 0x00
    header = b"\x1f\x8b\x08\x00" + struct.pack("<I", GZIP_MTIME) + bytes([xfl, GZIP_OS_UNKNOWN])
    body = zlib.compress(payload, level=level, wbits=-15)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    isize = len(payload) & 0xFFFFFFFF
    trailer = struct.pack("<II", crc, isize)
    return header + body + trailer


def anonymize_upstream_bytes(data: bytes, old_label: str, new_label: str) -> bytes:
    """Replace a single structure display label and re-encode with stable gzip."""
    import gzip

    text = gzip.decompress(data).decode("utf-8")
    if old_label not in text:
        raise ValueError(f"upstream missing expected structure label {old_label!r}")
    if text.count(old_label) != 1:
        raise ValueError(f"expected exactly one occurrence of {old_label!r}")
    text = text.replace(old_label, new_label)
    return stable_gzip_compress(text.encode("utf-8"))