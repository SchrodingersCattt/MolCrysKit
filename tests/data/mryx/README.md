# Public Mercury `.mryx` validation fixtures

These entries support **independent** Mercury format checks for PR #228. They are
not checked into git.

`public_manifest.json` lists download URLs, hashes, licenses, and provenance.
`scripts/validate_mryx_public_roundtrip.py` and
`tests/unit/io/test_mryx_public_roundtrip.py` fetch fixtures on demand (Zenodo
is CC-BY-4.0 and allows redistribution when attributed).

Synthetic XML in `tests/unit/io/test_io_mryx.py` remains contract coverage only;
it does not satisfy the real-format P1 gate.