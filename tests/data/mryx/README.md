# Public Mercury `.mryx` validation fixtures

Tier-(A) validation uses **git-tracked, anonymized** Mercury exports. CI and
default tests read bundled files only; they do not download from Zenodo or other
hosts at runtime.

- `public_manifest.json` — fixture ids, derived/upstream hashes, tier-(A) expectations, provenance.
- `fixtures/` — gzip Mercury `.mryx` bytes (derived from CC-BY Zenodo deposits).
- `ATTRIBUTION.md` — required licence and source citation for vendored data.
- `fixture_paths.py` — offline path resolution and SHA-256 verification.

Maintainers may refresh the anonymized copy from an upstream file with matching
upstream SHA-256 using `scripts/maintain_mryx_anonymized_fixture.py` and
`derive_fixture.py` (`gzip.compress` with `mtime=0` for byte-stable output;
local/manual; not part of CI).

Synthetic XML in `tests/unit/io/test_io_mryx.py` remains contract coverage only;
it does not satisfy the real-format tier-(A) gate.