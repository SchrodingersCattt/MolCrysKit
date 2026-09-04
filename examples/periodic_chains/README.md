# Periodic-chain fixtures

These JSON requests exercise `mck build chain`. They are geometry fixtures,
not chemistry templates: charge, valence, force fields, and relaxation are
intentionally absent. Each generated bundle contains `structure.cif` and
`structure.json`; the CIF is the default structure deliverable and the JSON
sidecar restores the geometry-native annotations needed for validation. The
set covers a synthetic non-zero-winding chain, a red phosphorus local chain,
a polyethylene-like chain, and an alpha-Se local chain.

Generated bundles:

- `bundles/synthetic_nonzero_winding/`
- `bundles/red_phosphorus_local_chain/`
- `bundles/polyethylene_like_chain/`
- `bundles/alpha_se_local_chain/`
