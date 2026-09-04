# Periodic-chain fixtures

These JSON requests exercise `mck build chain`. They are geometry fixtures,
not chemistry templates: charge, valence, force fields, and relaxation are
intentionally absent. Running a request produces a local bundle containing
`structure.cif` and `structure.json`; generated bundles are intentionally not
tracked in source control. The set is limited to geometry fixtures: a
synthetic non-zero-winding chain, a red phosphorus local-chain template, a
polyethylene-like template, and an alpha-Se local-chain template. These
templates are not validated material models or force-field-ready structures.
