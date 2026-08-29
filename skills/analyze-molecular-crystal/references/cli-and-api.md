# CLI and Python API

Read this before selecting an analysis interface.

## Default: CLI

Read the repository `docs/cli.md`, then probe the installed command:

```bash
mck analyze --help
mck analyze interactions --help
```

Use the CLI for BFDH ranking, aggregate interaction profiles, coordination-polyhedra enumeration, and structural sanity checks. Prefer `--json` for machine-readable results.

Use `mck analyze summary INPUT --json` for composition, cell, symmetry, Wyckoff sites, and disorder facts.

## Use the API for detail or missing CLI coverage

Read `docs/api.md` and source docstrings before using public imports. Use the API for custom criteria, detailed interaction records, CShM, stoichiometry, volume/boundary, formal charge, and local chemical environments.

Example: the CLI reports the aggregate interaction profile:

```bash
mck analyze interactions input.cif --json
```

Use public detectors for standalone CH-pi and H-H contact records:

```python
from molcrys_kit.analysis import find_ch_pi, find_h_h_contacts
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
ch_pi_records = find_ch_pi(crystal)
h_h_records = find_h_h_contacts(crystal)
```

Do not describe these standalone API detectors as families emitted by `mck analyze interactions`.

See [analysis API examples](./analysis-api.md) for API-only workflows and the parameters that must be reported.
