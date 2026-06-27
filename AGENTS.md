# MolCrysKit – Agent / AI-Coding Guidelines

Conventions and constraints for AI coding agents *modifying* this repo.
Design rationale and implementation details live in source-code docstrings;
read the relevant module before modifying it.

> **For AI agents that want to *use* molcrys_kit** (not modify it),
> see [API & Capabilities](docs/api.md) instead.

---

## Repository Layout

```
molcrys_kit/
  analysis/           Disorder resolution, packing shells, cluster invariants
  io/                 Structural parser
  operations/         Cluster carving, surface, hydrogens, perturbation, …
  structures/         MolecularCrystal, Molecule, CrystalCluster types
  constants/          Element radii, bond lengths, config thresholds
examples/             
scripts/              Best practice; one-off diagnosis scripts
tests/unit/           Pytest regression suite
```

---

## Testing Rules

- **Full suite must stay green.** Run `pytest tests/` before pushing.
- Assume test cases may involve unpublished research;
  sanitize committed data and avoid revealing system details in PRs.
- Every bug-fix or new structural motif must add a regression case in
  `tests/unit/test_disorder_regression.py`: a `CifCase` entry **and** a
  targeted `test_<material>_topology()` that asserts per-molecule formula
  counts.
- The corresponding CIF must be copied to `tests/data/cif/`.
- `xfail_reason` is allowed for known-broken cases but must be removed
  once fixed.

---

## Changing Code

- **Read the module docstring first.** Most modules document their design
  constraints (two-path architecture, solver modes, carve invariants, etc.)
  in the module or class docstring. Respect those constraints.
- **Prefer additive changes.** Extend existing code paths rather than
  rewriting core logic; the regression suite is brittle to rewrites.
  Separate refactor PRs with full test validation are fine when needed.
- **No magic numbers.** Put thresholds and constants in
  `molcrys_kit/constants/`; reuse existing ones before adding new ones.
- **Reuse before reinventing.** Check existing modules and utilities
  before writing new logic.
- **Do not hardcode version strings.** Version is derived from git tags
  via `setuptools_scm` → `molcrys_kit/_version.py` (gitignored).

---

## Documentation Maintenance

- When adding or changing a **public API** function or class, update
  `docs/api.md` — both the "Capability Map" and the "Module Index".
- When adding a new **user-facing feature**, add it to the Capability Map
  and optionally add a detailed tutorial in `docs/tutorials.md`.
- When adding or changing a CLI subcommand, update the README CLI overview
  if the command tree changes and add a smoke or regression test in
  `tests/unit/test_cli.py`.
- The Module Index should stay in sync with each sub-package `__init__.py`
  `__all__` list.
- **CI validates this automatically**: `tests/unit/test_docs_api_sync.py`
  compares every `__all__` symbol against `docs/api.md` Module Index sections.
  Run `pytest tests/unit/test_docs_api_sync.py -v` locally before pushing.

---

## Versioning & Release

Version is owned by **`setuptools_scm`** from the latest `vX.Y.Z` git tag.

Release steps (manual → automated):
1. Branch off `main`, bump `CITATION.cff` `version:` field, PR → merge.
2. `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`.
3. Tag push **automatically** triggers `release.yml`, which:
  - validates `CITATION.cff` against the tag,
  - runs unit tests,
  - builds and uploads PyPI distributions via trusted publishing,
  - builds and pushes the standard Docker image to GHCR,
  - creates the GitHub Release with generated notes.

CI checkouts use `fetch-depth: 0` so `setuptools_scm` sees full tag history;
do not lower it.
