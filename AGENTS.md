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

## Skills and Docs as Code

- Each directory under `skills/` is independently installable. Keep its
  Markdown links and bundled assets inside that skill directory.
- `docs/cli.md` is the canonical repository CLI reference. Skill workflows
  should prefer `mck` commands and probe the installed command with `--help`.
- Python examples must import names exported through the target module's
  `__all__`; update `docs/api.md` when changing public exports.
- Run `pytest tests/unit/contracts/test_docs_api_sync.py tests/unit/contracts/test_skill_docs_sync.py`
  after changing CLI commands, public APIs, skills, or bundled skill scripts.

---

## Testing Rules

- **Full suite must stay green.** Run `pytest tests/` before pushing.
- Assume test cases may involve unpublished research;
  sanitize committed data and avoid revealing system details in PRs.
- Every bug-fix or new structural motif must add a regression case in
  `tests/unit/analysis/test_disorder_regression.py`: a `CifCase` entry **and** a
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

Stable documentation ships in the same pull request as the change it describes.
The full rule, out-of-scope list, and `Docs: N/A` declaration are in
[CONTRIBUTING.md](CONTRIBUTING.md). Pull request expectations are also stated
in [docs/architecture.md](docs/architecture.md).

- **In-scope changes** (public Python API, CLI, CIF and other I/O contracts,
  bond / chemistry / symmetry semantics, architecture, or how-to-use flows)
  must update `docs/api.md`, `docs/cli.md`, `docs/architecture.md`,
  `docs/bonds.md`, `docs/tutorials.md`, or the focused design note that owns
  the change. Missing docs on an in-scope pull request is review **P1**.
  Update this guide when a standing rule changes.
- **Out of scope** (pure refactor with identical public behavior, test-only
  or CI-only change, private helper rename, docs-only pull request): put
  `Docs: N/A` and a one-line reason in the pull request body. Ephemeral
  change logs are not a substitute for those stable pages. Prefer API docs,
  golden files, and contract tests as the source of truth.
- A warn-only CI job may annotate a pull request that edits `molcrys_kit/`
  without touching `docs/` and without `Docs: N/A`. That job does not fail
  the build and does not replace the P1 review.
- When adding or changing a **public API** function or class, update
  `docs/api.md` — both the "Capability Map" and the "Module Index".
- When adding a new **user-facing feature**, add it to the Capability Map
  and optionally add a detailed tutorial in `docs/tutorials.md`.
- When adding or changing a CLI subcommand, update `docs/cli.md` and the
  README CLI overview if the command tree changes, and add a smoke or
  regression test in `tests/unit/cli/test_cli.py`.
- The Module Index should stay in sync with each sub-package `__init__.py`
  `__all__` list.
- **CI validates this automatically**: `tests/unit/contracts/test_docs_api_sync.py`
  compares every `__all__` symbol against `docs/api.md` Module Index sections
  and verifies all CLI subcommands are listed in `docs/cli.md`.
  Run `pytest tests/unit/contracts/test_docs_api_sync.py -v` locally before pushing.

---

## Versioning & Release

Version is owned by **`setuptools_scm`** from the latest release tag. Stable
releases use `vX.Y.Z`; pre-releases use PEP 440 tags such as `vX.Y.Zb1`.

Release steps (manual → automated):
1. Branch off `main`, bump `CITATION.cff` `version:` field, PR → merge.
2. Create a stable tag (`vX.Y.Z`) or beta tag (`vX.Y.Zb1`) and push it:
   `git tag -a vX.Y.Zb1 -m "Release vX.Y.Zb1" && git push origin vX.Y.Zb1`.
3. Tag push **automatically** triggers `release.yml`, which:
  - validates `CITATION.cff` against the tag,
  - runs unit tests,
  - builds and uploads PyPI distributions via trusted publishing,
  - builds and pushes the standard Docker image to GHCR,
  - creates the GitHub Release with generated notes; PEP 440 pre-release tags
    are marked as GitHub pre-releases.

CI checkouts use `fetch-depth: 0` so `setuptools_scm` sees full tag history;
do not lower it.
