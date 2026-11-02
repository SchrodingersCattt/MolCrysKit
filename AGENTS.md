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

---

## Cursor Cloud specific instructions

The Cloud Agent image needs the `python3.12-venv` apt package. Environment install creates `/opt/molcrys-venv` and runs `pip install -e ".[dev]"` from the repository root (Python 3.12, the same interpreter CI uses for coverage). `mck`, `pytest`, `ruff`, and `pre-commit` are linked onto the default `PATH`. `python` and `python3` are wrapper scripts that exec `/opt/molcrys-venv/bin/python3`; that path is what loads the virtualenv site-packages. Keep those wrappers. A direct symlink of the venv interpreter into `/usr/local/bin` resolves to system Python and drops the installed packages.

- Lint: `ruff==0.15.14` from the `dev` extra. Pre-commit runs `ruff-check` on staged files. CI does not lint the whole tree.
- Tests: `COVERAGE_CORE=sysmon pytest tests/unit --cov=molcrys_kit --cov-report=term-missing -n 4 --dist worksteal --timeout=180`. GitHub Actions uses `-n 8` on larger runners. This VM has 4 CPUs, so 4 workers matches the machine.
- Product check: build the README two-water crystal with `MolecularCrystal.from_ase`, then run `mck io info tests/data/cif/MAP.cif` and `mck analyze summary tests/data/cif/PETN_PERYTN10.cif`.
- Nothing has to stay running after install. The Cambridge Structural Database Python API is not part of this environment. Unit tests that mention it supply their own stand-in. Listing skills with `mck skills list` works; running `query-csd-structures` needs a licensed CSD install.
- `setuptools_scm` writes gitignored `molcrys_kit/_version.py` at install time from git tags. Checkouts that must report a real version need tag history.
