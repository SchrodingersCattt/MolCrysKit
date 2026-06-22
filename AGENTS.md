# MolCrysKit – Agent / AI-Coding Guidelines

Conventions and constraints for AI coding agents working in this repo.
Design rationale and implementation details live in source-code docstrings;
read the relevant module before modifying it.

---

## Repository Layout

```
molcrys_kit/
  analysis/           Disorder resolution, packing shells, cluster invariants
  io/                 CIF parser
  operations/         Cluster carving, surface, hydrogens, perturbation, …
  structures/         MolecularCrystal, Molecule, CrystalCluster types
  constants/          Element radii, bond lengths, config thresholds
examples/             CIF files (also used by regression tests)
scripts/              One-off diagnostics / repro scripts
tests/unit/           Pytest regression suite
```

---

## Testing Rules

- **Full suite must stay green.** Run `pytest tests/` before pushing.
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
- **Prefer additive changes.** When fixing a new edge case, add a new
  processing step or extend an existing code path rather than rewriting
  core logic. The regression suite is large and brittle to rewrites.
- **Do not hardcode version strings.** Version is derived from git tags
  via `setuptools_scm` → `molcrys_kit/_version.py` (gitignored).

---

## Versioning & Release

Version is owned by **`setuptools_scm`** from the latest `vX.Y.Z` git tag.

Release steps:
1. Branch off `main`, bump `CITATION.cff` `version:` field.
2. PR → merge → `git tag -a vX.Y.Z <merge-commit>` → `git push origin vX.Y.Z`.
3. Tag push triggers `.github/workflows/publish-pypi.yml` (PyPI) and
   `.github/workflows/publish-ghcr.yml` (Docker).
4. `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file ...`.

CI checkouts use `fetch-depth: 0` so `setuptools_scm` sees full tag history;
do not lower it.
