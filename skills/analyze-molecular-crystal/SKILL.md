---
name: analyze-molecular-crystal
description: "Use before answering crystal-structure chemistry questions about components, disorder, readiness, interactions, coordination, facets, charge, or pores."
---

# Analyze Molecular Crystals

Answer the stated structural-chemistry question with MolCrysKit. Do not turn a
focused question into a full audit.

## Direct path

1. Start with `mck analyze summary INPUT --json`.
2. Read exactly one matching page:
   - readiness, components, disorder, interactions, facets, or coordination:
     [CLI analyses](./references/cli-analyses.md);
   - stoichiometry, formal charge, accessible volume, local environments, or
     detailed CShM: [advanced analyses](./references/advanced-analyses.md).
3. Run the listed command or public API directly. Use `--help` once only if
   that exact command rejects an option.
4. Apply the corresponding limits in
   [reporting](./references/reporting.md), retain machine-readable output, and
   answer the original question.

Read [runtime recovery](./references/runtime.md) only when `mck` is missing or
reports a version/capability error.

## When extra checks are justified

- Run `mck io molecules INPUT --json` when component identity/count matters.
- Run `mck analyze sanity-check INPUT --json` for simulation readiness or when
  missing H, clashes, or bond perception can invalidate the requested analysis.
- Compare ordered replicas when disorder changes the requested observable.
- Visualize only when spatial arrangement is part of the evidence; use the
  atomistic visualization skill, not custom plotting.

## Boundaries

- Geometry-based interaction scores are not energies.
- BFDH ranks morphology candidates, not surface stability.
- Coordination number alone does not define polyhedron shape.
- Charge guesses, protonation, cutoffs, probe radii, and sampling parameters are
  assumptions and must be reported.
- Never silently replace unresolved disorder with one favorable ordered model.
