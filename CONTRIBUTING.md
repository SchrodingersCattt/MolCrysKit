# Contributing

## Documentation ships with the change

A pull request that changes user-facing behavior, the public API, the CLI,
CIF or other I/O contracts, or crystallographic invariants must update the
stable documentation in the **same** pull request. A docs-only follow-up is
the exception, not the default.

The pull request description is the process gate. Use the template sections
**Summary**, **How to verify**, and **Documentation** so a reviewer can see
what changed and how to verify it.

### In scope

Update a stable docs entry point in the same pull request when the change
touches any of the following:

- Public Python API surface (`molcrys_kit` exports, signatures, defaults)
- CLI flags or subcommands (`docs/cli.md`)
- CIF, forward, disorder, or companion contracts
- Bond, chemistry, or symmetry semantics
- Architecture or how-to-use flows

Primary homes:

- `docs/api.md`
- `docs/cli.md`
- `docs/architecture.md`
- `docs/bonds.md`
- `docs/tutorials.md`
- a focused design note when that note is the stable home for the change

Update `AGENTS.md` when the pull request adds or changes a standing rule for
people and agents who modify this repository.

### Out of scope

These changes do not need a stable-doc edit. Write the following declaration
on its own line in the pull request body, outside HTML comments and fenced
code blocks, and add a one-line reason:

```
Docs: N/A — one-line reason
```

Out of scope:

- Pure refactors with identical public behavior
- Test-only or CI-only changes
- Private helper renames
- Docs-only pull requests

### Review bar

Missing documentation on an in-scope pull request is review **P1**. The
warn-only check below does not replace that review.

### What counts as truth

Prefer machine-checkable sources of truth where they exist: API docs, golden
files, and contract tests. Prose covers what those checks cannot.

Do not treat an ephemeral change log, release note, or pull request comment
as a substitute for the stable handbook, API, CLI, or contract pages.

### Pull request template

`.github/pull_request_template.md` asks for:

- docs paths touched, or the `Docs: N/A` declaration and a one-line reason
- whether the public API, CLI, CIF/I-O contracts, or crystallographic invariants changed (yes/no)
- whether `AGENTS.md` was updated if a standing rule changed
- a project-agnostic checkbox confirming the pull request has no internal project names, private paths, or unpublished data or CIF files

The example declaration in the template sits inside a code fence on purpose.
Copy it out of the fence when you use it.

### Warn-only check

On pull requests, `Docs ship with code (warn)` in
`.github/workflows/unit-tests.yml` warns when files under `molcrys_kit/`
change, no file under `docs/` changes, and the body has no `Docs: N/A`
declaration outside HTML comments and fenced code blocks. The job emits a
warning annotation and does not fail the build. Silence from that job is not
approval: reviewers still apply the P1 bar when an in-scope change updates
the wrong page, or when a contract changes without a `molcrys_kit/` diff.
