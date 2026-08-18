# Install and Verify MolCrysKit

Read this before analyzing structures in a new environment.

## Install

```bash
python -m pip install molcrys-kit
```

For a source checkout:

```bash
python -m pip install -e .
```

Use the same interpreter for installation and execution.

## Verify

Read the repository `docs/cli.md` first, then verify that the active installation matches it:

```bash
mck --version
mck --help
mck io --help
mck analyze --help
python - <<'PY'
import importlib.metadata
import sys
import molcrys_kit

print("python=", sys.executable)
print("distribution=", importlib.metadata.version("molcrys-kit"))
print("module=", molcrys_kit.__file__)
PY
```

Record the interpreter, version, and module path. Probe the intended analysis:

```bash
mck analyze bfdh --help
mck analyze interactions --help
mck analyze polyhedra --help
mck analyze sanity-check --help
```

The installed CLI is authoritative when it differs from repository documentation. Read `docs/api.md` and source docstrings before using API-only stoichiometry, volume, accessible-boundary, formal-charge, detailed CShM, or chemical-environment analysis.
