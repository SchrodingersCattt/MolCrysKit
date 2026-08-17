# Install and Verify MolCrysKit

Read this before using either MolCrysKit skill in a new environment.

## Install

Use the Python environment that will execute the work:

```bash
python -m pip install molcrys-kit
```

For development from a MolCrysKit checkout, install the repository root in editable mode instead:

```bash
python -m pip install -e .
```

Do not silently switch interpreters between installation and execution.

## Verify the active installation

```bash
mck --version
mck --help
mck io --help
mck operate --help
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

Record the interpreter, distribution version, module path, and relevant live help. The installed CLI is authoritative: do not invent commands or options from a different release.

## Capability probe

Before committing to a workflow, inspect the exact subcommand:

```bash
mck operate disorder --help
mck operate slab --help
mck analyze sanity-check --help
```

Some capabilities are Python-API-only. In particular, molecule manipulation, topology-aware stoichiometry, volume and accessible-boundary calculations, formal-charge assignment, and detailed chemical-environment analysis do not have complete CLI front ends.
