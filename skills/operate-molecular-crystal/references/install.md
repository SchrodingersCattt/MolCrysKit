# Install and Verify MolCrysKit

Read this before operating on structures in a new environment.

## Install

```bash
python -m pip install molcrys-kit
```

For a source checkout, run this from the repository root:

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
mck operate --help
python - <<'PY'
import importlib.metadata
import sys
import molcrys_kit

print("python=", sys.executable)
print("distribution=", importlib.metadata.version("molcrys-kit"))
print("module=", molcrys_kit.__file__)
PY
```

Record the interpreter, distribution version, and module path. Probe the exact operation before using it:

```bash
mck operate disorder --help
mck operate slab --help
mck operate cluster --help
```

The installed CLI is authoritative when it differs from repository documentation. Molecule translation, rotation, and replacement are Python-API-only; read `docs/api.md` and the installed source docstrings for them.
