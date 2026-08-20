# CSD Environment

Read this before querying the Cambridge Structural Database.

## Requirements

- a licensed CSD installation;
- the vendor-provided CCDC Python API;
- access to the intended CSD data version;
- MolCrysKit only if exported structures will be handed to `mck`.

The CCDC API commonly lives in its own conda environment. Activate it using the local CSD installation instructions; do not hard-code a machine-specific activation path in a reusable workflow.

## Verify

```python
import sys
import ccdc
from ccdc.io import EntryReader

print("python=", sys.executable)
print("ccdc=", ccdc.__file__)
with EntryReader("CSD") as reader:
    print("entries=", len(reader))
```

Also record the CSD database version exposed by the installed release or CSD installation metadata. Verify that the same interpreter can run the bundled exporter.

## Environment boundary

If MolCrysKit is not installed in the CCDC environment, export CIF files there and analyze them in a separate MolCrysKit environment. The CIF plus retrieval manifest is the explicit boundary. Do not copy Python objects or rely on an implicit interpreter switch.
