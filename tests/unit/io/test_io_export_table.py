"""The shared I/O table keeps Mercury readers without eager desktop imports."""

import json
import subprocess
import sys


def test_mryx_is_exported_lazily_beside_cif():
    script = """
import json
import sys
import molcrys_kit.io as io

payload = {
    "exports": ["read_mryx", "read_mryx_atoms", "read_mol_crystal"],
    "present": [name in io.__all__ for name in (
        "read_mryx", "read_mryx_atoms", "read_mol_crystal"
    )],
    "mryx_loaded_before_use": "molcrys_kit.io.mryx" in sys.modules,
}
reader = io.read_mryx
payload["reader_callable"] = callable(reader)
payload["mryx_loaded_after_use"] = "molcrys_kit.io.mryx" in sys.modules
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["present"] == [True, True, True]
    assert payload["mryx_loaded_before_use"] is False
    assert payload["reader_callable"] is True
    assert payload["mryx_loaded_after_use"] is True
