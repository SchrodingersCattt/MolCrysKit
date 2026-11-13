"""CIF results stay the same when pymatgen, ASE, SciPy, and spglib are absent."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    ROOT / "tests" / "data" / "test_full_coords.cif",
    ROOT / "tests" / "data" / "cif" / "synthetic_two_assemblies.cif",
    ROOT / "tests" / "data" / "cif" / "DAP-4.cif",
]


def _snapshot(path: Path):
    from molcrys_kit.io.cif import read_mol_crystal

    crystal = read_mol_crystal(str(path), attach_chemistry=False)
    rows = []
    for molecule in crystal.molecules:
        occupancy = molecule.arrays.get("occupancy")
        groups = molecule.arrays.get("disorder_group")
        symbols = molecule.get_chemical_symbols()
        positions = molecule.get_positions()
        cell = np.asarray(molecule.get_cell(), dtype=float)
        fractional = positions @ np.linalg.inv(cell)
        fractional = fractional - np.floor(fractional + 1e-8)
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "occ": None if occupancy is None else round(float(occupancy[index]), 5),
                    "group": None if groups is None else int(groups[index]),
                    "frac": [round(float(value), 3) for value in fractional[index]],
                }
            )
    rows.sort(key=lambda row: (row["symbol"], row["occ"], row["group"], tuple(round(v, 2) for v in row["frac"])))
    return rows


@pytest.mark.parametrize("path", CASES, ids=lambda path: path.name)
def test_cif_snapshot_matches_without_desktop_dependencies(path: Path):
    expected = _snapshot(path)
    script = textwrap.dedent(
        f"""
        import json
        import sys

        class _Blocker:
            _prefixes = ("pymatgen", "ase", "scipy", "spglib")

            def find_spec(self, name, path, target=None):
                if name in ("numpy", "networkx") or name.startswith(("numpy.", "networkx.")):
                    return None
                if name in self._prefixes or any(name.startswith(prefix + ".") for prefix in self._prefixes):
                    raise ImportError(name)
                return None

        sys.meta_path.insert(0, _Blocker())
        from molcrys_kit.io.cif import read_mol_crystal

        crystal = read_mol_crystal({str(path)!r}, attach_chemistry=False)
        rows = []
        for molecule in crystal.molecules:
            occupancy = molecule.arrays.get("occupancy")
            groups = molecule.arrays.get("disorder_group")
            symbols = molecule.get_chemical_symbols()
            positions = molecule.get_positions()
            cell = __import__("numpy").asarray(molecule.get_cell(), dtype=float)
            fractional = positions @ __import__("numpy").linalg.inv(cell)
            fractional = fractional - __import__("numpy").floor(fractional + 1e-8)
            for index, symbol in enumerate(symbols):
                rows.append({{
                    "symbol": symbol,
                    "occ": None if occupancy is None else round(float(occupancy[index]), 5),
                    "group": None if groups is None else int(groups[index]),
                    "frac": [round(float(value), 3) for value in fractional[index]],
                }})
        rows.sort(key=lambda row: (row["symbol"], row["occ"], row["group"], tuple(round(v, 2) for v in row["frac"])))
        print(json.dumps(rows))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    actual = json.loads(completed.stdout)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got["symbol"] == want["symbol"]
        assert got["occ"] == want["occ"]
        assert got["group"] == want["group"]
        assert got["frac"] == pytest.approx(want["frac"], abs=0.002)
