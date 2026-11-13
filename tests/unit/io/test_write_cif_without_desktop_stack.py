"""P1 CIF export and re-import work with SciPy, ASE, pymatgen and spglib absent."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
DAP4 = ROOT / "tests" / "data" / "cif" / "DAP-4.cif"

_SCRIPT = textwrap.dedent(
    """
    import json, sys, warnings
    warnings.simplefilter("ignore")

    class _Blocker:
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("scipy", "ase", "pymatgen", "spglib"):
                raise ImportError(name)

    sys.meta_path.insert(0, _Blocker())

    import numpy as np
    from molcrys_kit.io import read_mol_crystal, write_cif

    def snapshot(crystal):
        symbols, positions, occupancies = [], [], []
        for molecule in crystal.molecules:
            cart = molecule.get_positions()
            occ = molecule.arrays.get("occupancy")
            for i, symbol in enumerate(molecule.get_chemical_symbols()):
                symbols.append(symbol)
                positions.append(list(map(float, cart[i])))
                occupancies.append(1.0 if occ is None else float(occ[i]))
        return {{
            "lattice": np.asarray(crystal.lattice).tolist(),
            "symbols": symbols,
            "positions": positions,
            "occupancies": occupancies,
        }}

    original = read_mol_crystal({path!r})
    before = snapshot(original)
    out = {out!r}
    write_cif(original, out, representation="p1")
    reread = read_mol_crystal(out)
    after = snapshot(reread)
    blocked = [m for m in ("ase", "scipy", "pymatgen", "spglib") if m in sys.modules]
    print("RESULT" + json.dumps({{"before": before, "after": after, "loaded": blocked}}))
    """
)


def _sort_key(snap):
    order = sorted(
        range(len(snap["symbols"])),
        key=lambda i: (snap["symbols"][i], *np.round(snap["positions"][i], 3)),
    )
    return order


def test_p1_cif_roundtrip_without_desktop_stack(tmp_path):
    out = tmp_path / "roundtrip_p1.cif"
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(path=str(DAP4), out=str(out))],
        capture_output=True,
        text=True,
        timeout=170,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    line = next(x for x in completed.stdout.splitlines() if x.startswith("RESULT"))
    payload = json.loads(line[len("RESULT") :])
    assert payload["loaded"] == []
    assert out.exists()

    before, after = payload["before"], payload["after"]
    np.testing.assert_allclose(after["lattice"], before["lattice"], atol=1e-4)
    assert sorted(after["symbols"]) == sorted(before["symbols"])
    assert len(after["symbols"]) == len(before["symbols"])
    # The export re-orders atoms, so compare in a canonical order.
    ib, ia = _sort_key(before), _sort_key(after)
    assert [after["symbols"][i] for i in ia] == [before["symbols"][i] for i in ib]
    np.testing.assert_allclose(
        np.asarray(after["positions"])[ia],
        np.asarray(before["positions"])[ib],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(after["occupancies"])[ia],
        np.asarray(before["occupancies"])[ib],
        atol=1e-4,
    )
    # DAP-4 is disordered: the check is only meaningful with partial occupancies.
    assert min(before["occupancies"]) < 1.0


def test_poscar_and_xyz_writers_still_work_with_ase(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms

    import molcrys_kit as mck
    from molcrys_kit.io import write_poscar, write_trajectory

    atoms = Atoms(
        "OH2",
        positions=[[1, 1, 1], [1.8, 1, 1], [0.7, 1.6, 1]],
        cell=[8, 8, 8],
        pbc=True,
    )
    crystal = mck.MolecularCrystal.from_ase(atoms)
    text = write_poscar(crystal)
    assert "O" in text and "Direct" in text
    path = write_trajectory([crystal], str(tmp_path / "t.xyz"), format="xyz")
    assert Path(path).exists()
