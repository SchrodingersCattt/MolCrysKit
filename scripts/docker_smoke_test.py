"""
docker_smoke_test.py
====================
Minimal end-to-end smoke test for the MolCrysKit Docker image.
Exercises the public API without requiring any internet access.

Run inside the container:
    python /opt/molcryskit/docker_smoke_test.py

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed.
"""

import sys
import traceback

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append(True)
    except Exception:
        print(f"  [{FAIL}] {name}")
        traceback.print_exc()
        results.append(False)


# ── 1. Basic imports ───────────────────────────────────────────────────────────
print("\n=== MolCrysKit Docker Smoke Test ===\n")
print("[ imports ]")


def test_import_top_level():
    import molcrys_kit  # noqa: F401
    from molcrys_kit import MolecularCrystal, CrystalMolecule, read_mol_crystal  # noqa: F401


def test_import_operations():
    from molcrys_kit.operations.hydrogen_completion import add_hydrogens  # noqa: F401
    from molcrys_kit.operations.surface import generate_topological_slab  # noqa: F401


def test_import_io():
    from molcrys_kit.io.cif import read_mol_crystal  # noqa: F401


check("import molcrys_kit top-level symbols", test_import_top_level)
check("import operations (hydrogen_completion, surface)", test_import_operations)
check("import io.cif.read_mol_crystal", test_import_io)

# ── 2. Dependency imports ──────────────────────────────────────────────────────
print("\n[ dependencies ]")


def test_ase():
    from ase import Atoms  # noqa: F401


def test_pymatgen():
    import pymatgen  # noqa: F401


def test_networkx():
    import networkx  # noqa: F401


def test_numpy():
    import numpy  # noqa: F401


def test_scipy():
    import scipy  # noqa: F401


check("ase", test_ase)
check("pymatgen", test_pymatgen)
check("networkx", test_networkx)
check("numpy", test_numpy)
check("scipy", test_scipy)

# ── 3. Core API: build a toy crystal from scratch ─────────────────────────────
print("\n[ core API – in-memory ]")


def test_toy_crystal_two_water():
    from ase import Atoms
    from molcrys_kit.structures.crystal import MolecularCrystal

    atoms = Atoms(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [1.0, 1.0, 1.0],
            [1.8, 1.0, 1.0],
            [0.7, 1.6, 1.0],  # molecule 1
            [5.0, 5.0, 5.0],
            [5.8, 5.0, 5.0],
            [4.7, 5.6, 5.0],  # molecule 2
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    assert len(crystal.molecules) == 2, (
        f"Expected 2 molecules, got {len(crystal.molecules)}"
    )
    lp = crystal.get_lattice_parameters()
    assert lp is not None, "get_lattice_parameters() returned None"


def test_molecule_formula():
    from ase import Atoms
    from molcrys_kit.structures.crystal import MolecularCrystal

    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[1.0, 1.0, 1.0], [1.8, 1.0, 1.0], [0.7, 1.6, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    crystal = MolecularCrystal.from_ase(atoms)
    mol = crystal.molecules[0]
    formula = mol.get_chemical_formula()
    assert formula is not None and len(formula) > 0, "Chemical formula is empty"


check("MolecularCrystal.from_ase() – two-water system", test_toy_crystal_two_water)
check("mol.get_chemical_formula()", test_molecule_formula)

# ── 4. CIF round-trip using bundled example ───────────────────────────────────
print("\n[ CIF I/O – bundled MAP.cif ]")


def test_cif_read_map():
    import os
    import warnings
    from molcrys_kit.io.cif import read_mol_crystal

    cif_path = "/workspace/notebook/example/H_lacking_structures/MAP.cif"
    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"{cif_path} not found – verify that notebook/ was COPY'd into the image"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal = read_mol_crystal(cif_path)
    assert crystal is not None, "read_mol_crystal returned None"
    assert len(crystal.molecules) > 0, (
        f"read_mol_crystal found 0 molecules in MAP.cif"
    )


check("read_mol_crystal('.../notebook/example/H_lacking_structures/MAP.cif')", test_cif_read_map)

# ── 5. Jupyter kernel availability ────────────────────────────────────────────
print("\n[ Jupyter ]")


def test_jupyter_installed():
    import subprocess
    result = subprocess.run(
        ["jupyter", "notebook", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"'jupyter notebook --version' failed: {result.stderr.strip()}"
    )
    version = result.stdout.strip()
    print(f"       (version: {version})", end="")


check("jupyter notebook is installed and executable", test_jupyter_installed)

# ── 6. Summary ────────────────────────────────────────────────────────────────
print()
total = len(results)
passed = sum(results)
failed = total - passed
print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)\n")
    sys.exit(1)
else:
    print("  – all OK\n")
    sys.exit(0)
