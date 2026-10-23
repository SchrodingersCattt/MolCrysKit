"""Structure-facing topological slab API for the pymatgen addon."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Structure

from molcrys_kit.io import read_mol_crystal
from pymatgen.analysis.molcrys import TopologicalSlabGenerator, generate_topological_slab

REPO_ROOT = Path(__file__).resolve().parents[3]
CIF_PATH = REPO_ROOT / "tests" / "data" / "cif" / "Acetaminophen_HXACAN.cif"


@pytest.fixture(scope="module")
def acetaminophen_structure() -> Structure:
    crystal = read_mol_crystal(str(CIF_PATH), attach_chemistry=False)
    return crystal.to_pymatgen()


def test_generate_topological_slab_returns_structure(acetaminophen_structure):
    slab = generate_topological_slab(
        acetaminophen_structure,
        (0, 0, 1),
        layers=2,
        vacuum=12.0,
    )
    assert isinstance(slab, Structure)
    assert len(slab) > 0
    assert slab.lattice.c > acetaminophen_structure.lattice.c


def test_topological_slab_generator_class(acetaminophen_structure):
    gen = TopologicalSlabGenerator(acetaminophen_structure, miller_index=(1, 0, 0))
    slab = gen.get_slab(layers=2, vacuum=10.0)
    assert isinstance(slab, Structure)
    assert len(slab) >= len(acetaminophen_structure)


def test_get_slab_requires_miller_index(acetaminophen_structure):
    gen = TopologicalSlabGenerator(acetaminophen_structure)
    with pytest.raises(ValueError, match="miller_index"):
        gen.get_slab(layers=1)


def test_import_path_is_pymatgen_namespace():
    import pymatgen.analysis.molcrys as molcrys

    assert hasattr(molcrys, "generate_topological_slab")
    assert hasattr(molcrys, "TopologicalSlabGenerator")


def test_facade_accepts_partial_occupancy_structure(acetaminophen_structure):
    """Facade conversion path must not reject disordered Structure inputs."""
    from pymatgen.analysis.molcrys._convert import as_molecular_crystal

    species = list(acetaminophen_structure.species)
    species[0] = {str(species[0]): 0.7}
    disordered = Structure(
        acetaminophen_structure.lattice,
        species,
        acetaminophen_structure.frac_coords,
        coords_are_cartesian=False,
    )
    crystal = as_molecular_crystal(disordered)
    assert len(crystal.to_ase()) == len(disordered)
