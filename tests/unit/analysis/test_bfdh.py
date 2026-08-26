import math

import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup

from molcrys_kit.analysis.bfdh import (
    BFDHFacetInfo,
    _d_hkl,
    enumerate_bfdh_facets,
    enumerate_low_index_millers,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
)


def _explicit_symmetry(symbol: str) -> CrystalSymmetry:
    group = SpaceGroup(symbol)
    return CrystalSymmetry(
        operations=tuple(
            FractionalAffineOperation(
                operation.rotation_matrix,
                operation.translation_vector,
            )
            for operation in group.symmetry_ops
        ),
        space_group_number=int(group.int_number),
        space_group_symbol=str(group.symbol),
        source="test_parent",
    )


def test_low_index_enumerator_deterministic_and_sign_canonical():
    hkls = enumerate_low_index_millers(max_index=1)
    assert (0, 0, 0) not in hkls
    assert (1, 0, 0) in hkls
    assert (-1, 0, 0) not in hkls
    assert hkls == enumerate_low_index_millers(max_index=1)


def test_low_index_enumerator_keeps_higher_order_planes():
    hkls = enumerate_low_index_millers(max_index=2)
    assert (1, 0, 0) in hkls
    assert (2, 0, 0) in hkls


def test_low_index_enumerator_rejects_invalid_max_index():
    with pytest.raises(ValueError):
        enumerate_low_index_millers(max_index=0)


def test_bfdh_cubic_ranking_explicit_millers():
    lattice = Lattice.cubic(4.0)
    facets = enumerate_bfdh_facets(
        lattice,
        miller_indices=[(1, 1, 1), (1, 1, 0), (1, 0, 0)],
    )
    assert [facet.miller_index for facet in facets] == [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    assert math.isclose(facets[0].d_hkl, 4.0)
    assert math.isclose(facets[1].d_hkl, lattice.d_hkl((1, 1, 0)))
    assert facets[0].relative_morphological_importance == 1.0
    assert facets[0].relative_growth_rate < facets[-1].relative_growth_rate


def test_bfdh_top_n_and_dataclass_dict():
    lattice = Lattice.cubic(4.0)
    [facet] = enumerate_bfdh_facets(
        lattice,
        miller_indices=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        top_n=1,
    )
    assert isinstance(facet, BFDHFacetInfo)
    assert facet.rank == 0
    as_dict = facet.as_dict()
    assert as_dict["miller_index"] == [1, 0, 0]
    assert as_dict["rank"] == 0


def test_bfdh_rejects_invalid_arguments():
    lattice = Lattice.cubic(4.0)
    with pytest.raises(ValueError):
        enumerate_bfdh_facets(lattice, max_index=0)
    with pytest.raises(ValueError):
        enumerate_bfdh_facets(lattice, top_n=0)


def test_bfdh_explicit_millers_ignore_max_index():
    facets = enumerate_bfdh_facets(
        Lattice.cubic(5.0),
        max_index=0,
        miller_indices=[(1, 0, 0)],
        extinction_filter=False,
    )

    assert [facet.miller_index for facet in facets] == [(1, 0, 0)]


def test_d_hkl_rejects_zero_miller_index():
    with pytest.raises(ValueError, match="zero length"):
        _d_hkl(Lattice.cubic(4.0), (0, 0, 0))


def test_bfdh_uses_pymatgen_symmetry_backend_for_structure():
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na"], [[0, 0, 0]])
    facets = enumerate_bfdh_facets(structure, max_index=2, top_n=3, extinction_filter=False)
    assert facets
    assert all(facet.source == "pymatgen" for facet in facets)
    assert facets[0].miller_index == (1, 0, 0)


def test_bfdh_include_equivalents_for_cubic_family():
    lattice = Lattice.cubic(4.0)
    [facet] = enumerate_bfdh_facets(
        lattice,
        miller_indices=[(1, 0, 0)],
        include_equivalents=True,
    )
    assert facet.multiplicity == 3
    assert set(facet.equivalent_millers) == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_bfdh_internal_fallback_can_be_forced():
    lattice = Lattice.cubic(4.0)
    facets = enumerate_bfdh_facets(
        lattice,
        max_index=1,
        use_pymatgen_symmetry=False,
        top_n=3,
    )
    assert facets
    assert all(facet.source == "internal" for facet in facets)
    assert facets[0].d_hkl >= facets[-1].d_hkl


def test_bfdh_accepts_molecular_crystal_input():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 5.0, 6.0], pbc=True)
    crystal = MolecularCrystal.from_ase(atoms)
    facets = enumerate_bfdh_facets(
        crystal,
        miller_indices=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    assert facets[0].miller_index == (0, 0, 1)
    assert math.isclose(facets[0].d_hkl, 6.0)


def test_bfdh_default_extinction_filter_for_p21c():
    lattice = Lattice.monoclinic(19.7404, 14.3294, 21.2948, 90.075)
    x, y, z = 0.11, 0.22, 0.33
    structure = Structure(
        lattice,
        ["C"] * 4,
        [
            [x, y, z],
            [-x, -y, -z],
            [-x + 0.5, y + 0.5, -z + 0.5],
            [x + 0.5, -y + 0.5, z + 0.5],
        ],
        to_unit_cell=True,
    )
    facets = enumerate_bfdh_facets(structure, max_index=2, top_n=6)
    assert [facet.miller_index for facet in facets] == [
        (1, 0, -1),
        (1, 0, 1),
        (0, 1, -1),
        (1, 1, 0),
        (0, 0, 2),
        (1, 1, -1),
    ]
    assert math.isclose(facets[0].d_hkl, 14.48637910, rel_tol=1e-6)


def test_bfdh_default_extinction_filter_for_pnma():
    lattice = Lattice.orthorhombic(10.26733, 14.7004, 20.9914)
    structure = Structure.from_spacegroup("Pnma", lattice, ["C"], [[0.11, 0.25, 0.33]])
    facets = enumerate_bfdh_facets(structure, max_index=2, top_n=6)
    assert [facet.miller_index for facet in facets] == [
        (0, 1, -1),
        (0, 0, 2),
        (1, 0, -1),
        (1, -1, -1),
        (0, 2, 0),
        (1, 0, -2),
    ]
    assert math.isclose(facets[0].d_hkl, 12.04130654, rel_tol=1e-6)


def test_bfdh_explicit_parent_symmetry_matches_structure_extinctions():
    lattice = Lattice.orthorhombic(10.26733, 14.7004, 20.9914)
    symmetry = _explicit_symmetry("Pnma")
    facets = enumerate_bfdh_facets(
        lattice,
        max_index=2,
        top_n=6,
        symmetry=symmetry,
    )

    assert [facet.miller_index for facet in facets] == [
        (0, 1, -1),
        (0, 0, 2),
        (1, 0, -1),
        (1, -1, -1),
        (0, 2, 0),
        (1, 0, -2),
    ]
    assert facets[0].equivalent_millers == ((0, 1, -1), (0, 1, 1))


def test_bfdh_explicit_p21c_symmetry_expands_and_filters_facets():
    facets = enumerate_bfdh_facets(
        Lattice.monoclinic(19.7404, 14.3294, 21.2948, 90.075),
        miller_indices=[(1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
        symmetry=_explicit_symmetry("P2_1/c"),
    )

    assert [facet.miller_index for facet in facets] == [(1, 0, 0), (0, 1, -1)]
    assert facets[1].equivalent_millers == ((0, 1, -1), (0, 1, 1))


def test_bfdh_explicit_symmetry_can_skip_extinction_filter():
    lattice = Lattice.orthorhombic(10.26733, 14.7004, 20.9914)
    symmetry = _explicit_symmetry("Pnma")
    filtered = enumerate_bfdh_facets(
        lattice,
        miller_indices=[(1, 0, 0)],
        symmetry=symmetry,
    )
    unfiltered = enumerate_bfdh_facets(
        lattice,
        miller_indices=[(1, 0, 0)],
        symmetry=symmetry,
        extinction_filter=False,
    )

    assert filtered == []
    assert [facet.miller_index for facet in unfiltered] == [(1, 0, 0)]
