"""Tests for topology-preserving implicit-shape nanoclusters."""

from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure

from molcrys_kit.operations import (
    ImplicitShape,
    NanoClusterCarver,
    NanoShape,
    carve_nanocluster,
)
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def _single_atom_crystal(
    lattice: np.ndarray | None = None,
    *,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> MolecularCrystal:
    cell = np.asarray(lattice if lattice is not None else np.eye(3), dtype=float)
    position = 0.5 * np.sum(cell, axis=0)
    atom = Atoms("He", positions=[position], cell=cell, pbc=pbc)
    atom.set_array("site_tag", np.array([7], dtype=int))
    return MolecularCrystal(cell, [atom], pbc=pbc)


def _adn_like_crystal() -> MolecularCrystal:
    """One 5-atom cation plus one 7-atom anion in a 10 A cell."""
    cell = 10.0 * np.eye(3)
    cation = Atoms(
        ["N", "H", "H", "H", "H"],
        positions=[
            [2.0, 2.0, 2.0],
            [2.9, 2.0, 2.0],
            [1.1, 2.0, 2.0],
            [2.0, 2.9, 2.0],
            [2.0, 2.0, 2.9],
        ],
        cell=cell,
        pbc=True,
    )
    anion = Atoms(
        ["N", "N", "N", "O", "O", "O", "O"],
        positions=[
            [6.0, 6.0, 6.0],
            [7.1, 6.0, 6.0],
            [4.9, 6.0, 6.0],
            [7.8, 6.6, 6.0],
            [7.8, 5.4, 6.0],
            [4.2, 6.6, 6.0],
            [4.2, 5.4, 6.0],
        ],
        cell=cell,
        pbc=True,
    )
    cation.set_array("site_tag", np.arange(5, dtype=int))
    anion.set_array("site_tag", np.arange(5, 12, dtype=int))
    cation.info["component_label"] = "cation"
    anion.info["component_label"] = "anion"
    for molecule in (cation, anion):
        positions = molecule.get_positions()
        molecule.set_array("frac_x", positions[:, 0] / 10.0)
        molecule.set_array("frac_y", positions[:, 1] / 10.0)
        molecule.set_array("frac_z", positions[:, 2] / 10.0)
        molecule.set_array("image_shift", np.zeros((len(molecule), 3), dtype=int))
    return MolecularCrystal(
        cell,
        [cation, anion],
        extra_arrays={"source_rank": np.arange(12, dtype=int)},
    )


def _graph_signature(molecule: CrystalMolecule) -> tuple[str, int, tuple[int, ...]]:
    graph = molecule.graph
    return (
        molecule.get_chemical_formula(),
        graph.number_of_edges(),
        tuple(sorted(dict(graph.degree()).values())),
    )


def test_shape_presets_define_expected_fields() -> None:
    sphere = NanoShape.sphere(2.0)
    box = NanoShape.box((2.0, 4.0, 6.0))
    ellipsoid = NanoShape.ellipsoid((1.0, 2.0, 3.0))
    cylinder = NanoShape.cylinder(2.0, 6.0, axis="x")

    points = np.array([0.0, 1.0, 3.0])
    zeros = np.zeros(3)
    assert np.allclose(sphere.field(points, zeros, zeros), [-1.0, -0.75, 1.25])
    assert np.allclose(box.bounds, [[-1, 1], [-2, 2], [-3, 3]])
    assert ellipsoid.field(np.array([1.0]), np.array([0.0]), np.array([0.0]))[0] == 0
    assert np.allclose(cylinder.bounds, [[-3, 3], [-2, 2], [-2, 2]])


def test_bfdh_shape_cubic_100_is_scaled_cube() -> None:
    shape = ImplicitShape.bfdh(
        Lattice.cubic(4.0),
        60.0,
        miller_indices=[(1, 0, 0)],
        extinction_filter=False,
    )

    assert shape.name == "bfdh"
    assert np.max(np.ptp(shape.bounds, axis=1)) == 60.0
    assert shape.field(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0] == -1.0
    assert np.isclose(
        shape.field(np.array([30.0]), np.array([0.0]), np.array([0.0]))[0],
        0.0,
    )
    assert len(shape.parameters["planes"]) == 6
    assert len(shape.parameters["vertices_A"]) == 8
    json.dumps(shape.parameters)


def test_bfdh_shape_without_explicit_symmetry_uses_lattice_metric() -> None:
    structure = Structure.from_spacegroup(
        "Pnma",
        Lattice.orthorhombic(8.0, 9.0, 10.0),
        ["C"],
        [[0.11, 0.25, 0.33]],
    )
    shape = ImplicitShape.bfdh(structure, 24.0, max_index=1)

    assert shape.parameters["symmetry"] == {"kind": "lattice_metric"}


def test_bfdh_shape_rejects_unbounded_or_zero_millers() -> None:
    lattice = Lattice.from_parameters(4.0, 5.0, 6.0, 70.0, 80.0, 75.0)
    with pytest.raises(ValueError, match="do not enclose"):
        ImplicitShape.bfdh(
            lattice,
            20.0,
            miller_indices=[(1, 0, 0)],
            extinction_filter=False,
        )
    with pytest.raises(ValueError, match="cannot all be zero"):
        ImplicitShape.bfdh(lattice, 20.0, miller_indices=[(0, 0, 0)])


def test_bfdh_shape_plane_distances_follow_inverse_d_hkl() -> None:
    shape = ImplicitShape.bfdh(
        Lattice.orthorhombic(4.0, 5.0, 6.0),
        30.0,
        miller_indices=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        extinction_filter=False,
    )
    distances = {
        tuple(plane["miller_index"]): plane["distance_A"]
        for plane in shape.parameters["planes"]
    }

    assert np.isclose(distances[(1, 0, 0)] / distances[(0, 1, 0)], 5.0 / 4.0)
    assert np.isclose(distances[(0, 1, 0)] / distances[(0, 0, 1)], 6.0 / 5.0)


def test_bfdh_shape_normals_follow_sheared_reciprocal_lattice() -> None:
    lattice = Lattice.from_parameters(4.0, 5.0, 6.0, 72.0, 81.0, 76.0)
    millers = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    shape = ImplicitShape.bfdh(
        lattice,
        30.0,
        miller_indices=millers,
        extinction_filter=False,
    )
    normals = {
        tuple(plane["miller_index"]): np.asarray(plane["normal_cartesian"])
        for plane in shape.parameters["planes"]
    }

    reciprocal = lattice.reciprocal_lattice_crystallographic
    for hkl in millers:
        reciprocal_vector = np.asarray(reciprocal.get_cartesian_coords(hkl))
        assert np.allclose(normals[hkl], reciprocal_vector / np.linalg.norm(reciprocal_vector))


def test_bfdh_nanocluster_selects_representatives_inside_shape() -> None:
    crystal = _single_atom_crystal(5.0 * np.eye(3))
    shape = ImplicitShape.bfdh(
        crystal,
        20.0,
        miller_indices=[(1, 0, 0)],
        extinction_filter=False,
    )
    result = carve_nanocluster(crystal, shape, topology_unit="molecule")
    info = result.metadata["nanocluster"]
    source_positions = (
        result.to_ase().get_positions() - np.asarray(info["output_shift_A"])
    )
    local_positions = source_positions - np.asarray(info["source_center_A"])

    assert not any(result.pbc)
    assert np.all(
        shape.field(
            local_positions[:, 0], local_positions[:, 1], local_positions[:, 2]
        )
        <= 1e-12
    )
    assert info["shape"] == "bfdh"
    assert info["shape_parameters"]["symmetry"]["kind"] == "lattice_metric"


def test_custom_superellipsoid_fixed_geometry() -> None:
    crystal = _single_atom_crystal(5.0 * np.eye(3))

    def superellipsoid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (np.abs(x) / 8.0) ** 4 + (np.abs(y) / 5.0) ** 4 + (np.abs(z) / 3.0) ** 4 - 1.0

    shape = NanoShape(superellipsoid, ((-8, 8), (-5, 5), (-3, 3)), "superellipsoid")
    result = carve_nanocluster(crystal, shape, topology_unit="unit_cell", vacuum=2.0)
    info = result.metadata["nanocluster"]

    assert info["selection_mode"] == "fixed_geometry"
    assert info["shape"] == "superellipsoid"
    assert info["selected_unit_count"] > 0
    assert not any(result.pbc)
    positions = result.to_ase().get_positions()
    assert np.all(positions.min(axis=0) >= 2.0 - 1e-12)


def test_fixed_count_adn_models_have_identical_6444_atom_composition() -> None:
    crystal = _adn_like_crystal()
    source_positions = [molecule.get_positions().copy() for molecule in crystal.molecules]
    source_signatures = [_graph_signature(molecule) for molecule in crystal.molecules]
    shapes = [
        NanoShape.box((30.0, 30.0, 600.0)),
        NanoShape.box((100.0, 100.0, 60.0)),
        NanoShape.sphere(60.0),
    ]

    for shape in shapes:
        result = carve_nanocluster(
            crystal,
            shape,
            topology_unit="unit_cell",
            target_units=537,
        )
        info = result.metadata["nanocluster"]
        assert info["selected_unit_count"] == 537
        assert info["selected_molecule_count"] == 1074
        assert info["selected_atom_count"] == 6444
        assert sum(len(molecule) for molecule in result.molecules) == 6444
        assert len(result.extra_arrays["source_rank"]) == 6444
        assert np.array_equal(result.extra_arrays["source_rank"][:12], np.arange(12))
        assert [
            _graph_signature(molecule) for molecule in result.molecules
        ] == source_signatures * 537
        for index, molecule in enumerate(result.molecules):
            source = crystal.molecules[index % len(crystal.molecules)]
            assert molecule.get_chemical_symbols() == source.get_chemical_symbols()
            assert np.array_equal(molecule.arrays["site_tag"], source.arrays["site_tag"])
            assert molecule.info["component_label"] == source.info["component_label"]
            assert "frac_x" not in molecule.arrays
            assert "frac_y" not in molecule.arrays
            assert "frac_z" not in molecule.arrays
            assert "image_shift" not in molecule.arrays
    for molecule, positions in zip(crystal.molecules, source_positions):
        assert np.array_equal(molecule.get_positions(), positions)


def test_molecule_mode_preserves_metadata_and_warns_for_mixed_species() -> None:
    crystal = _adn_like_crystal()
    with pytest.warns(UserWarning, match="does not guarantee charge neutrality"):
        result = carve_nanocluster(
            crystal,
            NanoShape.sphere(20.0),
            topology_unit="molecule",
            target_units=5,
        )

    assert len(result.molecules) == 5
    for molecule in result.molecules:
        assert "site_tag" in molecule.arrays
        assert "frac_x" not in molecule.arrays
        assert "image_shift" not in molecule.arrays


def test_center_of_mass_and_centroid_are_distinct_selection_modes() -> None:
    cell = 20.0 * np.eye(3)
    molecule = Atoms(
        ["H", "Au"],
        positions=[[8.0, 10.0, 10.0], [12.0, 10.0, 10.0]],
        cell=cell,
        pbc=True,
    )
    crystal = MolecularCrystal(cell, [molecule])
    shape = NanoShape.sphere(1.0)

    centroid_result = carve_nanocluster(crystal, shape, center_kind="centroid")
    assert len(centroid_result.molecules) == 1
    with pytest.raises(ValueError, match="selected no topology units"):
        carve_nanocluster(crystal, shape, center_kind="com")


def test_triclinic_translation_bounds_match_brute_force() -> None:
    lattice = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.5, 0.75, 3.0]])
    crystal = _single_atom_crystal(lattice)
    shape = NanoShape.sphere(6.0)
    result = carve_nanocluster(crystal, shape, topology_unit="unit_cell")

    brute_count = 0
    for shift in itertools_product_range(-4, 5):
        local = np.asarray(shift) @ lattice
        if float(np.dot(local, local)) <= 36.0 + 1e-12:
            brute_count += 1
    assert result.metadata["nanocluster"]["selected_unit_count"] == brute_count


def itertools_product_range(start: int, stop: int):
    for i in range(start, stop):
        for j in range(start, stop):
            for k in range(start, stop):
                yield i, j, k


def test_fixed_count_ties_are_deterministic() -> None:
    crystal = _single_atom_crystal(2.0 * np.eye(3))
    shape = NanoShape(
        lambda x, y, z: np.zeros_like(x),
        ((-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)),
        "flat",
    )
    first = carve_nanocluster(crystal, shape, topology_unit="unit_cell", target_units=7)
    second = carve_nanocluster(crystal, shape, topology_unit="unit_cell", target_units=7)
    assert np.array_equal(first.to_ase().get_positions(), second.to_ase().get_positions())
    assert np.allclose(
        first.to_ase().get_positions(),
        np.column_stack((np.zeros(7), np.zeros(7), 2.0 * np.arange(7))),
    )


@pytest.mark.parametrize(
    "bounds",
    [
        ((-1, 1), (-1, 1)),
        ((-1, 1), (-1, 1), (0, 0)),
        ((-1, 1), (-1, 1), (-1, np.inf)),
    ],
)
def test_shape_rejects_invalid_bounds(bounds) -> None:
    with pytest.raises(ValueError, match="bounds"):
        NanoShape(lambda x, y, z: x, bounds)


@pytest.mark.parametrize(
    "field, error_type, message",
    [
        (lambda x, y, z: np.ones_like(x, dtype=bool), TypeError, "not booleans"),
        (lambda x, y, z: np.ones_like(x, dtype=complex), TypeError, "real numeric"),
        (lambda x, y, z: 0.0, ValueError, "one value per input point"),
        (lambda x, y, z: np.full_like(x, np.nan), ValueError, "non-finite"),
    ],
)
def test_invalid_shape_field_contract(field, error_type, message) -> None:
    crystal = _single_atom_crystal()
    shape = NanoShape(field, ((-1, 1), (-1, 1), (-1, 1)))
    with pytest.raises(error_type, match=message):
        carve_nanocluster(crystal, shape, topology_unit="unit_cell")


def test_target_count_larger_than_bounds_fails() -> None:
    crystal = _single_atom_crystal()
    with pytest.raises(ValueError, match="exceeds"):
        carve_nanocluster(
            crystal,
            NanoShape.sphere(1.0),
            topology_unit="unit_cell",
            target_units=100,
        )


def test_partial_periodicity_is_rejected() -> None:
    with pytest.raises(ValueError, match="all three dimensions"):
        NanoClusterCarver(_single_atom_crystal(pbc=(True, True, False)))


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(batch_size) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NanoClusterCarver(_single_atom_crystal(), batch_size=batch_size)


@pytest.mark.parametrize("target_units", [0, -1, 1.5, True])
def test_invalid_target_units_is_rejected(target_units) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        carve_nanocluster(
            _single_atom_crystal(),
            NanoShape.sphere(1.0),
            topology_unit="unit_cell",
            target_units=target_units,
        )


def test_million_candidate_selection_is_batched_and_copies_only_hit(monkeypatch) -> None:
    crystal = _single_atom_crystal()
    field_batch_sizes: list[int] = []

    def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        field_batch_sizes.append(len(x))
        return x * x + y * y + z * z

    shape = NanoShape(field, ((-50, 50), (-50, 50), (-50, 50)), "million-grid")
    copy_count = 0
    original_copy = CrystalMolecule.copy

    def counted_copy(self):
        nonlocal copy_count
        copy_count += 1
        return original_copy(self)

    monkeypatch.setattr(CrystalMolecule, "copy", counted_copy)
    result = NanoClusterCarver(crystal, batch_size=100_000).carve(
        shape,
        topology_unit="unit_cell",
        target_units=1,
    )
    info = result.metadata["nanocluster"]

    assert info["candidate_count"] == 101**3
    assert sum(field_batch_sizes) == 101**3
    assert max(field_batch_sizes) <= 100_000
    assert len(field_batch_sizes) == math.ceil((101**3) / 100_000)
    assert copy_count == 2  # selected molecule + MolecularCrystal ownership copy
    assert info["selected_atom_count"] == 1


def test_graph_isomorphism_survives_translation_and_output_reboxing() -> None:
    crystal = _adn_like_crystal()
    result = carve_nanocluster(
        crystal,
        NanoShape.sphere(25.0),
        topology_unit="unit_cell",
        target_units=3,
        vacuum=3.0,
    )
    for index, molecule in enumerate(result.molecules):
        source = crystal.molecules[index % len(crystal.molecules)]
        assert nx.is_isomorphic(source.graph, molecule.graph)
        assert source.get_chemical_formula() == molecule.get_chemical_formula()
