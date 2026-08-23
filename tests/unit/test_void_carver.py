"""Tests for topology-preserving implicit-shape void carving."""

from __future__ import annotations

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.disorder import UnresolvedDisorderWarning
from molcrys_kit.io import read_extxyz, read_mol_crystal, write_extxyz
from molcrys_kit.operations import (
    ImplicitShape,
    NanoShape,
    VacancyGenerator,
    VoidCarver,
    carve_void,
)
from molcrys_kit.operations.implicit_shape import evaluate_shape_field
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


DAP4 = Path(__file__).parents[1] / "data" / "cif" / "DAP-4.cif"


@pytest.fixture(scope="module")
def ordered_dap4() -> MolecularCrystal:
    return read_mol_crystal(DAP4, resolve_disorder=True)


def _one_atom_crystal(
    entries: list[tuple[str, tuple[float, float, float]]],
    *,
    lattice: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> MolecularCrystal:
    cell = np.asarray(lattice if lattice is not None else 10.0 * np.eye(3))
    molecules = []
    source_rank = []
    for index, (symbol, position) in enumerate(entries):
        molecule = Atoms(symbol, positions=[position], cell=cell, pbc=pbc)
        molecule.set_array("site_tag", np.asarray([index]))
        molecules.append(molecule)
        source_rank.append(index)
    return MolecularCrystal(
        cell,
        molecules,
        pbc=pbc,
        metadata={"source": "synthetic"},
        extra_arrays={"source_rank": np.asarray(source_rank)},
    )


def _adn_like_supercell() -> MolecularCrystal:
    cell = 8.0 * np.eye(3)
    cation = Atoms(
        ["N", "H", "H", "H", "H"],
        positions=[[2, 2, 2], [2.8, 2, 2], [1.2, 2, 2], [2, 2.8, 2], [2, 2, 2.8]],
        cell=cell,
        pbc=True,
    )
    anion = Atoms(
        ["N", "N", "N", "O", "O", "O", "O"],
        positions=[[6, 6, 6], [6.8, 6, 6], [5.2, 6, 6], [6, 6.8, 6], [6, 5.2, 6], [6, 6, 6.8], [6, 6, 5.2]],
        cell=cell,
        pbc=True,
    )
    cation.set_array("site_tag", np.arange(5))
    anion.set_array("site_tag", np.arange(5, 12))
    crystal = MolecularCrystal(
        cell,
        [cation, anion],
        extra_arrays={"source_rank": np.arange(12)},
    )
    supercell = crystal.get_supercell(4, 4, 4)
    supercell.extra_arrays["source_rank"] = np.arange(
        sum(len(molecule) for molecule in supercell.molecules)
    )
    return supercell


def test_implicit_shape_alias_and_arbitrary_axis_cylinder() -> None:
    assert NanoShape is ImplicitShape
    shape = ImplicitShape.cylinder(2.0, 6.0, axis=(1.0, 1.0, 0.0))
    axis = np.asarray(shape.parameters["axis_cartesian"])
    np.testing.assert_allclose(axis, [2**-0.5, 2**-0.5, 0.0])
    assert evaluate(shape, 0.0, 0.0, 0.0) < 0
    assert evaluate(shape, *(axis * 3.0)) == pytest.approx(0.0)


def test_through_cylinder_uses_primitive_lattice_direction() -> None:
    lattice = np.diag([10.0, 12.0, 14.0])
    shape = ImplicitShape.through_cylinder(2.0, lattice, (2, 2, 0))
    assert shape.parameters["direction_hkl"] == [1, 1, 0]
    expected_height = float(np.linalg.norm(np.asarray([10.0, 12.0, 0.0])))
    assert shape.parameters["height_A"] == pytest.approx(expected_height)

    opposite = ImplicitShape.through_cylinder(2.0, lattice, (-2, -2, 0))
    diagonal = ImplicitShape.through_cylinder(2.0, lattice, (-2, 2, 0))
    assert opposite.parameters["direction_hkl"] == [1, 1, 0]
    assert diagonal.parameters["direction_hkl"] == [1, -1, 0]


def test_shape_field_rejects_complex_values() -> None:
    shape = ImplicitShape(
        lambda x, y, z: x.astype(complex) + 1j,
        ((-1, 1), (-1, 1), (-1, 1)),
    )
    with pytest.raises(TypeError, match="real numeric"):
        evaluate_shape_field(shape, np.zeros((2, 3)))


def evaluate(shape: ImplicitShape, x: float, y: float, z: float) -> float:
    return float(shape.field(np.asarray([x]), np.asarray([y]), np.asarray([z]))[0])


def test_fixed_count_preserves_ionic_stoichiometry_topology_and_arrays() -> None:
    source = _adn_like_supercell()
    source_positions = [molecule.get_positions().copy() for molecule in source.molecules]
    source_graphs = {}
    for molecule in source.molecules:
        source_graphs.setdefault(molecule.get_chemical_formula(), molecule.graph.copy())
    result, removed = carve_void(
        source,
        ImplicitShape.sphere(11.0),
        center_frac=(0.5, 0.5, 0.5),
        target_units=7,
        species_charge_map={"H4N_1": 1, "N3O4_1": -1},
        return_removed_cluster=True,
    )
    info = result.metadata["void"]

    assert info["selected_unit_count"] == 7
    assert info["removed_species_counts"] == {"H4N_1": 7, "N3O4_1": 7}
    assert info["removed_atom_count"] == 84
    assert info["removed_net_charge_e"] == 0.0
    assert sum(map(len, removed.molecules)) == 84
    assert len(result.extra_arrays["source_rank"]) + len(removed.extra_arrays["source_rank"]) == sum(map(len, source.molecules))
    for molecule in result.molecules + removed.molecules:
        assert "site_tag" in molecule.arrays
        assert molecule.info.get("atom_indices") is None
        assert nx.is_isomorphic(
            molecule.graph,
            source_graphs[molecule.get_chemical_formula()],
        )
    for molecule, positions in zip(source.molecules, source_positions):
        np.testing.assert_array_equal(molecule.get_positions(), positions)


def test_inside_and_cover_round_stoichiometry_on_opposite_sides() -> None:
    crystal = _one_atom_crystal(
        [
            ("H", (5.0, 5.0, 5.0)),
            ("H", (5.9, 5.0, 5.0)),
            ("H", (8.0, 5.0, 5.0)),
            ("He", (5.2, 5.0, 5.0)),
            ("He", (6.2, 5.0, 5.0)),
            ("He", (8.5, 5.0, 5.0)),
        ]
    )
    shape = ImplicitShape.sphere(1.0)
    inside = carve_void(crystal, shape, center=(5, 5, 5), boundary_policy="inside")
    cover, removed = carve_void(
        crystal,
        shape,
        center=(5, 5, 5),
        boundary_policy="cover",
        return_removed_cluster=True,
    )
    assert inside.metadata["void"]["removed_species_counts"] == {"H_1": 1, "He_1": 1}
    assert cover.metadata["void"]["removed_species_counts"] == {"H_1": 2, "He_1": 2}
    np.testing.assert_array_equal(cover.extra_arrays["source_rank"], [2, 5])
    np.testing.assert_array_equal(removed.extra_arrays["source_rank"], [0, 1, 3, 4])


def test_any_atom_and_all_atoms_use_complete_molecule_envelopes() -> None:
    cell = 10.0 * np.eye(3)
    crossing = Atoms("H2", positions=[[5.2, 5, 5], [5.8, 5, 5]], cell=cell, pbc=True)
    compact = Atoms("H2", positions=[[4.8, 5, 5], [5.2, 5, 5]], cell=cell, pbc=True)
    outside = Atoms("H2", positions=[[8.0, 5, 5], [8.5, 5, 5]], cell=cell, pbc=True)
    crystal = MolecularCrystal(cell, [crossing, compact, outside])
    shape = ImplicitShape.sphere(0.75)

    any_result = carve_void(crystal, shape, center=(5, 5, 5), hit_mode="any_atom")
    all_result = carve_void(crystal, shape, center=(5, 5, 5), hit_mode="all_atoms")
    assert any_result.metadata["void"]["raw_inside_species_counts"]["H2_1"] == 2
    assert all_result.metadata["void"]["raw_inside_species_counts"]["H2_1"] == 1
    assert len(any_result.molecules) == 1
    assert len(all_result.molecules) == 2


def test_periodic_shape_image_wraps_across_boundary() -> None:
    crystal = _one_atom_crystal(
        [("He", (0.2, 5, 5)), ("He", (5.0, 5, 5))]
    )
    wrapped = carve_void(
        crystal,
        ImplicitShape.sphere(0.5),
        center=(9.8, 5, 5),
        periodic_images=True,
    )
    assert len(wrapped.molecules) == 1
    with pytest.raises(ValueError, match="does not contain enough"):
        carve_void(
            crystal,
            ImplicitShape.sphere(0.5),
            center=(9.8, 5, 5),
            periodic_images=False,
        )


def test_all_atoms_uses_one_periodic_shape_image_per_molecule() -> None:
    cell = 10.0 * np.eye(3)
    split = Atoms("H2", positions=[[0.2, 5, 5], [9.8, 5, 5]], cell=cell, pbc=True)
    compact = Atoms("H2", positions=[[0.1, 5, 5], [0.2, 5, 5]], cell=cell, pbc=True)
    outside = Atoms("H2", positions=[[5.0, 5, 5], [5.1, 5, 5]], cell=cell, pbc=True)
    split_molecule = CrystalMolecule(split, check_pbc=False)
    # Preserve a known H-H topology while retaining deliberately wrapped coordinates.
    split_molecule._graph = nx.Graph()
    split_molecule._graph.add_nodes_from([(0, {"symbol": "H"}), (1, {"symbol": "H"})])
    split_molecule._graph.add_edge(0, 1)
    crystal = MolecularCrystal(
        cell,
        [split_molecule, compact, outside],
    )
    shape = ImplicitShape.sphere(0.3)

    any_result = carve_void(
        crystal,
        shape,
        center=(0, 5, 5),
        hit_mode="any_atom",
    )
    all_result = carve_void(
        crystal,
        shape,
        center=(0, 5, 5),
        hit_mode="all_atoms",
    )
    assert any_result.metadata["void"]["raw_inside_species_counts"] == {"H2_1": 2}
    assert all_result.metadata["void"]["raw_inside_species_counts"] == {"H2_1": 1}


def test_cartesian_and_fractional_centers_are_equivalent() -> None:
    crystal = _one_atom_crystal(
        [("He", (5.0, 5, 5)), ("He", (8.0, 5, 5))]
    )
    shape = ImplicitShape.sphere(1.0)
    cart = carve_void(crystal, shape, center=(5, 5, 5))
    frac = carve_void(crystal, shape, center_frac=(0.5, 0.5, 0.5))
    np.testing.assert_array_equal(cart.to_ase().positions, frac.to_ase().positions)
    with pytest.raises(ValueError, match="mutually exclusive"):
        carve_void(crystal, shape, center=(5, 5, 5), center_frac=(0.5, 0.5, 0.5))


def test_partial_pbc_only_wraps_enabled_directions() -> None:
    crystal = _one_atom_crystal(
        [("He", (0.2, 0.2, 5)), ("He", (5.0, 5.0, 5))],
        pbc=(True, False, False),
    )
    shape = ImplicitShape.sphere(0.6)
    result = carve_void(crystal, shape, center=(9.8, 0.2, 5))
    assert len(result.molecules) == 1
    with pytest.raises(ValueError, match="does not contain enough"):
        carve_void(crystal, shape, center=(0.2, 9.8, 5))


def test_charge_map_is_complete_and_neutral() -> None:
    crystal = _one_atom_crystal(
        [("H", (5, 5, 5)), ("He", (5.2, 5, 5)), ("H", (8, 5, 5)), ("He", (8.2, 5, 5))]
    )
    shape = ImplicitShape.sphere(1.0)
    with pytest.raises(ValueError, match="missing target species"):
        carve_void(crystal, shape, species_charge_map={"H_1": 1})
    with pytest.raises(ValueError, match="non-zero net charge"):
        carve_void(crystal, shape, species_charge_map={"H_1": 1, "He_1": 1})
    with pytest.raises(ValueError, match="positive integer"):
        carve_void(crystal, shape, target_spec={"H_1": 1.0}, target_units=1)


def test_through_cylinder_rejects_nonperiodic_hkl_component() -> None:
    crystal = _one_atom_crystal(
        [("He", (5, 5, 5)), ("He", (8, 8, 8))],
        pbc=(True, False, True),
    )
    shape = ImplicitShape.through_cylinder(1.0, crystal.lattice, (1, 1, 0))
    with pytest.raises(ValueError, match="non-periodic"):
        carve_void(crystal, shape)


def test_incomplete_topology_unit_is_rejected() -> None:
    crystal = _one_atom_crystal([("He", (5, 5, 5)), ("He", (8, 8, 8))])
    crystal.molecules[0].info["unwrap_completed"] = False
    with pytest.raises(ValueError, match="Periodic 3-D frameworks/MOFs"):
        VoidCarver(crystal)


def test_dap4_unresolved_warns_and_resolved_replica_is_ordered(
    ordered_dap4: MolecularCrystal,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unresolved = read_mol_crystal(DAP4)
    with pytest.warns(UnresolvedDisorderWarning, match="unresolved disorder"):
        VoidCarver(unresolved)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        carver = VoidCarver(ordered_dap4)
    assert carver._input_disorder["all_atom_ordered"]
    assert not [item for item in caught if item.category is UnresolvedDisorderWarning]


def test_dap4_void_extxyz_roundtrip_preserves_valid_arrays(
    ordered_dap4: MolecularCrystal,
    tmp_path: Path,
) -> None:
    source = ordered_dap4.get_supercell(2, 1, 1)
    source_atom_count = sum(len(molecule) for molecule in source.molecules)
    source.extra_arrays["source_rank"] = np.arange(source_atom_count)
    remaining, removed = carve_void(
        source,
        ImplicitShape.sphere(5.0),
        center_frac=(0.5, 0.5, 0.5),
        target_units=1,
        return_removed_cluster=True,
    )
    output = tmp_path / "dap4_void.extxyz"
    write_extxyz(remaining, str(output))
    reread = read_extxyz(str(output))

    assert sum(map(len, removed.molecules)) == 42
    assert sum(map(len, reread.molecules)) == source_atom_count - 42
    np.testing.assert_array_equal(
        reread.extra_arrays["source_rank"],
        remaining.extra_arrays["source_rank"],
    )
    flat = reread.to_ase()
    assert np.allclose(flat.arrays["occupancy"], 1.0)
    for key in ("disorder_group", "assembly", "label", "image_shift"):
        assert key in flat.arrays
    for stale_key in ("frac_x", "frac_y", "frac_z"):
        assert stale_key not in flat.arrays


def test_dap4_vacancy_regression_preserves_formula_packet(
    ordered_dap4: MolecularCrystal,
) -> None:
    remaining, removed = VacancyGenerator(ordered_dap4).generate_vacancy(
        target_spec={"C6H14N2_1": 1, "ClO4_1": 3, "H4N_1": 1},
        random_seed=0,
        return_removed_cluster=True,
    )
    formulas = [molecule.get_chemical_formula() for molecule in removed.molecules]
    assert formulas.count("C6H14N2") == 1
    assert formulas.count("ClO4") == 3
    assert formulas.count("H4N") == 1
    assert sum(map(len, remaining.molecules)) == 294
    assert sum(map(len, removed.molecules)) == 42


def test_field_calls_respect_batch_size_for_atom_hit_modes() -> None:
    cell = 30.0 * np.eye(3)
    molecules = [
        Atoms("H6", positions=np.column_stack((np.arange(6) + index, np.zeros(6), np.zeros(6))), cell=cell, pbc=False)
        for index in range(6)
    ]
    crystal = MolecularCrystal(cell, molecules, pbc=(False, False, False))
    batch_lengths: list[int] = []

    def field(x, y, z):
        batch_lengths.append(len(x))
        return x * x + y * y + z * z - 1.0

    with pytest.raises(ValueError, match="does not contain enough"):
        carve_void(
            crystal,
            ImplicitShape(field, ((-2, 2), (-2, 2), (-2, 2))),
            center=(15, 15, 15),
            hit_mode="any_atom",
            batch_size=10,
        )
    assert batch_lengths
    assert max(batch_lengths) <= 10


def test_single_molecule_larger_than_batch_size_is_chunked() -> None:
    cell = 100.0 * np.eye(3)
    near = Atoms(
        "H25",
        positions=np.column_stack((np.arange(25), np.zeros(25), np.zeros(25))),
        cell=cell,
        pbc=False,
    )
    far = Atoms(
        "H25",
        positions=np.column_stack((50 + np.arange(25), np.zeros(25), np.zeros(25))),
        cell=cell,
        pbc=False,
    )
    crystal = MolecularCrystal(cell, [near, far], pbc=(False, False, False))
    batch_lengths: list[int] = []

    def field(x, y, z):
        batch_lengths.append(len(x))
        return x * x + y * y + z * z - 0.25

    result = carve_void(
        crystal,
        ImplicitShape(field, ((-1, 1), (-1, 1), (-1, 1))),
        center=(0, 0, 0),
        hit_mode="any_atom",
        batch_size=10,
    )
    assert len(result.molecules) == 1
    assert batch_lengths
    assert max(batch_lengths) <= 10


def test_partition_rejects_non_per_atom_extra_array() -> None:
    crystal = _one_atom_crystal(
        [("He", (5, 5, 5)), ("He", (8, 8, 8))]
    )
    crystal.extra_arrays["invalid"] = np.asarray([1])
    with pytest.raises(ValueError, match="expected 2 per-atom values"):
        carve_void(
            crystal,
            ImplicitShape.sphere(1.0),
            center=(5, 5, 5),
            target_units=1,
        )
