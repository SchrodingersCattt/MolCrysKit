"""Regression and invariance checks for geometry-only periodic paths."""

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic

from molcrys_kit.operations import (
    PeriodicPathConfig,
    PeriodicPathError,
    interpolate_periodic_paths,
)
from molcrys_kit.operations._periodic_path_geometry import (
    canonical_periodic_basis,
    interpolate_cells,
    nearest_image,
)
from molcrys_kit.operations.periodic_path import _image_choices, _objective
from molcrys_kit.structures.crystal import MolecularCrystal


def _crystal(
    symbols, positions, cell, pbc=(True, True, True), groups=None, records=None
):
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=pbc)
    atoms.set_array(
        "molecule_index",
        np.asarray(groups if groups is not None else range(len(atoms))),
    )
    atoms.set_array("atom_id", np.arange(len(atoms)))
    if records is not None:
        atoms.info["_molcrys_bond_records"] = records
    return MolecularCrystal.from_ase_atoms(atoms)


def _config(**values):
    return PeriodicPathConfig(n_images=5, max_candidates=1, **values)


def test_true_minimum_image_for_skew_cell_and_partial_periodicity():
    cell = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    delta = np.array([0.49, 0.49, 0.0]) @ cell
    vector, shift = nearest_image(delta, cell, (True, True, False))
    expected, _ = find_mic(delta, cell, pbc=(True, True, False))
    np.testing.assert_allclose(vector, expected, atol=1e-12)
    np.testing.assert_array_equal(shift, [0, -1, 0])
    assert np.linalg.norm(vector) < 0.6


def test_minimum_image_accepts_zero_nonperiodic_cell_vectors():
    cell = np.diag([10.0, 0.0, 0.0])
    vector, shift = nearest_image(
        np.array([-8.0, 2.0, 3.0]), cell, (True, False, False)
    )
    np.testing.assert_allclose(vector, [2.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_array_equal(shift, [1, 0, 0])


@pytest.mark.parametrize(
    "pbc", [(True, True, True), (False, True, True), (True, False, False)]
)
def test_basis_representations_are_normalized(pbc):
    cell = np.array([[4.0, 0.2, 0.1], [0.4, 4.4, 0.3], [0.7, 0.8, 5.1]])
    axes = np.flatnonzero(pbc)
    transform = np.eye(3, dtype=int)
    if len(axes) > 1:
        transform[axes[0], axes[1]] = 2
    else:
        transform[axes[0], axes[0]] = -1
    first, _ = canonical_periodic_basis(cell, pbc)
    second, _ = canonical_periodic_basis(transform @ cell, pbc)
    np.testing.assert_allclose(first[axes], second[axes], atol=1e-10)


def test_equivalent_rotated_cell_has_no_singular_intermediate():
    first = np.eye(3) * 3.0
    last = np.diag([-3.0, -3.0, 3.0])
    cells, transform = interpolate_cells(
        first, last, (True, True, True), np.linspace(0, 1, 7)
    )
    np.testing.assert_allclose(cells[0], first, atol=1e-12)
    np.testing.assert_allclose(cells[-1], transform @ last, atol=1e-12)
    assert all(np.linalg.det(cell) > 0 for cell in cells)


def test_partial_periodic_cells_can_have_zero_rows():
    first = np.diag([3.0, 0.0, 4.0])
    last = np.diag([4.0, 0.0, 5.0])
    cells, _ = interpolate_cells(first, last, (True, False, True), np.linspace(0, 1, 5))
    for cell in cells:
        np.testing.assert_allclose(cell[1], 0.0)
        assert np.linalg.matrix_rank(cell[[0, 2]]) == 2


def test_invalid_periodic_subspace_is_rejected():
    with pytest.raises(ValueError, match="linearly independent"):
        interpolate_cells(
            np.zeros((3, 3)), np.zeros((3, 3)), (True, False, False), [0, 1]
        )


def test_exact_atom_exchange_is_not_falsely_converged():
    cell = np.eye(3) * 10.0
    start = _crystal("HH", [[-1.0, 0, 0], [1.0, 0, 0]], cell)
    end = _crystal("HH", [[1.0, 0, 0], [-1.0, 0, 0]], cell)
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0, 1], config=_config()
    )
    candidate = result.candidates[0]
    assert candidate.diagnostics["accepted"]
    assert not candidate.diagnostics["exact_overlap"]
    assert (
        np.linalg.norm(
            candidate.images[2].positions[1] - candidate.images[2].positions[0]
        )
        > 0.3
    )
    np.testing.assert_allclose(candidate.images[0].positions, start.to_ase().positions)
    np.testing.assert_allclose(candidate.images[-1].positions, end.to_ase().positions)


def test_collision_refinement_is_bounded_and_reported():
    cell = np.eye(3) * 10.0
    start = _crystal("HH", [[-1.0, 0, 0], [1.0, 0, 0]], cell)
    end = _crystal("HH", [[1.0, 0, 0], [-1.0, 0, 0]], cell)
    strict = _config(collision_tolerance_A=1e-4, smoothness_weight=0.2)
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0, 1], config=strict
    )
    diagnostic = result.candidates[0].diagnostics
    assert (
        1
        <= diagnostic["collision_refinement_steps"]
        <= strict.max_collision_refinements
    )
    assert diagnostic["maximum_collision_penetration_A"] <= strict.collision_tolerance_A
    assert diagnostic["total_iterations"] == sum(
        stage["iterations"] for stage in diagnostic["refinement_history"]
    )


def test_endpoint_collision_is_a_reported_failure():
    cell = np.eye(3) * 10.0
    start = _crystal("HH", [[0, 0, 0], [0, 0, 0]], cell)
    end = _crystal("HH", [[0, 0, 0], [1, 0, 0]], cell)
    with pytest.raises(PeriodicPathError) as error:
        interpolate_periodic_paths(
            start, end, product_index_by_reactant=[0, 1], config=_config()
        )
    assert error.value.diagnostics["rejected_candidates"]
    assert error.value.diagnostics["rejected_candidates"][0]["exact_overlap"]


def test_component_rotation_keeps_a_finite_group_intact():
    cell = np.eye(3) * 10.0
    start = _crystal(
        "CHH", [[0, 0, 0], [1.05, 0, 0], [-1.05, 0, 0]], cell, groups=[0, 0, 0]
    )
    end = _crystal(
        "CHH", [[0, 0, 0], [-1.05, 0, 0], [1.05, 0, 0]], cell, groups=[0, 0, 0]
    )
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0, 1, 2], config=_config()
    )
    for image in result.candidates[0].images:
        assert np.linalg.norm(image.positions[1] - image.positions[2]) > 1.9
    assert result.candidates[0].diagnostics["maximum_conserved_bond_deviation_A"] < 0.1


def test_one_periodic_axis_unwraps_the_short_displacement():
    cell = np.diag([10.0, 0.0, 0.0])
    start = _crystal("C", [[9.0, 1.0, 2.0]], cell, pbc=(True, False, False))
    end = _crystal("C", [[0.0, 1.0, 2.0]], cell, pbc=(True, False, False))
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0], config=_config()
    )
    x = [image.positions[0, 0] for image in result.candidates[0].images]
    np.testing.assert_allclose(x, np.linspace(9.0, 10.0, 5), atol=1e-6)
    assert result.candidates[0].product_image_shifts == ((1, 0, 0),)


def test_distinct_component_windings_are_not_deduplicated():
    cell = np.diag([10.0, 0.0, 0.0])
    start = _crystal("C", [[1.0, 0.0, 0.0]], cell, pbc=(True, False, False))
    end = _crystal("C", [[2.0, 0.0, 0.0]], cell, pbc=(True, False, False))
    config = PeriodicPathConfig(n_images=5, max_candidates=3)
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0], config=config
    )
    assert len(result.candidates) == 3
    assert {
        candidate.product_image_shifts[0][0] for candidate in result.candidates
    } == {-1, 0, 1}


def test_rewrapped_endpoints_yield_the_same_physical_path():
    cell = np.array([[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [0.0, 0.0, 9.0]])
    first = np.array([[1.0, 1.0, 1.0], [4.0, 3.0, 2.0]])
    last = np.array([[1.4, 1.2, 1.0], [4.3, 3.1, 2.0]])
    start = _crystal("CH", first, cell)
    end = _crystal("CH", last, cell)
    start_rewrapped = _crystal(
        "CH", first + np.array([[1, -1, 0], [0, 1, 0]]) @ cell, cell
    )
    end_rewrapped = _crystal(
        "CH", last + np.array([[0, 1, 0], [-1, 0, 0]]) @ cell, cell
    )
    result1 = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0, 1], config=_config()
    )
    result2 = interpolate_periodic_paths(
        start_rewrapped,
        end_rewrapped,
        product_index_by_reactant=[0, 1],
        config=_config(),
    )
    for first_image, second_image in zip(
        result1.candidates[0].images, result2.candidates[0].images
    ):
        delta = first_image.positions - second_image.positions
        vectors, _ = find_mic(delta, cell, pbc=True)
        np.testing.assert_allclose(vectors, 0.0, atol=1e-7)


def test_rewrapped_periodic_bond_preserves_a_finite_component():
    cell = np.eye(3) * 10.0

    def make(positions, shift):
        records = [
            {
                "molecule_index": 0,
                "left": 0,
                "right": 1,
                "right_image_shift": shift,
                "vector": [1.0, 0, 0],
                "role": "covalent",
            }
        ]
        return _crystal("CH", positions, cell, groups=[0, 0], records=records)

    start = make([[9.6, 4, 4], [0.6, 4, 4]], [1, 0, 0])
    wrapped = make([[9.6, 4, 4], [10.6, 4, 4]], [0, 0, 0])
    end = make([[0.2, 4, 4], [1.2, 4, 4]], [0, 0, 0])
    original = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[0, 1], config=_config()
    )
    changed = interpolate_periodic_paths(
        wrapped, end, product_index_by_reactant=[0, 1], config=_config()
    )
    for image1, image2 in zip(
        original.candidates[0].images, changed.candidates[0].images
    ):
        delta, _ = find_mic(image1.positions - image2.positions, cell, pbc=True)
        np.testing.assert_allclose(delta, 0.0, atol=1e-7)
        bond, _ = find_mic(image1.positions[1] - image1.positions[0], cell, pbc=True)
        assert np.linalg.norm(bond) == pytest.approx(1.0, abs=1e-5)


def test_equivalent_cell_basis_yields_the_same_physical_path():
    cell = np.array([[5.0, 0.2, 0.1], [0.4, 5.4, 0.3], [0.7, 0.8, 6.1]])
    transform = np.array([[1, 2, -1], [0, 1, 1], [0, 0, 1]])
    first = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 2.0]])
    last = np.array([[1.3, 1.1, 1.0], [3.2, 3.1, 2.0]])
    original = interpolate_periodic_paths(
        _crystal("CH", first, cell),
        _crystal("CH", last, cell),
        product_index_by_reactant=[0, 1],
        config=_config(),
    )
    rebased = interpolate_periodic_paths(
        _crystal("CH", first, transform @ cell),
        _crystal("CH", last, transform @ cell),
        product_index_by_reactant=[0, 1],
        config=_config(),
    )
    for image1, image2 in zip(
        original.candidates[0].images, rebased.candidates[0].images
    ):
        vectors, _ = find_mic(image1.positions - image2.positions, cell, pbc=True)
        np.testing.assert_allclose(vectors, 0.0, atol=1e-7)


def test_variable_cell_with_independent_basis_changes_is_equivariant():
    first_cell = np.array([[6.0, 0.1, 0.2], [0.0, 7.0, 0.2], [0.1, 0.2, 8.0]])
    last_cell = np.array([[6.4, 0.2, 0.1], [0.1, 7.3, 0.3], [0.2, 0.1, 8.4]])
    first_basis = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    last_basis = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    first = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    last = np.array([[1.3, 1.1, 1.0], [3.2, 3.1, 3.2]])
    original = interpolate_periodic_paths(
        _crystal("CH", first, first_cell),
        _crystal("CH", last, last_cell),
        product_index_by_reactant=[0, 1],
        config=_config(),
    )
    rebased = interpolate_periodic_paths(
        _crystal("CH", first, first_basis @ first_cell),
        _crystal("CH", last, last_basis @ last_cell),
        product_index_by_reactant=[0, 1],
        config=_config(),
    )
    for a, b in zip(original.candidates[0].images, rebased.candidates[0].images):
        np.testing.assert_allclose(b.cell.array, first_basis @ a.cell.array, atol=1e-8)
        difference, _ = find_mic(a.positions - b.positions, a.cell, pbc=True)
        np.testing.assert_allclose(difference, 0.0, atol=1e-7)


def test_self_image_edge_survives_variable_cell():
    def make(length):
        cell = np.diag([length, 0.0, 0.0])
        records = [
            {
                "molecule_index": 0,
                "left": 0,
                "right": 0,
                "right_image_shift": [1, 0, 0],
                "vector": [length, 0, 0],
                "role": "covalent",
            }
        ]
        return _crystal(
            "C", [[0, 0, 0]], cell, pbc=(True, False, False), records=records
        )

    result = interpolate_periodic_paths(
        make(3.0), make(3.5), product_index_by_reactant=[0], config=_config()
    )
    assert len(result.reactant_graph.edges) == 1
    assert result.reactant_graph.edges[0].image_shift == (1, 0, 0)
    assert len(result.product_graph.edges) == 1
    assert len(result.reactant_bond_records) == 1
    assert result.reactant_bond_records[0].role == "covalent"
    assert all(
        "_molcrys_bond_records" not in image.info
        for image in result.candidates[0].images
    )
    assert result.candidates[0].diagnostics["accepted"]


def test_parallel_self_image_edges_are_kept_separate():
    cell = np.diag([3.0, 4.0, 0.0])
    records = [
        {
            "molecule_index": 0,
            "left": 0,
            "right": 0,
            "right_image_shift": [1, 0, 0],
            "vector": [3.0, 0, 0],
            "role": "covalent",
        },
        {
            "molecule_index": 0,
            "left": 0,
            "right": 0,
            "right_image_shift": [0, 1, 0],
            "vector": [0, 4.0, 0],
            "role": "covalent",
        },
    ]
    crystal = _crystal("C", [[0, 0, 0]], cell, pbc=(True, True, False), records=records)
    result = interpolate_periodic_paths(
        crystal, crystal, product_index_by_reactant=[0], config=_config()
    )
    assert {edge.image_shift for edge in result.reactant_graph.edges} == {
        (1, 0, 0),
        (0, 1, 0),
    }


def test_missing_or_invalid_atom_mapping_fails():
    cell = np.eye(3) * 8
    first = _crystal("CH", [[0, 0, 0], [2, 0, 0]], cell)
    with pytest.raises(ValueError, match="explicit"):
        interpolate_periodic_paths(
            first, first, product_index_by_reactant=None, config=_config()
        )
    with pytest.raises(ValueError, match="permutation"):
        interpolate_periodic_paths(
            first, first, product_index_by_reactant=[0, 0], config=_config()
        )
    with pytest.raises(ValueError, match="element"):
        interpolate_periodic_paths(
            first, first, product_index_by_reactant=[1, 0], config=_config()
        )


def test_product_permutation_preserves_explicit_identity_and_inputs():
    cell = np.eye(3) * 10.0
    start = _crystal("CH", [[1.0, 1, 1], [4.0, 4, 4]], cell)
    end = _crystal("HC", [[4.2, 4, 4], [1.2, 1, 1]], cell)
    start_before = start.to_ase().positions.copy()
    end_before = end.to_ase().positions.copy()
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=[1, 0], config=_config()
    )
    np.testing.assert_allclose(start.to_ase().positions, start_before)
    np.testing.assert_allclose(end.to_ase().positions, end_before)
    np.testing.assert_allclose(
        result.candidates[0].images[-1].positions, end_before[[1, 0]]
    )
    assert result.product_index_by_reactant == (1, 0)
    for image in result.candidates[0].images:
        np.testing.assert_array_equal(image.arrays["atom_id"], [0, 1])
        assert image.get_chemical_symbols() == ["C", "H"]


def test_objective_gradient_matches_finite_difference():
    cell = np.eye(3) * 10.0
    cells = [cell] * 3
    seed = np.array(
        [
            [[0.0, 0, 0], [1.0, 0, 0]],
            [[0.0, 0, 0], [0.3, 0.04, 0]],
            [[0.0, 0, 0], [1.0, 0, 0]],
        ]
    )
    config = _config()
    x = seed[1:-1].ravel().copy()
    args = (
        seed,
        cells,
        np.array([True, True, True]),
        np.array([1, 1]),
        [],
        np.array([]),
        np.array([]),
        config,
    )
    _, gradient = _objective(x, *args)
    numerical = np.zeros_like(x)
    for index in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (
            _objective(plus, *args)[0] - _objective(minus, *args)[0]
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_fixed_atom_stays_fixed_while_free_atom_avoids_overlap():
    cell = np.eye(3) * 10
    start = _crystal("CH", [[0, 0, 0], [-1.2, 0, 0]], cell)
    end = _crystal("CH", [[0, 0, 0], [1.2, 0, 0]], cell)
    cfg = _config(fixed_atom_indices=(0,))
    result = interpolate_periodic_paths(start, end, product_index_by_reactant=[0, 1], config=cfg)
    for image in result.candidates[0].images:
        np.testing.assert_array_equal(image.positions[0], start.to_ase().positions[0])
        assert image.constraints
    assert result.diagnostics["fixed_atom_indices"] == [0]


def test_fixed_atom_accepts_equivalent_periodic_image():
    cell = np.eye(3) * 10
    start = _crystal("C", [[9.0, 1, 1]], cell)
    end = _crystal("C", [[-1.0, 1, 1]], cell)
    result = interpolate_periodic_paths(start, end, product_index_by_reactant=[0], config=_config(fixed_atom_indices=(0,)))
    for image in result.candidates[0].images:
        np.testing.assert_allclose(image.positions[0], [9, 1, 1])


def test_fixed_atom_rejects_incompatible_endpoints():
    cell = np.eye(3) * 10
    start = _crystal("C", [[1.0, 1, 1]], cell)
    end = _crystal("C", [[1.5, 1, 1]], cell)
    with pytest.raises(ValueError, match="fixed atom endpoints"):
        interpolate_periodic_paths(start, end, product_index_by_reactant=[0], config=_config(fixed_atom_indices=(0,)))


def test_all_fixed_atoms_with_identical_endpoints():
    cell = np.eye(3) * 10
    crystal = _crystal("CH", [[1, 1, 1], [4, 4, 4]], cell)
    result = interpolate_periodic_paths(crystal, crystal, product_index_by_reactant=[0, 1], config=_config(fixed_atom_indices=(0, 1)))
    assert result.candidates[0].diagnostics["optimizer_success"]
    assert result.candidates[0].diagnostics["total_iterations"] == 0
    for image in result.candidates[0].images:
        np.testing.assert_allclose(image.positions, crystal.to_ase().positions)


@pytest.mark.parametrize("indices", [(-1,), (0,0), (True,), (0.5,)])
def test_reject_invalid_fixed_indices(indices):
    with pytest.raises(ValueError, match="fixed atom indices"):
        _config(fixed_atom_indices=indices)


def test_reject_out_of_range_fixed_index():
    cell = np.eye(3) * 10
    crystal = _crystal("C", [[1, 1, 1]], cell)
    with pytest.raises(ValueError, match="out of range"):
        interpolate_periodic_paths(crystal, crystal, product_index_by_reactant=[0], config=_config(fixed_atom_indices=(1,)))



def test_fixed_atom_candidates_are_deduplicated_before_the_limit():
    shifts = np.zeros((2, 3), dtype=int)
    positions = np.zeros((2, 3))
    enumeration = _image_choices(
        shifts,
        [([0], True), ([1], True)],
        positions,
        positions,
        np.eye(3) * 10,
        np.eye(3, dtype=int),
        np.ones(3, dtype=bool),
        3,
        np.array([0]),
        np.zeros((1, 3), dtype=int),
    )
    choices = enumeration.choices
    assert enumeration.family_unique_count >= 3
    assert len(choices) == 3
    if enumeration.family_unique_count > 3:
        assert enumeration.truncated
    assert all(np.array_equal(choice[0], [0, 0, 0]) for choice in choices)
    assert len({tuple(choice[1]) for choice in choices}) == 3


def test_all_fixed_candidate_count_reports_unique_choices():
    cell = np.eye(3) * 10
    crystal = _crystal("CH", [[1, 1, 1], [4, 4, 4]], cell)
    config = PeriodicPathConfig(
        n_images=5, max_candidates=3, fixed_atom_indices=(0, 1)
    )
    result = interpolate_periodic_paths(
        crystal, crystal, product_index_by_reactant=[0, 1], config=config
    )
    assert result.diagnostics["candidate_count_evaluated"] == 1
    assert len(result.candidates) == 1


def test_image_shift_enumeration_reports_truncation_within_finite_family():
    cell = np.eye(3) * 10
    crystal = _crystal("CH", [[1, 1, 1], [4, 4, 4]], cell)
    config = PeriodicPathConfig(n_images=5, max_candidates=1)
    result = interpolate_periodic_paths(
        crystal, crystal, product_index_by_reactant=[0, 1], config=config
    )
    meta = result.diagnostics["image_shift_enumeration"]
    assert meta["family_unique_count"] > config.max_candidates
    assert meta["enumeration_truncated"] is True
    assert meta["search_complete_within_family"] is False
    assert meta["candidates_returned_for_search"] == 1


def test_geometric_screening_documents_gradient_acceptance_factor():
    cell = np.eye(3) * 10
    crystal = _crystal("C", [[1, 1, 1]], cell)
    result = interpolate_periodic_paths(
        crystal, crystal, product_index_by_reactant=[0], config=_config()
    )
    diag = result.candidates[0].diagnostics
    from molcrys_kit.constants.periodic_path import PERIODIC_PATH_GRADIENT_ACCEPTANCE_FACTOR

    assert diag["gradient_acceptance_factor"] == PERIODIC_PATH_GRADIENT_ACCEPTANCE_FACTOR
    assert diag["gradient_screening_threshold"] == pytest.approx(
        diag["gradient_tolerance_optimizer"] * PERIODIC_PATH_GRADIENT_ACCEPTANCE_FACTOR
    )
    assert diag["passes_geometric_screening"] is diag["accepted"]
    assert (
        diag["passes_gradient_screening"]
        == (diag["maximum_gradient_component"] <= diag["gradient_screening_threshold"])
    )


@pytest.mark.parametrize("angle_deg", [150.0, 180.0])
def test_default_path_keeps_known_large_rigid_rotation_undistorted(angle_deg):
    """Cartesian smoothing must not shrink an otherwise exact rigid arc."""
    from scipy.spatial.transform import Rotation

    cell = np.array([[12.0, 0, 0], [2.0, 13.0, 0], [0, 0, 14.0]])
    center = np.array([11.6, 6.5, 7.0])
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    tetra = tetra / np.sqrt(3.0) * np.array([1.09, 1.35, 1.78, 1.94])[:, None]
    initial = np.vstack([center, center + tetra])
    rotation = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(angle_deg)])
    final = center + np.array([1.0, 0.2, 0.0]) + rotation.apply(initial - center)

    def endpoint(positions):
        atoms = Atoms("CHFClBr", positions=positions, cell=cell, pbc=True)
        atoms.set_array("molecule_index", np.zeros(5, dtype=int))
        atoms.set_array("atom_id", np.arange(5))
        atoms.wrap()
        records = []
        for j in range(1, 5):
            vector, shift = nearest_image(atoms.positions[j] - atoms.positions[0], cell, atoms.pbc)
            records.append({"molecule_index": 0, "left": 0, "right": j,
                            "right_image_shift": shift.tolist(), "vector": vector.tolist(),
                            "role": "covalent"})
        atoms.info["_molcrys_bond_records"] = records
        return MolecularCrystal.from_ase_atoms(atoms)

    start, end = endpoint(initial), endpoint(final)
    result = interpolate_periodic_paths(
        start, end, product_index_by_reactant=range(5),
        config=PeriodicPathConfig(n_images=9, max_candidates=1),
    )
    first, second = np.triu_indices(5, 1)
    _, reference = find_mic(start.to_ase().positions[second] - start.to_ase().positions[first], cell, pbc=True)
    for image in result.candidates[0].images:
        _, distances = find_mic(image.positions[second] - image.positions[first], cell, pbc=True)
        np.testing.assert_allclose(distances, reference, atol=1e-6)
