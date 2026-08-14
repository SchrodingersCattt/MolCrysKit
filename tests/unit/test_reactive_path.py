"""Tests for atom-mapped reactive initial paths."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.operations import (
    BondChange,
    ReactivePathConfig,
    RigidGroup,
    interpolate_reactive_path,
)
from molcrys_kit.structures.crystal import MolecularCrystal


CELL = np.array([[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.5, 0.7, 8.0]])
SYMBOLS = ["N", "C", "F", "H", "O", "S", "Cl"]
START = np.array(
    [
        [9.0, 5.0, 5.0],
        [9.0, 6.2, 5.0],
        [8.2, 5.0, 5.4],
        [9.8, 5.0, 5.0],
        [1.0, 5.0, 5.0],
        [1.0, 3.6, 5.0],
        [2.0, 5.0, 5.5],
    ]
)


def _crystal(positions, symbols=SYMBOLS, pbc=(True, True, True)):
    atoms = Atoms(symbols=symbols, positions=positions, cell=CELL, pbc=pbc)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=int))
    atoms.set_array("molecule_index", np.array([0, 0, 0, 0, 1, 1, 1]))
    return MolecularCrystal.from_ase_atoms(atoms)


def _rotate(points, angle_deg, translation):
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    center = np.mean(points, axis=0)
    return (points - center) @ rotation.T + center + np.asarray(translation)


def _endpoint_pair():
    target = START.copy()
    target[:3] = _rotate(START[:3], 20.0, [0.3, 0.1, 0.0])
    target[4:] = _rotate(START[4:], -15.0, [-0.2, 0.1, 0.0])
    target[3] = target[4] + np.array([-0.95, 0.0, 0.0])
    return _crystal(START), _crystal(target)


def _groups():
    return [RigidGroup((0, 1, 2), "donor"), RigidGroup((4, 5, 6), "acceptor")]


def test_two_rigid_groups_and_free_atom_reach_exact_endpoints():
    reactant, product = _endpoint_pair()
    result = interpolate_reactive_path(
        reactant,
        product,
        rigid_groups=_groups(),
        config=ReactivePathConfig(n_images=5, validate_bond_changes=False),
    )

    assert len(result.images) == 5
    np.testing.assert_allclose(result.images[0].to_ase().positions, reactant.to_ase().positions)
    shifts = np.asarray(result.product_image_shifts) @ CELL
    expected_end = product.to_ase().positions + shifts
    np.testing.assert_allclose(result.images[-1].to_ase().positions, expected_end, atol=1e-12)

    for group in ((0, 1, 2), (4, 5, 6)):
        reference = reactant.to_ase().get_all_distances(mic=False)[np.ix_(group, group)]
        for frame in result.images:
            current = frame.to_ase().get_all_distances(mic=False)[np.ix_(group, group)]
            np.testing.assert_allclose(current, reference, atol=1e-10)

    midpoint = result.images[2].to_ase().positions[3]
    np.testing.assert_allclose(
        midpoint,
        0.5 * (reactant.to_ase().positions[3] + expected_end[3]),
        atol=1e-12,
    )


def test_product_permutation_preserves_reactant_order():
    reactant, product = _endpoint_pair()
    product_atoms = product.to_ase()
    permutation = np.array([4, 5, 6, 3, 0, 1, 2])
    permuted = MolecularCrystal.from_ase_atoms(product_atoms[permutation])
    inverse = tuple(int(np.where(permutation == index)[0][0]) for index in range(len(permutation)))

    result = interpolate_reactive_path(
        reactant,
        permuted,
        rigid_groups=_groups(),
        product_index_by_reactant=inverse,
        config=ReactivePathConfig(n_images=3, validate_bond_changes=False),
    )
    assert result.images[-1].to_ase().get_chemical_symbols() == SYMBOLS
    np.testing.assert_array_equal(result.images[-1].to_ase().arrays["atom_id"], np.arange(7))


def test_free_atom_uses_continuous_periodic_image():
    reactant = _crystal(START)
    target = START.copy()
    target[3] = [0.2, 5.0, 5.0]
    product = _crystal(target)
    result = interpolate_reactive_path(
        reactant,
        product,
        rigid_groups=_groups(),
        config=ReactivePathConfig(n_images=5, validate_bond_changes=False),
    )
    x = np.array([frame.to_ase().positions[3, 0] for frame in result.images])
    np.testing.assert_allclose(x, [9.8, 9.9, 10.0, 10.1, 10.2], atol=1e-12)
    assert result.product_image_shifts[3] == (1, 0, 0)


def test_declared_bond_break_and_formation_are_validated():
    initial = START.copy()
    initial[4:, 0] += 0.8
    reactant = _crystal(initial)
    target = initial.copy()
    target[3] = [0.85, 5.0, 5.0]
    product = _crystal(target)
    result = interpolate_reactive_path(
        reactant,
        product,
        rigid_groups=_groups(),
        bond_changes=[
            BondChange(0, 3, True, False),
            BondChange(3, 4, False, True),
        ],
        config=ReactivePathConfig(n_images=3),
    )
    assert all(record["validated"] for record in result.metadata["bond_changes"])


def test_non_rigid_group_is_rejected():
    reactant, product = _endpoint_pair()
    target = product.to_ase()
    target.positions[1, 0] += 0.2
    deformed = MolecularCrystal.from_ase_atoms(target)
    with pytest.raises(ValueError, match="not rigid-reachable"):
        interpolate_reactive_path(
            reactant,
            deformed,
            rigid_groups=_groups(),
            config=ReactivePathConfig(validate_bond_changes=False),
        )


@pytest.mark.parametrize(
    "groups, message",
    [
        ([RigidGroup((0, 1, 2))], "at least two"),
        ([RigidGroup(()), RigidGroup((4, 5, 6))], "empty"),
        ([RigidGroup((0, 1, 2)), RigidGroup((2, 4, 5))], "overlap"),
        ([RigidGroup((0, 0)), RigidGroup((4, 5, 6))], "duplicate"),
    ],
)
def test_invalid_rigid_groups_are_rejected(groups, message):
    reactant, product = _endpoint_pair()
    with pytest.raises(ValueError, match=message):
        interpolate_reactive_path(
            reactant,
            product,
            rigid_groups=groups,
            config=ReactivePathConfig(validate_bond_changes=False),
        )


def test_inputs_and_atom_partition_are_preserved():
    reactant, product = _endpoint_pair()
    for crystal in (reactant, product):
        atoms = crystal.to_ase()
        scaled = atoms.get_scaled_positions(wrap=False)
        atoms.set_array("frac_x", scaled[:, 0])
        atoms.set_array("frac_y", scaled[:, 1])
        atoms.set_array("frac_z", scaled[:, 2])
        rebuilt = MolecularCrystal.from_ase_atoms(atoms)
        if crystal is reactant:
            reactant = rebuilt
        else:
            product = rebuilt
    positions_before = reactant.to_ase().positions.copy()
    result = interpolate_reactive_path(
        reactant,
        product,
        rigid_groups=_groups(),
        config=ReactivePathConfig(n_images=3, validate_bond_changes=False),
    )
    np.testing.assert_allclose(reactant.to_ase().positions, positions_before, atol=0.0)
    expected_partition = reactant.to_ase().arrays["molecule_index"]
    for frame in result.images:
        atoms = frame.to_ase()
        assert atoms.get_chemical_symbols() == SYMBOLS
        np.testing.assert_array_equal(atoms.arrays["molecule_index"], expected_partition)
        np.testing.assert_array_equal(atoms.arrays["atom_id"], np.arange(7))
        assert not {"frac_x", "frac_y", "frac_z"}.intersection(atoms.arrays)
        assert frame.metadata["path_kind"] == "reactive"


def test_endpoint_exclusion_uses_existing_frame_count_semantics():
    reactant, product = _endpoint_pair()
    config = ReactivePathConfig(n_images=4, include_endpoints=False, validate_bond_changes=False)
    result = interpolate_reactive_path(reactant, product, rigid_groups=_groups(), config=config)
    assert len(result.images) == 4
    lambdas = [frame.metadata["path_lambda"] for frame in result.images]
    np.testing.assert_allclose(lambdas, [0.2, 0.4, 0.6, 0.8])
