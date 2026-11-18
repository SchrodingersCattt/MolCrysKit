"""Concrete multi-species reactive endpoint paths."""

import json
from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from molcrys_kit.analysis.interactions import find_hydrogen_bonds
from molcrys_kit.operations import (
    FreeAtomArc,
    ReactivePathConfig,
    RigidGroup,
    build_bond_cleavage_path,
    build_proton_transfer_path,
    interpolate_reactive_path,
    write_reactive_path,
)
from molcrys_kit.operations.bond_rotation import RingBondRotationError
from molcrys_kit.structures.crystal import MolecularCrystal

CELL = [10.0, 10.0, 10.0]
# The extra spectator H makes the nuclear charge even, so multiplicity 1 is a
# closed-shell singlet (36 electrons at cell charge 0).
PROTON_SYMBOLS = ["N", "H", "C", "O", "C", "C", "H", "H"]
PROTON_POSITIONS = np.array(
    [
        [0.00, 0.00, 0.00],
        [1.01, 0.00, 0.00],
        [0.00, 1.47, 0.00],
        [3.20, 0.00, 0.00],
        [4.50, 0.00, 0.00],
        [0.00, 6.00, 0.00],
        [1.09, 6.00, 0.00],
        [0.00, 6.00, 1.09],
    ]
)
PROTON_MOLECULES = np.array([0, 0, 0, 1, 1, 2, 2, 2])
REACTANT_CHARGES = ((0, 1), (1, -1), (2, 0))
PRODUCT_CHARGES = ((0, 0), (1, 0), (2, 0))

CHAIN_SYMBOLS = ["C", "C", "C", "C", "N", "H"]
CHAIN_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 1.0, 0.0],
        [3.5, 1.0, 1.0],
        [0.0, 5.0, 0.0],
        [1.01, 5.0, 0.0],
    ]
)
CHAIN_MOLECULES = np.array([0, 0, 0, 0, 1, 1])
CHAIN_CHARGES = ((0, 0), (1, 0))
CHAIN_PRODUCT_CHARGES = ((0, 0), (1, 0), (2, 0))


def _crystal(symbols, positions, molecule_index) -> MolecularCrystal:
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=CELL,
        pbc=True,
    )
    atoms.set_array("molecule_index", np.asarray(molecule_index, dtype=int))
    return MolecularCrystal.from_ase_atoms(atoms)


def _proton_crystal(positions=PROTON_POSITIONS) -> MolecularCrystal:
    return _crystal(PROTON_SYMBOLS, positions, PROTON_MOLECULES)


def _chain_crystal() -> MolecularCrystal:
    return _crystal(CHAIN_SYMBOLS, CHAIN_POSITIONS, CHAIN_MOLECULES)


def _counts(symbols, molecule_index, mol_index) -> Counter:
    return Counter(
        symbols[index]
        for index, owner in enumerate(molecule_index)
        if int(owner) == mol_index
    )


def _proton_path(crystal=None, **overrides):
    kwargs = dict(
        donor_molecule=0,
        hydrogen_index=1,
        acceptor_molecule=1,
        acceptor_index=3,
        cell_charge=0,
        multiplicity=1,
        reactant_species_charges=REACTANT_CHARGES,
        product_species_charges=PRODUCT_CHARGES,
        config=ReactivePathConfig(n_images=5),
    )
    kwargs.update(overrides)
    return build_proton_transfer_path(crystal or _proton_crystal(), **kwargs)


def test_proton_transfer_moves_only_the_hydrogen_and_records_species():
    crystal = _proton_crystal()
    bonds = find_hydrogen_bonds(crystal)
    assert len(bonds) == 1
    assert bonds[0].donor.molecule_index == 0
    assert bonds[0].hydrogen.crystal_atom_index == 1
    assert bonds[0].acceptor.crystal_atom_index == 3

    before = crystal.to_ase().positions.copy()
    result = _proton_path(crystal)
    np.testing.assert_allclose(crystal.to_ase().positions, before)

    symbols = result.images[0].get_chemical_symbols()
    reactant_index = result.images[0].arrays["reactant_molecule_index"]
    product_index = result.metadata["product_molecule_index"]
    assert product_index == [0, 1, 0, 1, 1, 2, 2, 2]
    assert _counts(symbols, reactant_index, 0) == Counter({"N": 1, "C": 1, "H": 1})
    assert _counts(symbols, reactant_index, 1) == Counter({"O": 1, "C": 1})
    assert _counts(symbols, reactant_index, 2) == Counter({"C": 1, "H": 2})
    assert _counts(symbols, product_index, 0) == Counter({"N": 1, "C": 1})
    assert _counts(symbols, product_index, 1) == Counter({"O": 1, "C": 1, "H": 1})
    assert _counts(symbols, product_index, 2) == Counter({"C": 1, "H": 2})
    assert result.metadata["n_electrons"] == 36

    pairs = [tuple(record["atoms"]) for record in result.metadata["bond_changes"]]
    assert pairs == [(0, 1), (1, 3)]
    assert all(record["validated"] for record in result.metadata["bond_changes"])
    assert result.metadata["schema"] == "molcrys_kit.reactive_path.v3"
    assert result.metadata["cell_charge"] == 0
    assert result.metadata["multiplicity"] == 1

    expected_h = np.linspace(1.01, 2.24, 5)
    for frame, hydrogen_x in zip(result.images, expected_h):
        np.testing.assert_allclose(frame.positions[1, 0], hydrogen_x, atol=1e-8)
        np.testing.assert_allclose(frame.positions[1, 1:], 0.0, atol=1e-8)
        np.testing.assert_allclose(
            frame.positions[[0, 2, 3, 4, 5, 6, 7]],
            before[[0, 2, 3, 4, 5, 6, 7]],
            atol=1e-8,
        )
        assert frame.info["cell_charge"] == 0
        assert frame.info["multiplicity"] == 1
        assert frame.info["path_schema"] == "molcrys_kit.reactive_path.v3"

    for group in ((0, 2), (3, 4), (5, 6, 7)):
        reference_points = before[list(group)]
        reference = np.linalg.norm(
            reference_points[:, None, :] - reference_points[None, :, :],
            axis=-1,
        )
        for frame in result.images:
            current_points = frame.positions[list(group)]
            current = np.linalg.norm(
                current_points[:, None, :] - current_points[None, :, :],
                axis=-1,
            )
            np.testing.assert_allclose(current, reference, atol=1e-8)


def test_proton_transfer_round_trip_keeps_coordinates_and_charges(tmp_path):
    result = _proton_path()
    trajectory = tmp_path / "proton-transfer.extxyz"
    write_reactive_path(result, trajectory)
    frames = read(trajectory, index=":", format="extxyz")
    payload = json.loads(trajectory.with_suffix(".json").read_text(encoding="utf-8"))

    assert len(frames) == 5
    expected_h = np.linspace(1.01, 2.24, 5)
    for frame, hydrogen_x in zip(frames, expected_h):
        np.testing.assert_allclose(frame.positions[1, 0], hydrogen_x, atol=1e-8)
    assert [tuple(record["atoms"]) for record in payload["bond_changes"]] == [
        (0, 1),
        (1, 3),
    ]
    assert payload["schema"] == "molcrys_kit.reactive_path.v3"
    assert [
        (row["molecule_index"], row["formal_charge"])
        for row in payload["reactant_species"]
    ] == [(0, 1), (1, -1), (2, 0)]
    assert [
        (row["molecule_index"], row["formal_charge"])
        for row in payload["product_species"]
    ] == [(0, 0), (1, 0), (2, 0)]


def test_proton_transfer_arc_about_the_acceptor_keeps_the_endpoint():
    result = _proton_path(free_atom_arcs=[FreeAtomArc(1, 3)])
    delta = result.images[-1].positions[1] - result.images[-1].positions[3]
    assert float(np.linalg.norm(delta)) == pytest.approx(0.96, abs=1e-6)
    assert result.metadata["endpoint_max_error_A"] <= 1.0e-8


def test_proton_transfer_rejects_unbalanced_charge():
    with pytest.raises(ValueError, match="cell charge"):
        _proton_path(reactant_species_charges=((0, 1), (1, 0), (2, 0)))


def test_proton_transfer_rejects_spectator_charge_change():
    with pytest.raises(ValueError, match="spectator molecule 2"):
        _proton_path(product_species_charges=((0, -1), (1, 0), (2, 1)))


def test_proton_transfer_rejects_distant_acceptor():
    positions = PROTON_POSITIONS.copy()
    positions[3] = [6.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="hydrogen-bond limit"):
        _proton_path(_proton_crystal(positions))


def test_proton_transfer_rejects_hydrogen_on_another_molecule():
    with pytest.raises(ValueError, match="not on donor"):
        _proton_path(hydrogen_index=6)


def test_proton_transfer_rejects_an_unintended_new_bond():
    """A spectator that the moving hydrogen reaches is not a declared bond."""
    positions = np.vstack([PROTON_POSITIONS, [[2.24, 1.20, 0.0]]])
    crystal = _crystal(
        PROTON_SYMBOLS + ["C"],
        positions,
        np.concatenate([PROTON_MOLECULES, [3]]),
    )
    with pytest.raises(ValueError, match="endpoint connectivity changes"):
        _proton_path(
            crystal,
            reactant_species_charges=REACTANT_CHARGES + ((3, 0),),
            product_species_charges=PRODUCT_CHARGES + ((3, 0),),
        )


def test_bridge_cleavage_separates_the_moving_fragment():
    crystal = _chain_crystal()
    before = crystal.to_ase().positions.copy()
    result = build_bond_cleavage_path(
        crystal,
        molecule_index=0,
        atom_i=1,
        atom_j=2,
        separation_A=1.0,
        cell_charge=0,
        multiplicity=1,
        electron_pairing="closed_shell",
        reactant_species_charges=CHAIN_CHARGES,
        product_species_charges=CHAIN_PRODUCT_CHARGES,
        config=ReactivePathConfig(n_images=5),
    )

    symbols = result.images[0].get_chemical_symbols()
    reactant_index = result.images[0].arrays["reactant_molecule_index"]
    product_index = result.metadata["product_molecule_index"]
    assert product_index == [0, 0, 2, 2, 1, 1]
    assert _counts(symbols, reactant_index, 0) == Counter({"C": 4})
    assert _counts(symbols, reactant_index, 1) == Counter({"H": 1, "N": 1})
    assert _counts(symbols, product_index, 0) == Counter({"C": 2})
    assert _counts(symbols, product_index, 2) == Counter({"C": 2})
    assert _counts(symbols, product_index, 1) == Counter({"H": 1, "N": 1})
    assert result.metadata["electron_pairing"] == "closed_shell"
    assert [tuple(record["atoms"]) for record in result.metadata["bond_changes"]] == [
        (1, 2)
    ]

    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    start = before[2]
    bond_lengths = []
    for frame, lam in zip(result.images, np.linspace(0.0, 1.0, 5)):
        np.testing.assert_allclose(frame.positions[2], start + lam * axis, atol=1e-8)
        np.testing.assert_allclose(
            frame.positions[3], before[3] + lam * axis, atol=1e-8
        )
        np.testing.assert_allclose(
            frame.positions[[0, 1, 4, 5]], before[[0, 1, 4, 5]], atol=1e-8
        )
        bond_lengths.append(np.linalg.norm(frame.positions[2] - frame.positions[1]))
        fragment = np.linalg.norm(frame.positions[3] - frame.positions[2])
        np.testing.assert_allclose(fragment, np.sqrt(2.0), atol=1e-8)
        assert frame.info["electron_pairing"] == "closed_shell"
        assert frame.info["cell_charge"] == 0
        assert frame.info["multiplicity"] == 1
    np.testing.assert_allclose(
        bond_lengths,
        np.sqrt(2.0) + np.linspace(0.0, 1.0, 5),
        atol=1e-8,
    )


def test_ring_bond_cleavage_is_refused():
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ring = np.column_stack([1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)])
    positions = np.vstack([ring, [[0.0, 5.0, 0.0], [1.01, 5.0, 0.0]]])
    crystal = _crystal(["C"] * 6 + ["N", "H"], positions, [0] * 6 + [1, 1])
    with pytest.raises(RingBondRotationError, match="ring"):
        build_bond_cleavage_path(
            crystal,
            molecule_index=0,
            atom_i=0,
            atom_j=1,
            separation_A=1.0,
            cell_charge=0,
            multiplicity=1,
            electron_pairing="diradical",
            reactant_species_charges=((0, 0), (1, 0)),
            product_species_charges=((0, 0), (1, 0), (2, 0)),
        )


def test_cleavage_follows_requested_separation_beyond_half_cell():
    crystal = _crystal(
        ["C", "C"],
        [[0.0, 5.0, 5.0], [1.5, 5.0, 5.0]],
        [0, 0],
    )
    result = build_bond_cleavage_path(
        crystal,
        molecule_index=0,
        atom_i=0,
        atom_j=1,
        separation_A=6.0,
        cell_charge=0,
        multiplicity=1,
        electron_pairing="closed_shell",
        reactant_species_charges=((0, 0),),
        product_species_charges=((0, 0), (1, 0)),
        config=ReactivePathConfig(n_images=5),
    )
    start = np.array([1.5, 5.0, 5.0])
    fixed = np.array([0.0, 5.0, 5.0])
    distances = []
    for frame, lam in zip(result.images, np.linspace(0.0, 1.0, 5)):
        np.testing.assert_allclose(frame.positions[0], fixed, atol=1e-8)
        np.testing.assert_allclose(
            frame.positions[1], start + [lam * 6.0, 0.0, 0.0], atol=1e-8
        )
        distances.append(np.linalg.norm(frame.positions[1] - frame.positions[0]))
    np.testing.assert_allclose(distances, 1.5 + np.linspace(0.0, 6.0, 5), atol=1e-8)
    assert result.product_image_shifts[1] == (0, 0, 0)


def _odd_proton_crystal() -> MolecularCrystal:
    """Seven-atom cell with 35 electrons, before the spectator H that closes the shell."""
    return _crystal(
        PROTON_SYMBOLS[:7],
        PROTON_POSITIONS[:7],
        PROTON_MOLECULES[:7],
    )


def test_proton_transfer_rejects_a_charge_change_that_is_not_one_proton():
    with pytest.raises(ValueError, match="donor formal charge"):
        _proton_path(product_species_charges=((0, 1), (1, -1), (2, 0)))


def test_proton_transfer_rejects_singlet_with_odd_electron_count():
    with pytest.raises(ValueError, match="incompatible with 35 electrons"):
        _proton_path(_odd_proton_crystal(), multiplicity=1)


def test_proton_transfer_accepts_doublet_with_odd_electron_count():
    result = _proton_path(_odd_proton_crystal(), multiplicity=2)
    assert result.metadata["multiplicity"] == 2
    assert result.metadata["n_electrons"] == 35


def test_cleavage_rejects_doublet_with_even_electron_count():
    with pytest.raises(ValueError, match="incompatible with 32 electrons"):
        build_bond_cleavage_path(
            _chain_crystal(),
            molecule_index=0,
            atom_i=1,
            atom_j=2,
            separation_A=1.0,
            cell_charge=0,
            multiplicity=2,
            electron_pairing="diradical",
            reactant_species_charges=CHAIN_CHARGES,
            product_species_charges=CHAIN_PRODUCT_CHARGES,
        )


def test_cleavage_accepts_doublet_with_odd_electron_count():
    positions = np.vstack([CHAIN_POSITIONS, [[0.0, 5.0, 1.01]]])
    crystal = _crystal(
        CHAIN_SYMBOLS + ["H"],
        positions,
        np.concatenate([CHAIN_MOLECULES, [1]]),
    )
    result = build_bond_cleavage_path(
        crystal,
        molecule_index=0,
        atom_i=1,
        atom_j=2,
        separation_A=1.0,
        cell_charge=0,
        multiplicity=2,
        electron_pairing="diradical",
        reactant_species_charges=CHAIN_CHARGES,
        product_species_charges=CHAIN_PRODUCT_CHARGES,
        config=ReactivePathConfig(n_images=3),
    )
    assert result.metadata["n_electrons"] == 33
    assert result.metadata["multiplicity"] == 2


def test_cleavage_accepts_a_bridge_that_crosses_a_cell_face():
    crystal = _crystal(
        ["N", "H", "C"],
        [[9.5, 5.0, 5.0], [0.5, 5.0, 5.0], [5.0, 5.0, 1.0]],
        [0, 0, 1],
    )
    result = build_bond_cleavage_path(
        crystal,
        molecule_index=0,
        atom_i=0,
        atom_j=1,
        separation_A=1.0,
        cell_charge=0,
        multiplicity=1,
        electron_pairing="closed_shell",
        reactant_species_charges=((0, 0), (1, 0)),
        product_species_charges=((0, 0), (1, 0), (2, 0)),
        config=ReactivePathConfig(n_images=5),
    )

    def mic_distance(left, right) -> float:
        delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
        delta -= np.round(delta / 10.0) * 10.0
        return float(np.linalg.norm(delta))

    distances = [
        mic_distance(frame.positions[0], frame.positions[1]) for frame in result.images
    ]
    assert distances[0] == pytest.approx(1.0, abs=1e-6)
    assert distances[-1] == pytest.approx(2.0, abs=1e-6)
    assert distances == sorted(distances)
    np.testing.assert_allclose(
        result.images[0].positions[2], [5.0, 5.0, 1.0], atol=1e-8
    )
    np.testing.assert_allclose(
        result.images[-1].positions[2], [5.0, 5.0, 1.0], atol=1e-8
    )


def test_many_spectators_do_not_require_an_all_pairs_bond_scan(monkeypatch):
    symbols = list(PROTON_SYMBOLS)
    positions = [row.tolist() for row in PROTON_POSITIONS]
    molecules = list(PROTON_MOLECULES)
    next_molecule = int(PROTON_MOLECULES.max()) + 1
    for ix in range(8):
        for iy in range(8):
            symbols.append("C")
            positions.append([4.0 * ix, 4.0 * iy, 8.0])
            molecules.append(next_molecule)
            next_molecule += 1
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[40.0, 40.0, 16.0],
        pbc=True,
    )
    atoms.set_array("molecule_index", np.asarray(molecules, dtype=int))
    crystal = MolecularCrystal.from_ase_atoms(atoms)
    charges = REACTANT_CHARGES + tuple((index, 0) for index in range(3, next_molecule))
    product_charges = PRODUCT_CHARGES + tuple(
        (index, 0) for index in range(3, next_molecule)
    )

    import molcrys_kit.operations.reactive_path as reactive_path

    calls = {"n": 0}
    original = reactive_path._pair_is_bonded

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reactive_path, "_pair_is_bonded", _counting)
    result = build_proton_transfer_path(
        crystal,
        donor_molecule=0,
        hydrogen_index=1,
        acceptor_molecule=1,
        acceptor_index=3,
        cell_charge=0,
        multiplicity=1,
        reactant_species_charges=charges,
        product_species_charges=product_charges,
        config=ReactivePathConfig(n_images=3),
    )
    n_atoms = len(symbols)
    assert calls["n"] < 10 * n_atoms
    assert result.metadata["n_electrons"] == 36 + 6 * 64
    assert [tuple(record["atoms"]) for record in result.metadata["bond_changes"]] == [
        (0, 1),
        (1, 3),
    ]


def test_bond_cleavage_requires_electron_pairing():
    with pytest.raises(ValueError, match="electron_pairing"):
        build_bond_cleavage_path(
            _chain_crystal(),
            molecule_index=0,
            atom_i=1,
            atom_j=2,
            separation_A=1.0,
            cell_charge=0,
            multiplicity=1,
            reactant_species_charges=CHAIN_CHARGES,
            product_species_charges=CHAIN_PRODUCT_CHARGES,
        )


def _mic(cell, left, right) -> float:
    lattice = np.diag(np.asarray(cell, dtype=float))
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    frac = np.linalg.solve(lattice.T, delta)
    frac -= np.round(frac)
    return float(np.linalg.norm(lattice.T @ frac))


def test_cell_face_cleavage_keeps_cartesian_separation():
    cell = [10.0, 10.0, 10.0]
    # N at 9.5 and H at 0.5 are 1 Å apart through the cell face.
    positions = np.array(
        [
            [9.5, 5.0, 5.0],
            [0.5, 5.0, 5.0],
            [5.0, 1.0, 1.0],
            [6.1, 1.0, 1.0],
        ]
    )
    crystal = _crystal(["N", "H", "C", "C"], positions, [0, 0, 1, 1])
    result = build_bond_cleavage_path(
        crystal,
        molecule_index=0,
        atom_i=0,
        atom_j=1,
        separation_A=6.0,
        cell_charge=0,
        multiplicity=1,
        electron_pairing="closed_shell",
        reactant_species_charges=((0, 0), (1, 0)),
        product_species_charges=((0, 0), (1, 0), (2, 0)),
        config=ReactivePathConfig(n_images=5),
    )
    start = result.images[0].positions
    end = result.images[-1].positions
    cartesian = float(np.linalg.norm(end[1] - end[0]))
    assert cartesian == pytest.approx(7.0, abs=0.05)
    assert _mic(cell, end[0], end[1]) > _mic(cell, start[0], start[1])


def test_polar_arc_stays_on_the_unwrapped_pivot_image():
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array(
        [
            [0.20, 5.0, 5.0],
            [9.60, 5.0, 5.0],
            [9.70, 6.0, 5.0],
            [4.0, 1.0, 1.0],
            [5.2, 1.0, 1.0],
        ]
    )
    crystal = MolecularCrystal.from_ase_atoms(
        Atoms(
            symbols=["C", "C", "H", "O", "H"],
            positions=positions,
            cell=cell,
            pbc=True,
        )
    )
    # The pivot's bond walk crosses the cell; the hydrogen is stored next to
    # the wrapped pivot, 1 Å away, not next to the unwrapped pivot.
    result = interpolate_reactive_path(
        crystal,
        crystal,
        rigid_groups=[RigidGroup((0, 1), "chain"), RigidGroup((3, 4), "spectator")],
        free_atom_arcs=[FreeAtomArc(2, 1)],
        config=ReactivePathConfig(n_images=5, validate_bond_changes=False),
    )
    radii = []
    for frame in result.images:
        radii.append(_mic([10, 10, 10], frame.positions[2], frame.positions[1]))
    assert min(radii) == pytest.approx(max(radii), abs=0.05)
    assert min(radii) == pytest.approx(1.0, abs=0.15)


def test_product_image_shifts_reconstruct_unwrapped_frames():
    cell = np.diag([12.0, 12.0, 12.0])
    wrapped = np.array(
        [
            [0.4, 5.0, 5.0],
            [11.6, 5.0, 5.0],
            [0.4, 6.2, 5.0],
            [6.0, 1.0, 1.0],
            [7.2, 1.0, 1.0],
            [6.0, 2.2, 1.0],
        ]
    )
    atoms = Atoms(
        symbols=["C", "C", "C", "O", "O", "O"], positions=wrapped, cell=cell, pbc=True
    )
    atoms.set_array("molecule_index", np.array([0, 0, 0, 1, 1, 1]))
    crystal = MolecularCrystal.from_ase_atoms(atoms)
    result = interpolate_reactive_path(
        crystal,
        crystal,
        rigid_groups=[RigidGroup((0, 1, 2), "a"), RigidGroup((3, 4, 5), "b")],
        config=ReactivePathConfig(n_images=3, validate_bond_changes=False),
    )
    shifts = np.asarray(result.product_image_shifts, dtype=float)
    reconstructed = crystal.to_ase().positions + shifts @ cell
    error = np.max(np.linalg.norm(result.images[-1].positions - reconstructed, axis=1))
    assert error < 1e-6


def test_long_reactant_bond_is_not_a_new_contact():
    # Cs-N bonding threshold sits above the old 3.5 Å neighbour cutoff.
    # The pair is split across rigid groups so the contact diagnostic sees it.
    positions = np.array(
        [
            [0.0, 5.0, 5.0],
            [3.55, 5.0, 5.0],
            [0.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
        ]
    )
    product_positions = positions.copy()
    product_positions[1, 0] = 3.20
    cell = [12.0, 12.0, 12.0]

    def _packed(coords):
        atoms = Atoms(
            symbols=["Cs", "N", "O", "H"],
            positions=coords,
            cell=cell,
            pbc=True,
        )
        atoms.set_array("molecule_index", np.array([0, 1, 2, 2]))
        return MolecularCrystal.from_ase_atoms(atoms)

    result = interpolate_reactive_path(
        _packed(positions),
        _packed(product_positions),
        rigid_groups=[
            RigidGroup((0,), "donor"),
            RigidGroup((1,), "acceptor"),
            RigidGroup((2, 3), "spectator"),
        ],
        config=ReactivePathConfig(n_images=3, validate_bond_changes=False),
    )
    for image in result.metadata["images"]:
        contacts = [tuple(pair) for pair in image["new_contacts"]]
        assert (0, 1) not in contacts
