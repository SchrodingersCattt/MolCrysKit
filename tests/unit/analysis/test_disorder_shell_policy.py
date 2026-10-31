"""Disorder policy for molecule identity and site-centred ligand shells.

The fixtures are synthetic. A molecular cation is disordered over alternative
orientations that share an ordered pivot, so bond perception would otherwise
fuse them into one invalid molecule. Ligands form an octahedron around the
major orientation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.analysis.disorder.diagnostics import UnresolvedDisorderWarning
from molcrys_kit.analysis.disorder.policy import (
    disorder_realizations,
    normalize_disorder_policy,
)
from molcrys_kit.analysis.packing_shell import find_polyhedra
from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_OCCUPANCY,
)
from molcrys_kit.io.cif import (
    identify_molecule_indices,
    identify_molecules,
    read_mol_crystal,
)
from molcrys_kit.structures.crystal import MolecularCrystal

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CELL = 20.0
SITE = np.array([10.0, 10.0, 10.0])
LIGAND_DISTANCE = 5.0
SEARCH_RADIUS = 8.0


def _tetrahedral_offsets(distance: float = 1.45) -> np.ndarray:
    raw = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    raw /= np.linalg.norm(raw, axis=1)[:, None]
    return raw * distance


def _rotate_z(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return points @ rotation.T


def _orientation(degrees: float, shift) -> np.ndarray:
    return _rotate_z(_tetrahedral_offsets(), degrees) + np.asarray(shift, dtype=float)


def _ligand_positions(site: np.ndarray) -> list[np.ndarray]:
    positions = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            offset = np.zeros(3)
            offset[axis] = sign * LIGAND_DISTANCE
            positions.append(site + offset)
    return positions


def _cation_atoms(orientations: list[tuple[np.ndarray, int, float]]) -> Atoms:
    """N at the site plus one carbon set per orientation.

    The nitrogen is ordered (group 0, full occupancy) and is within bonding
    distance of every carbon, so the orientations fuse unless a disorder
    policy separates them.
    """
    symbols = ["N"]
    positions = [SITE.copy()]
    groups = [0]
    occupancies = [1.0]
    assemblies = [""]
    for offsets, group, occupancy in orientations:
        for offset in offsets:
            symbols.append("C")
            positions.append(SITE + np.asarray(offset, dtype=float))
            groups.append(int(group))
            occupancies.append(float(occupancy))
            assemblies.append("")
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.asarray(groups, dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_ASSEMBLY, np.asarray(assemblies, dtype=object))
    return atoms


def _ligand_atoms() -> list[Atoms]:
    cell = np.eye(3) * CELL
    return [
        Atoms(symbols=["Cl"], positions=[position], cell=cell, pbc=True)
        for position in _ligand_positions(SITE)
    ]


def _crystal_with(*molecules: Atoms) -> MolecularCrystal:
    return MolecularCrystal(np.eye(3) * CELL, list(molecules), pbc=(True, True, True))


def _shells(crystal: MolecularCrystal, **kwargs):
    return find_polyhedra(
        crystal,
        "N C4",
        "Cl",
        level="molecule",
        cutoff=SEARCH_RADIUS,
        score_shape=True,
        **kwargs,
    )


def _assert_octahedron(record) -> None:
    assert record["coordination_number"] == 6
    assert record["enclosed"] is True
    assert record["center_formula"] == "C4N"
    assert record["shell_formula"] == "Cl"
    assert record["best_match"]["name"] == "octahedron"
    assert record["best_match"]["angular_rmsd"] < 1e-6
    assert np.allclose(record["center_position"], SITE, atol=1e-6)
    assert np.allclose(record["shell_distances"], LIGAND_DISTANCE, atol=1e-6)


def test_major_policy_recovers_ligand_shell_around_disordered_cation():
    major = _orientation(0.0, (0.0, 0.0, 0.0))
    minor = _orientation(55.0, (0.25, 0.0, 0.0))
    cation = _cation_atoms([(major, 1, 0.65), (minor, 2, 0.35)])
    crystal = _crystal_with(cation, *_ligand_atoms())

    fused = _shells(crystal, disorder_policy="as_refined")
    shells = _shells(crystal)

    assert fused == []
    assert len(shells) == 1
    _assert_octahedron(shells[0])
    assert shells[0]["disorder_policy"] == "major"
    assert shells[0]["disorder_component_ids"] == ["group:1"]


def test_ordered_cation_shell_matches_undisordered_control():
    cation = Atoms(
        symbols=["N", "C", "C", "C", "C"],
        positions=np.vstack([SITE, SITE + _tetrahedral_offsets()]),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    crystal = _crystal_with(cation, *_ligand_atoms())

    default = _shells(crystal)
    explicit = _shells(crystal, disorder_policy="as_refined")
    exclusive = _shells(crystal, disorder_policy="exclusive")

    assert len(default) == len(explicit) == len(exclusive) == 1
    for record in (default[0], explicit[0], exclusive[0]):
        _assert_octahedron(record)
        assert record["disorder_component_ids"] == []


def test_multi_component_occupancy_selects_highest():
    orientations = [
        (_orientation(0.0, (0.0, 0.0, 0.0)), 1, 0.55),
        (_orientation(55.0, (0.25, 0.0, 0.0)), 2, 0.30),
        (_orientation(-40.0, (0.0, 0.20, 0.0)), 3, 0.15),
    ]
    cation = _cation_atoms(orientations)
    # Occupancy is the selection key even when disorder groups are absent.
    cation.set_array(KEY_DISORDER_GROUP, np.zeros(len(cation), dtype=int))
    crystal = _crystal_with(cation, *_ligand_atoms())

    shells = _shells(crystal)
    realization = disorder_realizations(cation, "major")[0]
    kept_symbols = [
        symbol
        for symbol, keep in zip(cation.get_chemical_symbols(), realization.mask)
        if keep
    ]

    assert realization.drops_atoms is True
    assert realization.component_ids == ("occupancy:0.55000",)
    assert sorted(kept_symbols) == ["C", "C", "C", "C", "N"]
    assert len(shells) == 1
    _assert_octahedron(shells[0])
    assert shells[0]["disorder_component_ids"] == ["occupancy:0.55000"]


def test_exclusive_policy_emits_one_shell_per_orientation():
    major = _orientation(0.0, (0.0, 0.0, 0.0))
    minor = _orientation(55.0, (0.25, 0.0, 0.0))
    third = _orientation(-40.0, (0.0, 0.20, 0.0))
    cation = _cation_atoms([(major, 1, 0.55), (minor, 2, 0.30), (third, 3, 0.15)])
    crystal = _crystal_with(cation, *_ligand_atoms())

    shells = _shells(crystal, disorder_policy="exclusive")

    assert len(shells) == 3
    by_component = {record["disorder_component_ids"][0]: record for record in shells}
    assert set(by_component) == {"group:1", "group:2", "group:3"}
    _assert_octahedron(by_component["group:1"])
    assert by_component["group:2"]["coordination_number"] >= 4
    assert by_component["group:3"]["coordination_number"] >= 4


def test_separate_orientations_at_one_site_keep_the_major_shell():
    def _oriented(origin, group: int, occupancy: float) -> Atoms:
        atoms = Atoms(
            symbols=["N", "C", "C", "C", "C"],
            positions=np.vstack([origin, origin + _tetrahedral_offsets()]),
            cell=np.eye(3) * CELL,
            pbc=True,
        )
        atoms.set_array(KEY_DISORDER_GROUP, np.full(5, group, dtype=int))
        atoms.set_array(KEY_OCCUPANCY, np.full(5, occupancy, dtype=float))
        atoms.set_array(KEY_ASSEMBLY, np.array([""] * 5, dtype=object))
        return atoms

    major = _oriented(SITE, group=1, occupancy=0.70)
    minor = _oriented(SITE + np.array([0.30, 0.0, 0.0]), group=2, occupancy=0.30)
    crystal = _crystal_with(major, minor, *_ligand_atoms())

    collapsed = _shells(crystal, disorder_policy="major")
    each = _shells(crystal, disorder_policy="exclusive")

    assert len(collapsed) == 1
    _assert_octahedron(collapsed[0])
    assert collapsed[0]["center_molecule_index"] == 0
    assert len(each) == 2
    assert {record["center_molecule_index"] for record in each} == {0, 1}


def test_ordered_neighbours_are_not_collapsed_by_the_major_policy():
    def _plain(origin) -> Atoms:
        return Atoms(
            symbols=["N", "C", "C", "C", "C"],
            positions=np.vstack([origin, origin + _tetrahedral_offsets()]),
            cell=np.eye(3) * CELL,
            pbc=True,
        )

    crystal = _crystal_with(
        _plain(SITE),
        _plain(SITE + np.array([6.0, 0.0, 0.0])),
        *_ligand_atoms(),
    )

    shells = _shells(crystal)

    assert len(shells) == 2
    at_site = min(
        shells,
        key=lambda record: np.linalg.norm(np.asarray(record["center_position"]) - SITE),
    )
    _assert_octahedron(at_site)


def test_identify_molecules_major_policy_does_not_fuse_orientations():
    major = _orientation(0.0, (0.0, 0.0, 0.0))
    minor = _orientation(55.0, (0.25, 0.0, 0.0))
    atoms = _cation_atoms([(major, 1, 0.65), (minor, 2, 0.35)])

    fused = identify_molecules(atoms, disorder_policy="as_refined")
    selected = identify_molecules(atoms, disorder_policy="major")
    each = identify_molecules(atoms, disorder_policy="exclusive")

    assert len(fused) == 1
    assert len(fused[0]) == 9
    assert len(selected) == 1
    assert sorted(selected[0].get_chemical_symbols()) == ["C", "C", "C", "C", "N"]
    assert selected[0].info["disorder_component_ids"] == ["group:1"]
    assert len(selected[0].graph.edges()) == 4
    assert len(each) == 2
    assert all(len(molecule) == 5 for molecule in each)
    assert all(0 in molecule.info["atom_indices"] for molecule in each)
    assert {molecule.info["disorder_component_ids"][0] for molecule in each} == {
        "group:1",
        "group:2",
    }


def test_unlabelled_groups_stay_independent_when_they_do_not_overlap():
    atoms = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.array([1, 2], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([0.80, 0.20]))
    atoms.set_array(KEY_ASSEMBLY, np.array(["", ""], dtype=object))

    realization = disorder_realizations(atoms, "major")[0]

    assert realization.drops_atoms is False
    assert realization.mask.tolist() == [True, True]


def test_labelled_assembly_selects_the_major_group_without_a_contact():
    atoms = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.array([1, 2], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([0.80, 0.20]))
    atoms.set_array(KEY_ASSEMBLY, np.array(["A", "A"], dtype=object))

    realization = disorder_realizations(atoms, "major")[0]

    assert realization.component_ids == ("assembly:A:group:1",)
    assert realization.mask.tolist() == [True, False]


def test_overlapping_unlabelled_groups_keep_the_major_component():
    atoms = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.array([1, 2], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([0.70, 0.30]))
    atoms.set_array(KEY_ASSEMBLY, np.array(["", ""], dtype=object))

    realization = disorder_realizations(atoms, "major")[0]

    assert realization.component_ids == ("group:1",)
    assert realization.mask.tolist() == [True, False]


def test_atom_level_major_policy_skips_minor_assembly_ligands():
    atoms = Atoms(
        symbols=["N", "Cl", "Cl"],
        positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [2.4, 0.0, 0.0]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    atoms.set_array(KEY_ASSEMBLY, np.array(["", "A", "A"], dtype=object))
    atoms.set_array(KEY_DISORDER_GROUP, np.array([0, 1, 2], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([1.0, 0.75, 0.25]))

    major = find_polyhedra(atoms, "N", "Cl", cutoff=3.0, disorder_policy="major")
    refined = find_polyhedra(atoms, "N", "Cl", cutoff=3.0, disorder_policy="as_refined")
    each = find_polyhedra(atoms, "N", "Cl", cutoff=3.0, disorder_policy="exclusive")

    # Minor ligand positions are not shell vertices under any policy.
    for record in (major[0], refined[0], each[0]):
        assert record["coordination_number"] == 1
        assert record["shell_indices"] == [1]


def test_identify_molecules_default_keeps_every_refined_atom():
    """General identification must not drop atoms unless the caller asks."""
    major = _orientation(0.0, (0.0, 0.0, 0.0))
    minor = _orientation(55.0, (0.25, 0.0, 0.0))
    atoms = _cation_atoms([(major, 1, 0.65), (minor, 2, 0.35)])

    molecules = identify_molecules(atoms)
    indices = identify_molecule_indices(atoms)
    covered = [index for group in indices for index in group]

    assert len(molecules) == 1
    assert len(molecules[0]) == len(atoms)
    assert sorted(covered) == list(range(len(atoms)))
    assert len(covered) == len(set(covered))


def test_far_apart_unlabelled_parts_keep_each_sites_major():
    """The same PART labels at two distant sites are independent choices."""
    symbols = ["C", "C", "C", "C", "C", "C", "C", "C"]
    positions = [
        [0.0, 0.0, 0.0],  # site A, part 1, occ 0.8
        [0.30, 0.0, 0.0],  # site A, part 2, occ 0.2
        [1.6, 0.0, 0.0],
        [3.1, 0.0, 0.0],
        [4.6, 0.0, 0.0],
        [6.1, 0.0, 0.0],
        [8.0, 0.0, 0.0],  # site B, part 1, occ 0.3
        [8.30, 0.0, 0.0],  # site B, part 2, occ 0.7
    ]
    groups = [1, 2, 0, 0, 0, 0, 1, 2]
    occupancies = [0.8, 0.2, 1.0, 1.0, 1.0, 1.0, 0.3, 0.7]
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.asarray(groups, dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * len(atoms), dtype=object))

    realization = disorder_realizations(atoms, "major")[0]

    assert realization.mask.tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    assert "group:1:site:0" in realization.component_ids
    assert "group:2:site:7" in realization.component_ids


def test_partial_occupancy_bonds_are_not_alternatives():
    """Unequal occupancy on a real bond is not an alternative orientation."""
    water = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    water.set_array(KEY_OCCUPANCY, np.array([0.50, 0.50, 0.25]))
    water.set_array(KEY_DISORDER_GROUP, np.zeros(3, dtype=int))
    water.set_array(KEY_ASSEMBLY, np.array(["", "", ""], dtype=object))

    chain_positions = [[index * 1.4, 0.0, 0.0] for index in range(6)]
    chain = Atoms(
        symbols=["C"] * 6,
        positions=chain_positions,
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    chain.set_array(KEY_OCCUPANCY, np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4]))
    chain.set_array(KEY_DISORDER_GROUP, np.zeros(6, dtype=int))
    chain.set_array(KEY_ASSEMBLY, np.array([""] * 6, dtype=object))

    assert disorder_realizations(water, "major")[0].drops_atoms is False
    assert disorder_realizations(chain, "major")[0].drops_atoms is False
    assert disorder_realizations(water, "exclusive")[0].drops_atoms is False


def test_equal_occupancy_alternatives_keep_the_lower_index():
    """50:50 split sites still produce one drawable shell.

    The two orientations share an occupancy and the same angle spread, so the
    tie keeps the shell closer to the centre. In this fixture that is the
    inner set.
    """
    major = _tetrahedral_offsets(1.45)
    minor = _tetrahedral_offsets(1.80)
    symbols = ["N"]
    positions = [SITE.copy()]
    occupancies = [1.0]
    for offset in major:
        symbols.append("C")
        positions.append(SITE + offset)
        occupancies.append(0.5)
    for offset in minor:
        symbols.append("C")
        positions.append(SITE + offset)
        occupancies.append(0.5)
    cation = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    cation.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    cation.set_array(KEY_DISORDER_GROUP, np.zeros(len(cation), dtype=int))
    cation.set_array(KEY_ASSEMBLY, np.array([""] * len(cation), dtype=object))
    crystal = _crystal_with(cation, *_ligand_atoms())

    realization = disorder_realizations(cation, "major")[0]
    shells = _shells(crystal)

    assert realization.drops_atoms is True
    assert realization.mask.tolist() == [True, True, True, True, True, False, False, False, False]
    assert len(shells) == 1
    _assert_octahedron(shells[0])


def test_minor_ligand_position_is_not_a_shell_vertex():
    cation = Atoms(
        symbols=["N", "C", "C", "C", "C"],
        positions=np.vstack([SITE, SITE + _tetrahedral_offsets()]),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    cell = np.eye(3) * CELL
    ordered_offsets = [
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ]
    ligands = [
        Atoms(symbols=["Cl"], positions=[SITE + np.asarray(offset)], cell=cell, pbc=True)
        for offset in ordered_offsets
    ]
    # The +x vertex is disordered. The minor image sits 0.35 Å farther out
    # and must not become a seventh shell vertex.
    major_ligand = Atoms(
        symbols=["Cl"],
        positions=[SITE + np.array([5.0, 0.0, 0.0])],
        cell=cell,
        pbc=True,
    )
    major_ligand.set_array(KEY_DISORDER_GROUP, np.array([1], dtype=int))
    major_ligand.set_array(KEY_OCCUPANCY, np.array([0.6]))
    major_ligand.set_array(KEY_ASSEMBLY, np.array([""], dtype=object))
    minor = Atoms(
        symbols=["Cl"],
        positions=[SITE + np.array([5.35, 0.0, 0.0])],
        cell=cell,
        pbc=True,
    )
    minor.set_array(KEY_DISORDER_GROUP, np.array([2], dtype=int))
    minor.set_array(KEY_OCCUPANCY, np.array([0.4]))
    minor.set_array(KEY_ASSEMBLY, np.array([""], dtype=object))
    crystal = _crystal_with(cation, *ligands, major_ligand, minor)

    for policy in ("major", "exclusive", "as_refined"):
        shells = _shells(crystal, disorder_policy=policy)
        assert len(shells) == 1
        assert shells[0]["coordination_number"] == 6
        assert max(shells[0]["shell_distances"]) < 5.2


def _occupancy_cation(rotation_deg: float, first: float, second: float, *, interleave: bool = False) -> Atoms:
    """Ordered N plus two tetrahedral carbon sets, group 0, given occupancies."""
    primary = _tetrahedral_offsets(1.45)
    secondary = _rotate_z(primary, rotation_deg)
    symbols = ["N"]
    positions = [SITE.copy()]
    occupancies = [1.0]
    if interleave:
        for left, right in zip(primary, secondary):
            symbols.extend(["C", "C"])
            positions.extend([SITE + left, SITE + right])
            occupancies.extend([first, second])
    else:
        for offset in primary:
            symbols.append("C")
            positions.append(SITE + offset)
            occupancies.append(first)
        for offset in secondary:
            symbols.append("C")
            positions.append(SITE + offset)
            occupancies.append(second)
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(len(atoms), dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * len(atoms), dtype=object))
    return atoms


def test_rotated_occupancy_only_orientations_have_a_shell():
    """60° and 90° duplicates are still one shell, not an empty match."""
    for rotation in (60.0, 90.0):
        cation = _occupancy_cation(rotation, 0.62, 0.38)
        shells = _shells(_crystal_with(cation, *_ligand_atoms()))
        assert len(shells) == 1, rotation
        _assert_octahedron(shells[0])
        kept = disorder_realizations(cation, "major")[0]
        assert int(np.count_nonzero(kept.mask)) == 5


def test_occupancy_sum_slightly_above_one_still_selects_major():
    cation = _occupancy_cation(25.0, 0.55, 0.50)
    shells = _shells(_crystal_with(cation, *_ligand_atoms()))
    assert len(shells) == 1
    _assert_octahedron(shells[0])


def test_interleaved_equal_occupancy_keeps_one_orientation():
    cation = _occupancy_cation(20.0, 0.5, 0.5, interleave=True)
    major = disorder_realizations(cation, "major")[0]
    each = disorder_realizations(cation, "exclusive")
    primary = _tetrahedral_offsets(1.45)
    secondary = _rotate_z(primary, 20.0)
    kept_positions = cation.get_positions()[major.mask][1:]  # drop N
    primary_at_site = SITE + primary
    secondary_at_site = SITE + secondary

    def _matches(reference) -> bool:
        if len(kept_positions) != len(reference):
            return False
        unused = list(reference)
        for position in kept_positions:
            distances = [np.linalg.norm(position - other) for other in unused]
            nearest = int(np.argmin(distances))
            if distances[nearest] > 1e-6:
                return False
            unused.pop(nearest)
        return not unused

    assert _matches(primary_at_site) or _matches(secondary_at_site)
    # Exactly one pure orientation, not a mix and not one swap per atom.
    assert len(each) == 2
    for realization in each:
        carbons = int(np.count_nonzero(realization.mask)) - 1
        assert carbons == 4


def test_six_tetrahedral_orientations_keep_four_hydrogens():
    symbols = ["N"]
    positions = [np.zeros(3)]
    occupancies = [1.0]
    for step in range(6):
        for offset in _rotate_z(_tetrahedral_offsets(1.01), step * 30.0):
            symbols.append("H")
            positions.append(offset)
            occupancies.append(1.0 / 6.0)
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(len(atoms), dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * len(atoms), dtype=object))

    major = disorder_realizations(atoms, "major")[0]
    each = disorder_realizations(atoms, "exclusive")
    kept_h = [
        symbol
        for symbol, keep in zip(atoms.get_chemical_symbols(), major.mask)
        if keep and symbol == "H"
    ]
    assert len(kept_h) == 4
    assert len(each) == 6
    assert all(
        sum(symbol == "H" for symbol, keep in zip(atoms.get_chemical_symbols(), item.mask) if keep)
        == 4
        for item in each
    )


def test_unlabelled_minor_ligand_is_not_an_atom_shell_vertex():
    """A nearby unlabelled minor ligand must not add a shell vertex."""
    positions = [[0.0, 0.0, 0.0]]
    symbols = ["N"]
    groups = [0]
    occupancies = [1.0]
    assemblies = [""]
    ligand_offsets = [
        [2.6, 0.0, 0.0],
        [-2.6, 0.0, 0.0],
        [0.0, 2.6, 0.0],
        [0.0, -2.6, 0.0],
        [0.0, 0.0, 2.6],
        [0.0, 0.0, -2.6],
    ]
    for offset in ligand_offsets:
        symbols.append("Cl")
        positions.append(offset)
        groups.append(0)
        occupancies.append(1.0)
        assemblies.append("")
    # Minor image of the +x ligand, unlabelled assembly, PART 2.
    symbols.append("Cl")
    positions.append([2.95, 0.0, 0.0])
    groups.append(2)
    occupancies.append(0.4)
    assemblies.append("")
    groups[1] = 1
    occupancies[1] = 0.6
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * 16.0,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.asarray(groups, dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_ASSEMBLY, np.asarray(assemblies, dtype=object))

    for policy in ("major", "exclusive", "as_refined"):
        records = find_polyhedra(
            atoms,
            central="N",
            ligand="Cl",
            cutoff=3.5,
            disorder_policy=policy,
        )
        assert len(records) == 1
        assert records[0]["coordination_number"] == 6
        assert 7 not in records[0]["shell_indices"]


@pytest.mark.parametrize(
    ("metal", "first", "second"),
    [("Na", 0.5, 0.5), ("Zn", 0.5, 0.5), ("Ag", 0.6, 0.4)],
)
def test_partial_metals_bridging_ordered_oxygen_stay(metal, first, second):
    atoms = Atoms(
        symbols=["C", "O", metal, metal],
        positions=[
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [1.3, 2.2, 0.0],
            [1.3, -2.2, 0.0],
        ],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([1.0, 1.0, first, second]))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(4, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * 4, dtype=object))

    realization = disorder_realizations(atoms, "major")[0]

    assert realization.drops_atoms is False
    assert realization.mask.tolist() == [True, True, True, True]


def test_coordinated_water_keeps_two_hydrogens():
    oxygen = np.zeros(3)
    hydrogens = [
        oxygen + 0.97 * np.array([0.0, np.cos(index * np.pi / 2), np.sin(index * np.pi / 2)])
        for index in range(4)
    ]
    atoms = Atoms(
        symbols=["Zn", "O", "H", "H", "H", "H"],
        positions=np.vstack([[-2.1, 0.0, 0.0], oxygen, *hydrogens]),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.5]))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(6, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * 6, dtype=object))

    kept = [
        symbol
        for symbol, keep in zip(
            atoms.get_chemical_symbols(),
            disorder_realizations(atoms, "major")[0].mask,
        )
        if keep
    ]

    assert kept.count("H") == 2
    assert "O" in kept and "Zn" in kept


def test_bonded_partial_atoms_in_a_tight_ring_stay():
    angle = np.deg2rad(60.0)
    atoms = Atoms(
        symbols=["C", "C", "C"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [1.4 * np.cos(angle), 1.4 * np.sin(angle), 0.0],
        ],
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([1.0, 0.6, 0.4]))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(3, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(["", "", ""], dtype=object))

    assert disorder_realizations(atoms, "major")[0].drops_atoms is False


def test_shifted_equal_occupancy_orientation_still_has_a_shell():
    primary = _tetrahedral_offsets(1.45)
    secondary = _rotate_z(primary, 50.0) + np.array([0.0, 0.0, 0.15])
    symbols = ["N"]
    positions = [SITE.copy()]
    occupancies = [1.0]
    for offset in (*primary, *secondary):
        symbols.append("C")
        positions.append(SITE + offset)
        occupancies.append(0.5)
    cation = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    cation.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    cation.set_array(KEY_DISORDER_GROUP, np.zeros(len(cation), dtype=int))
    cation.set_array(KEY_ASSEMBLY, np.array([""] * len(cation), dtype=object))

    shells = _shells(_crystal_with(cation, *_ligand_atoms()))
    each = disorder_realizations(cation, "exclusive")

    assert len(shells) == 1
    _assert_octahedron(shells[0])
    assert len(each) == 2


def test_unresolved_equal_occupancy_class_warns_and_keeps_atoms():
    oxygen = np.zeros(3)
    # Five hydrogens in a tight cone cannot form two regular shells.
    hydrogens = [
        oxygen + np.array([0.97, 0.02 * index, 0.0])
        for index in range(5)
    ]
    atoms = Atoms(
        symbols=["O", "H", "H", "H", "H", "H"],
        positions=np.vstack([oxygen, *hydrogens]),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5]))
    atoms.set_array(KEY_DISORDER_GROUP, np.zeros(6, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * 6, dtype=object))

    with pytest.warns(UnresolvedDisorderWarning):
        realization = disorder_realizations(atoms, "major")[0]

    assert realization.drops_atoms is False


def test_negative_part_tetrahedral_shell_keeps_four_hydrogens():
    symbols = ["N"]
    positions = [np.zeros(3)]
    groups = [0]
    occupancies = [1.0]
    for step in range(6):
        for offset in _rotate_z(_tetrahedral_offsets(1.01), step * 30.0):
            symbols.append("H")
            positions.append(offset)
            groups.append(-1)
            occupancies.append(1.0 / 6.0)
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3) * CELL,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.asarray(groups, dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    atoms.set_array(KEY_ASSEMBLY, np.array([""] * len(atoms), dtype=object))

    kept = disorder_realizations(atoms, "major")[0]
    hydrogen_count = sum(
        symbol == "H" and bool(keep)
        for symbol, keep in zip(atoms.get_chemical_symbols(), kept.mask)
    )

    assert hydrogen_count == 4


def test_periodic_boundary_pair_is_one_ligand_site():
    atoms = Atoms(
        symbols=["Cl", "Cl"],
        positions=[[0.2, 5.0, 5.0], [9.8, 5.0, 5.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.set_array(KEY_DISORDER_GROUP, np.array([1, 2], dtype=int))
    atoms.set_array(KEY_OCCUPANCY, np.array([0.6, 0.4]))
    atoms.set_array(KEY_ASSEMBLY, np.array(["", ""], dtype=object))

    folded = disorder_realizations(atoms, "major", minimum_image=True)[0]
    separate = disorder_realizations(atoms, "major", minimum_image=False)[0]

    assert folded.drops_atoms is True
    assert int(np.count_nonzero(folded.mask)) == 1
    assert separate.drops_atoms is False


def test_orientation_choice_does_not_follow_atom_order():
    def _kept_vectors(order: np.ndarray) -> set[tuple]:
        cation = _occupancy_cation(20.0, 0.5, 0.5, interleave=True)
        cation = cation[order]
        cation.set_array(KEY_OCCUPANCY, cation.arrays[KEY_OCCUPANCY])
        realization = disorder_realizations(cation, "major")[0]
        origin = cation.get_positions()[0]
        kept = []
        for index, (symbol, keep) in enumerate(
            zip(cation.get_chemical_symbols(), realization.mask)
        ):
            if symbol != "C" or not keep:
                continue
            vector = np.round(cation.get_positions()[index] - origin, 4)
            kept.append(tuple(float(value) for value in vector))
        return set(kept)

    forward = _kept_vectors(np.arange(9))
    backward = _kept_vectors(np.array([0, 8, 7, 6, 5, 4, 3, 2, 1]))

    assert len(forward) == 4
    assert forward == backward


def test_invalid_disorder_policy_raises():
    atoms = Atoms(symbols=["N"], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="disorder_policy"):
        normalize_disorder_policy("merge-both")
    with pytest.raises(ValueError, match="disorder_policy"):
        disorder_realizations(atoms, "merge-both")
    crystal = _crystal_with(
        Atoms(symbols=["N"], positions=[SITE], cell=np.eye(3) * CELL, pbc=True)
    )
    with pytest.raises(ValueError, match="disorder_policy"):
        find_polyhedra(
            crystal, "N", "Cl", level="molecule", disorder_policy="merge-both"
        )


def test_read_mol_crystal_major_policy_on_synthetic_cif():
    major = _orientation(0.0, (0.0, 0.0, 0.0))
    minor = _orientation(55.0, (0.25, 0.0, 0.0))
    cation = _cation_atoms([(major, 1, 0.65), (minor, 2, 0.35)])
    ligands = _ligand_positions(SITE)
    symbols = list(cation.get_chemical_symbols()) + ["Cl"] * len(ligands)
    positions = np.vstack([cation.get_positions(), np.asarray(ligands, dtype=float)])
    groups = list(cation.arrays[KEY_DISORDER_GROUP]) + [0] * len(ligands)
    occupancies = list(cation.arrays[KEY_OCCUPANCY]) + [1.0] * len(ligands)
    fractional = positions / CELL
    lines = [
        "data_synthetic_disordered_cation",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_space_group_name_H-M_alt 'P 1'",
        "_space_group_IT_number 1",
        f"_cell_length_a {CELL}",
        f"_cell_length_b {CELL}",
        f"_cell_length_c {CELL}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "x,y,z",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_disorder_group",
    ]
    for index, (symbol, frac, occupancy, group) in enumerate(
        zip(symbols, fractional, occupancies, groups)
    ):
        lines.append(
            f"{symbol}{index + 1} {symbol} "
            f"{frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} "
            f"{float(occupancy):.4f} {int(group)}"
        )
    cif_text = "\n".join(lines) + "\n"

    refined = read_mol_crystal(
        cif_text=cif_text,
        attach_chemistry=False,
        disorder_policy="as_refined",
    )
    major_crystal = read_mol_crystal(
        cif_text=cif_text,
        attach_chemistry=False,
        disorder_policy="major",
    )

    refined_cation = max(refined.molecules, key=len)
    selected_cation = max(major_crystal.molecules, key=len)
    assert len(refined_cation) == 9
    assert sorted(selected_cation.get_chemical_symbols()) == ["C", "C", "C", "C", "N"]

    hidden = _shells(refined, disorder_policy="as_refined")
    recovered = _shells(refined)
    from_major_read = _shells(major_crystal)

    assert hidden == []
    assert len(recovered) == 1
    assert len(from_major_read) == 1
    _assert_octahedron(recovered[0])
    _assert_octahedron(from_major_read[0])
