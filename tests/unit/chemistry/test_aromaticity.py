"""Graph aromaticity is independent of bond-length windows."""

import math

import pytest

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    infer_bond_orders,
)
from molcrys_kit.chemistry.aromaticity import apply_aromaticity


def _ring(elements, *, hydrogens=()):
    atoms = [ChemicalAtom(f"A{i}", element) for i, element in enumerate(elements)]
    atoms.extend(ChemicalAtom(f"H{i}", "H") for i in range(len(hydrogens)))
    bonds = []
    for i in range(len(elements)):
        bonds.append(
            ChemicalBond(f"A{i}", f"A{(i + 1) % len(elements)}", kind=BondKind.COVALENT)
        )
    for index, owner in enumerate(hydrogens):
        bonds.append(ChemicalBond(f"A{owner}", f"H{index}", kind=BondKind.COVALENT))
    return FiniteChemicalEntity("ring", tuple(atoms), tuple(bonds))


def _orders(result):
    return {(bond.atom1_id, bond.atom2_id): bond.order for bond in result.entity.bonds}


def _geometry_ring(elements, side, *, radial_extensions=(), z_offsets=()):
    radius = side / (2 * math.sin(math.pi / len(elements)))
    coordinates = [
        (
            f"A{i}",
            (
                radius * math.cos(i * 2 * math.pi / len(elements)),
                radius * math.sin(i * 2 * math.pi / len(elements)),
                0.0,
            ),
        )
        for i in range(len(elements))
    ]
    for atom_index, extension in radial_extensions:
        anchor = coordinates[(atom_index - 1) % len(elements)][1]
        point = coordinates[atom_index][1]
        vector = tuple(point[axis] - anchor[axis] for axis in range(3))
        length = math.sqrt(sum(value * value for value in vector))
        coordinates[atom_index] = (
            coordinates[atom_index][0],
            tuple(
                point[axis] + extension * vector[axis] / length for axis in range(3)
            ),
        )
    for atom_index, offset in z_offsets:
        atom_id, point = coordinates[atom_index]
        coordinates[atom_index] = (atom_id, (*point[:2], point[2] + offset))
    atoms = tuple(ChemicalAtom(f"A{i}", element) for i, element in enumerate(elements))
    bonds = tuple(
        ChemicalBond(f"A{i}", f"A{(i + 1) % len(elements)}", kind=BondKind.COVALENT)
        for i in range(len(elements))
    )
    return FiniteChemicalEntity(
        "geometry-ring", atoms, bonds, embedding=Embedding(coordinates)
    )


def _polygon_ring(elements, edge_lengths, *, z_offsets=(), extra_atoms=()):
    """Build a planar cyclic embedding with the requested edge lengths."""
    # For a cyclic polygon, the central angle for edge ``d`` is
    # 2*asin(d/(2R)).  Solve for R so those angles close exactly at 2*pi.
    lower = max(edge_lengths) / 2 * (1 + 1e-8)
    upper = 100.0

    def angle_sum(radius):
        return sum(2 * math.asin(length / (2 * radius)) for length in edge_lengths)

    for _ in range(80):
        radius = (lower + upper) / 2
        if angle_sum(radius) > 2 * math.pi:
            lower = radius
        else:
            upper = radius
    radius = (lower + upper) / 2
    coordinates = []
    angle = 0.0
    z_by_index = dict(z_offsets)
    for index, length in enumerate(edge_lengths):
        coordinates.append(
            (
                f"A{index}",
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    float(z_by_index.get(index, 0.0)),
                ),
            )
        )
        angle += 2 * math.asin(length / (2 * radius))
    atoms = [ChemicalAtom(f"A{i}", element) for i, element in enumerate(elements)]
    bonds = [
        ChemicalBond(f"A{i}", f"A{(i + 1) % len(elements)}", kind=BondKind.COVALENT)
        for i in range(len(elements))
    ]
    for atom_id, element, point, _owner in extra_atoms:
        atoms.append(ChemicalAtom(atom_id, element))
        coordinates.append((atom_id, point))
    for atom_id, _element, _point, owner in extra_atoms:
        bonds.append(ChemicalBond(owner, atom_id, kind=BondKind.COVALENT))
    return FiniteChemicalEntity(
        "polygon-ring",
        tuple(atoms),
        tuple(bonds),
        embedding=Embedding(tuple(coordinates)),
    )


def test_pyridine_and_pyrrole_follow_huckel_not_bond_lengths():
    pyridine = _ring("NCCCCC")
    result = infer_bond_orders(pyridine)
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)
    nitrogen = next(atom for atom in result.entity.atoms if atom.element == "N")
    assert nitrogen.formal_charge == 0
    pyrrole_atoms = [ChemicalAtom("N", "N", explicit_hydrogens=1)]
    pyrrole_atoms.extend(ChemicalAtom(f"C{i}", "C") for i in range(4))
    pairs = [("N", "C0"), ("C0", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "N")]
    pyrrole = FiniteChemicalEntity(
        "pyrrole",
        tuple(pyrrole_atoms),
        tuple(ChemicalBond(a, b, kind=BondKind.COVALENT) for a, b in pairs),
    )
    result = infer_bond_orders(pyrrole)
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)
    nitrogen_orders = [
        bond.order
        for bond in result.entity.bonds
        if "N" in {bond.atom1_id, bond.atom2_id}
    ]
    assert sum(nitrogen_orders) == 2.0


def test_tropylium_empty_p_orbital_is_aromatic():
    atoms = [ChemicalAtom("C0", "C", formal_charge=1, explicit_hydrogens=1)]
    atoms.extend(ChemicalAtom(f"C{i}", "C", explicit_hydrogens=1) for i in range(1, 7))
    bonds = [
        ChemicalBond(f"C{i}", f"C{(i + 1) % 7}", kind=BondKind.COVALENT)
        for i in range(7)
    ]
    result = infer_bond_orders(
        FiniteChemicalEntity("tropylium", tuple(atoms), tuple(bonds))
    )
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)
    assert result.entity.net_charge == 1


def test_benzene_is_aromatic_kekule_not_a_triple_pattern():
    entity = _ring("CCCCCC")
    result = infer_bond_orders(entity)
    ring = [bond for bond in result.entity.bonds]
    assert all(bond.aromatic for bond in ring)
    assert all(bond.order in {1.0, 2.0} for bond in ring)
    sums = {}
    for bond in ring:
        sums[bond.atom1_id] = sums.get(bond.atom1_id, 0) + bond.order
        sums[bond.atom2_id] = sums.get(bond.atom2_id, 0) + bond.order
    assert set(sums.values()) == {3.0}
    assert all(atom.formal_charge == 0 for atom in result.entity.atoms)
    assert all(atom.implicit_hydrogens == 1 for atom in result.entity.atoms)
    assert all(atom.explicit_hydrogens is None for atom in result.entity.atoms)
    assert result.status is InferenceStatus.INFERRED


def test_thiophene_sulfur_lone_pair_makes_the_ring_aromatic():
    entity = _ring("CCCSC")
    result = infer_bond_orders(entity)
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(bond.aromatic for bond in result.entity.bonds)
    assert all(bond.order in {1.0, 2.0} for bond in result.entity.bonds)
    sulfur = next(
        bond for bond in result.entity.bonds if "A3" in {bond.atom1_id, bond.atom2_id}
    )
    # Every Kekulé form keeps the two C-S orders summing to 2.
    sulfur_orders = [
        bond.order
        for bond in result.entity.bonds
        if "A3" in {bond.atom1_id, bond.atom2_id}
    ]
    assert sum(sulfur_orders) == 2.0
    assert sulfur.order in {1.0, 2.0}


def test_geometry_allows_long_c_s_edges_in_a_valid_heteroaromatic_ring():
    atoms = tuple(ChemicalAtom(f"A{i}", "S" if i == 4 else "C") for i in range(5))
    coordinates = {
        "A0": (0.0, 0.0, 0.0),
        "A1": (1.4, 0.0, 0.0),
        "A2": (2.8, 0.0, 0.0),
        "A3": (2.8, 1.4, 0.0),
    }
    diagonal = math.hypot(2.8, 1.4)
    offset = math.sqrt(1.7**2 - (diagonal / 2) ** 2)
    coordinates["A4"] = (
        2.8 / 2 - offset * 1.4 / diagonal,
        1.4 / 2 + offset * 2.8 / diagonal,
        0.0,
    )
    bonds = tuple(
        ChemicalBond(f"A{i}", f"A{(i + 1) % 5}", kind=BondKind.COVALENT)
        for i in range(5)
    )
    result = infer_bond_orders(
        FiniteChemicalEntity(
            "thiophene-geometry",
            atoms,
            bonds,
            embedding=Embedding(tuple(coordinates.items())),
        )
    )
    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 1
    assert result.diagnostics["uncertain_hydrogen_atoms"] == ()
    assert all(bond.aromatic for bond in result.entity.bonds)


def test_isoxazole_n_o_edge_at_1_415_angstrom_stays_aromatic():
    # Start from a 1.39 Å pentagon and extend O away from N by 0.025 Å. This
    # gives N--O = 1.415 Å while keeping the adjacent C--O edge below 1.40 Å.
    result = infer_bond_orders(
        _geometry_ring("NOCCC", 1.39, radial_extensions=((1, 0.025),))
    )

    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 1
    assert result.diagnostics["uncertain_hydrogen_atoms"] == ()
    assert all(bond.aromatic and bond.order in {1.0, 2.0} for bond in result.entity.bonds)


@pytest.mark.parametrize("side", (1.37, 1.38, 1.39))
def test_short_c_o_edges_remain_aromatic(side):
    result = infer_bond_orders(_geometry_ring("OCCCC", side))

    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 1
    assert result.diagnostics["uncertain_hydrogen_atoms"] == ()
    assert all(bond.aromatic and bond.order in {1.0, 2.0} for bond in result.entity.bonds)


@pytest.mark.parametrize(
    ("elements", "side"),
    (("NOCCC", 1.415), ("ONCCC", 1.39), ("OCCCC", 1.37)),
)
def test_planar_heteroaromatics_use_ring_metric_near_pair_boundaries(elements, side):
    result = infer_bond_orders(_geometry_ring(elements, side))

    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 1
    assert result.diagnostics["uncertain_hydrogen_atoms"] == ()
    assert all(bond.aromatic for bond in result.entity.bonds)


@pytest.mark.parametrize(("elements", "side"), (("NOCCC", 1.45), ("OCCCC", 1.46)))
def test_saturated_heterocycles_are_abstained_in_the_explicit_ambiguous_band(
    elements, side
):
    result = infer_bond_orders(_geometry_ring(elements, side))

    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"A{i}" for i in range(len(elements))
    )
    assert result.status is not InferenceStatus.INFERRED


def test_puckered_thf_does_not_become_aromatic_from_short_edges():
    result = infer_bond_orders(
        _geometry_ring(
            "OCCCC",
            1.43,
            radial_extensions=((2, 0.40), (4, -0.30)),
            z_offsets=((2, 0.40), (4, -0.30)),
        )
    )

    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"]


def test_benzofuran_keeps_both_planar_rings_aromatic():
    coordinates = {
        "C0": (0.0, 0.0, 0.0),
        "C1": (1.4, 0.0, 0.0),
        "C2": (2.1, -1.2, 0.0),
        "C3": (1.4, -2.4, 0.0),
        "C4": (0.0, -2.4, 0.0),
        "C5": (-0.7, -1.2, 0.0),
        "O": (-0.4, 1.0, 0.0),
        "C6": (0.7, 1.8, 0.0),
        "C7": (2.1, 1.2, 0.0),
    }
    atoms = tuple(
        ChemicalAtom(atom_id, "O" if atom_id == "O" else "C")
        for atom_id in coordinates
    )
    pairs = (
        ("C0", "C1"),
        ("C1", "C2"),
        ("C2", "C3"),
        ("C3", "C4"),
        ("C4", "C5"),
        ("C5", "C0"),
        ("O", "C0"),
        ("C1", "C7"),
        ("C7", "C6"),
        ("C6", "O"),
    )
    bonds = tuple(ChemicalBond(left, right, kind=BondKind.COVALENT) for left, right in pairs)
    result = infer_bond_orders(
        FiniteChemicalEntity(
            "benzofuran-geometry",
            atoms,
            bonds,
            embedding=Embedding(tuple(coordinates.items())),
        )
    )

    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 2
    assert all(bond.aromatic for bond in result.entity.bonds)


@pytest.mark.parametrize(
    ("elements", "edge_lengths"),
    (
        ("CCCCCC", (1.33, 1.50, 1.50, 1.33, 1.50, 1.50)),  # 1,4-cyclohexadiene
        ("CCCCC", (1.34, 1.47, 1.34, 1.50, 1.50)),  # cyclopentadiene
        ("OCCCCC", (1.42, 1.34, 1.50, 1.34, 1.50, 1.42)),  # 2H-pyran
        ("CCCCCC", (1.34, 1.46, 1.34, 1.50, 1.53, 1.50)),  # 1,3-cyclohexadiene
        ("NCCCCC", (1.47, 1.50, 1.34, 1.45, 1.35, 1.38)),  # 1,2-dihydropyridine
    ),
    ids=(
        "1,4-cyclohexadiene",
        "cyclopentadiene",
        "2H-pyran",
        "1,3-cyclohexadiene",
        "1,2-dihydropyridine",
    ),
)
def test_planar_alternating_rings_remain_unresolved(elements, edge_lengths):
    result = infer_bond_orders(_polygon_ring(elements, edge_lengths))

    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"A{i}" for i in range(len(elements))
    )
    assert all(not bond.aromatic and bond.order is None for bond in result.entity.bonds)


def test_puckered_1_3_cyclohexadiene_is_not_planar_aromatic():
    result = infer_bond_orders(
        _polygon_ring(
            "CCCCCC",
            (1.40,) * 6,
            z_offsets=((2, 0.15), (4, -0.15)),
        )
    )

    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"]


def test_planar_alternating_o_benzoquinone_is_not_aromatic():
    result = infer_bond_orders(
        _polygon_ring(
            "CCCCCC",
            (1.54, 1.47, 1.35, 1.46, 1.34, 1.47),
            extra_atoms=(
                ("O0", "O", (0.0, 1.22, 0.0), "A0"),
                ("O3", "O", (0.0, -1.22, 0.0), "A3"),
            ),
        )
    )

    assert result.status is not InferenceStatus.INFERRED
    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"A{i}" for i in range(6)
    )


def test_geometry_aromatic_but_non_huckel_ring_abstains():
    result = infer_bond_orders(_geometry_ring("CCCCC", 1.40))

    assert result.status is InferenceStatus.INDETERMINATE
    assert result.diagnostics["aromatic_rings"] == 0
    assert result.diagnostics["uncertain_hydrogen_atoms"] == tuple(
        f"A{i}" for i in range(5)
    )


def test_naphthalene_bridge_is_aromatic_and_bridgeheads_sum_to_four():
    # Numbering: 0-1-2-3-4-5 six-ring, 4-6-7-8-9-5 fused ring, bridge 4-5.
    atoms = [ChemicalAtom(f"C{i}", "C") for i in range(10)]
    pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (4, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    bonds = [ChemicalBond(f"C{a}", f"C{b}", kind=BondKind.COVALENT) for a, b in pairs]
    entity = FiniteChemicalEntity("naphthalene", tuple(atoms), tuple(bonds))
    result = infer_bond_orders(entity)
    assert result.diagnostics["aromatic_rings"] == 2
    assert all(bond.aromatic for bond in result.entity.bonds)
    sums = {atom.atom_id: 0.0 for atom in result.entity.atoms}
    for bond in result.entity.bonds:
        sums[bond.atom1_id] += bond.order
        sums[bond.atom2_id] += bond.order
    assert sums["C4"] == 4.0 and sums["C5"] == 4.0
    assert sums["C0"] == 3.0


def test_biphenyl_bridge_is_not_aromatic():
    atoms = [ChemicalAtom(f"C{i}", "C") for i in range(12)]
    pairs = (
        [(i, (i + 1) % 6) for i in range(6)]
        + [(6 + i, 6 + (i + 1) % 6) for i in range(6)]
        + [(0, 6)]
    )
    bonds = [ChemicalBond(f"C{a}", f"C{b}", kind=BondKind.COVALENT) for a, b in pairs]
    entity = FiniteChemicalEntity("biphenyl", tuple(atoms), tuple(bonds))
    result = infer_bond_orders(entity)
    bridge = next(
        bond
        for bond in result.entity.bonds
        if {bond.atom1_id, bond.atom2_id} == {"C0", "C6"}
    )
    assert bridge.aromatic is False
    assert bridge.order == 1.0
    assert result.diagnostics["aromatic_rings"] == 2


def test_cyclohexane_with_explicit_hydrogens_is_not_aromatic():
    entity = _ring("CCCCCC", hydrogens=(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5))
    result = infer_bond_orders(entity)
    ring = [
        bond
        for bond in result.entity.bonds
        if bond.atom1_id.startswith("A") and bond.atom2_id.startswith("A")
    ]
    assert result.diagnostics["aromatic_rings"] == 0
    assert all(bond.aromatic is False for bond in ring)
    assert all(bond.order == 1.0 for bond in result.entity.bonds)


def test_cyclobutadiene_is_outside_the_huckel_window():
    entity = _ring("CCCC")
    result = infer_bond_orders(entity)
    assert result.diagnostics["aromatic_rings"] == 0
    assert all(bond.aromatic is False for bond in result.entity.bonds)


def test_user_confirmed_single_bond_blocks_a_false_aromatic_ring():
    atoms = [ChemicalAtom(f"C{i}", "C") for i in range(6)]
    bonds = []
    for i in range(6):
        bond = ChemicalBond(f"C{i}", f"C{(i + 1) % 6}", kind=BondKind.COVALENT)
        if i == 0:
            bond = ChemicalBond(
                f"C{i}",
                f"C{(i + 1) % 6}",
                order=1.0,
                kind=BondKind.COVALENT,
                aromatic=False,
                evidence=(
                    Evidence(EvidenceSource.USER_CONFIRMED, "user", "not aromatic"),
                ),
            )
        bonds.append(bond)
    assignment = apply_aromaticity(atoms, bonds)
    assert assignment.ring_count == 0
    assert assignment.bonds[0].aromatic is False
