"""Section 4.1 micro-family matrix for bond-order perception.

Each family in issue #174 is built from an explicit graph and checked against
the Lewis outcome. Geometry ranks resonance forms. It does not create an order
that valence forbids, and missing or contradictory evidence stays unresolved.
"""

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
from molcrys_kit.chemistry.topology import analyze_periodic_topology


def _atom(atom_id, element, **kwargs):
    return ChemicalAtom(atom_id, element, **kwargs)


def _bond(left, right, **kwargs):
    return ChemicalBond(left, right, kind=BondKind.COVALENT, **kwargs)


def _entity(name, atoms, bonds, coords=None):
    embedding = Embedding(tuple(coords)) if coords else None
    return FiniteChemicalEntity(name, tuple(atoms), tuple(bonds), embedding=embedding)


def _orders(result):
    return {
        frozenset((bond.atom1_id, bond.atom2_id)): bond.order
        for bond in result.entity.bonds
    }


def _charges(result):
    return {atom.atom_id: atom.formal_charge for atom in result.entity.atoms}


def _alkane(n):
    atoms = []
    bonds = []
    for index in range(n):
        hydrogens = 3 if index in {0, n - 1} else 2
        atoms.append(_atom(f"C{index}", "C", explicit_hydrogens=hydrogens))
    for index in range(n - 1):
        bonds.append(_bond(f"C{index}", f"C{index + 1}"))
    return _entity(f"C{n}", atoms, bonds)


def _ring(name, elements, *, extras=(), flags=None):
    atoms = []
    for index, element in enumerate(elements):
        kwargs = dict(flags.get(index, {}) if flags else {})
        atoms.append(_atom(f"A{index}", element, **kwargs))
    bonds = [
        _bond(f"A{index}", f"A{(index + 1) % len(elements)}")
        for index in range(len(elements))
    ]
    atoms.extend(extras)
    return _entity(name, atoms, bonds)


@pytest.mark.parametrize("n", [1, 2, 3, 6, 13, 64, 256, 1024])
def test_saturated_family_assigns_single_bonds_without_truncation(n):
    if n == 1:
        entity = _entity("methane", [_atom("C", "C", explicit_hydrogens=4)], [])
    elif n == 6:
        entity = _entity(
            "cyclohexane",
            [_atom(f"C{i}", "C", explicit_hydrogens=2) for i in range(6)],
            [_bond(f"C{i}", f"C{(i + 1) % 6}") for i in range(6)],
        )
    else:
        entity = _alkane(n)
    result = infer_bond_orders(entity)
    assert result.status is InferenceStatus.INFERRED
    assert result.diagnostics["search_complete"] is True
    assert result.diagnostics["aromatic_rings"] == 0
    assert set(result.diagnostics["solver"] for _ in (0,)) <= {
        "tree_dp",
        "fixed",
        "search",
    }
    if n >= 13:
        assert result.diagnostics["solver"] == "tree_dp"
        assert result.diagnostics["search_states"] < 20 * n
    assert all(bond.order == 1.0 for bond in result.entity.bonds) or n == 1
    assert all(atom.formal_charge == 0 for atom in result.entity.atoms)


@pytest.mark.parametrize(
    "name,builder,expected",
    [
        (
            "ethene",
            lambda: _entity(
                "ethene",
                [
                    _atom("C1", "C", explicit_hydrogens=2),
                    _atom("C2", "C", explicit_hydrogens=2),
                ],
                [_bond("C1", "C2")],
            ),
            {frozenset({"C1", "C2"}): 2.0},
        ),
        (
            "ethyne",
            lambda: _entity(
                "ethyne",
                [
                    _atom("C1", "C", explicit_hydrogens=1),
                    _atom("C2", "C", explicit_hydrogens=1),
                ],
                [_bond("C1", "C2")],
            ),
            {frozenset({"C1", "C2"}): 3.0},
        ),
        (
            "formaldehyde",
            lambda: _entity(
                "h2co",
                [_atom("C", "C", explicit_hydrogens=2), _atom("O", "O")],
                [_bond("C", "O")],
            ),
            {frozenset({"C", "O"}): 2.0},
        ),
        (
            "carbon_dioxide",
            lambda: _entity(
                "co2",
                [_atom("C", "C"), _atom("O1", "O"), _atom("O2", "O")],
                [_bond("C", "O1"), _bond("C", "O2")],
            ),
            {frozenset({"C", "O1"}): 2.0, frozenset({"C", "O2"}): 2.0},
        ),
        (
            "hcn",
            lambda: _entity(
                "hcn",
                [_atom("C", "C", explicit_hydrogens=1), _atom("N", "N")],
                [_bond("C", "N")],
            ),
            {frozenset({"C", "N"}): 3.0},
        ),
        (
            "acetonitrile",
            lambda: _entity(
                "mecn",
                [
                    _atom("C1", "C", explicit_hydrogens=3),
                    _atom("C2", "C"),
                    _atom("N", "N"),
                ],
                [_bond("C1", "C2"), _bond("C2", "N")],
            ),
            {frozenset({"C1", "C2"}): 1.0, frozenset({"C2", "N"}): 3.0},
        ),
        (
            "allene",
            lambda: _entity(
                "allene",
                [
                    _atom("C1", "C", explicit_hydrogens=2),
                    _atom("C2", "C"),
                    _atom("C3", "C", explicit_hydrogens=2),
                ],
                [_bond("C1", "C2"), _bond("C2", "C3")],
            ),
            {frozenset({"C1", "C2"}): 2.0, frozenset({"C2", "C3"}): 2.0},
        ),
    ],
)
def test_unsaturation_family(name, builder, expected):
    result = infer_bond_orders(builder())
    assert result.status is InferenceStatus.INFERRED, name
    assert _orders(result) == expected
    assert all(charge == 0 for charge in _charges(result).values())


def test_carbonyl_and_heteroatom_family():
    acetamide = _entity(
        "acetamide",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("C2", "C"),
            _atom("O", "O"),
            _atom("N", "N", explicit_hydrogens=2),
        ],
        [_bond("C1", "C2"), _bond("C2", "O"), _bond("C2", "N")],
        coords=(
            ("C1", (0, 0, 0)),
            ("C2", (1.51, 0, 0)),
            ("O", (2.74, 0, 0)),
            ("N", (1.51, 1.47, 0)),
        ),
    )
    urea = _entity(
        "urea",
        [
            _atom("N1", "N", explicit_hydrogens=2),
            _atom("C", "C"),
            _atom("O", "O"),
            _atom("N2", "N", explicit_hydrogens=2),
        ],
        [_bond("N1", "C"), _bond("C", "O"), _bond("C", "N2")],
        coords=(
            ("N1", (0, 1.47, 0)),
            ("C", (0, 0, 0)),
            ("O", (1.23, 0, 0)),
            ("N2", (0, -1.47, 0)),
        ),
    )
    acetic = _entity(
        "acetic",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("C2", "C"),
            _atom("O1", "O"),
            _atom("O2", "O", explicit_hydrogens=1),
        ],
        [_bond("C1", "C2"), _bond("C2", "O1"), _bond("C2", "O2")],
        coords=(
            ("C1", (0, 0, 0)),
            ("C2", (1.51, 0, 0)),
            ("O1", (2.74, 0, 0)),
            ("O2", (1.51, 1.43, 0)),
        ),
    )
    ester = _entity(
        "ester",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("C2", "C"),
            _atom("O1", "O"),
            _atom("O2", "O"),
            _atom("C3", "C", explicit_hydrogens=3),
        ],
        [_bond("C1", "C2"), _bond("C2", "O1"), _bond("C2", "O2"), _bond("O2", "C3")],
        coords=(
            ("C1", (0, 0, 0)),
            ("C2", (1.51, 0, 0)),
            ("O1", (2.74, 0, 0)),
            ("O2", (1.51, 1.43, 0)),
            ("C3", (1.51, 2.86, 0)),
        ),
    )
    sulfoxide = _entity(
        "sulfoxide",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("S", "S"),
            _atom("C2", "C", explicit_hydrogens=3),
            _atom("O", "O"),
        ],
        [_bond("C1", "S"), _bond("C2", "S"), _bond("S", "O")],
    )
    sulfone = _entity(
        "sulfone",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("S", "S"),
            _atom("C2", "C", explicit_hydrogens=3),
            _atom("O1", "O"),
            _atom("O2", "O"),
        ],
        [_bond("C1", "S"), _bond("C2", "S"), _bond("S", "O1"), _bond("S", "O2")],
        coords=(
            ("C1", (-1.82, 0, 0)),
            ("S", (0, 0, 0)),
            ("C2", (0, 1.82, 0)),
            ("O1", (1.48, 0, 0)),
            ("O2", (0, 0, 1.48)),
        ),
    )
    phosphate = _entity(
        "phosphate",
        [
            _atom("P", "P"),
            _atom("O1", "O"),
            _atom("O2", "O", explicit_hydrogens=1),
            _atom("O3", "O", explicit_hydrogens=1),
            _atom("O4", "O", explicit_hydrogens=1),
        ],
        [_bond("P", "O1"), _bond("P", "O2"), _bond("P", "O3"), _bond("P", "O4")],
        coords=(
            ("P", (0, 0, 0)),
            ("O1", (1.48, 0, 0)),
            ("O2", (0, 1.60, 0)),
            ("O3", (0, 0, 1.60)),
            ("O4", (-1.60, 0, 0)),
        ),
    )
    expected = {
        "acetamide": {
            frozenset({"C1", "C2"}): 1.0,
            frozenset({"C2", "O"}): 2.0,
            frozenset({"C2", "N"}): 1.0,
        },
        "urea": {
            frozenset({"N1", "C"}): 1.0,
            frozenset({"C", "O"}): 2.0,
            frozenset({"C", "N2"}): 1.0,
        },
        "acetic": {
            frozenset({"C1", "C2"}): 1.0,
            frozenset({"C2", "O1"}): 2.0,
            frozenset({"C2", "O2"}): 1.0,
        },
        "ester": {
            frozenset({"C1", "C2"}): 1.0,
            frozenset({"C2", "O1"}): 2.0,
            frozenset({"C2", "O2"}): 1.0,
            frozenset({"O2", "C3"}): 1.0,
        },
        "sulfoxide": {
            frozenset({"C1", "S"}): 1.0,
            frozenset({"C2", "S"}): 1.0,
            frozenset({"S", "O"}): 2.0,
        },
        "sulfone": {
            frozenset({"C1", "S"}): 1.0,
            frozenset({"C2", "S"}): 1.0,
            frozenset({"S", "O1"}): 2.0,
            frozenset({"S", "O2"}): 2.0,
        },
        "phosphate": {
            frozenset({"P", "O1"}): 2.0,
            frozenset({"P", "O2"}): 1.0,
            frozenset({"P", "O3"}): 1.0,
            frozenset({"P", "O4"}): 1.0,
        },
    }
    for entity in (acetamide, urea, acetic, ester, sulfoxide, sulfone, phosphate):
        result = infer_bond_orders(entity)
        assert result.status is InferenceStatus.INFERRED, entity.entity_id
        assert _orders(result) == expected[entity.entity_id]
        assert all(charge == 0 for charge in _charges(result).values())


def test_aromatic_and_antiaromatic_family():
    benzene = _ring("benzene", "CCCCCC")
    pyridine = _ring("pyridine", "NCCCCC")
    pyrrole = _ring("pyrrole", "NCCCC", flags={0: {"explicit_hydrogens": 1}})
    thiophene = _ring("thiophene", "SCCCC")
    furan = _ring("furan", "OCCCC")
    imidazole = _ring("imidazole", "NCNCC", flags={0: {"explicit_hydrogens": 1}})
    result = infer_bond_orders(benzene)
    assert result.diagnostics["aromatic_rings"] == 1
    assert all(
        bond.aromatic and bond.order in {1.0, 2.0} for bond in result.entity.bonds
    )
    assert all(
        atom.implicit_hydrogens == 1 and atom.formal_charge == 0
        for atom in result.entity.atoms
    )
    for entity in (pyridine, pyrrole, thiophene, furan, imidazole):
        perceived = infer_bond_orders(entity)
        assert perceived.diagnostics["aromatic_rings"] == 1, entity.entity_id
        assert all(bond.aromatic for bond in perceived.entity.bonds)
        assert perceived.status is InferenceStatus.INFERRED

    # Naphthalene: two fused rings, bridgeheads sum to 4.
    atoms = [_atom(f"C{i}", "C") for i in range(10)]
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
    naphthalene = infer_bond_orders(
        _entity("naphthalene", atoms, [_bond(f"C{a}", f"C{b}") for a, b in pairs])
    )
    assert naphthalene.diagnostics["aromatic_rings"] == 2
    assert all(bond.aromatic for bond in naphthalene.entity.bonds)

    # Indole: pyrrole fused to benzene. The bridge is aromatic; the N–C bonds sum to 2.
    indole_atoms = [_atom(f"C{i}", "C") for i in range(8)] + [
        _atom("N", "N", explicit_hydrogens=1)
    ]
    indole_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (4, 6),
        (6, 7),
        (7, "N"),
        ("N", 5),
    ]
    indole_bonds = []
    for left, right in indole_pairs:
        indole_bonds.append(
            _bond(
                f"C{left}" if isinstance(left, int) else left,
                f"C{right}" if isinstance(right, int) else right,
            )
        )
    indole = infer_bond_orders(_entity("indole", indole_atoms, indole_bonds))
    assert indole.diagnostics["aromatic_rings"] == 2
    assert all(bond.aromatic for bond in indole.entity.bonds)
    assert (
        sum(
            bond.order
            for bond in indole.entity.bonds
            if "N" in {bond.atom1_id, bond.atom2_id}
        )
        == 2.0
    )

    # Biphenyl bridge is a single bond and is not part of either ring.
    biphenyl_atoms = [_atom(f"C{i}", "C") for i in range(12)]
    biphenyl_pairs = (
        [(i, (i + 1) % 6) for i in range(6)]
        + [(6 + i, 6 + (i + 1) % 6) for i in range(6)]
        + [(0, 6)]
    )
    biphenyl = infer_bond_orders(
        _entity(
            "biphenyl",
            biphenyl_atoms,
            [_bond(f"C{a}", f"C{b}") for a, b in biphenyl_pairs],
        )
    )
    bridge = next(
        bond
        for bond in biphenyl.entity.bonds
        if {bond.atom1_id, bond.atom2_id} == {"C0", "C6"}
    )
    assert bridge.aromatic is False and bridge.order == 1.0
    assert biphenyl.diagnostics["aromatic_rings"] == 2

    cyclobutadiene = infer_bond_orders(
        _entity(
            "cbd",
            [_atom(f"C{i}", "C", explicit_hydrogens=1) for i in range(4)],
            [_bond(f"C{i}", f"C{(i + 1) % 4}") for i in range(4)],
        )
    )
    assert cyclobutadiene.diagnostics["aromatic_rings"] == 0
    assert all(
        bond.aromatic is False and bond.order in {1.0, 2.0}
        for bond in cyclobutadiene.entity.bonds
    )

    tropylium = infer_bond_orders(
        _entity(
            "tropylium",
            [_atom("C0", "C", formal_charge=1, explicit_hydrogens=1)]
            + [_atom(f"C{i}", "C", explicit_hydrogens=1) for i in range(1, 7)],
            [_bond(f"C{i}", f"C{(i + 1) % 7}") for i in range(7)],
        )
    )
    assert tropylium.diagnostics["aromatic_rings"] == 1
    assert tropylium.entity.net_charge == 1
    assert all(bond.aromatic for bond in tropylium.entity.bonds)


def test_resonance_and_ion_family():
    acetate = infer_bond_orders(
        _entity(
            "acetate",
            [
                _atom("C1", "C", explicit_hydrogens=3),
                _atom("C2", "C"),
                _atom("O1", "O"),
                _atom("O2", "O"),
            ],
            [_bond("C1", "C2"), _bond("C2", "O1"), _bond("C2", "O2")],
            coords=(
                ("C1", (-1.54, 0, 0)),
                ("C2", (0, 0, 0)),
                ("O1", (1.23, 0, 0)),
                ("O2", (0, 1.36, 0)),
            ),
        )
    )
    carbonate = infer_bond_orders(
        _entity(
            "carbonate",
            [_atom("C", "C"), _atom("O1", "O"), _atom("O2", "O"), _atom("O3", "O")],
            [_bond("C", "O1"), _bond("C", "O2"), _bond("C", "O3")],
            coords=(
                ("C", (0, 0, 0)),
                ("O1", (1.23, 0, 0)),
                ("O2", (-0.68, 1.18, 0)),
                ("O3", (-0.68, -1.18, 0)),
            ),
        )
    )
    nitrate = infer_bond_orders(
        _entity(
            "nitrate",
            [_atom("N", "N"), _atom("O1", "O"), _atom("O2", "O"), _atom("O3", "O")],
            [_bond("N", "O1"), _bond("N", "O2"), _bond("N", "O3")],
            coords=(
                ("N", (0, 0, 0)),
                ("O1", (1.21, 0, 0)),
                ("O2", (-0.70, 1.22, 0)),
                ("O3", (-0.70, -1.22, 0)),
            ),
        )
    )
    nitromethane = infer_bond_orders(
        _entity(
            "nitromethane",
            [
                _atom("C", "C", explicit_hydrogens=3),
                _atom("N", "N"),
                _atom("O1", "O"),
                _atom("O2", "O"),
            ],
            [_bond("C", "N"), _bond("N", "O1"), _bond("N", "O2")],
            coords=(
                ("C", (-1.47, 0, 0)),
                ("N", (0, 0, 0)),
                ("O1", (1.21, 0, 0)),
                ("O2", (-0.2, 1.40, 0)),
            ),
        )
    )
    sulfate = infer_bond_orders(
        _entity(
            "sulfate",
            [
                _atom("S", "S"),
                _atom("O1", "O"),
                _atom("O2", "O"),
                _atom("O3", "O"),
                _atom("O4", "O"),
            ],
            [_bond("S", f"O{i}") for i in range(1, 5)],
            coords=(
                ("S", (0, 0, 0)),
                ("O1", (1.48, 0, 0)),
                ("O2", (0, 1.48, 0)),
                ("O3", (0, 0, 1.64)),
                ("O4", (-1.64, 0, 0)),
            ),
        )
    )
    phosphate = infer_bond_orders(
        _entity(
            "phosphate",
            [
                _atom("P", "P"),
                _atom("O1", "O"),
                _atom("O2", "O"),
                _atom("O3", "O"),
                _atom("O4", "O"),
            ],
            [_bond("P", f"O{i}") for i in range(1, 5)],
            coords=(
                ("P", (0, 0, 0)),
                ("O1", (1.48, 0, 0)),
                ("O2", (0, 1.60, 0)),
                ("O3", (0, 0, 1.60)),
                ("O4", (-1.60, 0, 0)),
            ),
        )
    )
    ammonium = infer_bond_orders(
        _entity("ammonium", [_atom("N", "N", explicit_hydrogens=4)], [])
    )
    glycine = infer_bond_orders(
        _entity(
            "glycine",
            [
                _atom("N", "N", explicit_hydrogens=3),
                _atom("CA", "C", explicit_hydrogens=2),
                _atom("C", "C"),
                _atom("O1", "O"),
                _atom("O2", "O"),
            ],
            [_bond("N", "CA"), _bond("CA", "C"), _bond("C", "O1"), _bond("C", "O2")],
            coords=(
                ("N", (0, 0, 0)),
                ("CA", (1.47, 0, 0)),
                ("C", (2.54, 1.1, 0)),
                ("O1", (3.7, 1.4, 0)),
                ("O2", (2.1, 2.3, 0)),
            ),
        )
    )
    assert _orders(acetate)[frozenset({"C2", "O1"})] == 2.0
    assert _orders(acetate)[frozenset({"C2", "O2"})] == 1.0
    assert acetate.entity.net_charge == -1
    assert _orders(carbonate)[frozenset({"C", "O1"})] == 2.0
    assert carbonate.entity.net_charge == -2
    assert _orders(nitrate)[frozenset({"N", "O1"})] == 2.0
    assert _charges(nitrate)["N"] == 1
    assert nitrate.entity.net_charge == -1
    assert _orders(nitromethane)[frozenset({"C", "N"})] == 1.0
    assert _orders(nitromethane)[frozenset({"N", "O1"})] == 2.0
    assert _orders(nitromethane)[frozenset({"N", "O2"})] == 1.0
    assert _orders(sulfate)[frozenset({"S", "O1"})] == 2.0
    assert _orders(sulfate)[frozenset({"S", "O3"})] == 1.0
    assert sulfate.entity.net_charge == -2
    assert _orders(phosphate)[frozenset({"P", "O1"})] == 2.0
    assert _orders(phosphate)[frozenset({"P", "O2"})] == 1.0
    assert phosphate.entity.net_charge == -3
    assert ammonium.entity.net_charge == 1
    assert ammonium.status is InferenceStatus.INFERRED
    assert glycine.entity.net_charge == 0
    assert _charges(glycine)["N"] == 1
    assert _orders(glycine)[frozenset({"C", "O1"})] == 2.0

    # Nitrobenzene: aromatic ring plus a geometry-ranked nitro group.
    ring = [_atom(f"C{i}", "C") for i in range(6)]
    nitro_atoms = ring + [_atom("N", "N"), _atom("O1", "O"), _atom("O2", "O")]
    nitro_bonds = [_bond(f"C{i}", f"C{(i + 1) % 6}") for i in range(6)]
    nitro_bonds += [_bond("C0", "N"), _bond("N", "O1"), _bond("N", "O2")]
    coords = [
        ("C0", (0, 0, 0)),
        ("C1", (1.40, 0.2, 0)),
        ("C2", (2.1, 1.4, 0)),
        ("C3", (1.4, 2.6, 0)),
        ("C4", (0, 2.4, 0)),
        ("C5", (-0.7, 1.2, 0)),
        ("N", (-1.47, 0, 0)),
        ("O1", (-2.68, 0, 0)),
        ("O2", (-1.2, -1.35, 0)),
    ]
    nitrobenzene = infer_bond_orders(
        _entity("nitrobenzene", nitro_atoms, nitro_bonds, coords=coords)
    )
    assert nitrobenzene.diagnostics["aromatic_rings"] == 1
    assert _orders(nitrobenzene)[frozenset({"C0", "N"})] == 1.0
    assert _orders(nitrobenzene)[frozenset({"N", "O1"})] == 2.0
    assert _orders(nitrobenzene)[frozenset({"N", "O2"})] == 1.0
    assert _charges(nitrobenzene)["N"] == 1
    assert nitrobenzene.entity.net_charge == 0


def test_ambiguity_family_does_not_invent_certainty():
    carbon_monoxide = infer_bond_orders(
        _entity("co", [_atom("C", "C"), _atom("O", "O")], [_bond("C", "O")])
    )
    assert carbon_monoxide.status is InferenceStatus.INDETERMINATE
    assert all(bond.order is None for bond in carbon_monoxide.entity.bonds)

    nitric = infer_bond_orders(
        _entity(
            "no",
            [
                _atom("N", "N", radical_electrons=1),
                _atom("O", "O", radical_electrons=1),
            ],
            [_bond("N", "O")],
        )
    )
    oxygen = infer_bond_orders(
        _entity(
            "o2",
            [
                _atom("O1", "O", radical_electrons=1),
                _atom("O2", "O", radical_electrons=1),
            ],
            [_bond("O1", "O2")],
        )
    )
    assert nitric.status is InferenceStatus.PROVISIONAL
    assert oxygen.status is InferenceStatus.PROVISIONAL
    assert nitric.alternatives
    assert oxygen.alternatives

    missing = infer_bond_orders(
        _entity(
            "fragment",
            [_atom("C", "C"), _atom("N", "N"), _atom("H", "H")],
            [_bond("C", "N"), _bond("N", "H")],
        )
    )
    assert missing.status is InferenceStatus.INDETERMINATE
    assert all(bond.order is None for bond in missing.entity.bonds)

    distorted = infer_bond_orders(
        _entity(
            "short-ethane",
            [
                _atom("C1", "C", explicit_hydrogens=3),
                _atom("C2", "C", explicit_hydrogens=3),
            ],
            [_bond("C1", "C2")],
            coords=(("C1", (0, 0, 0)), ("C2", (1.20, 0, 0))),
        )
    )
    assert _orders(distorted)[frozenset({"C1", "C2"})] == 1.0

    contradicted = infer_bond_orders(
        _entity(
            "contradiction",
            [
                _atom("C1", "C", explicit_hydrogens=2),
                _atom("C2", "C", explicit_hydrogens=2),
            ],
            [
                ChemicalBond(
                    "C1",
                    "C2",
                    order=1.0,
                    kind=BondKind.COVALENT,
                    evidence=(
                        Evidence(EvidenceSource.USER_CONFIRMED, "user", "order:1"),
                        Evidence(EvidenceSource.EXPLICIT_CIF, "cif", "bond_type:D"),
                    ),
                )
            ],
        )
    )
    assert _orders(contradicted)[frozenset({"C1", "C2"})] == 1.0
    assert contradicted.diagnostics["source_contradictions"] == 1

    metal = infer_bond_orders(
        _entity("znn", [_atom("Zn", "Zn"), _atom("N", "N")], [ChemicalBond("Zn", "N")])
    )
    assert metal.entity.bonds[0].kind is BondKind.COORDINATION
    assert metal.entity.bonds[0].order is None


def test_periodic_and_multiedge_family():
    face = _entity(
        "face",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("C2", "C", explicit_hydrogens=3),
        ],
        [_bond("C1", "C2", atom2_image_shift=(1, 0, 0))],
    )
    perceived = infer_bond_orders(
        face,
        cell=((10.0, 0.0, 0.0), (1.0, 10.0, 0.0), (0.2, 0.3, 10.0)),
    )
    assert perceived.entity.bonds[0].order == 1.0
    assert perceived.entity.bonds[0].atom2_image_shift == (1, 0, 0)
    assert analyze_periodic_topology(("C1", "C2"), perceived.entity.bonds).rank == 0

    chain = _entity(
        "chain",
        [_atom("C", "C", explicit_hydrogens=2)],
        [_bond("C", "C", atom2_image_shift=(1, 0, 0))],
    )
    chain_result = infer_bond_orders(chain)
    assert chain_result.entity.bonds[0].order == 1.0
    assert analyze_periodic_topology(("C",), chain_result.entity.bonds).rank == 1

    sheet = _entity(
        "sheet",
        [_atom("C", "C")],
        [
            _bond("C", "C", atom2_image_shift=(1, 0, 0)),
            _bond("C", "C", atom2_image_shift=(0, 1, 0)),
        ],
    )
    sheet_result = infer_bond_orders(sheet)
    assert [bond.order for bond in sheet_result.entity.bonds] == [1.0, 1.0]
    assert analyze_periodic_topology(("C",), sheet_result.entity.bonds).rank == 2

    framework = _entity(
        "framework",
        [_atom("S", "S")],
        [
            _bond("S", "S", atom2_image_shift=(1, 0, 0)),
            _bond("S", "S", atom2_image_shift=(0, 1, 0)),
            _bond("S", "S", atom2_image_shift=(0, 0, 1)),
        ],
    )
    framework_result = infer_bond_orders(framework)
    assert [bond.order for bond in framework_result.entity.bonds] == [1.0, 1.0, 1.0]
    assert analyze_periodic_topology(("S",), framework_result.entity.bonds).rank == 3

    multiedge = infer_bond_orders(
        _entity(
            "multiedge",
            [
                _atom("C1", "C", explicit_hydrogens=2),
                _atom("C2", "C", explicit_hydrogens=2),
            ],
            [_bond("C1", "C2"), _bond("C1", "C2")],
        )
    )
    assert [bond.order for bond in multiedge.entity.bonds] == [1.0, 1.0]


def test_robustness_family_preserves_orders_under_harmless_transforms():
    original = _entity(
        "ethene",
        [
            _atom("C1", "C", explicit_hydrogens=2),
            _atom("C2", "C", explicit_hydrogens=2),
        ],
        [_bond("C1", "C2")],
        coords=(("C1", (0.0, 0.0, 0.0)), ("C2", (1.34, 0.0, 0.0))),
    )
    reversed_entity = _entity(
        "ethene-reversed",
        [
            _atom("C2", "C", explicit_hydrogens=2),
            _atom("C1", "C", explicit_hydrogens=2),
        ],
        [_bond("C2", "C1")],
        coords=(("C2", (1.34, 0.02, 0.0)), ("C1", (0.0, 0.01, 0.0))),
    )
    translated = _entity(
        "ethene-moved",
        [
            _atom("C1", "C", explicit_hydrogens=2),
            _atom("C2", "C", explicit_hydrogens=2),
        ],
        [_bond("C1", "C2")],
        coords=(("C1", (4.0, -2.0, 1.0)), ("C2", (5.34, -2.0, 1.0))),
    )
    assert _orders(infer_bond_orders(original)) == _orders(
        infer_bond_orders(reversed_entity)
    )
    assert _orders(infer_bond_orders(original)) == _orders(
        infer_bond_orders(translated)
    )

    separate = _entity(
        "salt-like",
        [
            _atom("C1", "C", explicit_hydrogens=3),
            _atom("C2", "C", explicit_hydrogens=3),
            _atom("A0", "C"),
            _atom("A1", "C"),
            _atom("A2", "C"),
            _atom("A3", "C"),
            _atom("A4", "C"),
            _atom("A5", "C"),
        ],
        [_bond("C1", "C2")] + [_bond(f"A{i}", f"A{(i + 1) % 6}") for i in range(6)],
    )
    mixed = infer_bond_orders(separate)
    assert mixed.diagnostics["bond_components"] == 2
    assert mixed.diagnostics["tree_components"] == 1
    assert mixed.diagnostics["cyclic_components"] == 1
    assert _orders(mixed)[frozenset({"C1", "C2"})] == 1.0
    assert all(
        bond.aromatic
        for bond in mixed.entity.bonds
        if bond.atom1_id.startswith("A") and bond.atom2_id.startswith("A")
    )
