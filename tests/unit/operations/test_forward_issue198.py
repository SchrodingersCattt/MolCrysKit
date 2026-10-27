"""Synthetic regressions for the Issue #198 forward-construction contract."""

from itertools import product

import numpy as np
import pytest
from ase import Atoms
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.groups import SpaceGroup

from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder
from molcrys_kit.io.output import (
    describe_structure_export,
    write_cif_asu,
    write_cif_p1,
    write_disorder_cif,
    write_forward_cif,
    write_poscar,
    write_xyz,
)
from molcrys_kit.operations.forward import (
    ConstructionValidationError,
    DISORDER_COMPANION_SCHEMA,
    DISORDER_CONSTRAINT_SCHEMA,
    DisorderCompanionLossError,
    IncompatibleChemistryError,
    InvalidSymmetryError,
    MolecularPose,
    MolecularTemplate,
    PeriodicBondRecord,
    analyze_molecular_pose,
    construct_molecular_crystal,
    construct_statistical_disorder,
    intermolecular_contacts,
    prepare_construction_context,
    read_statistical_disorder_companion,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
    LatticeBasisChange,
)


def _operation(rotation, translation=(0, 0, 0), xyz=None):
    return FractionalAffineOperation(
        np.asarray(rotation, dtype=float),
        np.asarray(translation, dtype=float),
        xyz=xyz,
    )


def _context_from_number(number: int, lattice=None, **kwargs):
    group = SpaceGroup.from_int_number(number)
    operations = tuple(
        FractionalAffineOperation(
            operation.rotation_matrix, operation.translation_vector
        )
        for operation in group.symmetry_ops
    )
    if lattice is None:
        lattice = np.eye(3) * 10.0
    return prepare_construction_context(
        lattice,
        CrystalSymmetry(
            operations,
            space_group_number=number,
            space_group_symbol=group.symbol,
        ),
        **kwargs,
    )


def _p1():
    return _context_from_number(1)


def _tetrahedron(scale=0.7):
    return (
        np.array(
            [
                vector
                for vector in product((-1.0, 1.0), repeat=3)
                if np.prod(vector) == 1.0
            ],
            dtype=float,
        )
        * scale
    )


def _cif_column(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _weighted_composition(cif_text: str) -> dict[str, float]:
    info = scan_cif_disorder(cif_text=cif_text)
    totals: dict[str, float] = {}
    for symbol, occupancy in zip(info.symbols, info.occupancies):
        totals[symbol] = totals.get(symbol, 0.0) + float(occupancy)
    return totals


def test_special_methane_asu_keeps_two_molecules_and_eight_bonds():
    context = _context_from_number(217)
    template = MolecularTemplate(
        Atoms("CH4", positions=np.vstack((np.zeros(3), _tetrahedron(0.63)))),
        connectivity=tuple((0, index) for index in range(1, 5)),
    )
    crystal = construct_molecular_crystal(context, MolecularPose(template, np.zeros(3)))
    assert len(crystal.molecules) == 2
    cif = write_cif_asu(crystal)
    parsed = CifParser.from_str(cif, occupancy_tolerance=1.0).parse_structures(
        primitive=False
    )
    assert len(parsed[0]) == 10
    restored = read_mol_crystal(cif_text=cif, attach_chemistry=False)
    assert [molecule.get_chemical_formula() for molecule in restored.molecules] == [
        "CH4",
        "CH4",
    ]
    assert sorted(
        molecule.graph.number_of_edges() for molecule in restored.molecules
    ) == [4, 4]
    assert sum(molecule.graph.number_of_edges() for molecule in restored.molecules) == 8


def test_explicit_enantiopure_general_position_fails_and_racemate_reassigns_stereo():
    context = _context_from_number(
        2, lattice=np.eye(3) * 20.0, chemistry_policy="enantiopure"
    )
    template = MolecularTemplate(
        Atoms(
            ["C", "H", "F", "Cl", "Br"],
            positions=np.vstack((np.zeros(3), _tetrahedron())),
        ),
        connectivity=tuple((0, index) for index in range(1, 5)),
        stereochemistry=("R", "", "", "", ""),
        chemical_identity="explicit-enantiopure",
    )
    pose = MolecularPose(template, np.array([0.21, 0.29, 0.37]))
    analysis = analyze_molecular_pose(context, pose)
    assert analysis.ordered_compatible
    assert not analysis.group_chemistry_compatible
    with pytest.raises(IncompatibleChemistryError):
        construct_molecular_crystal(context, pose)

    racemic = _context_from_number(
        2, lattice=np.eye(3) * 20.0, chemistry_policy="racemic"
    )
    typed = MolecularTemplate(
        Atoms(
            ["C", "H", "F", "Cl", "Br"],
            positions=np.vstack((np.zeros(3), _tetrahedron())),
        ),
        connectivity=tuple(
            {
                "left": 0,
                "right": index,
                "order": 1,
                "bond_type": "single",
                "kind": "covalent",
            }
            for index in range(1, 5)
        ),
        stereochemistry=("S", "", "", "", ""),
    )
    crystal = construct_molecular_crystal(
        racemic, MolecularPose(typed, np.array([0.21, 0.29, 0.37]))
    )
    labels = [
        str(molecule.arrays["stereochemistry"][0]) for molecule in crystal.molecules
    ]
    assert sorted(labels) == ["R", "S"]

    alkene = MolecularTemplate(
        Atoms(
            "C2F2H2",
            positions=[
                [0.0, 0.0, 0.0],
                [1.34, 0.0, 0.0],
                [-0.70, 1.10, 0.0],
                [2.04, -1.10, 0.0],
                [-0.50, -0.90, 0.0],
                [1.84, 0.90, 0.0],
            ],
        ),
        connectivity=(
            (0, 1, {"order": 2, "bond_type": "double"}),
            (0, 2, {"order": 1, "bond_type": "single"}),
            (0, 4, {"order": 1, "bond_type": "single"}),
            (1, 3, {"order": 1, "bond_type": "single"}),
            (1, 5, {"order": 1, "bond_type": "single"}),
        ),
        stereochemistry=("E", "", "", "", "", ""),
    )
    alkene_crystal = construct_molecular_crystal(
        racemic, MolecularPose(alkene, np.array([0.2, 0.3, 0.4]))
    )
    improper = [
        molecule
        for molecule in alkene_crystal.molecules
        if molecule.info.get("forward_stereo_reassigned")
    ]
    assert improper
    assert all(
        str(molecule.arrays["stereochemistry"][0]) == "E"
        for molecule in alkene_crystal.molecules
    )

    untyped = MolecularTemplate(
        Atoms(
            ["C", "H", "F", "Cl", "Br"],
            positions=np.vstack((np.zeros(3), _tetrahedron())),
        ),
        connectivity=tuple((0, index) for index in range(1, 5)),
        stereochemistry=("R", "", "", "", ""),
    )
    untyped_crystal = construct_molecular_crystal(
        racemic, MolecularPose(untyped, np.array([0.21, 0.29, 0.37]))
    )
    improper = [
        molecule
        for molecule in untyped_crystal.molecules
        if molecule.info.get("forward_stereo_reassigned")
    ]
    assert improper
    assert all(
        str(molecule.arrays["stereochemistry"][0]) == "" for molecule in improper
    )


def test_typed_bonds_survive_template_graph_and_both_cif_paths():
    context = _p1()
    template = MolecularTemplate(
        Atoms("O2", positions=[[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]]),
        connectivity=((0, 1, {"order": 2, "bond_type": "double", "kind": "covalent"}),),
    )
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, np.array([0.2, 0.2, 0.2]))
    )
    edge = crystal.molecules[0].graph.get_edge_data(0, 1)
    assert edge["order"] == 2
    assert edge["bond_type"] == "double"
    for writer in (write_cif_p1, write_cif_asu):
        block = next(iter(CifParser.from_str(writer(crystal)).as_dict().values()))
        assert _cif_column(block["_geom_bond_type"]) == ["double"]
        restored = read_mol_crystal(cif_text=writer(crystal), attach_chemistry=False)
        restored_edge = restored.molecules[0].graph.get_edge_data(0, 1)
        assert restored_edge["bond_type"] == "double"
        assert restored_edge["order"] == 2

    legacy = MolecularTemplate(
        Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        connectivity=((0, 1),),
    )
    legacy_crystal = construct_molecular_crystal(
        context, MolecularPose(legacy, np.array([0.2, 0.2, 0.2]))
    )
    legacy_block = next(
        iter(CifParser.from_str(write_cif_p1(legacy_crystal)).as_dict().values())
    )
    assert _cif_column(legacy_block["_geom_bond_type"]) == ["."]

    aromatic = MolecularTemplate(
        Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        bonds=(
            PeriodicBondRecord(
                0, 1, bond_type="aromatic", kind="covalent", bond_type_known=True
            ),
        ),
    )
    coordination = MolecularTemplate(
        Atoms("NiO", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        bonds=(
            PeriodicBondRecord(
                0,
                1,
                bond_type="coordination",
                kind="coordination",
                bond_type_known=True,
            ),
        ),
    )
    for typed, expected in ((aromatic, "aromatic"), (coordination, "coordination")):
        built = construct_molecular_crystal(
            context, MolecularPose(typed, np.array([0.25, 0.25, 0.25]))
        )
        block = next(iter(CifParser.from_str(write_cif_p1(built)).as_dict().values()))
        assert _cif_column(block["_geom_bond_type"]) == [expected]


def test_isotope_and_explicit_charge_round_trip_and_do_not_merge():
    context = _p1()
    deuterium = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    deuterium.set_array("isotope", np.array([2, 2]))
    deuterium.set_masses([2.01410177812, 2.01410177812])
    crystal = construct_molecular_crystal(
        context,
        MolecularPose(
            MolecularTemplate(deuterium, connectivity=((0, 1),)),
            np.array([0.2, 0.2, 0.2]),
        ),
    )
    info = scan_cif_disorder(cif_text=write_cif_p1(crystal))
    assert info.isotopes == [2, 2]
    restored = read_mol_crystal(cif_text=write_cif_p1(crystal), attach_chemistry=False)
    assert np.allclose(
        restored.molecules[0].get_masses(), [2.01410177812, 2.01410177812]
    )

    poses = []
    for symbol, charge, anchor in (
        ("Na", 1, [0.2, 0.2, 0.2]),
        ("Cl", -1, [0.7, 0.7, 0.7]),
    ):
        atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
        atoms.set_array("formal_charge", np.array([charge]))
        atoms.set_array("formal_charge_known", np.array([True]))
        poses.append(MolecularPose(MolecularTemplate(atoms), np.array(anchor)))
    ions = construct_molecular_crystal(context, poses)
    charges = scan_cif_disorder(cif_text=write_cif_p1(ions)).site_formal_charges
    assert charges == [1, -1]

    neutral = Atoms("C", positions=[[0.0, 0.0, 0.0]])
    neutral.set_array("formal_charge", np.array([0]))
    neutral.set_array("formal_charge_known", np.array([True]))
    unknown = Atoms("C", positions=[[0.0, 0.0, 0.0]])
    pair = construct_molecular_crystal(
        context,
        [
            MolecularPose(MolecularTemplate(neutral), np.array([0.1, 0.1, 0.1])),
            MolecularPose(MolecularTemplate(unknown), np.array([0.6, 0.2, 0.2])),
        ],
    )
    assert scan_cif_disorder(cif_text=write_cif_p1(pair)).site_formal_charges == [
        0,
        None,
    ]

    protium = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    protium.set_array("isotope", np.array([1]))
    heavy = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    heavy.set_array("isotope", np.array([2]))
    model = construct_statistical_disorder(
        context,
        [
            MolecularPose(
                MolecularTemplate(protium, identifier="h"),
                np.array([0.2, 0.2, 0.2]),
                occupation=0.5,
            ),
            MolecularPose(
                MolecularTemplate(heavy, identifier="d"),
                np.array([0.2, 0.2, 0.2]),
                occupation=0.5,
            ),
        ],
    )
    assert len(model.average_sites) == 2
    assert {site.isotope for site in model.average_sites} == {1, 2}
    assert all(np.isclose(site.occupancy, 0.5) for site in model.average_sites)


def test_deuterium_and_tritium_aliases_and_conflicts():
    def cif_for(symbol: str) -> str:
        return (
            "data_isotope\n"
            "_symmetry_space_group_name_H-M 'P 1'\n"
            "_symmetry_Int_Tables_number 1\n"
            "loop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n"
            "_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "loop_\n"
            "_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
            f"H1 {symbol} 0.1 0.2 0.3\n"
        )

    deuterium = read_mol_crystal(cif_text=cif_for("D"), attach_chemistry=False)
    tritium = read_mol_crystal(cif_text=cif_for("T"), attach_chemistry=False)
    numbered = read_mol_crystal(cif_text=cif_for("2H"), attach_chemistry=False)
    assert deuterium.molecules[0].get_chemical_symbols() == ["H"]
    assert int(deuterium.molecules[0].arrays["isotope"][0]) == 2
    assert int(tritium.molecules[0].arrays["isotope"][0]) == 3
    assert int(numbered.molecules[0].arrays["isotope"][0]) == 2
    assert scan_cif_disorder(cif_text=cif_for("D")).raw_type_symbols == ["D"]
    with pytest.raises(ValueError, match="conflicting isotope"):
        scan_cif_disorder(cif_text=cif_for("3D"))


def test_disorder_companion_round_trip_and_legacy_loss():
    context = _context_from_number(2)
    water = MolecularTemplate(
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        anchor_index=0,
        connectivity=(
            (0, 1, {"order": 1, "bond_type": "single"}),
            (0, 2, {"order": 1, "bond_type": "single"}),
        ),
    )
    model = construct_statistical_disorder(context, MolecularPose(water, np.zeros(3)))
    payload = model.to_dict()
    assert payload["schema"] == DISORDER_COMPANION_SCHEMA
    assert payload["alternatives"][0]["atoms"]
    assert payload["alternatives"][0]["bonds"]
    restored = read_statistical_disorder_companion(payload)
    again = restored.to_dict()
    third = read_statistical_disorder_companion(again).to_dict()
    assert again["schema"] == payload["schema"]
    assert len(again["alternatives"]) == len(payload["alternatives"])
    assert again["alternatives"][0]["bonds"][0]["bond_type"] == "single"

    def _cartesians(document):
        return [
            np.asarray(atom["cartesian"], dtype=float)
            for alternative in document["alternatives"]
            for atom in alternative["atoms"]
        ]

    original_coordinates = _cartesians(payload)
    restored_coordinates = _cartesians(again)
    rewritten_coordinates = _cartesians(third)
    assert original_coordinates
    assert (
        len(original_coordinates)
        == len(restored_coordinates)
        == len(rewritten_coordinates)
    )
    for left, right in zip(original_coordinates, restored_coordinates):
        assert np.allclose(left, right, atol=1.0e-12, rtol=0.0)
    for left, right in zip(restored_coordinates, rewritten_coordinates):
        assert np.array_equal(left, right)
    selected = restored.select_ordered_realization()
    assert len(selected.molecules) == 1
    assert selected.molecules[0].get_chemical_formula() == "H2O"
    legacy = {
        "representation": "statistical_average",
        "alternatives": [
            {"orbit_id": "center0:orbit0", "alternative_id": "a", "weight": 1.0}
        ],
    }
    with pytest.raises(DisorderCompanionLossError):
        read_statistical_disorder_companion(legacy)


def test_statistical_true_asu_matches_weighted_inventory_without_double_expansion():
    context = _context_from_number(2)
    water = MolecularTemplate(
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0]]),
        anchor_index=0,
    )
    model = construct_statistical_disorder(context, MolecularPose(water, np.zeros(3)))
    asu = write_forward_cif(model, representation="asu")
    assert "_symmetry_Int_Tables_number 2" in asu
    assert _weighted_composition(asu) == pytest.approx({"O": 1.0, "H": 2.0})
    p1 = write_forward_cif(model, representation="p1")
    assert "_symmetry_Int_Tables_number 1" in p1
    assert _weighted_composition(p1) == pytest.approx({"O": 1.0, "H": 2.0})

    centred = _context_from_number(12, lattice=np.eye(3) * 12.0)
    carbon = MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]]), anchor_index=0)
    centred_model = construct_statistical_disorder(
        centred, MolecularPose(carbon, np.array([0.13, 0.21, 0.37]))
    )
    centred_asu = write_forward_cif(centred_model, representation="asu")
    centred_p1 = write_disorder_cif(centred_model)
    assert _weighted_composition(centred_asu) == pytest.approx(
        {"C": float(len(centred.operations))}
    )
    assert _weighted_composition(centred_p1) == pytest.approx(
        {"C": float(len(centred.operations))}
    )
    assert "statistical_average" in centred_p1
    assert "_symmetry_Int_Tables_number 1" in centred_p1


def test_zero_weight_and_partial_occupation_selection():
    context = _p1()
    water = MolecularTemplate(
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        anchor_index=0,
        connectivity=((0, 1), (0, 2)),
    )
    vacant = construct_statistical_disorder(
        context, MolecularPose(water, np.array([0.2, 0.2, 0.2])), center_occupation=0.0
    )
    assert [alternative.weight for alternative in vacant.alternatives] == [0.0]
    assert vacant.vacancies and np.isclose(vacant.vacancies[0].weight, 1.0)
    assert (
        sum(len(molecule) for molecule in vacant.select_ordered_realization().molecules)
        == 0
    )
    with pytest.raises(ConstructionValidationError, match="zero-weight"):
        vacant.select_ordered_realization(
            {vacant.alternatives[0].orbit_id: vacant.alternatives[0].alternative_id}
        )

    partial = construct_statistical_disorder(
        context, MolecularPose(water, np.array([0.2, 0.2, 0.2])), center_occupation=0.5
    )
    with pytest.raises(ConstructionValidationError, match="partial center occupation"):
        partial.select_ordered_realization()
    with pytest.raises(ConstructionValidationError, match="partial center occupation"):
        partial.select_ordered_realization({})
    orbit = partial.alternatives[0].orbit_id
    occupied = partial.select_ordered_realization(
        {orbit: partial.alternatives[0].alternative_id}
    )
    empty = partial.select_ordered_realization(
        {orbit: partial.vacancies[0].alternative_id}
    )
    assert sum(len(molecule) for molecule in occupied.molecules) == 3
    assert empty.molecules == []
    assert occupied.metadata["realized_symmetry"]["matches_source_group"] is True

    two_centers = construct_statistical_disorder(
        context,
        [
            MolecularPose(water, np.array([0.2, 0.2, 0.2]), occupation=0.5),
            MolecularPose(water, np.array([0.7, 0.3, 0.4]), occupation=0.5),
        ],
    )
    orbits = list(two_centers.alternatives_by_orbit)
    assert len(orbits) == 2
    first_choice = two_centers.alternatives_by_orbit[orbits[0]][0].alternative_id
    second_choice = two_centers.alternatives_by_orbit[orbits[1]][0].alternative_id
    with pytest.raises(ConstructionValidationError, match="partial center occupation"):
        two_centers.select_ordered_realization({orbits[0]: first_choice})
    both = two_centers.select_ordered_realization(
        {orbits[0]: first_choice, orbits[1]: second_choice}
    )
    assert len(both.molecules) == 2


def test_generated_bond_image_shift_follows_the_operation():
    context = _context_from_number(2, lattice=np.eye(3) * 20.0)
    template = MolecularTemplate(
        Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        bonds=(
            PeriodicBondRecord(
                0,
                1,
                order=1.0,
                bond_type="single",
                kind="covalent",
                image_shift=(1, 0, 0),
                order_known=True,
                bond_type_known=True,
            ),
        ),
    )
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, np.array([0.2, 0.3, 0.4]))
    )
    shifts = {}
    for molecule in crystal.molecules:
        operation = context.operations[molecule.info["forward_operation_index"]]
        shift = tuple(
            int(value) for value in molecule.graph.get_edge_data(0, 1)["image_shift"]
        )
        assert tuple(molecule.info["forward_bonds"][0]["image_shift"]) == shift
        shifts[int(np.sign(operation.determinant))] = shift
    assert shifts[1] == (1, 0, 0)
    assert shifts[-1] == (-1, 0, 0)


def test_stabilizer_matching_does_not_wrap_atoms_independently():
    context = prepare_construction_context(
        np.diag([2.4, 10.0, 10.0]),
        CrystalSymmetry(
            (
                _operation(np.eye(3), xyz="x,y,z"),
                _operation(-np.eye(3), xyz="-x,-y,-z"),
            ),
            space_group_number=2,
            space_group_symbol="P -1",
        ),
    )
    analysis = analyze_molecular_pose(
        context,
        MolecularPose(
            MolecularTemplate(
                Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
                anchor_index=0,
                connectivity=((0, 1),),
            ),
            np.zeros(3),
        ),
    )
    assert analysis.stabilizer_order == 1
    assert not analysis.ordered_compatible

    long_context = _context_from_number(2)
    long_analysis = analyze_molecular_pose(
        long_context,
        MolecularPose(
            MolecularTemplate(
                Atoms("C2", positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
                anchor_cartesian=np.zeros(3),
                connectivity=((0, 1),),
            ),
            np.zeros(3),
        ),
    )
    assert long_analysis.stabilizer_order == 2
    assert long_analysis.ordered_compatible


def test_declared_group_must_match_the_complete_operation_set():
    identity = _operation(np.eye(3), xyz="x,y,z")
    with pytest.raises(InvalidSymmetryError):
        prepare_construction_context(
            np.eye(3) * 10.0,
            CrystalSymmetry(
                (identity,), space_group_number=2, space_group_symbol="P -1"
            ),
        )
    mirror = _operation(np.diag([1.0, -1.0, 1.0]), xyz="x,-y,z")
    with pytest.raises(InvalidSymmetryError):
        prepare_construction_context(
            np.eye(3) * 10.0,
            CrystalSymmetry(
                (identity, mirror), space_group_number=2, space_group_symbol="P -1"
            ),
        )
    custom = prepare_construction_context(np.eye(3) * 10.0, (identity,))
    assert custom.space_group_number is None

    standard = _context_from_number(2)
    change = LatticeBasisChange(np.eye(3), np.array([0.0, 0.0, 0.25]))
    transformed = tuple(
        change.transform_operation(operation) for operation in standard.operations
    )
    with pytest.raises(InvalidSymmetryError):
        prepare_construction_context(
            np.eye(3) * 10.0,
            CrystalSymmetry(
                transformed, space_group_number=2, space_group_symbol="P -1"
            ),
        )
    accepted = prepare_construction_context(
        np.eye(3) * 10.0,
        CrystalSymmetry(transformed, space_group_number=2, space_group_symbol="P -1"),
        origin_choice="0,0,1/4",
    )
    assert len(accepted.operations) == 2
    mirror = _context_from_number(6)
    axis_swap = LatticeBasisChange(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    conjugated = tuple(
        axis_swap.transform_operation(operation) for operation in mirror.operations
    )
    original_rotations = {
        tuple(np.rint(operation.rotation).astype(int).ravel())
        for operation in mirror.operations
    }
    conjugated_rotations = {
        tuple(np.rint(operation.rotation).astype(int).ravel())
        for operation in conjugated
    }
    assert original_rotations != conjugated_rotations
    with pytest.raises(InvalidSymmetryError):
        prepare_construction_context(
            np.eye(3) * 10.0,
            CrystalSymmetry(
                conjugated,
                space_group_number=6,
                space_group_symbol=mirror.symmetry.space_group_symbol,
            ),
        )
    basis_ok = prepare_construction_context(
        np.eye(3) * 10.0,
        CrystalSymmetry(
            conjugated,
            space_group_number=6,
            space_group_symbol=mirror.symmetry.space_group_symbol,
        ),
        basis="[[0, 1, 0], [1, 0, 0], [0, 0, 1]]",
    )
    assert len(basis_ok.operations) == len(mirror.operations)
    with pytest.raises(InvalidSymmetryError):
        prepare_construction_context(
            np.eye(3) * 10.0,
            CrystalSymmetry(
                standard.operations,
                space_group_number=2,
                space_group_symbol="P -1",
                hall_symbol="-P 1",
            ),
            setting="P 2/m",
        )


def test_joint_constraints_distinguish_coupling_and_reject_contradictions():
    context = _context_from_number(2)
    water = MolecularTemplate(
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0]]),
        anchor_index=0,
    )
    poses = (
        MolecularPose(water, np.zeros(3)),
        MolecularPose(water, np.array([0.5, 0.5, 0.5])),
    )
    probe = construct_statistical_disorder(context, poses)
    by_orbit = probe.alternatives_by_orbit
    assert len(by_orbit) == 2
    orbits = list(by_orbit)
    first_states = [item.alternative_id for item in by_orbit[orbits[0]]]
    second_states = [item.alternative_id for item in by_orbit[orbits[1]]]
    same_sense = {
        "schema": DISORDER_CONSTRAINT_SCHEMA,
        "joints": [
            {
                "weight": 0.5,
                "assignment": {orbits[0]: first_states[0], orbits[1]: second_states[0]},
            },
            {
                "weight": 0.5,
                "assignment": {orbits[0]: first_states[1], orbits[1]: second_states[1]},
            },
        ],
    }
    opposite_sense = {
        "schema": DISORDER_CONSTRAINT_SCHEMA,
        "joints": [
            {
                "weight": 0.5,
                "assignment": {orbits[0]: first_states[0], orbits[1]: second_states[1]},
            },
            {
                "weight": 0.5,
                "assignment": {orbits[0]: first_states[1], orbits[1]: second_states[0]},
            },
        ],
    }
    coupled = construct_statistical_disorder(
        context, poses, correlation_status="feasible", correlations=same_sense
    )
    anticorrelated = construct_statistical_disorder(
        context, poses, correlation_status="feasible", correlations=opposite_sense
    )
    assert coupled.to_dict()["correlations"] != anticorrelated.to_dict()["correlations"]
    assert coupled.constraint_report["status"] == "feasible"
    with pytest.raises(ConstructionValidationError):
        construct_statistical_disorder(
            context,
            poses,
            correlation_status="feasible",
            correlations={
                "schema": DISORDER_CONSTRAINT_SCHEMA,
                "joints": [
                    {
                        "weight": 1.0,
                        "assignment": {
                            orbits[0]: first_states[0],
                            orbits[1]: second_states[0],
                        },
                    }
                ],
            },
        )


def test_acceptance_geometry_symmetry_and_export_contracts():
    lattice = np.array(
        [[10.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.4, 0.2, 8.0]],
        dtype=float,
    )
    context = _context_from_number(1, lattice=lattice)
    template = MolecularTemplate(
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]]),
        anchor_index=0,
        connectivity=((0, 1),),
    )
    anchor = np.array([0.17, 0.23, 0.31])
    orientation = np.eye(3)
    caller = template.atoms.copy()
    caller_positions = caller.positions.copy()
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, anchor, orientation=orientation)
    )
    assert np.allclose(caller.positions, caller_positions)
    expected = (
        np.asarray(template.atoms.positions, dtype=float) - template.atoms.positions[0]
    )
    expected = expected + anchor @ lattice
    assert np.allclose(crystal.molecules[0].positions, expected, atol=1.0e-8)

    p4 = _context_from_number(75)
    rotor = MolecularTemplate(
        Atoms("ClCCl", positions=[[1.7, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.7, 0.0, 0.0]]),
        anchor_index=1,
        connectivity=((0, 1), (1, 2)),
    )
    special = analyze_molecular_pose(p4, MolecularPose(rotor, np.zeros(3)))
    assert (
        special.site_symmetry_order,
        special.stabilizer_order,
        special.orientation_multiplicity,
    ) == (4, 2, 2)
    disordered = construct_statistical_disorder(p4, MolecularPose(rotor, np.zeros(3)))
    assert len(disordered.alternatives) == 2
    assert all(np.isclose(item.weight, 0.5) for item in disordered.alternatives)

    general = _context_from_number(217)
    carbon = MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]]), anchor_index=0)
    pose = MolecularPose(carbon, np.array([0.13, 0.17, 0.19]))
    high = analyze_molecular_pose(general, pose)
    built = construct_molecular_crystal(general, pose)
    assert high.site_symmetry_order == 1
    assert high.orientation_multiplicity == 1
    assert len(built.molecules) == len(general.operations)

    benzene = MolecularTemplate(
        Atoms(
            ["C"] * 6 + ["H"] * 6,
            positions=np.vstack(
                (
                    np.array(
                        [
                            [1.39 * np.cos(angle), 1.39 * np.sin(angle), 0.0]
                            for angle in np.linspace(
                                0.0, 2.0 * np.pi, 6, endpoint=False
                            )
                        ]
                    ),
                    np.array(
                        [
                            [2.48 * np.cos(angle), 2.48 * np.sin(angle), 0.0]
                            for angle in np.linspace(
                                0.0, 2.0 * np.pi, 6, endpoint=False
                            )
                        ]
                    ),
                )
            ),
        ),
        identifier="constructed-c6h6",
        anchor_cartesian=np.zeros(3),
    )
    benzene_crystal = construct_molecular_crystal(
        _p1(), MolecularPose(benzene, np.array([0.2, 0.2, 0.2]))
    )
    assert benzene_crystal.molecules[0].get_chemical_formula() == "C6H6"
    assert benzene_crystal.metadata["hydrogen_policy"] == "explicit_template"

    near = _context_from_number(2)
    on_site = analyze_molecular_pose(
        near,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])), np.zeros(3)
        ),
    )
    inside = analyze_molecular_pose(
        near,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
            np.array([1.0e-8, 0.0, 0.0]),
        ),
    )
    outside = analyze_molecular_pose(
        near,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
            np.array([1.0e-3, 0.0, 0.0]),
        ),
    )
    assert on_site.site_symmetry_order == 2
    assert on_site.near_special_distance == pytest.approx(0.0, abs=1.0e-8)
    assert on_site.near_special_distance_A == pytest.approx(0.0, abs=1.0e-6)
    assert inside.site_symmetry_order == 2
    assert outside.site_symmetry_order == 1
    assert outside.near_special_distance_A == pytest.approx(0.02, rel=1.0e-6)
    outside_crystal = construct_molecular_crystal(
        near,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
            np.array([1.0e-3, 0.0, 0.0]),
        ),
    )
    assert len(outside_crystal.molecules) == 2
    assert np.allclose(
        outside_crystal.molecules[0].positions[0] / 10.0,
        np.array([1.0e-3, 0.0, 0.0]),
        atol=1.0e-8,
    ) or np.allclose(
        outside_crystal.molecules[1].positions[0] / 10.0,
        np.array([1.0e-3, 0.0, 0.0]),
        atol=1.0e-8,
    )

    reversed_ops = prepare_construction_context(
        np.eye(3) * 10.0,
        (
            _operation(-np.eye(3), xyz="-x,-y,-z"),
            _operation(np.eye(3), xyz="x,y,z"),
        ),
        space_group_number=2,
        space_group_symbol="P -1",
    )
    forward_ops = _context_from_number(2)
    left = construct_molecular_crystal(
        reversed_ops,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
            np.array([0.2, 0.3, 0.4]),
        ),
    )
    right = construct_molecular_crystal(
        forward_ops,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
            np.array([0.2, 0.3, 0.4]),
        ),
    )
    assert np.allclose(
        np.sort(np.vstack([molecule.positions for molecule in left.molecules]), axis=0),
        np.sort(
            np.vstack([molecule.positions for molecule in right.molecules]), axis=0
        ),
    )

    clash = construct_molecular_crystal(
        _p1(),
        [
            MolecularPose(
                MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
                np.array([0.2, 0.2, 0.2]),
            ),
            MolecularPose(
                MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
                np.array([0.21, 0.2, 0.2]),
            ),
        ],
    )
    assert intermolecular_contacts(clash)
    with pytest.raises(ConstructionValidationError, match="intermolecular contact"):
        construct_molecular_crystal(
            _p1(),
            [
                MolecularPose(
                    MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
                    np.array([0.2, 0.2, 0.2]),
                ),
                MolecularPose(
                    MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]])),
                    np.array([0.21, 0.2, 0.2]),
                ),
            ],
            reject_contacts=True,
        )

    assert describe_structure_export("xyz")["lossy"] is True
    assert describe_structure_export("poscar")["occupancy"] is False
    assert describe_structure_export("companion_json")["disorder_alternatives"] is True
    partial_atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]])
    partial_atoms.set_array("occupancy", np.array([0.5]))
    partial = CrystalMolecule(partial_atoms, check_pbc=False)
    partial_crystal = MolecularCrystal(np.eye(3) * 10.0, [partial])
    with pytest.warns(UserWarning, match="not a selected physical microstate"):
        write_xyz(partial)
    with pytest.warns(UserWarning, match="not a selected physical microstate"):
        write_poscar(partial_crystal)
