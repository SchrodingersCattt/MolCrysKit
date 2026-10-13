"""Contract tests for the Issue #172 forward-construction kernel."""

import numpy as np
import pytest
from ase import Atoms
from pymatgen.io.cif import CifParser

from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_ATOM_ID,
    KEY_DISORDER_GROUP,
    KEY_LABEL,
    KEY_OCCUPANCY,
    KEY_SITE_SYMMETRY_ORDER,
    KEY_SYM_OP_INDEX,
)
from molcrys_kit.io.cif import read_mol_crystal, scan_cif_disorder
from molcrys_kit.io.output import write_cif_asu, write_cif_p1, write_disorder_cif
from molcrys_kit.operations.forward import (
    ConstructionValidationError,
    IncompatibleSiteError,
    InvalidOrientationError,
    MolecularPose,
    MolecularTemplate,
    analyze_molecular_pose,
    construct_molecular_crystal,
    construct_statistical_disorder,
    prepare_construction_context,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.symmetry import FractionalAffineOperation


def _operation(rotation, translation=(0, 0, 0), xyz=None, index=None):
    return FractionalAffineOperation(
        np.asarray(rotation, dtype=float),
        np.asarray(translation, dtype=float),
        xyz=xyz,
        index=index,
    )


def _pminus1_context(lattice=None):
    lattice = np.eye(3) * 10.0 if lattice is None else lattice
    return prepare_construction_context(
        lattice,
        (
            _operation(np.eye(3), xyz="x,y,z", index=0),
            _operation(-np.eye(3), xyz="-x,-y,-z", index=1),
        ),
        space_group_number=2,
        space_group_symbol="P -1",
    )


def test_general_position_expands_two_complete_molecules_without_disorder():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.2, 0.1]]),
        identifier="co",
        anchor_index=0,
    )
    pose = MolecularPose(template, [0.17, 0.23, 0.31])
    analysis = analyze_molecular_pose(context, pose)
    assert analysis.orbit_multiplicity == 2
    assert analysis.site_symmetry_order == 1
    assert analysis.ordered_compatible
    crystal = construct_molecular_crystal(context, pose)
    assert len(crystal.molecules) == 2
    assert sum(len(molecule) for molecule in crystal.molecules) == 4
    assert all(np.isclose(molecule.arrays["occupancy"], 1.0).all() for molecule in crystal.molecules)


def test_asu_round_trip_keeps_general_site_near_mirror_and_explicit_bonds():
    """A near-plane general site is distinct from its mirror image."""
    context = prepare_construction_context(
        np.eye(3) * 10.0,
        (
            _operation(np.eye(3), xyz="x,y,z", index=0),
            _operation(np.diag([1.0, -1.0, 1.0]), xyz="x,-y,z", index=1),
        ),
        space_group_number=6,
        space_group_symbol="P m",
    )
    template = MolecularTemplate(
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [4.0, 1.2, 0.0]]),
        identifier="near-mirror-co",
        anchor_index=0,
        connectivity=((0, 1),),
    )
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, [0.2, 0.00036042, 0.2])
    )
    asu = write_cif_asu(crystal)

    # Independent symmetry expansion and MolCrysKit read-back must agree on
    # the two complete C--O molecules.  The C image separation is 0.0072 Å,
    # below the old 0.01 Å Cartesian deduplication threshold.
    parsed = CifParser.from_str(asu, occupancy_tolerance=1.0).parse_structures(
        primitive=False
    )
    assert len(parsed) == 1
    assert len(parsed[0]) == 4
    expanded = scan_cif_disorder(cif_text=asu)
    assert len(expanded.symbols) == 4

    restored = read_mol_crystal(cif_text=asu, attach_chemistry=False)
    assert sum(len(molecule) for molecule in restored.molecules) == 4
    assert len(restored.molecules) == 2
    assert all(list(molecule.graph.edges()) == [(0, 1)] for molecule in restored.molecules)


def test_inversion_special_site_requires_explicit_disorder_and_merges_anchor_atom():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0]]),
        identifier="water",
        anchor_index=0,
    )
    pose = MolecularPose(template, [0.0, 0.0, 0.0])
    analysis = analyze_molecular_pose(context, pose)
    assert (analysis.site_symmetry_order, analysis.stabilizer_order, analysis.orientation_multiplicity) == (2, 1, 2)
    with pytest.raises(IncompatibleSiteError):
        construct_molecular_crystal(context, pose)
    model = construct_statistical_disorder(context, pose)
    assert len(model.alternatives) == 2
    assert model.expected_formula == {"O": 1.0, "H": 2.0}
    oxygen = [site for site in model.average_sites if site.symbol == "O"]
    hydrogen = [site for site in model.average_sites if site.symbol == "H"]
    assert len(oxygen) == 1 and np.isclose(oxygen[0].occupancy, 1.0)
    assert len(hydrogen) == 4 and all(np.isclose(site.occupancy, 0.5) for site in hydrogen)
    cif = write_disorder_cif(model)
    assert "_molcrys_representation 'statistical_average'" in cif
    assert "_atom_site_occupancy" in cif


def test_proper_rotation_is_required():
    template = MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), identifier="c")
    with pytest.raises(InvalidOrientationError):
        MolecularPose(template, [0, 0, 0], orientation=np.diag([1.0, 1.0, -1.0]))


def test_true_asu_writer_uses_the_declared_group_and_not_p1():
    context = _pminus1_context()
    template = MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), identifier="c", anchor_index=0)
    crystal = construct_molecular_crystal(context, MolecularPose(template, [0.13, 0.21, 0.37]))
    cif = write_cif_asu(crystal)
    assert "_symmetry_Int_Tables_number 2" in cif
    assert "-x,-y,-z" in cif
    assert cif.count("_atom_site_label") == 1
    # A general-position single atom has one ASU row; the inversion operation
    # supplies the second full-cell image on read.
    assert sum(line.strip().startswith("C") for line in cif.splitlines()) == 1


def test_true_asu_writer_keeps_independent_same_element_sites_and_occupancies():
    """Coordinate equivalence cannot erase independent molecular entities."""
    context = _pminus1_context()
    molecules = []
    for label, occupancy, group in (("C_A", 0.5, 1), ("C_B", 0.25, 2)):
        atoms = Atoms("C", positions=[[1.3, 2.1, 3.7]])
        atoms.set_array(KEY_LABEL, np.asarray([label], dtype=str))
        atoms.set_array(KEY_ATOM_ID, np.asarray([label], dtype=str))
        atoms.set_array(KEY_OCCUPANCY, np.asarray([occupancy], dtype=float))
        atoms.set_array(KEY_DISORDER_GROUP, np.asarray([group], dtype=int))
        atoms.set_array(KEY_ASSEMBLY, np.asarray(["A"], dtype=str))
        molecules.append(CrystalMolecule(atoms, check_pbc=False))
    crystal = MolecularCrystal(
        context.lattice,
        molecules,
        metadata={"forward_context": context},
    )

    cif = write_cif_asu(crystal)
    atom_rows = [
        line.split()
        for line in cif.splitlines()
        if line.startswith("  C_")
    ]
    assert {row[0] for row in atom_rows} == {"C_A", "C_B"}
    assert {float(row[5]) for row in atom_rows} == {0.5, 0.25}
    assert {int(row[6]) for row in atom_rows} == {1, 2}
    assert {row[7] for row in atom_rows} == {"A"}
    assert "_molcrys_atom_id" in cif
    assert "C_A" in cif and "C_B" in cif


def test_center_occupancy_is_canonical_across_template_identifiers():
    context = _pminus1_context()
    first = MolecularPose(
        MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), identifier="carbon"),
        [0.0, 0.0, 0.0],
    )
    second = MolecularPose(
        MolecularTemplate(Atoms("N", positions=[[0, 0, 0]]), identifier="nitrogen"),
        [0.0, 0.0, 0.0],
    )
    with pytest.raises(ConstructionValidationError, match="center orbit"):
        construct_molecular_crystal(context, [first, second])
    with pytest.raises(ConstructionValidationError, match="center orbit"):
        construct_statistical_disorder(
            context,
            [
                MolecularPose(first.template, [0.0, 0.0, 0.0], occupation=0.75),
                MolecularPose(second.template, [0.0, 0.0, 0.0], occupation=0.5),
            ],
        )


def test_stabilizer_must_preserve_connectivity_and_stereochemical_identity():
    context = _pminus1_context()
    square_path = MolecularTemplate(
        Atoms("C4", positions=[[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]),
        identifier="square-path",
        anchor_cartesian=np.zeros(3),
        connectivity=((0, 1), (1, 2), (2, 3)),
    )
    analysis = analyze_molecular_pose(context, MolecularPose(square_path, [0, 0, 0]))
    assert not analysis.ordered_compatible
    stereo_atoms = Atoms("CC", positions=[[1, 0, 0], [-1, 0, 0]])
    stereo = MolecularTemplate(
        stereo_atoms,
        identifier="stereo",
        anchor_cartesian=np.zeros(3),
        stereochemistry=("R", "S"),
    )
    stereo_analysis = analyze_molecular_pose(context, MolecularPose(stereo, [0, 0, 0]))
    assert not stereo_analysis.ordered_compatible


def test_cif_round_trip_retains_forward_atom_ids_and_source_operations():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("CO", positions=[[0, 0, 0], [1.2, 0.2, 0.1]]),
        identifier="co",
        anchor_index=0,
    )
    crystal = construct_molecular_crystal(context, MolecularPose(template, [0.13, 0.21, 0.37]))
    p1 = read_mol_crystal(cif_text=write_cif_p1(crystal), attach_chemistry=False)
    asu = read_mol_crystal(cif_text=write_cif_asu(crystal), attach_chemistry=False)
    assert len(p1.metadata["forward_context"].operations) == len(context.operations)
    assert len(asu.metadata["forward_context"].operations) == len(context.operations)
    p1_ids = {str(value) for molecule in p1.molecules for value in molecule.arrays[KEY_ATOM_ID]}
    asu_ids = {str(value) for molecule in asu.molecules for value in molecule.arrays[KEY_ATOM_ID]}
    assert {"pose0:op0:a0", "pose0:op0:a1"}.issubset(p1_ids)
    assert {"atom_orbit0", "atom_orbit1"}.issubset(asu_ids)


def test_special_position_asu_has_one_row_for_exchanged_atom_orbit():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("CC", positions=[[0.7, 0, 0], [-0.7, 0, 0]]),
        identifier="centred-pair",
        anchor_cartesian=np.zeros(3),
        connectivity=((0, 1),),
    )
    crystal = construct_molecular_crystal(context, MolecularPose(template, [0, 0, 0]))
    cif = write_cif_asu(crystal)
    assert " C1 C1 " not in cif
    block = next(iter(CifParser.from_str(cif).as_dict().values()))
    assert len(block["_atom_site_label"]) == 1
    assert "_geom_bond_atom_site_label_1" in cif
    expanded = scan_cif_disorder(cif_text=cif)
    assert len(expanded.symbols) == 2
    assert all(np.isclose(value, 1.0) for value in expanded.occupancies)
    reread = read_mol_crystal(cif_text=cif, attach_chemistry=False)
    assert len(reread.molecules) == 1
    assert reread.molecules[0].graph.has_edge(0, 1)


def test_inversion_centered_ring_asu_round_trip_preserves_cross_image_bonds():
    """A stabilizer-exchanged ring must not degrade into two dimers."""
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms(
            "C4",
            positions=[
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ],
        ),
        identifier="inversion-ring",
        anchor_cartesian=np.zeros(3),
        connectivity=((0, 1), (1, 2), (2, 3), (3, 0)),
    )
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, [0.0, 0.0, 0.0])
    )

    asu = write_cif_asu(crystal)
    bond_rows = [
        line.split()
        for line in asu.splitlines()
        if line.startswith("  C1 C2 ")
    ]
    assert len(bond_rows) == 2
    assert "." in {row[-1] for row in bond_rows}
    assert any(row[-1] != "." for row in bond_rows)
    parsed = CifParser.from_str(asu, occupancy_tolerance=1.0).parse_structures(
        primitive=False
    )
    assert len(parsed) == 1
    assert len(parsed[0]) == 4

    restored = read_mol_crystal(cif_text=asu, attach_chemistry=False)
    assert len(restored.molecules) == 1
    molecule = restored.molecules[0]
    assert len(molecule) == 4
    assert molecule.graph.number_of_edges() == 4
    assert sorted(dict(molecule.graph.degree()).values()) == [2, 2, 2, 2]


def test_asu_rejects_full_occupancy_independent_rows_related_by_symmetry():
    """Two independent full sites in one declared orbit cannot both expand."""
    context = _pminus1_context()
    molecules = []
    for index, fractional in enumerate(
        ([0.13, 0.21, 0.37], [0.87, 0.79, 0.63])
    ):
        atoms = Atoms("C", positions=[np.asarray(fractional) * 10.0])
        atoms.set_array(KEY_LABEL, np.asarray([f"C{index + 1}"], dtype=str))
        atoms.set_array(KEY_ATOM_ID, np.asarray([f"site{index + 1}"], dtype=str))
        molecules.append(CrystalMolecule(atoms, check_pbc=False))
    crystal = MolecularCrystal(
        context.lattice, molecules, metadata={"forward_context": context}
    )
    with pytest.raises(ValueError, match="overoccupies a symmetry-related"):
        write_cif_asu(crystal)


def test_asu_rejects_mixed_species_full_occupancy_collision():
    """Different species on one symmetry orbit cannot both be full sites."""
    context = _pminus1_context()
    molecules = []
    for symbol, fractional in (("C", [0.13, 0.21, 0.37]), ("N", [0.87, 0.79, 0.63])):
        atoms = Atoms(symbol, positions=[np.asarray(fractional) * 10.0])
        atoms.set_array(KEY_OCCUPANCY, np.asarray([1.0]))
        atoms.set_array(KEY_LABEL, np.asarray([f"{symbol}1"], dtype=str))
        molecules.append(CrystalMolecule(atoms, check_pbc=False))
    crystal = MolecularCrystal(
        context.lattice, molecules, metadata={"forward_context": context}
    )
    with pytest.raises(ValueError, match="overoccupies a symmetry-related"):
        write_cif_asu(crystal)


def test_asu_metadata_and_site_provenance_survive_round_trip():
    context = _pminus1_context()
    template = MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), anchor_index=0)
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, [0.0, 0.0, 0.0])
    )
    cif = write_cif_asu(
        crystal,
        metadata={"cell_transform": {"mode": "test", "source_hash": "abc"}},
    )
    restored = read_mol_crystal(cif_text=cif, attach_chemistry=False)
    assert restored.metadata["cell_transform"] == {
        "mode": "test",
        "source_hash": "abc",
    }
    assert np.all(
        np.concatenate(
            [
                molecule.arrays[KEY_SITE_SYMMETRY_ORDER]
                for molecule in restored.molecules
            ]
        )
        == 2
    )


def test_forward_molecules_preserve_site_symmetry_order_provenance():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("CC", positions=[[0.7, 0, 0], [-0.7, 0, 0]]),
        identifier="centred-pair",
        anchor_cartesian=np.zeros(3),
        connectivity=((0, 1),),
    )
    crystal = construct_molecular_crystal(context, MolecularPose(template, [0, 0, 0]))
    assert all(
        np.all(molecule.arrays[KEY_SITE_SYMMETRY_ORDER] == 2)
        for molecule in crystal.molecules
    )


def test_template_connectivity_is_written_and_used_for_round_trip_grouping():
    context = prepare_construction_context(
        np.eye(3) * 20.0,
        (_operation(np.eye(3), xyz="x,y,z", index=0),),
    )
    template = MolecularTemplate(
        Atoms("CC", positions=[[0, 0, 0], [4, 0, 0]]),
        identifier="long-bond",
        anchor_index=0,
        connectivity=((0, 1),),
    )
    crystal = construct_molecular_crystal(context, MolecularPose(template, [0.1, 0.1, 0.1]))
    assert list(crystal.molecules[0].graph.edges()) == [(0, 1)]
    cif = write_cif_p1(crystal)
    assert "_geom_bond_atom_site_label_1" in cif
    round_tripped = read_mol_crystal(cif_text=cif, attach_chemistry=False)
    assert len(round_tripped.molecules) == 1
    assert list(round_tripped.molecules[0].graph.edges()) == [(0, 1)]


def test_statistical_pminus1_alternatives_use_unique_group_operation_indices():
    context = _pminus1_context()
    template = MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), anchor_index=0)
    model = construct_statistical_disorder(
        context, MolecularPose(template, [0.13, 0.21, 0.37])
    )
    assert len(model.alternatives) == 2
    operation_indices = [alternative.operation_index for alternative in model.alternatives]
    atom_ids = [str(alternative.molecule.arrays[KEY_ATOM_ID][0]) for alternative in model.alternatives]
    sym_op_indices = [int(alternative.molecule.arrays[KEY_SYM_OP_INDEX][0]) for alternative in model.alternatives]
    assert len(set(operation_indices)) == 2
    assert len(set(atom_ids)) == 2
    assert operation_indices == sym_op_indices


def test_statistical_cif_is_materialized_p1_and_parser_occupancies_stay_bounded():
    """Average sites already cover every center orbit; readers must not expand twice."""
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("OH2", positions=[[0, 0, 0], [0.8, 0, 0], [-0.2, 0.7, 0]]),
        identifier="water",
        anchor_index=0,
    )
    model = construct_statistical_disorder(
        context, MolecularPose(template, [0, 0, 0])
    )
    cif = write_disorder_cif(model)
    parsed = CifParser.from_str(cif, occupancy_tolerance=1.0).parse_structures(
        primitive=False
    )
    assert len(parsed) == 1
    assert all(
        float(site.species[site.species.elements[0]]) <= 1.0 + 1e-8
        for site in parsed[0]
    )
    assert "_symmetry_Int_Tables_number 1" in cif
    assert "_molcrys_source_symop_xyz" in cif


def test_pose_occupations_control_duplicate_center_sum_when_unspecified():
    context = _pminus1_context()
    first = MolecularPose(
        MolecularTemplate(Atoms("C", positions=[[0, 0, 0]]), identifier="a"),
        [0, 0, 0],
        occupation=0.4,
    )
    second = MolecularPose(
        MolecularTemplate(Atoms("N", positions=[[0, 0, 0]]), identifier="b"),
        [0, 0, 0],
        occupation=0.4,
    )
    model = construct_statistical_disorder(context, (first, second))
    assert np.isclose(sum(model.orbit_probabilities.values()), 0.8)
    ids = [alternative.alternative_id for alternative in model.alternatives]
    assert len(ids) == len(set(ids))
    assert len(model.alternatives_by_orbit[next(iter(model.alternatives_by_orbit))]) == 2


def test_forward_special_provenance_and_template_bonds_round_trip():
    """Special-site provenance and explicit template edges survive P1 CIF IO."""
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        identifier="long-edge",
        anchor_index=0,
        connectivity=((0, 1),),
    )
    crystal = construct_molecular_crystal(
        context, MolecularPose(template, [0.13, 0.21, 0.37])
    )
    assert all(np.all(molecule.arrays["site_symmetry_order"] == 1) for molecule in crystal.molecules)
    assert all(molecule.graph.has_edge(0, 1) for molecule in crystal.molecules)

    cif = write_cif_p1(crystal)
    assert "_geom_bond_atom_site_label_1" in cif
    parsed = read_mol_crystal(cif_text=cif, attach_chemistry=False)
    assert len(parsed.molecules) == 2
    assert all(molecule.graph.has_edge(0, 1) for molecule in parsed.molecules)
    assert all(np.all(molecule.arrays["site_symmetry_order"] == 1) for molecule in parsed.molecules)
    assert len(parsed.metadata["forward_context"].operations) == 2

    parsed_asu = read_mol_crystal(
        cif_text=write_cif_asu(crystal), attach_chemistry=False
    )
    assert len(parsed_asu.molecules) == 2
    assert all(molecule.graph.has_edge(0, 1) for molecule in parsed_asu.molecules)

    special = construct_molecular_crystal(
        context,
        MolecularPose(
            MolecularTemplate(Atoms("C", positions=[[0.0, 0.0, 0.0]]), anchor_index=0),
            [0.0, 0.0, 0.0],
        ),
    )
    special_cif = write_cif_p1(special)
    special_parsed = read_mol_crystal(cif_text=special_cif, attach_chemistry=False)
    assert len(special_parsed.molecules) == 1
    assert np.all(special_parsed.molecules[0].arrays["site_symmetry_order"] == 2)


def test_statistical_operation_indices_are_group_unique_for_p_minus_1():
    context = _pminus1_context()
    template = MolecularTemplate(
        Atoms("C", positions=[[0.0, 0.0, 0.0]]),
        identifier="single",
        anchor_index=0,
    )
    model = construct_statistical_disorder(context, MolecularPose(template, [0.13, 0.21, 0.37]))
    indices = [alternative.operation_index for alternative in model.alternatives]
    atom_ids = [
        str(value)
        for alternative in model.alternatives
        for value in alternative.molecule.arrays[KEY_ATOM_ID]
    ]
    assert len(indices) == len(set(indices)) == 2
    assert len(atom_ids) == len(set(atom_ids)) == 2


def test_selected_realization_retains_forward_context_for_asu_export():
    context = _pminus1_context()
    model = construct_statistical_disorder(
        context,
        MolecularPose(
            MolecularTemplate(
                Atoms("C", positions=[[0, 0, 0]]), identifier="selected"
            ),
            [0.13, 0.21, 0.37],
        ),
    )
    selected = model.select_ordered_realization()
    assert selected.metadata["forward_context"] is context
    restored = read_mol_crystal(
        cif_text=write_cif_asu(selected), attach_chemistry=False
    )
    assert len(restored.molecules) == 2
