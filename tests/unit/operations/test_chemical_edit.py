"""Chemical-edit contracts using only synthetic ordered cells and molecules."""

import copy
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from molcrys_kit.io import read_extxyz, read_mol_crystal
from molcrys_kit.operations import (
    ChemicalAtomMapping,
    ChemicalEdit,
    CrystalEditSpec,
    CrystalInventory,
    CounterfactualStatus,
    PackingTransferPlan,
    PlacementPolicy,
    RelaxationProtocol,
    apply_chemical_edit,
    periodic_clash_diagnostics,
    transfer_packing,
    write_cf0,
)
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal
from molcrys_kit.constants.config import KEY_ATOM_ID


def ion(symbol):
    return CrystalMolecule(Atoms(symbol, positions=[[0, 0, 0]]), check_pbc=False)


def ammonium():
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    return CrystalMolecule(
        Atoms("NH4", positions=np.vstack([[0, 0, 0], tetra * 1.02])), check_pbc=False
    )


def hydroxylammonium():
    # Complete synthetic N-H3 / N-O-H geometry with a supplied bond graph.
    import networkx as nx

    m = CrystalMolecule(
        Atoms(
            "NH3OH",
            positions=[
                [0, 0, 0],
                [-0.34, 0.96, 0],
                [-0.34, -0.48, 0.83],
                [-0.34, -0.48, -0.83],
                [1.42, 0, 0],
                [1.74, 0.91, 0],
            ],
        ),
        check_pbc=False,
    )
    m._graph = nx.Graph()
    m._graph.add_nodes_from(range(6))
    m._graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)], order=1, order_known=True
    )
    return m


def synthetic_cell(ratio=(1, 1, 3), repeats=1, chemistry=False, skew=False):
    roles = tuple(
        r for r, count in zip(("A", "B", "X"), ratio) for _ in range(count * repeats)
    )
    lattice = np.array([[24.0, 0, 0], [4.0 if skew else 0.0, 24, 0], [0, 0, 24]])
    molecules = []
    for i, role in enumerate(roles):
        m = {
            "A": lambda: ion("Mg" if chemistry else "Ca"),
            "B": hydroxylammonium if chemistry else ammonium,
            "X": lambda: ion("F" if chemistry else "Cl"),
        }[role]()
        frac = np.array(
            [0.15 + 0.35 * (i % 3), 0.15 + 0.35 * ((i // 3) % 3), 0.2 + 0.5 * (i // 9)]
        )
        if skew:
            frac = frac[[1, 0, 2]] + [0.01, 0.02, 0.03]
        m.positions += frac @ lattice
        molecules.append(m)
    crystal = MolecularCrystal(lattice, molecules)
    inventory = CrystalInventory(
        roles, tuple({"A": 2, "B": 1, "X": -1}[r] for r in roles), tuple(roles)
    )
    return crystal, inventory


def spec_for(molecule=None, **kwargs):
    return CrystalEditSpec(
        ChemicalEdit(
            hydroxylammonium() if molecule is None else molecule,
            ammonium().get_chemical_formula(),
            1,
            1,
            PlacementPolicy(0, 0),
            ChemicalAtomMapping(((0, 0),)),
        ),
        role="B",
        **kwargs,
    )


def snapshot(crystal):
    return copy.deepcopy(
        (
            crystal.lattice,
            crystal.metadata,
            [(m.positions, m.arrays, m.info) for m in crystal.molecules],
        )
    )


def assert_snapshot(crystal, saved):
    lattice, metadata, molecules = saved
    np.testing.assert_array_equal(crystal.lattice, lattice)
    assert crystal.metadata == metadata
    for m, (positions, arrays, info) in zip(crystal.molecules, molecules):
        np.testing.assert_array_equal(m.positions, positions)
        assert set(m.arrays) == set(arrays)
        for k in arrays:
            np.testing.assert_array_equal(m.arrays[k], arrays[k])
        assert m.info == info


def test_role_wide_unequal_size_abx3_one_record_deterministic_immutable():
    parent, inventory = synthetic_cell(repeats=2)
    parent.molecules[0].info["nested"] = {"values": [1, 2]}
    saved = snapshot(parent)
    edit = spec_for(expected_formula={"Ca": 2, "Cl": 6, "H": 8, "N": 2, "O": 2})
    target_saved = snapshot(
        MolecularCrystal(np.eye(3) * 30, [edit.edit.replacement.copy()])
    )
    a = apply_chemical_edit(parent, inventory, edit, seed=42)
    b = apply_chemical_edit(parent, inventory, edit, seed=42)
    assert a.status == CounterfactualStatus.CONSTRUCTED, a.provenance
    assert a.provenance == b.provenance
    assert a.provenance["selected_sites"] == [2, 3]
    assert len(a.provenance["atom_changes"]) == 2
    assert a.provenance["charge"] == 0
    assert a.provenance["composition_change"]
    assert not a.provenance["clashes"]
    assert len(a.provenance["parent_hash"]) == 64
    for i in (2, 3):
        m = a.crystal.molecules[i]
        assert m.get_chemical_formula() == hydroxylammonium().get_chemical_formula()
        np.testing.assert_allclose(m.positions[0], parent.molecules[i].positions[0])
        assert m.arrays[KEY_ATOM_ID][0] == parent.molecules[i].arrays[KEY_ATOM_ID][0]
        assert len(a.provenance["atom_changes"][i - 2]["created"]) == 5
        np.testing.assert_array_equal(m.positions, b.crystal.molecules[i].positions)
        assert m.graph.number_of_edges() == 5
    c = apply_chemical_edit(parent, inventory, edit, seed=43)
    assert not np.allclose(
        a.crystal.molecules[2].positions, c.crystal.molecules[2].positions
    )
    a.crystal.molecules[0].info["nested"]["values"].append(3)
    assert_snapshot(parent, saved)
    # Replacement geometry in the spec is not modified by either construction.
    np.testing.assert_array_equal(
        edit.edit.replacement.positions, target_saved[2][0][0]
    )


def test_orbit_scope_expands_seeds_and_requires_evidence():
    parent, inventory = synthetic_cell(repeats=2)
    spec = replace(spec_for(), scope="orbit", role=None, indices=(2,))
    result = apply_chemical_edit(parent, inventory, spec, seed=4)
    assert result.status == CounterfactualStatus.CONSTRUCTED
    assert result.provenance["selected_sites"] == [2, 3]
    missing = apply_chemical_edit(
        parent, replace(inventory, orbit_ids=()), spec, seed=4
    )
    assert missing.status == CounterfactualStatus.INCOMPATIBLE
    assert "orbit" in missing.provenance["reason"]
    for i, m in enumerate(parent.molecules):
        m.info["forward_pose_index"] = inventory.roles[i]
    forward = apply_chemical_edit(
        parent, replace(inventory, orbit_ids=()), spec, seed=4
    )
    assert forward.status == CounterfactualStatus.CONSTRUCTED
    assert forward.provenance["selected_sites"] == [2, 3]


def test_silver_ammonium_explicit_policy_supported_both_directions():
    parent = MolecularCrystal(np.eye(3) * 12, [ion("Ag")])
    inventory = CrystalInventory(("B",), (1,))
    edit = ChemicalEdit(ammonium(), "Ag", 1, 1, PlacementPolicy(0, 0))
    result = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(edit, role="B"), seed=7
    )
    assert result.status == CounterfactualStatus.CONSTRUCTED
    assert len(result.crystal.molecules[0]) == 5
    assert result.provenance["atom_changes"][0]["retained"] == []
    reverse = ChemicalEdit(
        ion("Ag"), ammonium().get_chemical_formula(), 1, 1, PlacementPolicy("com", 0)
    )
    back = apply_chemical_edit(
        result.crystal, inventory, CrystalEditSpec(reverse, role="B"), seed=7
    )
    assert back.status == CounterfactualStatus.CONSTRUCTED
    assert back.crystal.molecules[0].get_chemical_formula() == "Ag"
    bad = replace(edit, mapping=ChemicalAtomMapping(((0, 0),)))
    refused = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(bad, role="B"), seed=7
    )
    assert refused.status == CounterfactualStatus.INCOMPATIBLE
    assert "element" in refused.provenance["reason"]


def test_same_stoichiometry_transfer_two_different_215_cells():
    composition, ci = synthetic_cell((2, 1, 5), chemistry=True)
    template, ti = synthetic_cell((2, 1, 5), skew=True)
    a_saved, b_saved = snapshot(composition), snapshot(template)
    plan = PackingTransferPlan(
        ci, ti, {r: PlacementPolicy("com", "com") for r in ("A", "B", "X")}
    )
    result = transfer_packing(composition, template, plan, seed=21)
    assert result.status == CounterfactualStatus.CONSTRUCTED, result.provenance
    np.testing.assert_array_equal(result.crystal.lattice, template.lattice)
    assert Counter(result.inventory.roles) == {"A": 2, "B": 1, "X": 5}
    assert result.provenance["charge"] == 0
    actual_ids = {
        str(atom_id)
        for m in result.crystal.molecules
        for atom_id in m.arrays[KEY_ATOM_ID]
    }
    recorded_ids = {
        atom_id
        for change in result.provenance["atom_changes"]
        for atom_id in change["created"]
    }
    assert actual_ids == recorded_ids
    assert result.provenance["formula"] == {"F": 5, "H": 4, "Mg": 2, "N": 1, "O": 1}
    for i, j in result.provenance["site_mapping"]:
        np.testing.assert_allclose(
            result.crystal.molecules[j].get_center_of_mass(),
            template.molecules[j].get_center_of_mass(),
        )
        assert (
            result.crystal.molecules[j].get_chemical_formula()
            == composition.molecules[i].get_chemical_formula()
        )
    again = transfer_packing(composition, template, plan, seed=21)
    assert result.provenance == again.provenance
    assert_snapshot(composition, a_saved)
    assert_snapshot(template, b_saved)
    for role in ("A", "B", "X"):
        assert role in result.provenance["edit_spec"]["placements"]


def test_cross_stoichiometry_flagged_and_refused():
    composition, ci = synthetic_cell((1, 1, 3))
    template, ti = synthetic_cell((2, 1, 5))
    result = transfer_packing(
        composition, template, PackingTransferPlan(ci, ti, {}), seed=0
    )
    assert result.status == CounterfactualStatus.INCOMPATIBLE
    assert result.crystal is None
    assert result.provenance["composition_change"]
    assert "vacancy/insertion" in result.provenance["reason"]


def test_cif_extxyz_roundtrip_and_provenance(tmp_path):
    parent, inventory = synthetic_cell(repeats=2)
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=42)
    paths = write_cf0(result, tmp_path)
    # Independent ASE CIF parser and package reader both see the full cell.
    atoms = read(paths["cif"])
    assert Counter(atoms.get_chemical_symbols()) == result.provenance["formula"]
    loaded = read_mol_crystal(str(paths["cif"]))
    assert Counter(m.get_chemical_formula() for m in loaded.molecules) == Counter(
        m.get_chemical_formula() for m in result.crystal.molecules
    )
    np.testing.assert_allclose(atoms.cell.array, result.crystal.lattice)
    np.testing.assert_allclose(
        atoms.get_scaled_positions(),
        result.crystal.to_ase().get_scaled_positions(),
        atol=1e-6,  # Existing CIF writer emits six fractional decimal places.
        rtol=0,
    )
    xyz = read_extxyz(str(paths["extxyz"]))
    assert Counter(m.get_chemical_formula() for m in xyz.molecules) == Counter(
        m.get_chemical_formula() for m in result.crystal.molecules
    )
    assert json.loads(paths["provenance"].read_text()) == result.provenance
    # A read-back crystal contains unknown numeric metadata; it must remain
    # a valid edit input and its provenance must stay strict JSON.
    roles = tuple(
        "A"
        if m.get_chemical_formula() == "Ca"
        else "X"
        if m.get_chemical_formula() == "Cl"
        else "B"
        for m in loaded.molecules
    )
    loaded_inventory = CrystalInventory(
        roles, tuple({"A": 2, "B": 1, "X": -1}[r] for r in roles)
    )
    reverse = ChemicalEdit(
        ammonium(),
        hydroxylammonium().get_chemical_formula(),
        1,
        1,
        PlacementPolicy(0, 0),
        ChemicalAtomMapping(((0, 0),)),
    )
    edited_again = apply_chemical_edit(
        loaded, loaded_inventory, CrystalEditSpec(reverse, role="B"), seed=2
    )
    assert edited_again.status == CounterfactualStatus.CONSTRUCTED, (
        edited_again.provenance["reason"]
    )
    json.dumps(edited_again.provenance, allow_nan=False)
    result.crystal.molecules[0].positions += 1
    with pytest.raises(ValueError, match="mutated"):
        write_cf0(result, tmp_path / "changed")


def test_charge_formula_occupancy_and_source_guards(tmp_path):
    parent, inventory = synthetic_cell()
    for spec in (
        replace(spec_for(), expected_formula={"H": 1}),
        replace(spec_for(), edit=replace(spec_for().edit, target_charge=2)),
        replace(spec_for(), edit=replace(spec_for().edit, source_formula="Ag")),
    ):
        result = apply_chemical_edit(parent, inventory, spec, seed=1)
        assert result.status == CounterfactualStatus.INCOMPATIBLE
        assert result.crystal is None
    m = parent.molecules[0]
    m.set_array("formal_charge", np.array([0]))
    m.set_array("formal_charge_known", np.array([True]))
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=1)
    assert "charge" in result.provenance["reason"]
    m.arrays["formal_charge"][0] = 2
    parent.molecules[1].arrays["occupancy"][:] = 0.5
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=1)
    assert "occupancy" in result.provenance["reason"]
    paths = write_cf0(result, tmp_path)
    assert set(paths) == {"provenance"}
    assert json.loads(paths["provenance"].read_text())["status"] == "INCOMPATIBLE"


def test_periodic_clashes_skew_cell_self_images_and_nonperiodic_axis():
    # Fractional rounding gives a long vector here; the closest image is short.
    lattice = np.array([[10.0, 0, 0], [9.5, 1, 0], [0, 0, 10]])
    crystal = MolecularCrystal(lattice, [ion("He"), ion("He")])
    crystal.molecules[1].positions[:] = [0.49, 0.49, 0] @ lattice
    assert np.linalg.norm(crystal.molecules[1].positions) > 9
    contacts = periodic_clash_diagnostics(crystal)
    assert any(c["left_molecule"] != c["right_molecule"] for c in contacts)
    single = MolecularCrystal(np.diag([1.0, 10, 10]), [ion("Cl")])
    assert any(
        c["left_molecule"] == c["right_molecule"] and any(c["image_shift"])
        for c in periodic_clash_diagnostics(single)
    )
    single.pbc = (False, True, True)
    assert not periodic_clash_diagnostics(single)


def test_clash_refusal_checks_periodic_neighbors_after_all_sites_placed():
    parent = MolecularCrystal(np.eye(3) * 10, [ion("Ag"), ion("Ag")])
    parent.molecules[0].positions[:] = [0.1, 0, 0]
    parent.molecules[1].positions[:] = [9.9, 0, 0]
    inventory = CrystalInventory(("B", "B"), (1, 1))
    edit = ChemicalEdit(ammonium(), "Ag", 1, 1, PlacementPolicy(0, 0, samples=3))
    result = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(edit, role="B"), seed=2
    )
    assert result.status == CounterfactualStatus.CLASH
    assert result.provenance["attempts"] == 3
    assert any(any(c["image_shift"]) for c in result.provenance["clashes"])
    assert result.crystal is None


def test_mapped_orientation_and_clean_degenerate_refusal():
    m = hydroxylammonium()
    m.positions += [6, 6, 6]
    parent = MolecularCrystal(np.eye(3) * 20, [m])
    inventory = CrystalInventory(("B",), (1,))
    edit = ChemicalEdit(
        hydroxylammonium(),
        m.get_chemical_formula(),
        1,
        1,
        PlacementPolicy(0, 0, orientation="mapped"),
        ChemicalAtomMapping(tuple((i, i) for i in range(6))),
    )
    result = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(edit, role="B"), seed=0
    )
    assert result.status == CounterfactualStatus.CONSTRUCTED
    np.testing.assert_allclose(result.crystal.molecules[0].positions, m.positions)
    edit = replace(edit, mapping=ChemicalAtomMapping(((0, 0),)))
    result = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(edit, role="B"), seed=0
    )
    assert result.status == CounterfactualStatus.INCOMPATIBLE
    assert "three non-collinear" in result.provenance["reason"]


def test_transfer_explicit_mapping_guard_and_missing_policy():
    composition, ci = synthetic_cell((2, 1, 5))
    template, ti = synthetic_cell((2, 1, 5), skew=True)
    result = transfer_packing(
        composition, template, PackingTransferPlan(ci, ti, {}), seed=0
    )
    assert result.status == CounterfactualStatus.INCOMPATIBLE
    plan = PackingTransferPlan(
        ci,
        ti,
        {r: PlacementPolicy("com", "com") for r in set(ci.roles)},
        site_mapping=((0, 0),),
    )
    result = transfer_packing(composition, template, plan, seed=0)
    assert result.status == CounterfactualStatus.INCOMPATIBLE
    pairs = list((i, i) for i in range(8))
    pairs[0], pairs[1] = (0, 1), (1, 0)
    result = transfer_packing(
        composition, template, replace(plan, site_mapping=tuple(pairs)), seed=0
    )
    assert result.status == CounterfactualStatus.CONSTRUCTED


def test_relaxation_protocol_dof_contract():
    crystal, _ = synthetic_cell()
    expected = [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, True, False),
        (True, True, True, True),
    ]
    for i, (internal, translation, orientation, cell) in enumerate(expected):
        mask = RelaxationProtocol(f"CF{i}").constraints(crystal)
        assert (
            mask["internal_free"],
            mask["translation_free"],
            mask["orientation_free"],
        ) == (internal, translation, orientation)
        assert mask["cell_free_mask"] == [cell] * 6
        assert sum(map(len, mask["molecule_atom_indices"])) == sum(
            map(len, crystal.molecules)
        )
        assert mask["symmetry_policy"] == "P1"


@pytest.mark.parametrize(
    "construct",
    [
        lambda: ChemicalAtomMapping(((0, 0), (1, 0))),
        lambda: PlacementPolicy("centroid", 0),
        lambda: PlacementPolicy(0, 0, samples=0),
        lambda: PlacementPolicy(0, 0, clash_scale=0),
        lambda: CrystalInventory(("B",), (0.5,)),
        lambda: RelaxationProtocol("CF4"),
    ],
)
def test_invalid_typed_inputs(construct):
    with pytest.raises(ValueError):
        construct()


def test_existing_chemistry_snapshot_ids_bonds_and_embedding_are_remapped():
    from molcrys_kit.chemistry import (
        ChemicalAtom,
        ChemicalBond,
        Embedding,
        FiniteChemicalEntity,
    )

    target = hydroxylammonium()
    ids = tuple(f"target:{i}" for i in range(len(target)))
    target.set_array(KEY_ATOM_ID, np.asarray(ids))
    target.chemical_entity = FiniteChemicalEntity(
        entity_id="prebuilt",
        atoms=tuple(
            ChemicalAtom(atom_id, symbol, formal_charge=int(i == 0))
            for i, (atom_id, symbol) in enumerate(
                zip(ids, target.get_chemical_symbols())
            )
        ),
        bonds=tuple(
            ChemicalBond(ids[i], ids[j], order=1) for i, j in target.graph.edges()
        ),
        embedding=Embedding(
            tuple((atom_id, tuple(pos)) for atom_id, pos in zip(ids, target.positions))
        ),
        net_charge=1,
    )
    parent, inventory = synthetic_cell()
    spec = spec_for(target)
    result = apply_chemical_edit(parent, inventory, spec, seed=6)
    assert result.status == CounterfactualStatus.CONSTRUCTED, result.provenance
    edited = result.crystal.molecules[1]
    entity = edited.chemical_entity
    assert {a.atom_id for a in entity.atoms} == set(edited.arrays[KEY_ATOM_ID])
    assert len(entity.bonds) == 5
    assert all(
        b.atom1_id in edited.arrays[KEY_ATOM_ID]
        and b.atom2_id in edited.arrays[KEY_ATOM_ID]
        for b in entity.bonds
    )
    for atom_id, position in zip(edited.arrays[KEY_ATOM_ID], edited.positions):
        np.testing.assert_allclose(entity.embedding.position(atom_id), position)
    assert entity.net_charge == 1
    assert spec.edit.replacement.chemical_entity is target.chemical_entity
    bad = replace(spec, edit=replace(spec.edit, target_charge=2), expected_charge=1)
    refused = apply_chemical_edit(parent, inventory, bad, seed=6)
    assert refused.status == CounterfactualStatus.INCOMPATIBLE
    assert "chemical entity" in refused.provenance["reason"]


def test_unknown_metadata_and_isotope_or_stereo_mismatch():
    parent, inventory = synthetic_cell()
    parent.molecules[1].set_array("isotope", np.zeros(5, dtype=int))
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    assert result.status == CounterfactualStatus.CONSTRUCTED
    parent.molecules[1].arrays["isotope"][0] = 15
    refused = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    assert refused.status == CounterfactualStatus.INCOMPATIBLE
    assert "isotope" in refused.provenance["reason"]
    parent.molecules[1].arrays.pop("isotope")
    parent.molecules[1].set_array("stereochemistry", np.array(["R", "", "", "", ""]))
    refused = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    assert refused.status == CounterfactualStatus.INCOMPATIBLE
    assert "stereochemistry" in refused.provenance["reason"]


def test_declared_framework_refused_but_finite_boundary_bond_allowed():
    parent, inventory = synthetic_cell()
    parent.molecules[1].info["forward_bonds"] = [
        {"left": 0, "right": 0, "image_shift": [1, 0, 0]},
    ]
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    assert result.status == CounterfactualStatus.INCOMPATIBLE
    assert "framework" in result.provenance["reason"]
    parent.molecules[1].info["forward_bonds"] = [
        {"left": 0, "right": 1, "image_shift": [1, 0, 0]},
    ]
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    assert result.status == CounterfactualStatus.CONSTRUCTED


def test_export_refuses_reusing_an_output_directory(tmp_path):
    parent, inventory = synthetic_cell()
    result = apply_chemical_edit(parent, inventory, spec_for(), seed=0)
    paths = write_cf0(result, tmp_path)
    before = paths["cif"].read_bytes()
    with pytest.raises(FileExistsError):
        write_cf0(result, tmp_path)
    assert before == paths["cif"].read_bytes()


def test_orientation_sampling_resolves_a_clash_without_internal_distortion():
    parent = MolecularCrystal(np.eye(3) * 15, [ion("Ag"), ion("F")])
    parent.molecules[0].positions[:] = [5, 5, 5]
    parent.molecules[1].positions[:] = [6.7, 5, 5]
    inventory = CrystalInventory(("B", "X"), (1, -1))
    target = ammonium()
    edit = ChemicalEdit(target, "Ag", 1, 1, PlacementPolicy(0, 0, samples=1))
    first = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(edit, role="B"), seed=4
    )
    assert first.status == CounterfactualStatus.CLASH
    sampled = replace(edit, placement=PlacementPolicy(0, 0, samples=4))
    result = apply_chemical_edit(
        parent, inventory, CrystalEditSpec(sampled, role="B"), seed=4
    )
    assert result.status == CounterfactualStatus.CONSTRUCTED
    assert result.provenance["attempts"] == 2
    assert not periodic_clash_diagnostics(result.crystal)
    np.testing.assert_allclose(
        result.crystal.molecules[0].get_all_distances(),
        target.get_all_distances(),
        atol=1e-14,
    )
    rotation = np.array(result.provenance["atom_changes"][0]["rotation_row_matrix"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_single_site_scope_is_explicit_local_defect_and_charge_change_is_opt_in():
    parent, inventory = synthetic_cell(repeats=2)
    spec = replace(spec_for(), scope="sites", role=None, indices=(2,))
    result = apply_chemical_edit(parent, inventory, spec, seed=4)
    assert result.status == CounterfactualStatus.CONSTRUCTED
    assert result.provenance["selected_sites"] == [2]
    assert (
        result.crystal.molecules[3].get_chemical_formula()
        == ammonium().get_chemical_formula()
    )
    charged = replace(spec, edit=replace(spec.edit, target_charge=2), expected_charge=1)
    result = apply_chemical_edit(parent, inventory, charged, seed=4)
    assert result.status == CounterfactualStatus.CONSTRUCTED
    assert result.provenance["charge"] == 1


def test_contiguous_target_rebases_legacy_boundary_bond_geometry():
    parent, inventory = synthetic_cell()
    target = hydroxylammonium()
    edges = list(target.graph.edges())
    target._graph = None
    target.info["atom_indices"] = list(range(10, 16))
    target.info["bond_records"] = [
        {
            "left": i + 10,
            "right": j + 10,
            "right_image_shift": [1, 0, 0],
            "vector": [99.0, 0, 0],
        }
        for i, j in edges
    ]
    result = apply_chemical_edit(parent, inventory, spec_for(target), seed=3)
    assert result.status == CounterfactualStatus.CONSTRUCTED, result.provenance
    placed = result.crystal.molecules[1]
    for bond in placed.info["bond_records"]:
        assert bond["right_image_shift"] == [0, 0, 0]
        np.testing.assert_allclose(
            bond["vector"],
            placed.positions[bond["right"]] - placed.positions[bond["left"]],
        )
    assert target.info["bond_records"][0]["vector"] == [99.0, 0, 0]
