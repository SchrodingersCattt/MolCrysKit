"""Licence-free product probes, plus an optional CCDC synthetic-reader probe.

From repository root: python -c "import runpy;runpy.run_path('benchmarks/csd_roundtrip/reports/csd601-sep2026-baseline-v2/reproduce_confirmed.py',run_name='__main__')"
Use --ccdc-reader in the vendor interpreter for that optional probe.
No production file is modified. Generated CIFs stay under ignored _local/.
"""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

from benchmarks.csd_roundtrip.common import LOCAL, write_json


def atom_cif(symbol="D"):
    return f"""data_synthetic_atom
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
X1 {symbol} .2 .2 .2 1
"""


def ccdc_reader_probe():
    import os
    import sys
    from ccdc.crystal import Crystal

    text = atom_cif("C").replace(
        "X1 C .2 .2 .2 1", "C1 C .2 .2 .2 1\nC2 C .334 .2 .2 1"
    )
    loop = "\nloop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n_geom_bond_distance\n{tag}\n_geom_bond_site_symmetry_2\nC1 C2 1.34 {token} .\n"
    rows = []
    keep = []
    for tag, token in [
        ("_geom_bond_type", "double"),
        ("_ccdc_geom_bond_type", "double"),
        ("_ccdc_geom_bond_type", "D"),
        ("_ccdc_geom_bond_type", "Double"),
    ]:
        c = Crystal.from_string(text + loop.format(tag=tag, token=token), "cif")
        keep.append(c)
        m = c.molecule
        keep.append(m)
        rows.append(
            {
                "tag": tag,
                "token": token,
                "bond_types": [str(b.bond_type) for b in m.bonds],
                "chirality": [a.chirality for a in m.atoms],
            }
        )
    import ccdc

    result = {"synthetic_only": True, "api_version": ccdc.__version__, "rows": rows}
    write_json(Path(__file__).resolve().parent / "ccdc_bond_tag_probe.json", result)
    print(result, flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--ccdc-reader", action="store_true")
    args = p.parse_args()
    if args.ccdc_reader:
        ccdc_reader_probe()
        return
    import numpy as np
    from ase import Atoms
    from ase.spacegroup import Spacegroup
    from pymatgen.io.cif import CifParser
    import molcrys_kit.io.cif as cif_module
    from molcrys_kit.io.output import (
        write_cif_asu,
        write_cif_p1,
        write_disorder_cif,
        write_forward_cif,
    )
    from molcrys_kit.operations.forward import (
        MolecularTemplate,
        MolecularPose,
        prepare_construction_context,
        construct_molecular_crystal,
        construct_statistical_disorder,
    )
    from molcrys_kit.structures.symmetry import (
        CrystalSymmetry,
        FractionalAffineOperation,
    )
    from benchmarks.csd_roundtrip.worker import chemistry_check

    folder = LOCAL / "triage_synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"synthetic_only": True, "probes": {}}
    ops = tuple(
        FractionalAffineOperation(r, t, index=i)
        for i, (r, t) in enumerate(Spacegroup(217).get_symop())
    )
    ctx = prepare_construction_context(
        np.eye(3) * 10,
        CrystalSymmetry(ops, space_group_number=217, space_group_symbol="I -4 3 m"),
    )
    tetra = np.array([v for v in product((-1, 1), repeat=3) if np.prod(v) == 1])
    template = MolecularTemplate(
        Atoms("CH4", positions=np.vstack((np.zeros(3), tetra * 0.63))),
        connectivity=tuple((0, j) for j in range(1, 5)),
    )
    crystal = construct_molecular_crystal(ctx, MolecularPose(template, np.zeros(3)))
    path = folder / "ch4_asu.cif"
    write_cif_asu(crystal, str(path))
    original = cif_module.identify_molecules
    capture = {}

    def record_pairs(atoms, **kw):
        pairs = kw.get("explicit_bonds", [])
        capture["published_pair_count"] = len(pairs)
        capture["pair_distance_histogram_A"] = {
            str(k): v
            for k, v in Counter(
                round(float(atoms.get_distance(a, b, mic=True)), 6) for a, b in pairs
            ).items()
        }
        return original(atoms, **kw)

    try:
        cif_module.identify_molecules = record_pairs
        read = cif_module.read_mol_crystal(str(path))
    finally:
        cif_module.identify_molecules = original
    expected_distance = float(np.sqrt(3) * 0.63)

    def diagnostic_filter(atoms, **kw):
        # Counterfactual for this known synthetic molecule only. This is NOT
        # a general repair for bond or periodic-image provenance.
        kw["explicit_bonds"] = [
            (a, b)
            for a, b in kw.get("explicit_bonds", [])
            if abs(atoms.get_distance(a, b, mic=True) - expected_distance) < 1e-4
        ]
        return original(atoms, **kw)

    try:
        cif_module.identify_molecules = diagnostic_filter
        filtered = cif_module.read_mol_crystal(str(path))
    finally:
        cif_module.identify_molecules = original
    result["probes"]["RT01"] = {
        "expected_molecules": 2,
        "constructed_molecules": len(crystal.molecules),
        "independent_atoms": len(
            CifParser(str(path)).parse_structures(primitive=False)[0]
        ),
        "readback_formulas": [m.get_chemical_formula() for m in read.molecules],
        "diagnostic_filtered_formulas": [
            m.get_chemical_formula() for m in filtered.molecules
        ],
        **capture,
    }
    inversion = (
        FractionalAffineOperation(np.eye(3), np.zeros(3)),
        FractionalAffineOperation(-np.eye(3), np.zeros(3)),
    )
    ctx_inv = prepare_construction_context(
        np.eye(3) * 20,
        CrystalSymmetry(inversion, space_group_number=2, space_group_symbol="P -1"),
        chemistry_policy="enantiopure",
    )
    atom = Atoms(
        ["C", "H", "F", "Cl", "Br"], positions=np.vstack((np.zeros(3), tetra * 0.7))
    )
    chiral = MolecularTemplate(
        atom,
        connectivity=tuple((0, j) for j in range(1, 5)),
        stereochemistry=("R", "", "", "", ""),
        chemical_identity="explicit-enantiopure-synthetic",
    )
    try:
        c = construct_molecular_crystal(
            ctx_inv, MolecularPose(chiral, np.array([0.21, 0.29, 0.37]))
        )
        result["probes"]["RT02"] = {
            "expected": "reject incompatible enantiopure request",
            "accepted": True,
            "molecules": len(c.molecules),
            "labels": [str(m.arrays["stereochemistry"][0]) for m in c.molecules],
        }
    except Exception as exc:
        result["probes"]["RT02"] = {"accepted": False, "error_type": type(exc).__name__}
    ctx_p1 = prepare_construction_context(
        np.eye(3) * 10,
        CrystalSymmetry(
            (inversion[0],), space_group_number=1, space_group_symbol="P 1"
        ),
    )
    o2 = construct_molecular_crystal(
        ctx_p1,
        MolecularPose(
            MolecularTemplate(
                Atoms("O2", positions=[[0, 0, 0], [1.21, 0, 0]]), connectivity=((0, 1),)
            ),
            np.array([0.2, 0.2, 0.2]),
        ),
    )
    o2.molecules[0].get_graph()[0][1].update(bond_type="double", order=2)
    block = next(iter(CifParser.from_str(write_cif_p1(o2)).as_dict().values()))
    result["probes"]["RT03"] = {
        "graph_bond_type": "double",
        "expected_export_type": "double",
        "observed_export_types": block.get("_geom_bond_type"),
    }
    isotope = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    isotope.set_array("isotope", np.array([2, 2]))
    isotope.set_masses([2.01410177812] * 2)
    c = construct_molecular_crystal(
        ctx_p1,
        MolecularPose(
            MolecularTemplate(isotope, connectivity=((0, 1),)),
            np.array([0.2, 0.2, 0.2]),
        ),
    )
    iso_info = cif_module.scan_cif_disorder(cif_text=write_cif_p1(c))
    ions = []
    for symbol, charge, anchor in [
        ("Na", 1, [0.2, 0.2, 0.2]),
        ("Cl", -1, [0.7, 0.7, 0.7]),
    ]:
        a = Atoms(symbol, positions=[[0, 0, 0]])
        a.set_array("formal_charge", np.array([charge]))
        a.set_array("formal_charge_known", np.array([True]))
        ions.append(MolecularPose(MolecularTemplate(a), np.array(anchor)))
    charge_info = cif_module.scan_cif_disorder(
        cif_text=write_cif_p1(construct_molecular_crystal(ctx_p1, ions))
    )
    result["probes"]["RT04"] = {
        "expected_isotopes": [2, 2],
        "readback_isotopes": iso_info.isotopes,
        "expected_formal_charges": [1, -1],
        "readback_formal_charges": charge_info.site_formal_charges,
    }
    try:
        c = cif_module.read_mol_crystal(cif_text=atom_cif())
        result["probes"]["RT05"] = {
            "read_succeeded": True,
            "symbols": [x.symbol for m in c.molecules for x in m],
        }
    except Exception as exc:
        result["probes"]["RT05"] = {
            "expected": "H with isotope 2",
            "read_succeeded": False,
            "error_type": type(exc).__name__,
        }
    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    model = construct_statistical_disorder(
        prepare_construction_context(
            np.eye(3) * 10,
            CrystalSymmetry(inversion, space_group_number=2, space_group_symbol="P -1"),
        ),
        MolecularPose(
            MolecularTemplate(water, anchor_index=0, connectivity=((0, 1), (0, 2))),
            np.zeros(3),
        ),
    )
    data = model.to_dict()
    result["probes"]["RT06"] = {
        "model_alternatives": len(model.alternatives),
        "serialized_alternative_keys": sorted(data["alternatives"][0]),
        "serialized_top_keys": sorted(data),
    }
    mean = write_disorder_cif(model)
    result["probes"]["RT07"] = {
        "average_is_materialized_p1": "_symmetry_Int_Tables_number 1" in mean
    }
    try:
        write_forward_cif(model, representation="asu")
        result["probes"]["RT07"]["asu_call_raised"] = False
    except Exception as exc:
        result["probes"]["RT07"].update(
            asu_call_raised=True, error_type=type(exc).__name__
        )
    reference = {
        "atoms": [
            {
                "symbol": "C",
                "xyz": [0.0, 0.0, 0.0],
                "occupancy": 1.0,
                "formal_charge": 0,
                "isotope": None,
                "stereo": None,
            }
        ],
        "bonds": [],
    }
    candidate = {"atoms": [dict(reference["atoms"][0], stereo="Error")], "bonds": []}
    result["probes"]["RT08"] = {
        "expected": "assignment failure is not a different valid R/S configuration",
        "baseline_checker": chemistry_check([reference], [candidate], np.eye(3) * 10),
    }
    vacant = construct_statistical_disorder(
        ctx_p1,
        MolecularPose(
            MolecularTemplate(water, anchor_index=0, connectivity=((0, 1), (0, 2))),
            np.array([0.2, 0.2, 0.2]),
        ),
        center_occupation=0.0,
    )
    selected = vacant.select_ordered_realization()
    result["probes"]["RT10"] = {
        "centre_occupation": 0.0,
        "alternative_weights": [a.weight for a in vacant.alternatives],
        "expected_selected_atoms": 0,
        "actual_selected_atoms": sum(len(m) for m in selected.molecules),
    }
    from molcrys_kit.operations.forward import analyze_molecular_pose
    from benchmarks.csd_roundtrip.match import molecule_mapping

    short_cell = np.diag([2.4, 10.0, 10.0])
    short_context = prepare_construction_context(
        short_cell,
        CrystalSymmetry(inversion, space_group_number=2, space_group_symbol="P -1"),
    )
    co = MolecularTemplate(
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        anchor_index=0,
        connectivity=((0, 1),),
    )
    site = analyze_molecular_pose(short_context, MolecularPose(co, np.zeros(3)))
    whole = {
        "atoms": [
            {"symbol": s, "xyz": p, "occupancy": 1.0}
            for s, p in zip(("C", "O"), ([0.0, 0.0, 0.0], [1.2, 0.0, 0.0]))
        ],
        "bonds": [[0, 1, None]],
    }
    inverted = {
        "atoms": [
            {"symbol": s, "xyz": p, "occupancy": 1.0}
            for s, p in zip(("C", "O"), ([0.0, 0.0, 0.0], [-1.2, 0.0, 0.0]))
        ],
        "bonds": [[0, 1, None]],
    }
    result["probes"]["RT11"] = {
        "expected_stabilizer_order": 1,
        "actual_stabilizer_order": site.stabilizer_order,
        "expected_ordered_compatible": False,
        "actual_ordered_compatible": site.ordered_compatible,
        "single_whole_molecule_image_match": molecule_mapping(
            whole, inverted, short_cell
        )
        is not None,
    }
    try:
        wrong = prepare_construction_context(
            np.eye(3) * 10,
            CrystalSymmetry(
                (inversion[0],), space_group_number=2, space_group_symbol="P -1"
            ),
        )
        result["probes"]["RT12"] = {
            "expected": "reject declaration P -1 with identity-only operations",
            "accepted": True,
            "declared_group": wrong.space_group_number,
            "actual_operations": len(wrong.operations),
        }
    except Exception as exc:
        result["probes"]["RT12"] = {"accepted": False, "error_type": type(exc).__name__}
    write_json(Path(__file__).resolve().parent / "synthetic_diagnostics.json", result)
    print(result)


if __name__ == "__main__":
    main()
