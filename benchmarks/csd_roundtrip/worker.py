"""One isolated benchmark case. Restricted diagnostics never leave _local."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
import subprocess
import time
import traceback

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .common import (
    LOCAL,
    ROOT,
    config,
    read_json,
    write_json,
    isotope_symbol,
    normalize_source_isotopes,
)
from .match import cell_parameters, compare_molecules, molecule_mapping, parse_operation


def stage_error(exc, code="exception"):
    return {"status": "error", "code": code, "error_type": type(exc).__name__}


def snapshot_mck(crystal, canonical_lattice=None):
    canonical_lattice = (
        crystal.lattice if canonical_lattice is None else canonical_lattice
    )
    basis_map = np.linalg.inv(crystal.lattice) @ canonical_lattice
    result = []
    for molecule in crystal.molecules:
        atoms = []
        for i, (symbol, xyz) in enumerate(
            zip(molecule.get_chemical_symbols(), molecule.positions)
        ):
            atom = {"symbol": symbol, "xyz": (xyz @ basis_map).tolist()}
            for key, destination in (
                ("occupancy", "occupancy"),
                ("formal_charge", "formal_charge"),
                ("isotope", "isotope"),
                ("stereochemistry", "stereo"),
            ):
                values = molecule.arrays.get(key)
                value = values[i] if values is not None else None
                if isinstance(value, np.generic):
                    value = value.item()
                atom[destination] = None if value in (None, "") else value
                if destination == "isotope" and value == 0:
                    atom[destination] = None
            if atom["occupancy"] is None:
                atom["occupancy"] = 1.0
            atoms.append(atom)
        graph = molecule.get_graph()
        bonds = (
            [
                [int(a), int(b), data.get("bond_type")]
                for a, b, data in graph.edges(data=True)
            ]
            if graph is not None
            else []
        )
        result.append({"atoms": atoms, "bonds": bonds})
    return result


def chemistry_check(reference, candidate, lattice):
    geometric = compare_molecules(reference, candidate, lattice)
    if geometric["status"] != "pass":
        return geometric
    missing = set()
    for molecules in (reference,):
        if any(
            len(e) > 2 and e[2] not in (None, "Unknown")
            for m in molecules
            for e in m["bonds"]
        ):
            if any(len(e) < 3 or e[2] is None for m in candidate for e in m["bonds"]):
                missing.add("bond_order")
    for key in ("formal_charge", "stereo", "isotope"):
        expected = any(a.get(key) is not None for m in reference for a in m["atoms"])
        observed = any(a.get(key) is not None for m in candidate for a in m["atoms"])
        if expected and not observed:
            missing.add(key)
    if missing:
        if missing == {"bond_order"}:
            left = [
                {**m, "bonds": [[e[0], e[1], None] for e in m["bonds"]]}
                for m in reference
            ]
            right = [
                {**m, "bonds": [[e[0], e[1], None] for e in m["bonds"]]}
                for m in candidate
            ]
            identity = compare_molecules(left, right, lattice, chemistry=True)
            if identity["status"] != "pass":
                return {
                    "status": "mismatch",
                    "code": "atomic_chemical_identity_not_preserved",
                    "missing_annotations": sorted(missing),
                }
        return {
            "status": "unsupported",
            "code": "known_chemical_annotations_not_preserved",
            "missing_annotations": sorted(missing),
        }
    result = compare_molecules(reference, candidate, lattice, chemistry=True)
    if result["status"] != "pass":
        result["code"] = "chemical_identity_or_bond_order_mismatch"
    return result


def make_poses(source, context, disorder=False):
    from molcrys_kit.operations.forward import MolecularPose, MolecularTemplate

    lattice = np.asarray(source["lattice"])
    inv = np.linalg.inv(lattice)
    rotation = Rotation.from_euler(
        "xyz", config()["template_rotation_xyz_deg"], degrees=True
    ).as_matrix()
    templates = source["source_templates"] if disorder else source["templates"]
    poses = []
    for index, molecule in enumerate(templates):
        xyz = np.asarray([a["xyz"] for a in molecule["atoms"]])
        center = xyz.mean(0)
        if disorder:
            scaffold = [
                a["xyz"]
                for a in molecule["atoms"]
                if a.get("group", 0) == 0
                and abs(a["occupancy"] - 1.0) <= config()["occupancy"]
            ]
            if scaffold:
                center = np.mean(scaffold, axis=0)
        atoms = Atoms(
            [a["symbol"] for a in molecule["atoms"]],
            positions=(xyz - center) @ rotation,
        )
        # Chemistry is supplied independently of CIF bond re-perception.
        annotation_source = molecule
        if not disorder:
            for original in source["source_templates"]:
                mapping = molecule_mapping(original, molecule, lattice, occupancy=False)
                if mapping:
                    inverse = np.argsort(mapping[0])
                    annotation_source = {
                        "atoms": [original["atoms"][j] for j in inverse]
                    }
                    break
        for key, destination, dtype in (
            ("formal_charge", "formal_charge", float),
            ("stereo", "stereochemistry", str),
        ):
            values = [a.get(key) for a in annotation_source["atoms"]]
            if all(v is not None for v in values) or key == "stereo":
                atoms.set_array(
                    destination,
                    np.asarray(
                        [v if v is not None else "" for v in values], dtype=dtype
                    ),
                )
        atoms.set_array(
            "label",
            np.asarray(
                [f"{a['symbol']}{i + 1}" for i, a in enumerate(molecule["atoms"])],
                dtype=str,
            ),
        )
        isotopes = [a.get("isotope") for a in annotation_source["atoms"]]
        if any(x is not None for x in isotopes):
            atoms.set_array(
                "isotope", np.asarray([x or 0 for x in isotopes], dtype=int)
            )
            masses = atoms.get_masses()
            for atom_index, isotope in enumerate(isotopes):
                if isotope in (2, 3) and atoms[atom_index].symbol == "H":
                    masses[atom_index] = {2: 2.01410177812, 3: 3.01604928199}[isotope]
            atoms.set_masses(masses)
        template = MolecularTemplate(
            atoms,
            identifier=f"component{index}",
            anchor_cartesian=np.zeros(3),
            connectivity=tuple((e[0], e[1]) for e in molecule["bonds"]),
        )
        occupation = 1.0
        if disorder:
            # Complete occupied molecules are the alternatives, not atoms
            # simultaneously assigned their marginal occupancies.
            from molcrys_kit.operations.forward import analyze_molecular_pose

            provisional = MolecularPose(template, center @ inv, orientation=rotation)
            analysis = analyze_molecular_pose(context, provisional)
            partials = [
                a["occupancy"]
                for a in molecule["atoms"]
                if a["occupancy"] < 1.0 - config()["occupancy"]
            ]
            if partials and analysis.orientation_multiplicity:
                occupation = min(partials) * analysis.orientation_multiplicity
                if occupation > 1.0 + config()["occupancy"]:
                    raise ValueError(
                        "source alternative weights are incompatible with symmetry-derived equal weights"
                    )
        poses.append(
            MolecularPose(
                template,
                center @ inv,
                orientation=rotation,
                occupation=min(occupation, 1.0),
            )
        )
    return poses


def compare_cell(reference, candidate):
    a, alpha = cell_parameters(reference)
    b, beta = cell_parameters(candidate)
    length_error = float(np.max(np.abs(a - b)))
    angle_error = float(np.max(np.abs(alpha - beta)))
    passed = (
        length_error <= config()["length_A"] and angle_error <= config()["angle_deg"]
    )
    return {
        "status": "pass" if passed else "mismatch",
        "code": "cell_parameters",
        "length_error_A": length_error,
        "angle_error_deg": angle_error,
    }


def symmetry_check(molecules, source):
    lattice = np.asarray(source["lattice"])
    inv = np.linalg.inv(lattice)
    for xyz in source["operations"]:
        rotation, shift = parse_operation(xyz)
        transformed = []
        for mol in molecules:
            positions = np.asarray([a["xyz"] for a in mol["atoms"]])
            transformed.append(
                {
                    **mol,
                    "atoms": [
                        dict(a, xyz=p.tolist())
                        for a, p in zip(
                            mol["atoms"],
                            (positions @ inv @ rotation.T + shift) @ lattice,
                        )
                    ],
                }
            )
        comparison = compare_molecules(molecules, transformed, lattice)
        if comparison["status"] != "pass":
            return {"status": "mismatch", "code": "source_operation_not_satisfied"}
    result = {
        "status": "pass",
        "code": "independent_operation_verification",
        "n_operations": len(source["operations"]),
    }
    try:
        import spglib

        atoms = [a for m in molecules for a in m["atoms"]]
        positions = np.asarray([a["xyz"] for a in atoms]) @ inv
        numbers = Atoms([a["symbol"] for a in atoms]).numbers
        dataset = spglib.get_symmetry_dataset(
            (lattice, positions % 1.0, numbers), symprec=config()["spglib_symprec_A"]
        )
        if dataset:
            result["detected_spacegroup"] = int(dataset.number)
            result["source_spacegroup"] = source["spacegroup_number"]
    except Exception:
        pass
    return result


def operation_set_check(observed, expected):
    if len(observed) != len(expected):
        return {
            "status": "mismatch",
            "code": "operation_count",
            "n_operations": len(observed),
        }
    for left in observed:
        a, b = parse_operation(left)
        if not any(
            np.max(np.abs(a - c)) <= config()["fractional"]
            and np.max(np.abs((b - d) - np.rint(b - d))) <= config()["fractional"]
            for c, d in map(parse_operation, expected)
        ):
            return {"status": "mismatch", "code": "operation_set"}
    return {"status": "pass", "code": "operation_set", "n_operations": len(observed)}


def check_ccdc(refcode, path):
    activation = os.environ.get("CSD_ROUNDTRIP_ACTIVATE")
    if not activation:
        raise RuntimeError("CSD_ROUNDTRIP_ACTIVATE is required")
    # Passing a quoted /c expression as a list makes Python escape its quotes
    # with backslashes, which cmd.exe does not accept as shell escaping.
    command = f'cmd /d /s /c ""{activation}" && python -B -m benchmarks.csd_roundtrip.ccdc_adapter check --refcode {refcode} --file "{path}""'
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT.parents[1],
            capture_output=True,
            text=True,
            timeout=config()["timeout_s"],
        )
    except subprocess.TimeoutExpired as exc:
        path.with_suffix(".ccdc.log").write_text(
            str(exc.stdout or "") + str(exc.stderr or ""), encoding="utf-8"
        )
        raise
    path.with_suffix(".ccdc.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError("independent CCDC CIF reader failed; see private log")
    return read_json(path.with_suffix(".ccdc.json"))


def pymatgen_check(path, source):
    from pymatgen.io.cif import CifParser

    structure = CifParser(str(path), occupancy_tolerance=1.0).parse_structures(
        primitive=False
    )[0]
    expected = Counter(a["symbol"] for m in source["reference"] for a in m["atoms"])
    actual = Counter()
    for site in structure:
        for element, occupancy in site.species.items():
            actual[element.symbol] += float(occupancy)
    passed = set(actual) == set(expected) and all(
        abs(actual[k] - expected[k]) <= config()["occupancy"] for k in expected
    )
    return {
        "status": "pass" if passed else "mismatch",
        "code": "independent_expanded_composition",
        "n_atoms": len(structure),
    }


def average_reference(source):
    """Expand original CCDC marginal sites, preserving alternative identity."""
    lattice = np.asarray(source["lattice"])
    inv = np.linalg.inv(lattice)
    sites = []
    for atom in source["all_sites"]["atoms"]:
        for r, t in map(parse_operation, source["operations"]):
            f = np.asarray(atom["xyz"]) @ inv @ r.T + t
            # One source atomic/disorder orbit contributes once to any site.
            if any(
                atom["symbol"] == x["symbol"]
                and all(
                    atom.get(k) == x.get(k)
                    for k in ("formal_charge", "isotope", "stereo")
                )
                and atom["group"] == x["group"]
                and atom["assembly"] == x["assembly"]
                and np.linalg.norm(
                    ((f - x["fractional"]) - np.rint(f - x["fractional"])) @ lattice
                )
                <= config()["position_A"]
                for x in sites
            ):
                continue
            sites.append({**atom, "fractional": f % 1.0})
    return sites


def compare_average(reference, candidate, lattice):
    from ase.geometry import find_mic

    if len(reference) != len(candidate):
        return {
            "status": "mismatch",
            "code": "average_site_count",
            "n_atoms": len(candidate),
        }
    if not reference:
        return {"status": "indeterminate", "code": "empty_average"}
    n = len(reference)
    cost = np.full((n, n), 1e9)
    for i, left in enumerate(reference):
        for j, right in enumerate(candidate):
            if (
                left["symbol"] != right["symbol"]
                or abs(left["occupancy"] - right["occupancy"]) > config()["occupancy"]
            ):
                continue
            delta = (np.asarray(left["fractional"]) - right["fractional"]) @ lattice
            _, distance = find_mic(delta, lattice, pbc=True)
            cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    residual = cost[rows, cols]
    passed = bool(np.max(residual) <= config()["position_A"])
    return {
        "status": "pass" if passed else "mismatch",
        "code": "average_site_geometry_occupancy",
        "max_A": float(residual.max()),
        "rmsd_A": float(np.sqrt(np.mean(residual**2))),
        "n_atoms": n,
    }


def average_from_cif(path, source):
    """Read raw sites independently; never normalize overoccupation away."""
    from pymatgen.io.cif import CifParser

    block = next(iter(CifParser(str(path)).as_dict().values()))
    symbols = block["_atom_site_type_symbol"]
    lattice = np.asarray(source["lattice"])
    rows = []
    for i, symbol in enumerate(symbols):
        symbol, isotope = isotope_symbol(symbol)
        f = np.array([float(block["_atom_site_fract_" + axis][i]) for axis in "xyz"])
        group = block.get("_atom_site_disorder_group", ["0"] * len(symbols))[i]
        assembly = block.get("_atom_site_disorder_assembly", ["."] * len(symbols))[i]
        rows.append(
            {
                "symbol": symbol,
                "isotope": isotope,
                "formal_charge": float(
                    block.get("_atom_site_charge", [0.0] * len(symbols))[i]
                ),
                "xyz": (f @ lattice).tolist(),
                "occupancy": float(
                    block.get("_atom_site_occupancy", [1.0] * len(symbols))[i]
                ),
                "group": int(group) if group not in (".", "?") else 0,
                "assembly": assembly,
            }
        )
    return average_reference({**source, "all_sites": {"atoms": rows}}), block


def run_disorder(source, context, stages, folder):
    from molcrys_kit.operations.forward import construct_statistical_disorder
    from molcrys_kit.io.output import write_disorder_cif, write_forward_cif

    poses = make_poses(source, context, disorder=True)
    model = construct_statistical_disorder(
        context, poses, correlation_status="unknown_from_source"
    )
    expected = average_reference(source)
    exported, _ = average_from_cif(folder / "input.cif", source)
    stages["export"] = compare_average(expected, exported, context.lattice)
    candidate = [
        {"symbol": a.symbol, "fractional": a.fractional, "occupancy": a.occupancy}
        for a in model.average_sites
    ]
    stages["memory_average"] = compare_average(expected, candidate, context.lattice)
    stages["memory_average"]["n_alternatives"] = len(model.alternatives)
    write_disorder_cif(model, filename=str(folder / "average.cif"))
    # The statistical writer emits full-cell P1 sites. Read its raw rows;
    # CifParser.parse_structures may merge or rescale coincident alternatives.
    parsed, block = average_from_cif(
        folder / "average.cif", {**source, "operations": ["x,y,z"]}
    )
    stages["p1_average_pymatgen"] = compare_average(expected, parsed, context.lattice)
    try:
        independent = check_ccdc(source["refcode"], folder / "average.cif")
        independent_sites = [
            dict(
                a,
                fractional=np.asarray(a["xyz"]) @ np.linalg.inv(independent["lattice"]),
            )
            for m in independent["molecules"]
            for a in m["atoms"]
        ]
        stages["p1_average_ccdc"] = compare_average(
            expected, independent_sites, context.lattice
        )
    except Exception as exc:
        stages["p1_average_ccdc"] = stage_error(exc, "independent_average_readback")
    stages["average_chemistry"] = {
        "status": "unsupported",
        "code": "average_sites_omit_known_chemical_annotations",
        "missing_annotations": ["formal_charge", "bond_order"],
    }
    import itertools

    orbits = list(model.alternatives_by_orbit.values())
    count = int(np.prod([len(x) for x in orbits], dtype=object))
    checked = 0
    for selection in itertools.islice(
        itertools.product(*orbits), config()["max_disorder_combinations"]
    ):
        choices = {a.orbit_id: a.alternative_id for a in selection}
        realized = model.select_ordered_realization(choices)
        if sum(len(m) for m in realized.molecules) != sum(
            len(a.molecule) for a in selection
        ):
            stages["microstate_exclusivity"] = {
                "status": "mismatch",
                "code": "excluded_alternative_materialized",
            }
            break
        checked += 1
    else:
        stages["microstate_exclusivity"] = {
            "status": "pass" if checked == count else "indeterminate",
            "code": "local_exclusivity_verified"
            if checked == count
            else "microstate_enumeration_truncated",
            "n_alternatives": checked,
        }
    stages["average_source_symmetry"] = operation_set_check(
        block.get("_molcrys_source_symop_xyz", []), source["operations"]
    )
    # The pinned API exposes no inverse sidecar reader; inspecting a JSON
    # key is not a model round trip. Keep this a required, unsupported gate.
    sidecar = read_json(folder / "average.cif.json")
    lossless = all(
        "molecule" in a or "atoms" in a for a in sidecar.get("alternatives", [])
    )
    stages["disorder_model_readback"] = {
        "status": "unsupported",
        "code": "no_lossless_disorder_sidecar_reader"
        if not lossless
        else "no_disorder_sidecar_reader",
    }
    try:
        asu = write_forward_cif(model, representation="asu")
        (folder / "average_asu.cif").write_text(asu, encoding="utf-8")
        stages["asu_average"] = {
            "status": "unsupported",
            "code": "true_asu_average_not_verified",
        }
    except Exception as exc:
        stages["asu_average"] = {
            "status": "unsupported",
            "code": "true_asu_average_unavailable",
            "error_type": type(exc).__name__,
        }
    stages["correlation_preservation"] = {
        "status": "pass"
        if sidecar.get("correlation_status") == "unknown_from_source"
        else "mismatch",
        "code": "source_correlations_remain_unknown",
    }
    if source["n_combinations"] > config()["max_disorder_combinations"]:
        stages["source_combinations"] = {
            "status": "indeterminate",
            "code": "source_combinations_truncated",
        }


class CheckpointStages(dict):
    """Keep completed stages if a later native parser hangs or crashes."""

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.checkpoint()


def run_case(case, run_metadata, result_path=None):
    from molcrys_kit.operations.forward import (
        prepare_context_from_cif,
        construct_molecular_crystal,
    )
    from molcrys_kit.io.output import write_cif_p1, write_cif_asu
    from molcrys_kit.io import read_mol_crystal
    from molcrys_kit.analysis.sanity_check import sanity_check

    started = time.monotonic()
    folder = LOCAL / "cases" / case["refcode"]
    result = {
        **case,
        **run_metadata,
        "stages": {},
        "diagnostics": {},
        "status": "indeterminate",
        "finished": False,
    }

    def checkpoint():
        if result_path:
            result["elapsed_s"] = round(time.monotonic() - started, 4)
            write_json(result_path, result)

    stages = CheckpointStages(checkpoint)
    result["stages"] = stages
    try:
        source = read_json(folder / "source.json")
        if (
            hashlib.sha256((folder / "input.cif").read_bytes()).hexdigest()
            != source["input_sha256"]
        ):
            raise ValueError("private input checksum mismatch")
        normalize_source_isotopes(source)
        result.update(
            {k: source[k] for k in ("missing_atoms", "input_sha256", "source_sha256")}
        )
        result["molecular_completeness"] = (
            "complete" if not source["missing_atoms"] else "missing_coordinates"
        )
        result["n_reference_molecules"] = len(source["reference"])
        result["n_reference_atoms"] = sum(len(m["atoms"]) for m in source["reference"])
        stages["export"] = source["export_check"]
        context = prepare_context_from_cif(
            str(folder / "input.cif"),
            tolerance_A=config()["position_A"],
            fractional_tolerance=config()["fractional"],
            chemistry_policy="source_crystal",
        )
        stages["context"] = {
            "status": "pass",
            "code": "strict_context",
            "n_operations": len(context.operations),
        }
        if "disordered" in case["tags"]:
            run_disorder(source, context, stages, folder)
        else:
            poses = make_poses(source, context)
            crystal = construct_molecular_crystal(context, poses)
            candidate = snapshot_mck(crystal)
            stages["memory_geometry"] = compare_molecules(
                source["reference"], candidate, context.lattice
            )
            stages["memory_cell"] = compare_cell(source["lattice"], crystal.lattice)
            stages["memory_chemistry"] = chemistry_check(
                source["reference"], candidate, context.lattice
            )
            stages["memory_symmetry"] = symmetry_check(candidate, source)
            for name, writer in (("p1", write_cif_p1), ("asu", write_cif_asu)):
                try:
                    path = folder / (name + ".cif")
                    writer(crystal, filename=str(path))
                    stages[name + "_written"] = {
                        "status": "pass",
                        "code": "file_written",
                    }
                    independent = check_ccdc(case["refcode"], path)
                    stages[name + "_ccdc_geometry"] = compare_molecules(
                        source["reference"], independent["molecules"], context.lattice
                    )
                    stages[name + "_ccdc_chemistry"] = chemistry_check(
                        source["reference"], independent["molecules"], context.lattice
                    )
                    stages[name + "_cell"] = compare_cell(
                        source["lattice"], independent["lattice"]
                    )
                    if name == "asu":
                        stages["asu_operations"] = operation_set_check(
                            independent["operations"], source["operations"]
                        )
                    else:
                        from pymatgen.io.cif import CifParser

                        block = next(iter(CifParser(str(path)).as_dict().values()))
                        stages["p1_source_operations"] = operation_set_check(
                            block.get("_molcrys_source_symop_xyz", []),
                            source["operations"],
                        )
                    stages[name + "_pymatgen"] = pymatgen_check(path, source)
                    reread = read_mol_crystal(str(path))
                    stages[name + "_mck_readback"] = compare_molecules(
                        source["reference"],
                        snapshot_mck(reread, context.lattice),
                        context.lattice,
                    )
                except Exception as exc:
                    stages[name + "_readback"] = stage_error(
                        exc, "serialization_or_independent_readback"
                    )
                    (folder / (name + "_error.txt")).write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
            # Plausibility is diagnostic: pre-existing source defects do not
            # change a faithful round-trip into a failure by themselves.
            for name, obj in (
                ("source", read_mol_crystal(str(folder / "input.cif"))),
                ("output", crystal),
            ):
                try:
                    report = sanity_check(obj)
                    result["diagnostics"][name + "_sanity"] = {
                        r.name: r.status for r in report.results
                    }
                except Exception as exc:
                    result["diagnostics"][name + "_sanity_error"] = type(exc).__name__
    except Exception as exc:
        stages["execution"] = stage_error(exc)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "worker_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    statuses = {s["status"] for s in stages.values()}
    result["status"] = next(
        (
            s
            for s in ("error", "mismatch", "unsupported", "indeterminate")
            if s in statuses
        ),
        "pass",
    )
    result["record_match"] = result["status"] == "pass"
    geometry_gates = (
        "export",
        "memory_geometry",
        "memory_cell",
        "memory_symmetry",
        "p1_ccdc_geometry",
        "p1_cell",
        "p1_pymatgen",
        "p1_mck_readback",
        "p1_source_operations",
        "asu_ccdc_geometry",
        "asu_cell",
        "asu_operations",
        "asu_pymatgen",
        "asu_mck_readback",
    )
    result["geometry_roundtrip_match"] = all(
        stages.get(k, {}).get("status") == "pass" for k in geometry_gates
    )
    result["elapsed_s"] = round(time.monotonic() - started, 4)
    result["finished"] = True
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--result", required=True)
    args = p.parse_args()
    write_json(
        args.result,
        run_case(read_json(args.case), read_json(args.metadata), args.result),
    )


if __name__ == "__main__":
    main()
