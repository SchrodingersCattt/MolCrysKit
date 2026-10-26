"""Run only with the licensed CCDC interpreter. All structures stay _local.

The metadata extractor reuses the maintained CSD exporter's disorder lookup.
The benchmark writes 12-decimal values rather than its teaching precision.
No coordinates are generated for atoms with null CCDC sites.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import traceback

import numpy as np

from .common import (
    LOCAL,
    REPO,
    ROOT,
    STRATA,
    config,
    digest,
    read_json,
    write_json,
    normalize_source_isotopes,
)
from .match import (
    compare_molecules,
    independent_representatives,
    lattice_from_parameters,
)


def metadata(crystal):
    spec = importlib.util.spec_from_file_location(
        "csd_exporter",
        REPO / "skills/query-csd-structures/scripts/demo_csd_disorder_workflow.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._disorder_lookup(crystal)[0]


def operations(crystal):
    return [
        (
            np.asarray(crystal.symmetry_rotation(s)).reshape(3, 3),
            np.asarray(crystal.symmetry_translation(s)),
        )
        for s in crystal.symmetry_operators
    ]


def snapshot(molecule, crystal, lattice, groups=None):
    """CCDC component in a canonical Cartesian basis, retaining unwrapped sites."""
    groups = groups or {}
    atoms = []
    indices = {}
    rotation = np.asarray(crystal.orthogonal_to_fractional.rotation)
    for atom in molecule.atoms:
        try:
            if atom.coordinates is None:
                continue
            # CCDC Transformation.rotation acts on row coordinates.
            frac = np.asarray(atom.coordinates) @ rotation
            xyz = frac @ lattice
            if not np.isfinite(xyz).all():
                continue
        except (RuntimeError, AttributeError, TypeError):
            continue
        indices[atom.index] = len(atoms)
        group, assembly = groups.get(atom.label, (0, "."))
        atoms.append(
            {
                "symbol": atom.atomic_symbol,
                "xyz": xyz.tolist(),
                "label": atom.label,
                "occupancy": float(
                    atom.occupancy if atom.occupancy is not None else 1.0
                ),
                "formal_charge": int(atom.formal_charge),
                "stereo": str(atom.chirality) or None,
                "isotope": None,
                "group": group or 0,
                "assembly": assembly or ".",
            }
        )
    bonds = []
    for bond in molecule.bonds:
        a, b = (x.index for x in bond.atoms)
        if a in indices and b in indices:
            bonds.append([indices[a], indices[b], str(bond.bond_type)])
    return {
        "atoms": atoms,
        "bonds": bonds,
        "missing_atoms": len(molecule.atoms) - len(atoms),
    }


def components(molecule, crystal, lattice, groups=None):
    return [
        s
        for component in molecule.components
        if (s := snapshot(component, crystal, lattice, groups))["atoms"]
    ]


def classification(entry):
    c = entry.crystal
    lattice = lattice_from_parameters(c.cell_lengths, c.cell_angles)
    ops = operations(c)
    selected = components(c.molecule, c, lattice)
    all_mol = c.disordered_molecule
    partial = any(
        a.occupancy is not None and float(a.occupancy) < 1.0 - config()["occupancy"]
        for a in all_mol.atoms
    )
    special = False
    for component in selected:
        anchor = np.mean(
            [a["xyz"] for a in component["atoms"]], axis=0
        ) @ np.linalg.inv(lattice)
        stabilizers = 0
        for rotation, shift in ops:
            delta = anchor @ rotation.T + shift - anchor
            if np.max(np.abs(delta - np.rint(delta))) <= config()["fractional"]:
                stabilizers += 1
        special |= stabilizers > 1
    charges = [sum(a["formal_charge"] for a in m["atoms"]) for m in selected]
    ionic = sum(q != 0 for q in charges) >= 2
    tags = []
    for flag, name in (
        (entry.has_disorder or partial, "disordered"),
        (partial, "partial_occupancy"),
        (special, "special_position"),
        (ionic, "multi_ion"),
        (len(selected) > 1, "multi_component"),
        (any(m["missing_atoms"] for m in selected), "missing_coordinates"),
        (any(a.is_metal for a in all_mol.atoms), "metal_containing"),
    ):
        if flag:
            tags.append(name)
    stratum = next((s for s in STRATA[:-1] if s in tags), "single_component")
    return {"stratum": stratum, "tags": tags, "n_components": len(selected)}


def high_precision_cif(crystal, lattice, groups):
    """Symmetry-complete molecular rows are reduced to source atomic orbits.

    Preserve distinct occupancies, chemical identities and disorder groups.
    The reference packing comparison detects any loss during this adapter.
    """
    from .match import parse_operation

    molecule = snapshot(crystal.disordered_molecule, crystal, lattice, groups)
    inv = np.linalg.inv(lattice)
    ops = [parse_operation(s) for s in crystal.symmetry_operators]
    rows = []
    source_to_row = {}
    for index, atom in enumerate(molecule["atoms"]):
        f = np.asarray(atom["xyz"]) @ inv
        matched = None
        for j, row in enumerate(rows):
            if any(
                atom.get(k) != row.get(k)
                for k in (
                    "symbol",
                    "formal_charge",
                    "stereo",
                    "group",
                    "assembly",
                    "occupancy",
                )
            ):
                continue
            rf = np.asarray(row["xyz"]) @ inv
            for rotation, shift in ops:
                delta = f @ rotation.T + shift - rf
                if (
                    np.linalg.norm((delta - np.rint(delta)) @ lattice)
                    < config()["position_A"]
                ):
                    matched = j
                    break
            if matched is not None:
                break
        if matched is None:
            matched = len(rows)
            rows.append(atom)
        source_to_row[index] = matched
    number, setting = crystal.spacegroup_number_and_setting
    lines = [
        "data_roundtrip",
        f"_symmetry_space_group_name_H-M '{crystal.spacegroup_symbol}'",
        f"_space_group_IT_number {number}",
        f"_space_group_setting {setting}",
    ]
    for axis, value in zip("abc", crystal.cell_lengths):
        lines.append(f"_cell_length_{axis} {value:.12f}")
    for axis, value in zip(("alpha", "beta", "gamma"), crystal.cell_angles):
        lines.append(f"_cell_angle_{axis} {value:.12f}")
    lines += ["loop_", "_space_group_symop_operation_xyz"]
    lines += [f"'{s}'" for s in crystal.symmetry_operators]
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_disorder_group",
        "_atom_site_disorder_assembly",
        "_atom_site_charge",
    ]
    for i, atom in enumerate(rows):
        f = np.asarray(atom["xyz"]) @ inv
        lines.append(
            f"{atom['symbol']}{i + 1} {atom['symbol']} "
            + " ".join(f"{x:.12f}" for x in f)
            + f" {atom['occupancy']:.12f} {atom['group']} {atom['assembly']} {atom['formal_charge']}"
        )
    # Bond perception is independently checked against the original CSD graph;
    # the chemical annotation companion below records data CIF cannot express.
    return "\n".join(lines) + "\n", molecule


def export_case(reader, case):
    from ccdc.crystal import Crystal

    ref = case["refcode"]
    folder = LOCAL / "cases" / ref
    folder.mkdir(parents=True, exist_ok=True)
    source = reader.crystal(ref)
    lattice = lattice_from_parameters(source.cell_lengths, source.cell_angles)
    groups = metadata(source)
    cif, all_molecule = high_precision_cif(source, lattice, groups)
    (folder / "input.cif").write_text(cif, encoding="utf-8")
    original = components(source.molecule, source, lattice, groups)
    reference = components(
        source.packing(inclusion="UniqueIncluded"), source, lattice, groups
    )
    rebuilt = Crystal.from_string(cif, "cif")
    reread = components(rebuilt.molecule, rebuilt, lattice)
    reread_packing = components(
        rebuilt.packing(inclusion="UniqueIncluded"), rebuilt, lattice
    )
    geometric_export = compare_molecules(reference, reread_packing, lattice)
    # Extraction and reconstruction consume the re-read CIF geometry, while
    # the original CSD chemistry companion remains independent known truth.
    representatives = independent_representatives(reread, operations(source), lattice)
    source_representatives = independent_representatives(
        original, operations(source), lattice
    )
    missing = sum(x["missing_atoms"] for x in original)
    disorder_groups = []
    if source.disorder:
        for assembly in source.disorder.assemblies:
            disorder_groups.append(
                {
                    "assembly": str(assembly.id),
                    "groups": [
                        {
                            "id": str(g.id),
                            "weight": float(g.occupancy),
                            "labels": [a.label for a in g.atoms],
                        }
                        for g in assembly.groups
                    ],
                }
            )
    alternatives = []
    n_combinations = 1
    if source.disorder:
        n_combinations = int(
            np.prod([len(a.groups) for a in source.disorder.assemblies])
        )
        for selected in itertools.islice(
            source.disorder.combinations, config()["max_disorder_combinations"]
        ):
            current = components(source.molecule, source, lattice, groups)
            alternatives.append(
                {"selection": [str(x.id) for x in selected], "molecules": current}
            )
    payload = {
        "schema": "csd-roundtrip.private.v1",
        "refcode": ref,
        "lattice": lattice.tolist(),
        "operations": list(source.symmetry_operators),
        "spacegroup_number": int(source.spacegroup_number_and_setting[0]),
        "spacegroup_symbol": source.spacegroup_symbol,
        "setting": int(source.spacegroup_number_and_setting[1]),
        "templates": representatives,
        "source_templates": source_representatives,
        "reference": reference,
        "all_sites": all_molecule,
        "export_check": geometric_export,
        "export_reread_reference": reread_packing,
        "disorder_groups": disorder_groups,
        "alternatives": alternatives,
        "n_combinations": n_combinations,
        "correlations": "unknown_from_source",
        "missing_atoms": missing,
        "input_sha256": hashlib.sha256((folder / "input.cif").read_bytes()).hexdigest(),
    }
    payload["source_sha256"] = digest(payload)
    write_json(folder / "source.json", payload)
    return {
        "refcode": ref,
        "export": geometric_export["status"],
        "n_templates": len(representatives),
    }


def environment(reader):
    import ccdc
    from ccdc import io
    import platform

    database = Path(io.csd_directory())
    active = (
        reader.file_name if isinstance(reader.file_name, list) else [reader.file_name]
    )
    return {
        "ccdc_version": ccdc.__version__,
        "python": platform.python_version(),
        "csd_current_version": io.csd_version(return_current_version=True),
        "active_databases": [Path(p).name for p in active],
        "entry_count": len(reader),
        "database_metadata": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(database.glob("*.inf"))
        ],
        "database_files": [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(database.glob("*.sqlite"))
        ],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare(targeted_only=False):
    from ccdc import io

    cfg = config()
    with io.EntryReader("CSD") as reader:
        env = environment(reader)
        if targeted_only:
            for ref in cfg["targeted"]:
                case = dict(
                    classification(reader.entry(ref)), refcode=ref, targeted=True
                )
                try:
                    print(json.dumps(export_case(reader, case)), flush=True)
                except Exception:
                    folder = LOCAL / "cases" / ref
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "export_error.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    print(ref, "export_error", flush=True)
            write_json(LOCAL / "environment.json", env)
            return
        manifest_path = ROOT / "manifest.json"
        if manifest_path.exists():
            raise ValueError("frozen manifest already exists; use export to resume")
        print("Enumerating exact refcodes", len(reader), flush=True)
        identifiers = sorted(reader.identifier(i) for i in range(len(reader)))
        population_sha = digest(identifiers)
        random.Random(cfg["seed"]).shuffle(identifiers)
        counts = Counter()
        screening = Counter()
        cases = []
        for ref in identifiers:
            if ref in cfg["targeted"]:
                continue
            screening["scanned"] += 1
            try:
                e = reader.entry(ref)
                if not e.has_3d_structure:
                    screening["no_3d"] += 1
                    continue
                if e.is_polymeric:
                    screening["polymeric"] += 1
                    continue
                if not any(a.atomic_symbol == "C" for a in e.molecule.atoms):
                    screening["no_carbon"] += 1
                    continue
                meta = classification(e)
                stratum = meta["stratum"]
                screening["eligible_" + stratum] += 1
                if counts[stratum] < cfg["per_stratum"]:
                    cases.append(dict(meta, refcode=ref, targeted=False))
                    counts[stratum] += 1
            except Exception:
                screening["metadata_error"] += 1
            if screening["scanned"] % 500 == 0:
                print("screen", screening["scanned"], dict(counts), flush=True)
            if all(counts[s] == cfg["per_stratum"] for s in STRATA):
                break
        if not all(counts[s] == cfg["per_stratum"] for s in STRATA):
            raise RuntimeError(
                "stratum shortage; refusing to report a complete 1000-case experiment"
            )
        for ref in cfg["targeted"]:
            cases.append(
                dict(classification(reader.entry(ref)), refcode=ref, targeted=True)
            )
        manifest = {
            "schema": "csd-roundtrip.manifest.v1",
            "config_sha256": digest(cfg),
            "target_commit": cfg["target_commit"],
            "population_sha256": population_sha,
            "environment": env,
            "screening": dict(screening),
            "strata": dict(counts),
            "cases": sorted(cases, key=lambda c: (not c["targeted"], c["refcode"])),
        }
        manifest["sha256"] = digest(manifest)
        write_json(manifest_path, manifest)
        print("FROZEN", len(cases), manifest["sha256"], flush=True)


def export_all():
    from ccdc import io

    manifest = read_json(ROOT / "manifest.json")
    with io.EntryReader("CSD") as reader:
        for i, case in enumerate(manifest["cases"]):
            folder = LOCAL / "cases" / case["refcode"]
            if (folder / "source.json").exists():
                continue
            try:
                result = export_case(reader, case)
                print(i + 1, json.dumps(result), flush=True)
            except Exception:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "export_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                print(i + 1, case["refcode"], "export_error", flush=True)


def check_output(refcode, filename):
    from ccdc.crystal import Crystal

    crystal = Crystal.from_string(Path(filename).read_text(encoding="utf-8"), "cif")
    lattice = lattice_from_parameters(crystal.cell_lengths, crystal.cell_angles)
    snapshot_value = {
        "lattice": lattice.tolist(),
        "molecules": components(
            crystal.packing(inclusion="UniqueIncluded"), crystal, lattice
        ),
        "operations": list(crystal.symmetry_operators),
    }
    normalize_source_isotopes(snapshot_value)
    write_json(Path(filename).with_suffix(".ccdc.json"), snapshot_value)
    # This command owns no persistent state. The snapshot is fully written
    # and atomically renamed before exit. On this Windows CCDC build native
    # object finalizers can hang after successful parsing/packing; avoid
    # interpreting a finalizer stall as a structural readback failure.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    import faulthandler

    faulthandler.dump_traceback_later(60, repeat=True)
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "action", choices=["prepare", "targeted", "export", "check", "environment"]
    )
    p.add_argument("--refcode")
    p.add_argument("--file")
    args = p.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "targeted":
        prepare(True)
    elif args.action == "export":
        export_all()
    elif args.action == "environment":
        from ccdc import io

        with io.EntryReader("CSD") as reader:
            write_json(ROOT / "source_environment.json", environment(reader))
    else:
        check_output(args.refcode, args.file)


if __name__ == "__main__":
    main()
