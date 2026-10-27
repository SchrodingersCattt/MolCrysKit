"""Local cached-data audit; publishes only counts and annotation transitions.

Run from the repository root with runpy.run_path(..., run_name='__main__').
No CCDC download or production file mutation occurs. Coordinates remain local.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from benchmarks.csd_roundtrip.common import (
    LOCAL,
    config,
    normalize_source_isotopes,
    read_json,
    write_json,
)
from benchmarks.csd_roundtrip.match import molecule_mapping
from benchmarks.csd_roundtrip.worker import make_poses, snapshot_mck
from molcrys_kit.operations.forward import (
    construct_molecular_crystal,
    prepare_context_from_cif,
)


def compare_properties(reference, candidate, lattice):
    inv = np.linalg.inv(lattice)
    centres = (
        np.asarray(
            [np.mean([a["xyz"] for a in m["atoms"]], axis=0) @ inv for m in candidate]
        )
        % 1.0
    )
    centres[centres >= 1.0 - 1e-12] = 0.0
    tree = cKDTree(centres, boxsize=1.0)
    radius = config()["position_A"] * np.linalg.norm(inv, 2) + 1e-10
    cost = np.full((len(reference), len(candidate)), 1e9)
    maps = {}
    for i, mol in enumerate(reference):
        centre = np.mean([a["xyz"] for a in mol["atoms"]], axis=0) @ inv % 1.0
        centre[centre >= 1.0 - 1e-12] = 0.0
        for j in tree.query_ball_point(centre, radius):
            mapping = molecule_mapping(mol, candidate[j], lattice)
            if mapping is not None:
                cost[i, j] = float(mapping[1].max())
                maps[i, j] = mapping[0]
    ri, cj = linear_sum_assignment(cost)
    if len(ri) != len(reference) or any((i, j) not in maps for i, j in zip(ri, cj)):
        return {"status": "unresolved_geometry_mapping"}
    fields = Counter()
    missing_fields = Counter()
    conflicting_fields = Counter()
    derived_fields = Counter()
    sentinels = Counter()
    transitions = Counter()
    bond_changes = Counter()
    per_mate = []
    for i, j in zip(ri, cj):
        counts = Counter()
        order = maps[i, j]
        for a, k in zip(reference[i]["atoms"], order):
            b = candidate[j]["atoms"][k]
            for key in ("formal_charge", "isotope", "stereo"):
                expected, actual = a.get(key), b.get(key)
                if key == "stereo":
                    for origin, value in (("source", expected), ("candidate", actual)):
                        if value in ("Error", "Unknown", "Mixed"):
                            sentinels[origin + ":" + value] += 1
                    if expected in ("Error", "Unknown", "Mixed", ""):
                        expected = None
                    if actual in ("Error", "Unknown", "Mixed", ""):
                        actual = None
                if expected != actual:
                    fields[key] += 1
                    counts[key] += 1
                    transitions[key + ": " + str(expected) + " -> " + str(actual)] += 1
                    if expected is not None and actual is None:
                        missing_fields[key] += 1
                    elif expected is None and actual is not None:
                        derived_fields[key] += 1
                    else:
                        conflicting_fields[key] += 1
        edges = {
            tuple(sorted(e[:2])): e[2] if len(e) > 2 else None
            for e in candidate[j]["bonds"]
        }
        for edge in reference[i]["bonds"]:
            expected = edge[2] if len(edge) > 2 else None
            actual = edges.get(tuple(sorted((order[edge[0]], order[edge[1]]))))
            if expected in ("Unknown", "UNKNOWN", "Unspecified"):
                expected = None
            if actual in ("Unknown", "UNKNOWN", "Unspecified"):
                actual = None
            if expected != actual:
                bond_changes[str(expected) + " -> " + str(actual)] += 1
        per_mate.append({"candidate_molecule_index": int(j), "fields": dict(counts)})
    return {
        "status": "mapped",
        "fields": dict(fields),
        "missing_fields": dict(missing_fields),
        "conflicting_fields": dict(conflicting_fields),
        "derived_fields": dict(derived_fields),
        "unavailable_stereo_sentinels": dict(sentinels),
        "charge_comparison_note": "Raw CCDC numbers only; an unassigned imported zero is not proof of explicitly encoded neutral charge.",
        "transitions": dict(transitions),
        "bond_changes": dict(bond_changes),
        "per_mate": per_mate,
    }


def main():
    folder = Path(__file__).resolve().parent
    rows = [
        json.loads(x)
        for x in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    output = []
    for n, row in enumerate(rows):
        ref = row["refcode"]
        source = None
        record = {"refcode": ref, "audits": {}}
        if (
            row["stages"].get("memory_chemistry", {}).get("code")
            == "atomic_chemical_identity_not_preserved"
        ):
            try:
                source = normalize_source_isotopes(
                    read_json(LOCAL / "cases" / ref / "source.json")
                )
                ctx = prepare_context_from_cif(
                    str(LOCAL / "cases" / ref / "input.cif"),
                    tolerance_A=config()["position_A"],
                    fractional_tolerance=config()["fractional"],
                    chemistry_policy="source_crystal",
                )
                crystal = construct_molecular_crystal(ctx, make_poses(source, ctx))
                audit = compare_properties(
                    source["reference"],
                    snapshot_mck(crystal),
                    np.array(source["lattice"]),
                )
                improper = Counter()
                proper = Counter()
                for m in audit.pop("per_mate", []):
                    molecule = crystal.molecules[m["candidate_molecule_index"]]
                    op = ctx.operations[molecule.info["forward_operation_index"]]
                    (improper if op.determinant < 0 else proper).update(m["fields"])
                audit["improper_operation_fields"] = dict(improper)
                audit["proper_operation_fields"] = dict(proper)
                record["audits"]["memory"] = audit
            except Exception as exc:
                record["audits"]["memory"] = {
                    "status": "audit_error",
                    "error_type": type(exc).__name__,
                }
        for mode in ("p1", "asu"):
            if row["stages"].get(mode + "_ccdc_chemistry", {}).get("code") not in {
                "chemical_identity_or_bond_order_mismatch",
                "known_chemical_annotations_not_preserved",
            }:
                continue
            try:
                if source is None:
                    source = normalize_source_isotopes(
                        read_json(LOCAL / "cases" / ref / "source.json")
                    )
                candidate = normalize_source_isotopes(
                    read_json(LOCAL / "cases" / ref / (mode + ".ccdc.json"))
                )
                lattice = np.array(source["lattice"])
                basis = np.linalg.inv(candidate["lattice"]) @ lattice
                for molecule in candidate["molecules"]:
                    for atom in molecule["atoms"]:
                        atom["xyz"] = (np.array(atom["xyz"]) @ basis).tolist()
                audit = compare_properties(
                    source["reference"], candidate["molecules"], lattice
                )
                audit.pop("per_mate", None)
                record["audits"][mode] = audit
            except Exception as exc:
                record["audits"][mode] = {
                    "status": "audit_error",
                    "error_type": type(exc).__name__,
                }
        if record["audits"]:
            output.append(record)
        if n % 100 == 0:
            print("audit", n, "/", len(rows), flush=True)
    summary = {}
    for mode in ("memory", "p1", "asu"):
        cases = [x["audits"][mode] for x in output if mode in x["audits"]]
        fields = Counter()
        bonds = Counter()
        proper = Counter()
        improper = Counter()
        missing = Counter()
        conflicts = Counter()
        derived = Counter()
        unavailable = Counter()
        for x in cases:
            fields.update(x.get("fields", {}))
            bonds.update(x.get("bond_changes", {}))
            proper.update(x.get("proper_operation_fields", {}))
            improper.update(x.get("improper_operation_fields", {}))
            missing.update(x.get("missing_fields", {}))
            conflicts.update(x.get("conflicting_fields", {}))
            derived.update(x.get("derived_fields", {}))
            unavailable.update(x.get("unavailable_stereo_sentinels", {}))
        summary[mode] = {
            "cases": len(cases),
            "status": dict(Counter(x["status"] for x in cases)),
            "field_differences": dict(fields),
            "bond_transitions": dict(bonds),
            "proper_operation_fields": dict(proper),
            "improper_operation_fields": dict(improper),
            "missing_fields": dict(missing),
            "conflicting_fields": dict(conflicts),
            "derived_fields": dict(derived),
            "unavailable_stereo_sentinels": dict(unavailable),
        }
    write_json(
        folder / "identity_audit.json",
        {
            "target_commit": config()["target_commit"],
            "records": output,
            "summary": summary,
            "contains_coordinates": False,
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
