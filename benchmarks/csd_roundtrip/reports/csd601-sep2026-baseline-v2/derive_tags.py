"""Supplementary coordinate-free coverage tags; never changes match scores.

Run from the repository root with:
python -c "import runpy; runpy.run_path('benchmarks/csd_roundtrip/reports/csd601-sep2026-baseline-v2/derive_tags.py', run_name='__main__')"
"""

from collections import Counter
import csv
import hashlib
from pathlib import Path

import networkx as nx
import numpy as np

from benchmarks.csd_roundtrip.common import (
    LOCAL,
    ROOT,
    normalize_source_isotopes,
    read_json,
    write_json,
)
from benchmarks.csd_roundtrip.match import graph, signature, parse_operation


SOLVENT_FORMULAE = {
    "H2O",
    "CH4O",
    "C2H6O",
    "C2H3N",
    "C3H6O",
    "C4H8O",
    "C4H10O",
    "C3H7NO",
    "C2H6OS",
    "CH2Cl2",
    "CHCl3",
    "C6H6",
    "C7H8",
}


def hill_formula(atoms):
    counts = Counter(a["symbol"] for a in atoms)
    order = [x for x in ("C", "H") if x in counts] + sorted(set(counts) - {"C", "H"})
    return "".join(x + (str(counts[x]) if counts[x] != 1 else "") for x in order)


def main():
    folder = Path(__file__).resolve().parent
    manifest = read_json(ROOT / "manifest.json")
    rows = []
    totals = Counter()
    for case in manifest["cases"]:
        source = normalize_source_isotopes(
            read_json(LOCAL / "cases" / case["refcode"] / "source.json")
        )
        representatives = []
        for component in source["source_templates"]:
            g = graph(component, chemistry=True)
            if not any(
                signature(component) == signature(old)
                and nx.is_isomorphic(
                    g,
                    old_graph,
                    node_match=lambda a, b: a["identity"] == b["identity"],
                    edge_match=lambda a, b: a["kind"] == b["kind"],
                )
                for old, old_graph in representatives
            ):
                representatives.append((component, g))
        compounds = [m for m, _ in representatives]
        charges = [sum(a["formal_charge"] for a in m["atoms"]) for m in compounds]
        neutral_organic = [
            m
            for m, q in zip(compounds, charges)
            if q == 0 and any(a["symbol"] == "C" for a in m["atoms"])
        ]
        formulae = [hill_formula(m["atoms"]) for m in compounds]
        tags = list(case["tags"])
        if any(q > 0 for q in charges) and any(q < 0 for q in charges):
            tags.append("oppositely_charged_components")
        if len(neutral_organic) > 1:
            tags.append("neutral_multispecies_cocrystal_or_solvate_candidate")
        if "H2O" in formulae:
            tags.append("water_component_hydrate_candidate")
        if len(compounds) > 1 and any(f in SOLVENT_FORMULAE for f in formulae):
            tags.append("observed_solvent_formula_candidate")
        if len(source["source_templates"]) > len(compounds):
            tags.append("repeated_independent_species")
        if any(
            np.allclose(r, np.eye(3), atol=1e-7, rtol=0.0)
            and np.max(np.abs(t - np.rint(t))) > 1e-7
            for r, t in map(parse_operation, source["operations"])
        ):
            tags.append("centred_conventional_cell")
        totals.update(tags)
        rows.append(
            {
                "refcode": case["refcode"],
                "stratum": case["stratum"],
                "targeted": case["targeted"],
                "n_independent_components": len(source["source_templates"]),
                "n_species_available_annotations": len(compounds),
                "tags": ";".join(sorted(set(tags))),
            }
        )
    with (folder / "coverage_tags.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        folder / "coverage_tags.json",
        {
            "schema": "csd-roundtrip.coverage-tags.v1",
            "cases": len(rows),
            "manifest_sha256": manifest["sha256"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tag_counts": dict(totals),
            "species_identity": "graph isomorphism using available element, charge, explicit D/T isotope, stereo and typed bonds",
            "role_policy": "Water and solvent-formula flags are candidates; neutral multispecies does not distinguish a coformer from a solvent. No crystallization role is inferred as verified truth.",
            "changes_frozen_strata_or_scores": False,
        },
    )
    print(dict(totals), flush=True)


if __name__ == "__main__":
    main()
