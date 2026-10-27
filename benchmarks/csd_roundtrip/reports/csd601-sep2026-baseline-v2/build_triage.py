"""Generate public refcode lists and a stage-to-repair catalogue.

Uses the frozen public manifest/results and count-only supplementary audits.
No coordinates, atom maps or raw exceptions are read or published.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path

from benchmarks.csd_roundtrip.common import ROOT, config, read_json, write_json

PRODUCT = "https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/"
HARNESS = "https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/"
ISSUES = {
    "RT01": {
        "title": "ASU bond expansion creates cross-molecule edges",
        "priority": "P1",
        "owner": "product",
        "locations": [
            ["molcrys_kit/io/cif.py", 2293, "_published_bond_pairs fallback"],
            ["molcrys_kit/io/cif.py", 2303, "independent ASU endpoint candidate loops"],
            ["molcrys_kit/io/cif.py", 2475, "published bonds enter component graph"],
        ],
        "expected": "Two CH4 components and 8 intramolecular C-H edges",
        "observed": "Special-site dedup removes identity-image representatives. Fallback at 2296/2299 selects ALL mates; the independent endpoint Cartesian product at 2303/2304 creates cross-molecule edges. The symmetry-coded branch bypasses distance checking. CH4 has 16 edges (8 valid plus 8 spurious 7.569062-A edges). HXMTAM10 has 384 explicit pairs, 336 longer than every published bond; a diagnostic removal restores [22,22] components.",
        "next_action": "Retain original ASU source coordinates/op-to-retained-atom mapping; transform both endpoints with the same group operation; preserve directed lattice images. Do not select every mate when identity images were deduplicated.",
    },
    "RT02": {
        "title": "Whole-group chirality policy and improper-mate labels",
        "priority": "P1",
        "owner": "product",
        "locations": [
            [
                "molcrys_kit/operations/forward.py",
                614,
                "analyze_molecular_pose only iterates the site subgroup",
            ],
            [
                "molcrys_kit/operations/forward.py",
                799,
                "constructor calls site-only compatibility",
            ],
            [
                "molcrys_kit/operations/forward.py",
                704,
                "_copy_generated_atoms copies stereo arrays unchanged",
            ],
        ],
        "expected": "Reject incompatible explicit enantiopure requests; allowed racemic mates have valid transformed stereo annotations",
        "observed": "General-position P-1 accepts enantiopure input and copies R to both mates; audited known R/S conflicts occur on improper operations",
        "next_action": "Validate the explicit chemistry policy against the full operation set before expansion; reassign stereo from complete typed graph plus transformed geometry for allowed improper mates, or preserve unknown status. Do not mechanically flip every stereo descriptor.",
    },
    "RT03": {
        "title": "Typed bonds are stripped and CIF output hardcodes single",
        "priority": "P1",
        "owner": "product",
        "locations": [
            [
                "molcrys_kit/operations/forward.py",
                204,
                "MolecularTemplate untyped connectivity",
            ],
            [
                "molcrys_kit/operations/forward.py",
                239,
                "graph-to-connectivity drops edge attributes",
            ],
            [
                "molcrys_kit/operations/forward.py",
                766,
                "generated edges omit bond order/kind",
            ],
            ["molcrys_kit/io/output.py", 658, "P1 literal single"],
            ["molcrys_kit/io/output.py", 1284, "ASU literal single"],
            [
                "molcrys_kit/io/cif.py",
                2200,
                "published typed records reduced to integer pairs",
            ],
            [
                "molcrys_kit/io/cif.py",
                588,
                "materialized graph omits parsed bond types",
            ],
        ],
        "expected": "Known double/triple/aromatic/coordination types and unknown states survive construction and both CIF paths",
        "observed": "An explicitly annotated double edge is serialized as single; template input accepts only two-index edges",
        "next_action": "Add optional typed periodic bond records, preserving legacy pair inputs; propagate order/kind/unknown masks through template, generated graph, CIF bond loop and reader. Validate encoded tags separately from CCDC's unassigned chemistry.",
    },
    "RT04": {
        "title": "Isotope and explicit charge metadata disappear on writing",
        "priority": "P1",
        "owner": "product",
        "locations": [
            ["molcrys_kit/io/output.py", 615, "P1 writes element-only type symbol"],
            [
                "molcrys_kit/io/output.py",
                837,
                "_forward_atom_rows omits isotope/charge fields",
            ],
            ["molcrys_kit/io/output.py", 1157, "ASU writes element-only type symbol"],
            [
                "molcrys_kit/operations/forward.py",
                1145,
                "average-site records omit chemical annotations",
            ],
        ],
        "expected": "D2 retains isotope 2; Na+/Cl- retain known +1/-1, with unknown distinguished from explicit zero",
        "observed": "Synthetic D2 reads isotopes [0,0]; synthetic ion charges read [None,None]; AMEGEF03 loses isotope metadata",
        "next_action": "Carry isotope, formal charge, known masks and permitted site identity into writer/ASU-reduction rows and averaged-site records; encode in supported CIF/companion fields and validate readback without charge guessing.",
    },
    "RT05": {
        "title": "D/T aliases are passed to ASE as element names",
        "priority": "P2",
        "owner": "product",
        "locations": [
            ["molcrys_kit/io/cif.py", 1299, "_parse_atom_type_symbol"],
            ["molcrys_kit/io/cif.py", 1741, "symbols/isotopes extracted"],
            ["molcrys_kit/io/cif.py", 2410, "ASE Atoms receives D"],
        ],
        "expected": "D maps to H/isotope 2; T to H/isotope 3; raw token remains auditable",
        "observed": "Synthetic one-D CIF and AMEGEF03 source sanity read raise KeyError",
        "next_action": "Normalize explicit isotope aliases before ASE construction; validate conflicts with leading mass numbers and preserve raw token, charge and known status.",
    },
    "RT06": {
        "title": "Statistical companion is not a lossless molecular model",
        "priority": "P1",
        "owner": "product",
        "locations": [
            [
                "molcrys_kit/operations/forward.py",
                913,
                "StatisticalDisorderModel.to_dict",
            ],
            ["molcrys_kit/io/output.py", 1407, "sidecar dump"],
        ],
        "expected": "A shipped reader reconstructs complete alternatives, graph/identity, context, weights, constraints and valid selections",
        "observed": "Alternative records contain IDs/weight/op/centre but no complete molecular record; no public inverse model reader is supplied",
        "next_action": "Define a versioned lossless schema and reader, including unwrapped atoms, typed graph, atom/site mappings, cell/setting/operations/tolerances, occupations and constraint status. Bind companion to CIF; fail explicitly on insufficient legacy data.",
    },
    "RT07": {
        "title": "Statistical true-ASU dispatch enters ordered writer",
        "priority": "P1",
        "owner": "product",
        "locations": [
            [
                "molcrys_kit/io/output.py",
                304,
                "ASU dispatch unconditionally calls write_cif_asu",
            ],
            [
                "molcrys_kit/io/output.py",
                767,
                "ordered writer expects crystal.molecules",
            ],
            ["molcrys_kit/io/output.py", 1353, "statistical writer always declares P1"],
        ],
        "expected": "True-source-group ASU marginal output independently expands to the same weighted inventory, with companion retaining alternative/correlation semantics",
        "observed": "write_forward_cif(statistical_model,representation='asu') raises AttributeError; supported statistical output is P1",
        "next_action": "Dispatch on representation AND object type; implement chemical-identity-aware reduction of average sites, correct multiplicities/occupancies and companion group action. Do not attach the full group to already expanded P1 rows.",
    },
    "RT08": {
        "title": "Benchmark treats unavailable CCDC chemistry as known labels",
        "priority": "P1",
        "owner": "benchmark",
        "locations": [
            [
                "benchmarks/csd_roundtrip/ccdc_adapter.py",
                90,
                "snapshot converts chirality sentinel into stereo value",
            ],
            [
                "benchmarks/csd_roundtrip/worker.py",
                75,
                "chemistry_check availability and identity logic",
            ],
            [
                "benchmarks/csd_roundtrip/worker.py",
                148,
                "annotation transfer falls back to CIF-derived chemistry",
            ],
        ],
        "expected": "Error/Unknown/Mixed statuses are separate from known R/S; unavailable bond types are not contradictory known orders; source annotations are bound without inference",
        "observed": "Identical geometry with stereo None versus Error is scored mismatch. CCDC synthetic CIF import returns bond Unknown/chirality Error for both tested tag aliases. Of 108 raw memory identity flags, 93 have known R/S conflicts, one has missing known stereo and 14 have no valid-label contradiction after sentinel normalization.",
        "next_action": "Preserve raw vendor status plus availability masks; compare valid assignments only and record missing/indeterminate separately. Check encoded bond data directly; do not repair validation by guessing bond orders or copying truth into outputs. Bind input annotations against symmetry-complete source instances when the CIF reader chooses another mate.",
    },
}


ISSUES.update(
    {
        "RT09": {
            "title": "Disorder seed/oracle is a limited benchmark probe",
            "priority": "P1",
            "owner": "benchmark",
            "locations": [
                [
                    "benchmarks/csd_roundtrip/worker.py",
                    129,
                    "one selected source template set",
                ],
                [
                    "benchmarks/csd_roundtrip/worker.py",
                    207,
                    "q inferred from minimum atomic occupancy",
                ],
                [
                    "benchmarks/csd_roundtrip/worker.py",
                    379,
                    "strict average-site count comparison",
                ],
            ],
            "expected": "All supplied source molecular alternatives/weights are explicit; unknown input relations remain indeterminate; valid split/merged shared sites are compared by justified identity",
            "observed": "Only the selected template seed is supplied; q is inferred; source-group site counts are compared without a complete shared-site equivalence oracle",
            "next_action": "Build the complete source input model before grading the constructor, or label this probe incomplete. Supply non-symmetry-related conformers explicitly, validate q and compare occupancy measures with permitted site identities. Never force the constructor to invent missing experimental alternatives.",
        },
        "RT10": {
            "title": "Zero-probability alternatives can be selected as occupied",
            "priority": "P1",
            "owner": "product",
            "locations": [
                [
                    "molcrys_kit/operations/forward.py",
                    1195,
                    "select_ordered_realization defaults to first state",
                ],
                [
                    "molcrys_kit/operations/forward.py",
                    1203,
                    "selected molecule appended without weight/occupation check",
                ],
            ],
            "expected": "q=0 yields a vacant centre or a documented typed refusal; zero-weight occupied alternatives cannot be selected",
            "observed": "Synthetic q=0 water has alternative weight 0 but selection materializes 3 atoms",
            "next_action": "Respect centre occupation and zero weights, represent vacancy explicitly, reject impossible selections, and validate declared coupling before reporting a physical realization. Recompute actual selected-state symmetry separately from source average symmetry.",
        },
        "RT11": {
            "title": "Stabilizer matching uses independent atom image shifts",
            "priority": "P1",
            "owner": "product",
            "locations": [
                [
                    "molcrys_kit/operations/forward.py",
                    466,
                    "common anchor shift computed",
                ],
                [
                    "molcrys_kit/operations/forward.py",
                    478,
                    "pairwise atom differences wrapped independently",
                ],
            ],
            "expected": "An operation stabilizes the complete unwrapped molecule under ONE common lattice shift and a valid atom permutation",
            "observed": "C-O at x=0,1.2 in a 2.4-A cell is declared inversion-compatible although C fixes shift 0 and O would require another image; the independent whole-molecule comparator rejects it",
            "next_action": "After the common shift, match complete unwrapped Cartesian geometry without per-atom MIC. Keep chemically valid permutations and graph/image invariants; unwrap input templates coherently rather than rescuing a bad pose by atom-wise wrapping.",
        },
        "RT12": {
            "title": "Declared group is not checked against the complete operation set",
            "priority": "P1",
            "owner": "product",
            "locations": [
                [
                    "molcrys_kit/operations/forward.py",
                    116,
                    "_validate_group validates only the supplied set",
                ],
                [
                    "molcrys_kit/operations/forward.py",
                    328,
                    "context copies declared group metadata after closure check",
                ],
            ],
            "expected": "P -1 / IT 2 with only identity is rejected as contradictory",
            "observed": "prepare_construction_context accepts declared group 2 with 1 operation",
            "next_action": "Validate the complete operation set against number/symbol/Hall/setting/origin/basis in the same authority used by the strict CIF reader. Reject inconsistent declarations, while allowing explicitly custom groups without invented crystallographic labels.",
        },
    }
)

REGRESSIONS = {
    "RT01": [
        "Synthetic I-43m CH4: 2 molecules, 10 atoms, exactly 8 C-H edges; no cross-centre edges.",
        "HXMTAM10: 2 components of 22 atoms; keep legitimate special-position and boundary-crossing bonds.",
        "Do not use distance filtering as the general fix; published endpoint images remain authoritative.",
    ],
    "RT02": [
        "Explicit enantiopure general-position P -1 request fails closed; a deliberately allowed racemate succeeds.",
        "Allowed improper mates have correct known stereo under their typed graph; do not blanket-swap all descriptors.",
        "Rerun the 93 confirmed CSD R/S cases and separate unknown/assignment-error states.",
    ],
    "RT03": [
        "C=C, C=O, aromatic and coordination examples retain order/kind through template, graph, P1, ASU and reader.",
        "Unknown bond type remains unknown; legacy pair inputs remain accepted with documented semantics.",
        "Verify raw encoded typed records before judging a native CCDC reader that may leave imported bonds unassigned.",
    ],
    "RT04": [
        "Synthetic D2 retains isotope 2 and known mass; Na+/Cl- retain explicit +1/-1.",
        "Unknown charge is distinct from explicit neutral zero; do not treat CCDC default zero as a known encoded value.",
        "ASU/average-site reduction never merges different known isotopes or incompatible site identities.",
    ],
    "RT05": [
        "D and T parse as H/isotope 2 and 3; leading-mass syntax and raw tokens remain supported.",
        "AMEGEF03 source read succeeds without losing isotope metadata; invalid conflicting annotations fail explicitly.",
    ],
    "RT06": [
        "Statistical model write/read/write preserves complete alternative atoms, typed connectivity, mappings, weights and context.",
        "A selected restored microstate contains only its chosen complete alternatives.",
        "Old incomplete companions report loss explicitly rather than inventing graph/correlation data.",
    ],
    "RT07": [
        "Inversion-site water: true-ASU output expands to the same O=1, H=2 weighted composition and correct marginal sites.",
        "Centred-cell multiplicity and shared-site identity are verified independently; P1 averages are never expanded a second time.",
        "CIF represents marginals; companion preserves alternative/group action and correlations where CIF tags cannot.",
    ],
    "RT08": [
        "Vendor Error/Unknown are unavailable statuses, not valid CIP configurations; Mixed is a qualified state.",
        "Known R versus unassigned imported chemistry is missing evidence, not an observed R-to-S conversion.",
        "Input/oracle availability is aligned (NUYVOU); direct template matches must be inspected before attributing a flag to failed transfer. A different CIF-selected mate needs an explicit source-instance map.",
        "An output validator reads only what was encoded; it must not copy source truth or perceive replacement bond types to hide losses.",
    ],
    "RT09": [
        "Two non-symmetry-related states at q=1 with weights .25/.75 are both supplied, not converted into a .25 vacancy model.",
        "Shared site identity permits justified split/merge equivalence while distinct coincident entities remain distinct.",
        "Insufficient source coupling/input data produce an indeterminate input-model status.",
    ],
    "RT10": [
        "q=0 cannot materialize an occupied molecule; explicitly requesting a zero-weight state is rejected.",
        "q<1 vacancy selection and known coupling are checked; local marginals alone do not establish global feasibility.",
    ],
    "RT11": [
        "The synthetic C-O inversion example has stabilizer order 1 and is not ordered-compatible.",
        "Valid long boundary-crossing molecules still match after one coherent whole-molecule translation.",
        "Retain the independent-atom-wrap negative regression in the benchmark matcher.",
    ],
    "RT12": [
        "Declared P -1 with identity-only operations raises InvalidSymmetryError.",
        "Closed same-order but wrong groups/settings/origins also fail; legitimate basis transforms and custom unlabelled groups remain valid.",
    ],
}


def render_triage(folder, registry, catalogue_rows):
    lines = [
        "# Actionable round-trip triage",
        "",
        "This is the developer entry point for #198 / PR #199. Product source links are pinned to 9bf0ffb; benchmark links to the frozen v2 producer. No production code or frozen results were changed.",
        "",
        "## Files",
        "",
        "- `../../refcodes.txt`: all 1004 exact IDs, one per line.",
        "- `../../refcodes.csv`: the same IDs with frozen stratum/targeted flags.",
        "- `failure_catalogue.csv`: every non-pass stage, repair ID, confidence, expected/observed record, code entry point and next action.",
        "- `problem_refcodes/*.confirmed.txt` and `*.related.txt`: confirmed associations versus candidates; synthetic-only defects need no CSD refcode.",
        "- `reproduce_confirmed.py` / `synthetic_diagnostics.json`: executable licence-free probes and measured outcomes.",
        "- `identity_audit.json`: per-field local-cache audit; contains counts/transitions only.",
        "",
        "## Interpretation correction",
        "",
        "The 920 original `mismatch` outcomes are historical pipeline outputs, not 920 confirmed product defects. CCDC returns `Error` when chirality cannot be assigned and `Unknown` for unassigned bond types. The v2 adapter compared these as if they were known labels. In a synthetic C2 CIF, all tested bond-tag/token variants import as Unknown/Error in CCDC 3.7.2; merely renaming a tag is not a demonstrated repair.",
        "",
        "Of 108 raw in-memory identity flags, the audit confirms **93 CSD cases / 450 known R/S conflicts, all on improper operations**. NUYVOU instead has 4 input/API-availability differences: a direct original-template match exists, so this is not evidence of a failed symmetry-mate lookup. The remaining 14 cases show no valid R/S contradiction after separating vendor unavailable/mixed statuses. Geometry results and the frozen raw rows are unchanged; chemical-cause claims must use this triage.",
        "",
        "## Repair map",
        "",
        "| ID | Owner | Priority | Problem | Confirmed refs | Related refs |",
        "|---|---|---|---|---:|---:|",
    ]
    for issue in registry:
        lines.append(
            "| {issue_id} | {owner} | {priority} | {title} | {n} | {m} |".format(
                **issue,
                n=len(issue["confirmed_refcodes"]),
                m=len(issue["related_refcodes"]),
            )
        )
    lines += [
        "",
        "A related refcode is a reproducible queue item, not proof that the same root cause explains every case. RT08 and RT09 are benchmark issues; do not change production chemistry to make those probes pass.",
        "",
        "## Reproduce",
        "",
        "From the repository root, without CCDC:",
        "",
        "```powershell",
        "python -c \"import runpy; runpy.run_path('benchmarks/csd_roundtrip/reports/csd601-sep2026-baseline-v2/reproduce_confirmed.py', run_name='__main__')\"",
        "```",
        "",
        "Run the same script with `--ccdc-reader` in the licensed vendor interpreter for the optional synthetic reader probe. During repair, these probes can run on changed production code. The frozen 1004-case evaluator intentionally refuses a different target package; create a new versioned experiment for future full-corpus comparisons.",
        "",
    ]
    for issue in registry:
        ident = issue["issue_id"]
        lines += [
            "## " + ident + " — " + issue["title"],
            "",
            "**Owner / priority:** " + issue["owner"] + " / " + issue["priority"],
            "",
            "**Modification entry points:**",
            "",
        ]
        for path, line, function in issue["locations"]:
            url = (
                (HARNESS if path.startswith("benchmarks/") else PRODUCT)
                + path
                + "#L"
                + str(line)
            )
            lines.append("- [{}:{} — {}]({})".format(path, line, function, url))
        lines += [
            "",
            "**Expected:** " + issue["expected"],
            "",
            "**Observed / mechanism:** " + issue["observed"],
            "",
            "**Repair direction:** " + issue["next_action"],
            "",
            "**Regression acceptance:**",
            "",
        ]
        lines += ["- " + x for x in REGRESSIONS[ident]]
        lines += [
            "",
            "**Refcodes:** "
            + (
                ", ".join(issue["confirmed_refcodes"][:6])
                if issue["confirmed_refcodes"]
                else "Synthetic-only or no individually confirmed CSD association."
            )
            + " Full lists: `problem_refcodes/"
            + ident
            + ".confirmed.txt` and `.related.txt`.",
            "",
        ]
    lines += [
        "## Unresolved queues",
        "",
        "U_STAGE_FAILURE / U_TIMEOUT / U_DISORDER_INPUT keep their raw stage and code. They have not been promoted to confirmed product bugs. Use the licensed local inputs and run-specific diagnostics to minimize each case; keep successful controls such as ABEDUH and ANOSUS green.",
        "",
        "No CIF, coordinate, atom map or raw exception is included here. All named source entries remain retrievable through licensed CSD access.",
        "",
    ]
    (folder / "triage.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    folder = Path(__file__).resolve().parent
    manifest = read_json(ROOT / "manifest.json")
    rows = [
        json.loads(x)
        for x in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    refs = sorted(c["refcode"] for c in manifest["cases"])
    assert len(refs) == len(set(refs)) == 1004
    (ROOT / "refcodes.txt").write_text(
        "\n".join(refs) + "\n", encoding="utf-8", newline="\n"
    )
    with (ROOT / "refcodes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["refcode", "stratum", "targeted"])
        w.writeheader()
        w.writerows(
            (
                {k: c[k] for k in w.fieldnames}
                for c in sorted(manifest["cases"], key=lambda c: c["refcode"])
            )
        )
    audits = {
        r["refcode"]: r["audits"]
        for r in read_json(folder / "identity_audit.json")["records"]
    }
    rs_refs = {
        ref
        for ref, a in audits.items()
        if a.get("memory", {}).get("conflicting_fields", {}).get("stereo", 0) > 0
    }
    empty_refs = {
        ref
        for ref, a in audits.items()
        if "memory" in a and not a["memory"].get("fields")
    }
    output = []
    related = defaultdict(set)
    confirmed = defaultdict(set)
    for row in rows:
        ref = row["refcode"]
        for stage, value in row["stages"].items():
            if value["status"] == "pass":
                continue
            assignments = []
            code = value.get("code", "")
            if stage == "asu_mck_readback" and code in {
                "molecule_count",
                "whole_molecule_geometry_or_graph",
            }:
                assignments.append(
                    (
                        "RT01",
                        "confirmed"
                        if ref == "HXMTAM10"
                        else "candidate_same_failure_family",
                    )
                )
            if stage == "memory_chemistry" and "bond_order" in value.get(
                "missing_annotations", []
            ):
                assignments.append(("RT03", "confirmed_interface_gap"))
            if stage == "memory_chemistry" and ref in rs_refs:
                assignments.append(("RT02", "confirmed_known_RS_conflict"))
            if (
                stage == "memory_chemistry"
                and code == "atomic_chemical_identity_not_preserved"
                and ref not in rs_refs
            ):
                assignments.append(
                    (
                        "RT08",
                        "confirmed_benchmark_availability_issue"
                        if ref in empty_refs
                        else "source_annotation_transfer_needs_review",
                    )
                )
            if "isotope" in value.get("missing_annotations", []):
                assignments.append(("RT04", "confirmed_annotation_loss"))
            if stage == "disorder_model_readback":
                assignments.append(("RT06", "confirmed_capability_gap"))
            if stage == "asu_average":
                assignments.append(("RT07", "confirmed_unsupported_dispatch"))
            if stage in ("memory_average", "p1_average_ccdc", "p1_average_pymatgen"):
                assignments.append(("RT09", "input_or_representation_audit_required"))
            if stage in ("p1_ccdc_chemistry", "asu_ccdc_chemistry"):
                mode = stage.split("_")[0]
                a = audits.get(ref, {}).get(mode, {})
                if a.get("unavailable_stereo_sentinels", {}).get("candidate:Error", 0):
                    assignments.append(
                        ("RT08", "confirmed_independent_reader_unavailability")
                    )
            if not assignments:
                assignments = [
                    (
                        "U_TIMEOUT"
                        if value["status"] == "timeout"
                        else "U_DISORDER_INPUT"
                        if "average" in stage or "disorder" in stage
                        else "U_STAGE_FAILURE",
                        "unresolved_root_cause",
                    )
                ]
            for issue, confidence in assignments:
                related[issue].add(ref)
                if confidence.startswith("confirmed"):
                    confirmed[issue].add(ref)
                data = ISSUES.get(issue, {})
                loc = data.get("locations", [])
                urls = [
                    "{}{}#L{}".format(
                        HARNESS if path.startswith("benchmarks/") else PRODUCT,
                        path,
                        line,
                    )
                    for path, line, _ in loc
                ]
                output.append(
                    {
                        "refcode": ref,
                        "stratum": row["stratum"],
                        "stage": stage,
                        "raw_status": value["status"],
                        "raw_code": code,
                        "issue_id": issue,
                        "confidence": confidence,
                        "expected": data.get(
                            "expected",
                            "See recorded stage contract; root cause not yet established",
                        ),
                        "observed": json.dumps(
                            {
                                k: v
                                for k, v in value.items()
                                if k
                                in (
                                    "status",
                                    "code",
                                    "error_type",
                                    "n_molecules",
                                    "n_atoms",
                                    "missing_annotations",
                                )
                            },
                            sort_keys=True,
                        ),
                        "code_locations": ";".join(urls),
                        "next_action": data.get(
                            "next_action",
                            "Inspect the recorded stage and local licensed repro; do not classify every stage failure as a product defect.",
                        ),
                    }
                )
        if (
            row.get("diagnostics", {}).get("source_sanity_error") == "KeyError"
            and ref == "AMEGEF03"
        ):
            related["RT05"].add(ref)
            confirmed["RT05"].add(ref)
    related["RT04"].add("AMEGEF03")
    confirmed["RT04"].add("AMEGEF03")
    assert len(rs_refs) == 93
    with (folder / "failure_catalogue.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    lists = folder / "problem_refcodes"
    lists.mkdir(exist_ok=True)
    for issue in sorted(ISSUES):
        (lists / (issue + ".confirmed.txt")).write_text(
            "\n".join(sorted(confirmed[issue])) + ("\n" if confirmed[issue] else ""),
            encoding="utf-8",
            newline="\n",
        )
        (lists / (issue + ".related.txt")).write_text(
            "\n".join(sorted(related[issue])) + ("\n" if related[issue] else ""),
            encoding="utf-8",
            newline="\n",
        )
    registry = []
    for issue, data in ISSUES.items():
        registry.append(
            {
                "issue_id": issue,
                **data,
                "confirmed_refcodes": sorted(confirmed[issue]),
                "related_refcodes": sorted(related[issue]),
                "synthetic_probe": "synthetic_diagnostics.json#" + issue
                if issue != "RT09"
                else None,
            }
        )
    write_json(
        folder / "problem_index.json",
        {
            "target_commit": config()["target_commit"],
            "raw_evaluator_hash": read_json(folder / "analysis.json")["harness_sha256"],
            "frozen_result_rows_unchanged": True,
            "issues": registry,
            "catalogue_rows": len(output),
            "confirmed_known_RS_cases": len(rs_refs),
            "raw_memory_flags_without_valid_RS_conflict": sorted(empty_refs),
            "interpretation": "Original v2 scores are historical pipeline outputs, not counts of confirmed product bugs. See RT08 before interpreting chemical mismatch totals.",
        },
    )
    render_triage(folder, registry, len(output))
    print(
        {
            "refcodes": len(refs),
            "catalogue_rows": len(output),
            "confirmed_refcode_counts": {
                k: len(v) for k, v in sorted(confirmed.items())
            },
        }
    )


if __name__ == "__main__":
    main()
