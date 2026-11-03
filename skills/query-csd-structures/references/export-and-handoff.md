# Export and Handoff

Read this before writing CIF files or passing CSD structures to MolCrysKit.

## Bundled exporter

Run from the skill directory:

```bash
python scripts/demo_csd_disorder_workflow.py \
  --refcodes ABACIR ABABUB \
  --output-dir csd_export
```

Or provide one identifier per line:

```bash
python scripts/demo_csd_disorder_workflow.py \
  --refcodes-file refcodes.txt \
  --output-dir csd_export
```

The script writes:

```text
csd_export/
├── cifs/<REFCODE>.cif
└── retrieval_manifest.json
```

It exports only. It does not resolve disorder, add hydrogen, or analyze structures.

The exporter records `_space_group_IT_number` and the CCDC setting when the API
exposes them. CCDC 3.6/3.7 does not expose a Hall symbol through `Crystal`, so
the CIF contains an explicit `?` for `_space_group_name_Hall` instead of an
inferred setting. Ordered entries whose `disordered_molecule` is unavailable
fall back to `molecule`; null fractional-coordinate or bond handles are
skipped with warnings (BENZEN02 has riding H sites without coordinates in this
API).

## Public fixture sanitization

Before committing a CSD-derived regression fixture, rebuild it with the
repository helper so that CSD names, refcodes, audit/citation records,
refinement statistics, and machine paths are not copied into the CIF:

```bash
python scripts/sanitize_forward_fixtures.py \
  --input-dir csd_export/cifs \
  --output-dir tests/data/cif/forward
```

The manifest retains the source refcode, CSD/API release, checksums, and
operation/site counts. It also records any raw-to-fixture disorder-group ID
mapping. This mapping matters because CCDC may number an alternative group
from zero, while MolCrysKit reserves group `0` for ordered atoms. The
sanitizer maps such groups to positive IDs without changing occupancy or
assembly membership. The resulting CIFs remain subject to the applicable
CCDC licence; sanitization does not grant redistribution rights.

## Why not use a built-in CIF writer blindly

Some CCDC writer paths omit `_atom_site_occupancy`, `_atom_site_disorder_group`, or `_atom_site_disorder_assembly`. The bundled exporter reads `crystal.disordered_molecule`, extracts those fields explicitly, and records partial-occupancy statistics.

After export, inspect the CIF and manifest. For a disordered entry, require the expected occupancy/disorder tags and compare atom counts with the CCDC object.

## MolCrysKit handoff

```bash
mck io info csd_export/cifs/ABACIR.cif
mck io molecules csd_export/cifs/ABACIR.cif --json
mck analyze sanity-check csd_export/cifs/ABACIR.cif --json \
  -o csd_export/ABACIR.sanity.json
```

Resolve disorder only through the maintained MolCrysKit CLI/API:

```bash
mck operate disorder csd_export/cifs/ABACIR.cif \
  -o csd_export/ABACIR_ordered.cif --method optimal
```

This keeps retrieval and structural modeling as separate, auditable stages.

## Disorder integrity

CCDC group IDs may start at zero or be absent. The exporter numbers the
actual groups with positive IDs and preserves distinct assemblies. Ambiguous
or contradictory labels require review; ungrouped partial occupancies may
represent independent vacancies and are reported explicitly. Check the
manifest and atom-site rows before ordering.

Check `mck operate disorder --help` first; these options require a MolCrysKit version that supports them.

Use `mck operate disorder ... --strict-formula` only when the CIF has a
verified, parseable `_chemical_formula_moiety` and `_cell_formula_units_Z`.
A successful match does not prove protonation, charge or connectivity. A
missing or inaccurate formula requires a separately justified comparison.

When the source establishes that multiple multi-group assemblies are
independent, use `mck operate disorder ... --independent-assemblies`;
otherwise retain legacy behavior and review the source model. Some CIFs use
assembly labels for alternative fragments rather than separate choices, and
this switch is not an automatic chemical-identity check.

`mck operate disorder-supercell` accepts the same two flags and forwards
them into replica resolution. Commands that load with `--resolve-disorder`
(`io info`, `io molecules`, `io extract-molecule`, `io convert`,
`operate supercell`, `operate nanocluster`, `operate void`) do too, and
those flags require `--resolve-disorder`. Leaving them off keeps legacy
resolution. Exporter group numbers are local to each assembly.
