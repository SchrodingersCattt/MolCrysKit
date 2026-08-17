# Verify Results and Deliverables

Read this before accepting any generated structure or reporting an analysis.

## Machine checks

Run the complete sanity suite on each output:

```bash
mck analyze sanity-check output.cif --json -o output.sanity.json
mck io info output.cif
mck io molecules output.cif --json
```

The seven sanity checks cover:

1. hard clashes;
2. intermolecular clashes;
3. isolated atoms;
4. hydrogen presence;
5. topology preservation;
6. formula consistency;
7. bond distances.

A warning can be acceptable only with a documented physical reason. Do not suppress a check solely to obtain a passing report.

## Operation-specific invariants

| Operation | Verify |
|---|---|
| Disorder resolution | full occupancy where required; no incompatible alternatives; expected replica count; seed/coupling recorded |
| Hydrogen completion | intended total H count and protonation; sensible X–H distances; no new clashes; hydrogen-bond network inspected |
| Slab | Miller orientation; intact molecules; thickness and vacuum; both surfaces; termination identity; charge/dipole suitability |
| QM cluster | seed retained; intended shells/cutoff; cap placement; no unintended metal caps; atom cap respected; sidecar delivered |
| Supercell | cell vectors scaled correctly; molecule and atom counts multiply by `A×B×C` |
| Vacancy/desolvation | only selected species removed; resulting composition and net charge understood |
| Reorientation | requested direction aligns with target Cartesian axis; handedness and periodicity preserved |
| Interpolation | endpoints match request; constant atom ordering/count; no unphysical midpoint clashes or molecule flips |
| Molecule manipulation | only selected molecule changed; replacement alignment and clash threshold satisfied |
| Format conversion | cell, PBC, coordinates, species, and molecule membership survive where the target format supports them |

## Analysis-specific checks

- **BFDH:** report `max_index`; describe results as empirical morphology candidates, not surface energies.
- **Interactions:** record the analyzed hydrogenation/protonation state; missing H changes donor detection.
- **Polyhedra/CShM:** record center, ligand, level, cutoff, coordination number, and reference shape.
- **Volume:** report radii type, overlap correction, voxel size, probe radius, and sampling density as applicable.
- **Formal charge:** report whether each value came from `user_map`, `auto_guess`, or fallback.

## Human inspection

Visualize the input and output when geometry has changed. Inspect periodic boundaries, whole-molecule reconstruction, atom labels, surface terminations, caps, vacancies, and close contacts. A successful write or a zero exit code does not establish chemical correctness.

## Delivery record

Provide:

- requested operation or analysis;
- exact input and output paths;
- exact parameters and software version;
- assumptions and warnings;
- verification report;
- requested versus effective result when a fallback occurred.
