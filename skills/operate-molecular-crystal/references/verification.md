# Verify the Written Structure

Run after an operation, not as a repeated preflight:

```bash
mck analyze summary output.cif --json
mck analyze sanity-check output.cif --json -o output.sanity.json
```

Also run `mck io molecules output.cif --json` when component identity or count
is part of the operation.

Check the invariant that matches the edit:

| Edit | Required evidence |
|---|---|
| Disorder | replica count, full occupancy where required, method/seed/coupling |
| Add H | actual formula/H count, protonation, X-H geometry, no new clashes |
| Slab | Miller orientation, intact units, thickness, vacuum, termination, charge |
| Cluster | seed, shells/cutoff, caps, sidecar |
| Supercell | vectors and atom/component counts scale correctly |
| Vacancy/desolvation | only selected components removed; resulting charge known |
| Reorientation | direction, handedness, cell, and PBC |
| Interpolation | endpoints, atom order, mapping, and every intermediate frame |
| Conversion | cell, PBC, coordinates, species, and frame count survive |

Visual inspection is required when geometry changed. Use the atomistic
visualization skill rather than custom plotting.

Deliver the original input, exact command, output, version, parameters,
assumptions, warnings, summary/sanity results, and sidecars. Report observed
composition and counts from the output; never copy an intended formula into the
report without checking it.
