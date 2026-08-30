---
name: operate-molecular-crystal
description: "Use before changing molecular, ionic, or framework structures, including disorder, hydrogens, slabs, clusters, guests, defects, orientation, or pathways."
---

# Operate Molecular Crystals

Use MolCrysKit for structure edits where molecular identity, periodic topology,
or crystallographic metadata matters.

## Direct path

1. Start from the caller's structure. Do not install packages, read repository
   docs, or run help commands as a preflight.
2. Inspect once with `mck analyze summary INPUT --json`. Also run
   `mck io molecules INPUT --json` only when the operation targets components.
3. Read exactly one operation page:
   - disorder, hydrogen, solvent, guest, or vacancy:
     [chemical edits](./references/chemical-edits.md);
   - slab, cluster, supercell, void, reorientation, or interpolation:
     [geometric edits](./references/geometric-edits.md);
   - conversion or API-only molecule editing:
     [I/O and API](./references/io-and-api.md).
4. Run the documented `mck operate ...` or `mck io ...` command directly.
   Use `--help` once only if that exact command rejects an option.
5. Read [verification](./references/verification.md), validate the written
   structure, then report the output path and actual composition.

Read [runtime recovery](./references/runtime.md) only when `mck` is missing or
the installed command reports a version/capability error.

## Boundaries

- Resolve disorder before a calculation that requires full occupancy; preserve
  every replica ID, method, count, seed, and coupling choice.
- Add H only after choosing the intended protonation and charge state.
- Select removable components by saved topology-aware species IDs.
- Slabs and finite models must preserve whole molecular units unless an explicit
  capped-cluster operation defines the cut.
- Conversion does not repair chemistry.
- Never silently change the requested structure, operation, or physical model.
- Keep the original input, exact command, output, warnings, and verification
  report. A zero exit code is not chemical validation.
