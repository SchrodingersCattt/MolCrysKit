# Chemical Edits

Use one matching command, then return to the verification step in `SKILL.md`.

## Disorder to ordered structures

```bash
mck operate disorder input.cif -o ordered.cif --method optimal
mck operate disorder input.cif -o ensemble.cif --method random --count 20 --seed 42
mck operate disorder input.cif -o states.cif --method enumerate --count 20
```

`optimal` gives one occupancy-favoured model, not a proven dominant state.
Use `random` for a reproducible ensemble and `enumerate` only for a bounded
state space. Symmetry-expanded copies choose independently by default; add
`--coupled` only when the intended model requires linked choices.

## Add missing hydrogen

```bash
mck operate add-h ordered.cif -o hydrogenated.cif
mck operate add-h ordered.cif -o hydrogenated.cif \
  --target-elements N --target-elements O --optimize-torsion
```

`--target-elements` is a whitelist. Formula-moiety metadata can constrain the
expected count but does not establish protonation. Compare the actual H count,
formula, X-H distances, clashes, and component charge before accepting.

## Remove solvent or guests

List component IDs once:

```bash
mck io molecules input.cif --json
mck operate desolvate input.cif -o dry.cif \
  --targets H2O_1 --targets C2H6O_1
```

Use the saved species IDs, not a guessed formula substring. Confirm that only
the selected complete components were removed and reassess composition, charge,
coordination, and the need for relaxation.

## Create a molecular vacancy

```bash
mck operate vacancy input.cif -o vacancy.cif \
  --species C2H6O_1 1 --seed-index 0 --random-seed 42
```

Record whether the removed object represents a neutral molecule, ion, correlated
cluster, or occupancy model. Preserve the seed and resulting periodic separation.
