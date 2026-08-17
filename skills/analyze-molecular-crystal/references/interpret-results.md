# Interpret and Report Analysis Results

Read this before drawing conclusions or delivering reports.

## Structural sanity

Run all seven checks first: hard clashes, intermolecular clashes, isolated atoms, hydrogen presence, topology preservation, formula consistency, and bond distances. Do not suppress a failure solely to obtain a pass.

## Interpretation limits

| Analysis | Required reporting and limits |
|---|---|
| Molecule inventory | bond scale, disorder state, formula, topology-aware species ID, counts |
| Weak interactions | hydrogen/protonation state, criteria, atom/molecule identities; geometry scores are not energies |
| BFDH | `max_index`, symmetry handling, `d_hkl`, rank; not surface energy or growth kinetics |
| Polyhedra/CShM | center, ligand, level, cutoff, neighbors, CN, reference shape, CShM; CN alone does not define shape |
| Stoichiometry | unit-cell counts and reduced ratio; topology assumptions; solvent matches are heuristic |
| Volume | radii type, overlap correction, voxel size; atomic-sphere volume is not cell volume |
| Accessible boundary | finite/non-periodic model, probe radius, radii, sampling density; converge quantitative comparisons |
| Formal charge | value and source; verify auto-guesses and cell electroneutrality |
| Chemical environment | molecule-local indices and geometric heuristic used; inspect unusual motifs |

## Disorder and ensembles

For ordered ensembles, report the distribution or range of results rather than selecting a favorable member. Preserve replica IDs, generation method, seed, and coupling assumption.

## Machine-readable delivery

Prefer `--json` and retain the exact input path, package version, command/options, thresholds, and warnings. If Python result objects are transformed into tables, preserve the raw results or a reproducible serialization step.

## Human review

Visualize structures when a result depends on periodic images, ring assignment, close contacts, coordination shells, or a surface orientation. A numerically ranked output is not sufficient chemical validation.
