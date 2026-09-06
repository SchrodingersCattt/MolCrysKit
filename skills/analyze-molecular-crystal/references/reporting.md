# Reporting Limits

Answer the requested question first, then report only the assumptions needed to
interpret it.

| Result | Required context |
|---|---|
| Components | disorder state, bond scale, species IDs, counts |
| Readiness | failed checks and which downstream claims they affect |
| Interactions | H/protonation, criteria, identities; scores are not energies |
| BFDH | max index, symmetry, d-spacing, rank; not surface energy |
| Polyhedra | center, ligand, level, neighbors, cutoff, CN, shape evidence |
| Stoichiometry | cell counts and reduced ratio; solvent matching is heuristic |
| Volume/boundary | finite model, radii, probe, voxel/sampling parameters |
| Formal charge | value and source; independently check electroneutrality |
| Local environment | atom indexing and geometric heuristic |

For disorder ensembles, preserve replica IDs, method, seed, and coupling, and
report the distribution instead of choosing a favorable member.

Prefer JSON output and retain the input path, package version, exact command or
public-API call, thresholds, warnings, and raw result. A successful parser does
not prove chemical correctness.

When a conclusion depends on periodic images, coordination shells, close
contacts, or surface orientation, include a MatterVis figure. Do not run a
visualization merely to decorate a text-only answer.
