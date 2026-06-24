# MolCrysKit API Reference

This document serves as a placeholder for the MolCrysKit API reference. The API reference provides detailed information about the classes, functions, and methods available in the MolCrysKit package.

## Modules

### mck.structures
- [CrystalMolecule](../molcrys_kit/structures/molecule.py): Molecular representation with graph connectivity
- [MolecularCrystal](../molcrys_kit/structures/crystal.py): Crystal structure composed of molecules; CIF readers may attach a raw `formula_moiety` string from `_chemical_formula_moiety`
- [Atom](../molcrys_kit/structures/atom.py): Individual atom representation

### mck.operations
- [add_hydrogens](../molcrys_kit/operations/hydrogen_completion.py): Function to add hydrogen atoms using heuristic placement, with optional `_chemical_formula_moiety` H-count correction (`use_formula_moiety=True` by default)
- [HydrogenCompleter](../molcrys_kit/operations/hydrogen_completion.py): Class for adding hydrogen atoms to molecular crystals using the same heuristic/moiety workflow
- [generate_topological_slab](../molcrys_kit/operations/surface.py): Function to generate surface slabs preserving molecular topology
- [TopologicalSlabGenerator](../molcrys_kit/operations/surface.py): Class for generating surface slabs while preserving molecular topology
- [TerminationInfo](../molcrys_kit/operations/surface.py): Dataclass holding metadata for a single surface termination (Miller index, shift, Tasker type, layer charges, dipole, charge source)
- [enumerate_terminations](../molcrys_kit/operations/surface.py): Enumerate topologically unique surface terminations for a Miller plane, with Tasker classification
- [generate_slabs_with_terminations](../molcrys_kit/operations/surface.py): Build slabs for selected terminations with configurable Tasker-preferred-first ordering
- [generate_vacancy](../molcrys_kit/operations/defects.py): Function to generate vacancies by removing molecular clusters

### mck.analysis
- [BFDHFacetInfo](../molcrys_kit/analysis/bfdh.py): Dataclass describing one Bravais-Friedel-Donnay-Harker facet candidate, including Miller index, interplanar spacing, normalized growth-rate proxy, normalized morphological-importance proxy, and optional symmetry equivalents
- [enumerate_bfdh_facets](../molcrys_kit/analysis/bfdh.py): Enumerate model-agnostic BFDH facet candidates from a molecular crystal, pymatgen lattice/structure, or 3x3 lattice matrix; by default searches low-index Miller indices up to `max_index=2` and applies Donnay-Harker-style systematic-absence filtering when structure symmetry is available
- [enumerate_low_index_millers](../molcrys_kit/analysis/bfdh.py): Deterministic fallback low-index Miller enumerator used when pymatgen symmetry enumeration is disabled or unavailable
- [DisorderSolver](../molcrys_kit/analysis/disorder/solver.py): Class for solving disorder problems using graph algorithms
- [generate_ordered_replicas_from_disordered_sites](../molcrys_kit/analysis/disorder/process.py): Resolve disordered CIFs with `method="optimal"` (single greedy MWIS), `method="random"` (occupancy-weighted PART/SP sampling, optionally seeded), or `method="enumerate"` (deterministic Cartesian enumeration of independent alternatives).  By default `coupled=False` lets symmetry-equivalent copies choose PART/SP orientations independently (e.g. a two-copy PART 1/2 site can enumerate `AA`, `AB`, `BA`, `BB`); pass `coupled=True` to preserve the legacy all-copies-locked behaviour.
- [DisorderGraphBuilder](../molcrys_kit/analysis/disorder/graph.py): Class for building exclusion graphs from disorder data
- [ChemicalEnvironment](../molcrys_kit/analysis/chemical_env.py): Class for analyzing chemical environments in molecular crystals
- [Fragment](../molcrys_kit/analysis/formula_moiety.py): Dataclass representing one parsed `_chemical_formula_moiety` fragment
- [parse_moiety_string](../molcrys_kit/analysis/formula_moiety.py): Parse CIF `_chemical_formula_moiety` values into fragments
- [match_molecule_to_fragment](../molcrys_kit/analysis/formula_moiety.py): Match a molecule to a unique formula-moiety fragment by heavy-atom signature
- [heavy_signature](../molcrys_kit/analysis/formula_moiety.py): Build a hydrogen-free composition signature for matching
- [MolChargeResult](../molcrys_kit/analysis/charge.py): Dataclass holding formal charge assignment result for one molecule topology type
- [assign_mol_formal_charges](../molcrys_kit/analysis/charge.py): Assign formal charges to all distinct molecule topologies in a crystal using hybrid strategy (user map → pymatgen BVAnalyzer → zero fallback)
- [compute_topo_signature](../molcrys_kit/analysis/charge.py): Compute a unique topology fingerprint for a CrystalMolecule combining formula and bond-graph degree sequence
- [find_polyhedra](../molcrys_kit/analysis/packing_shell.py): Enumerate A--B coordination polyhedra in a periodic structure. Two levels of A/B identity are supported through the `level` parameter: `level="atom"` (default, backward compatible) matches by chemical symbol on a flat ASE `Atoms` and is appropriate for atomic ionic crystals (Pb--I, Cs--Cl, etc.); `level="molecule"` requires a `MolecularCrystal` and matches by single-fragment moiety strings (e.g. `"N H4"`, `"Cl O4"`, `"C2 H10 N2"`) against each molecule's heavy-atom signature, then runs the same gap+enclosure CN selection on molecule centroids. Centroid choice is configurable via `center_kind` (`"centroid"`, `"com"`, or `"heavy_centroid"`). Use `level="molecule"` for hybrid molecular crystals such as ABX3 / ABX4 / A2BX5 hybrid perovskites where the A site is an organic cation and the B site is a polyatomic anion. The three radial kwargs are wired per level: at atom level `cutoff` is the historical hard radial cap and `hard_cutoff` is rejected; at molecule level `cutoff` (synonym: `search_cutoff`, mutually exclusive) is the candidate search radius feeding gap+enclosure, while `hard_cutoff` is the explicit opt-in for the historical "fill the ball" semantics. If `hard_cutoff` is larger than the search radius, the search radius is auto-bumped to `hard_cutoff` so the requested hard sphere is not silently truncated. Records always expose `search_cutoff`, `hard_cutoff` (`None` when not used), and `cutoff` (echo of what `detect_coordination_number` received).
- [detect_coordination_number](../molcrys_kit/analysis/packing_shell.py): Choose a coordination number from sorted neighbour distances using the gap+enclosure heuristic; shared by both `find_polyhedra` levels.
- [ScoringParams](../molcrys_kit/analysis/interactions/scoring.py): Continuous weak-interaction scoring defaults. Distances use Lorentzian kernels for long tails; angles use Gaussian kernels for directional selectivity. Hydrogen-bond and halogen-bond distances are normalized by vdW-radius sums from `constants/vdw_radii.json`; hydrogen bonds use a shorter normalized optimum than halogen bonds because H···A contacts commonly sit well inside the vdW-radius sum.
- [interaction_profile](../molcrys_kit/analysis/interactions/profile.py): Run all weak-interaction detectors and summarize each interaction kind as `{count, max, mean, sum}` while retaining raw scored interaction records. Raw H···H records keep their legacy `kind="h_h_contact"`, but profile summaries aggregate them under `close_contact` because their score describes geometric close-contact character rather than a uniformly attractive interaction.

### mck.io
- [read_mol_crystal](../molcrys_kit/io/cif.py): Function to read molecular crystals from CIF files, including `_chemical_formula_moiety` metadata when present
- [write_cif](../molcrys_kit/io/output.py): Function to write crystal structures to CIF files
- [identify_molecules](../molcrys_kit/io/cif.py): Function to identify individual molecules in a crystal structure

## Detailed API Documentation

For detailed API documentation, please refer to the docstrings in the source code files.