# MolCrysKit Architecture

## Symmetry-path architecture

Crystallographic path construction is split into four layers:

1. `io.cif.read_cif_symmetry` parses explicit or declared space-group
	operations without expanding atoms.
2. `structures.symmetry` owns fractional affine algebra, lattice-basis changes,
	subgroup validation, and coset/domain enumeration.
3. `operations.symmetry_path.build_symmetry_path_plan` performs global molecule
	assignment, atom graph mapping, periodic-image selection, and strict proper
	rigid-reachability checks.
4. `interpolate_symmetry_path` generates only proper rigid interior motions and
	copies exact endpoints.

This separation is deliberate.  A determinant-minus-one crystal operation is a
valid endpoint generator, but it is not itself a continuous molecular rotation.
It is accepted only when an allowed equivalent-atom permutation makes every
selected molecule superposable by an element of SO(3).  Otherwise the strict
planner fails and a separate internal-coordinate path model is required.

All numerical defaults are centralized in
`molcrys_kit.constants.symmetry_path`; no operation or structure module should
embed path tolerances.
The planner also applies a centralized minimum-displacement threshold and rejects
operations that reduce to a pure permutation of the full-cell molecule list.

## Core Philosophy

MolCrysKit is built around the concept of extending the Atomic Simulation Environment (ASE) with graph-based molecular representations. The central class, [CrystalMolecule](../molcrys_kit/structures/molecule.py), inherits from ASE [Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms) but adds a NetworkX graph for connectivity. This dual representation allows for both standard ASE operations and sophisticated graph-based molecular analysis.

The architecture enables:
- Full compatibility with ASE tools and workflows
- NetworkX graph algorithms for molecular analysis
- Efficient identification of molecular components within crystal structures
- Flexible representation of chemical connectivity with customizable bonding thresholds

## Disorder Solver (The "Referee" Logic)

The disorder handling pipeline in MolCrysKit is structured as a three-phase process that transforms raw disorder information into physically realistic ordered structures:

### Phase 1: Raw Data Extraction
The process begins with [scan_cif_disorder](../molcrys_kit/io/cif.py) which parses CIF files to extract disorder information. This phase identifies atoms belonging to different disorder groups (PART numbers), their occupancies, and assembly information. The extracted data is stored in a [DisorderInfo](../molcrys_kit/analysis/disorder/info.py) object (defined in [io/cif.py](../molcrys_kit/io/cif.py), re-exported from disorder/info.py) containing:
- Atomic symbols and labels
- Fractional coordinates
- Occupancy values
- Disorder group assignments
- Assembly identifiers

### Phase 2: Building the Exclusion Graph
The [DisorderGraphBuilder](../molcrys_kit/analysis/disorder/graph.py) constructs a conflict graph where atoms that cannot coexist in the same physical structure are connected by edges. This phase implements sophisticated conflict detection mechanisms:

- **Conformer conflicts**: Detects logical alternatives that cannot occupy the same space
- **Explicit conflicts**: Identifies atoms with identical assembly IDs or close proximity
- **Geometric conflicts**: Flags atoms that are too close to physically coexist
- **Valence conflicts**: Resolves chemically unrealistic coordination environments

The graph construction process uses precomputed distance matrices with Periodic Boundary Conditions (PBC) to efficiently evaluate all interatomic relationships.

### Phase 3: Solving for the Maximum Weight Independent Set (MWIS)
The [DisorderSolver](../molcrys_kit/analysis/disorder/solver.py) implements the final phase by solving the Maximum Weight Independent Set problem on the exclusion graph. The solver:

- Groups atoms into rigid bodies based on disorder group and assembly information
- Implements a greedy algorithm to select groups with high occupancy weights and low conflict degrees
- Samples PART/SP alternatives by occupancy with `method="random"` to generate reproducible ensembles when a seed is provided
- Enumerates Cartesian products of independent PART/SP alternatives with `method="enumerate"`
- Reconstructs complete molecular crystals from the selected atom sets

### Symmetry-Copy Decoupling
`generate_ordered_replicas_from_disordered_sites(..., coupled=False)` is the
default enumeration contract.  Explicit PART/assembly conflicts are scoped to
the same symmetry operation when symmetry provenance is available, so expanded
copies of one asymmetric-unit disorder model make independent decisions.  For
example, a two-copy PART 1/2 site enumerates `AA`, `AB`, `BA`, and `BB` rather
than only `AA` and `BB`.

The same flag applies to implicit special-position motifs.  In decoupled mode,
isolated X(H)n centres such as NH4+ expose multiple locally valid H
orientations as competing rigid groups, bounded by
`_MAX_MOTIF_ORIENTATIONS_PER_CENTER`.  Passing `coupled=True` restores the
legacy behaviour where symmetry copies are locked together and motif merge keeps
only the greedy best orientation.

The MWIS solution represents a compromise between maximizing total occupancy (thermodynamic stability) and minimizing steric clashes (geometric feasibility), resulting in physically realistic ordered structures from disordered crystal data.