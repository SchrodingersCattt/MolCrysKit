# MolCrysKit Architecture

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
The process begins with [scan_cif_disorder](../molcrys_kit/analysis/disorder/process.py) which parses CIF files to extract disorder information. This phase identifies atoms belonging to different disorder groups (PART numbers), their occupancies, and assembly information. The extracted data is stored in a [DisorderInfo](../molcrys_kit/analysis/disorder/info.py) object containing:
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
- Optionally applies randomized weights to generate ensembles of physically realistic structures
- Reconstructs complete molecular crystals from the selected atom sets

The MWIS solution represents a compromise between maximizing total occupancy (thermodynamic stability) and minimizing steric clashes (geometric feasibility), resulting in physically realistic ordered structures from disordered crystal data.