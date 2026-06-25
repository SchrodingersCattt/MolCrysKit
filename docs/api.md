# MolCrysKit API (Quick Reference & Recipes)

> **For AI agents** — this is your primary entry point.
> Read "At a Glance" for what the library can do, then jump to the recipe you need.
> For detailed parameter docs, read the source docstrings.

## Core Concepts

- **`MolecularCrystal`** is the central type. Nearly every operation takes one as input and returns one as output.
- **Data flow**: `CIF → read_mol_crystal() → MolecularCrystal → operation → write_cif / write_poscar / write_xyz`
- **ASE interop**: `MolecularCrystal.from_ase(atoms)` / `crystal.to_ase()`
- **Import convention**: `import molcrys_kit as mck`
- **Non-mutating**: All operations return a **new** object; the original crystal is never modified.

## At a Glance

| Capability | Entry Point | Input → Output |
|---|---|---|
| Parse CIF | `mck.read_mol_crystal(path)` | CIF → `MolecularCrystal` |
| Resolve disorder | `generate_ordered_replicas_from_disordered_sites(mc, method)` | MC → `list[MolecularCrystal]` |
| Add hydrogens | `add_hydrogens(mc)` | MC → MC |
| Generate surface slab | `generate_topological_slab(mc, miller, layers, vacuum)` | MC → MC |
| BFDH facet ranking | `enumerate_bfdh_facets(mc, top_n=5)` | MC → `list[BFDHFacetInfo]` |
| Enumerate terminations | `enumerate_terminations(mc, miller_index)` | MC → `list[TerminationInfo]` |
| Build slabs by termination | `generate_slabs_with_terminations(mc, miller_index)` | MC → `list[(MC, TerminationInfo)]` |
| Carve QM cluster | `ClusterCarver(mc).carve(seeds, bond_shells=N)` | MC → `CrystalCluster` |
| Create vacancy | `generate_vacancy(mc, mol_idx)` | MC → MC |
| Remove solvents | `Desolvator(mc).remove_solvents(...)` | MC → MC |
| Translate molecule | `translate_molecule(mc, mol_index, vector)` | MC → MC |
| Rotate molecule | `rotate_molecule(mc, mol_index, axis, angle)` | MC → MC |
| Replace molecule | `replace_molecule(mc, mol_index, new_molecule)` | MC → MC |
| Write CIF | `write_cif(mc, path)` | MC → file |
| Write ExtXYZ bundle | `write_extxyz(crystals, path)` | `list[MC]` → file |
| Find H-bonds | `find_hydrogen_bonds(mc)` | MC → `list[HydrogenBond]` |
| Find π-stacking | `find_pi_stacking(mc)` | MC → `list[PiStacking]` |
| Find halogen bonds | `find_halogen_bonds(mc)` | MC → `list[HalogenBond]` |
| Coordination polyhedra | `find_polyhedra(mc, A, B)` | MC → `list[PolyhedronRecord]` |

## Recipes

### 1. CIF → Molecule Identification

```python
import molcrys_kit as mck

crystal = mck.read_mol_crystal("structure.cif")

print(f"Molecules: {len(crystal.molecules)}")
print(f"Atoms: {crystal.get_total_nodes()}")
for i, mol in enumerate(crystal.molecules):
    print(f"  Mol {i}: {mol.get_chemical_formula()}, "
          f"atoms={len(mol)}, COM={mol.get_center_of_mass()}")
```

See [tutorials.md](tutorials.md) for extended extXYZ workflows.

### 2. Disorder Resolution Pipeline

```python
from molcrys_kit.analysis.disorder import generate_ordered_replicas_from_disordered_sites

# method: "optimal" (default, single greedy MWIS),
#         "random" (occupancy-weighted sampling),
#         "enumerate" (Cartesian product of independent choices)
replicas = generate_ordered_replicas_from_disordered_sites(
    crystal, method="optimal", coupled=False
)
for r in replicas:
    print(f"Replica: {len(r.molecules)} molecules, "
          f"atoms={r.get_total_nodes()}")
```

See [Architecture](architecture.md) for the full three-phase pipeline and solver modes.

### 3. Hydrogen Completion

```python
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.io.output import write_cif

crystal = mck.read_mol_crystal("bulk.cif")

# Default: heuristic placement + CIF _chemical_formula_moiety H-count correction
h_crystal = add_hydrogens(crystal)
print(f"Before: {crystal.get_total_nodes()} atoms, After: {h_crystal.get_total_nodes()} atoms")

# Custom rules for specific elements
h_crystal = add_hydrogens(
    crystal,
    target_elements=["O", "N", "C"],
    rules=[
        {"symbol": "N", "geometry": "trigonal_pyramidal", "target_coordination": 3},
        {"symbol": "O", "geometry": "bent", "target_coordination": 2},
    ],
    bond_lengths={"O-H": 0.96, "N-H": 1.01, "C-H": 1.09},
    use_formula_moiety=True,
)
write_cif(h_crystal, "hydrogenated.cif")
```

See [tutorials.md](tutorials.md) for geometry types, moiety correction details, and best practices.

### 4. Surface Slab Pipeline (BFDH → Termination → Slab)

```python
from molcrys_kit.analysis import enumerate_bfdh_facets
from molcrys_kit.operations import enumerate_terminations, generate_slabs_with_terminations
from molcrys_kit.io.output import write_cif

crystal = mck.read_mol_crystal("bulk.cif")

# Step 1: Rank BFDH facet candidates
facets = enumerate_bfdh_facets(crystal, top_n=5)
best = facets[0]
print(f"Best facet: {best.miller_index}, d={best.d_hkl:.3f} Å")

# Step 2: Enumerate unique terminations for the chosen Miller plane
term_infos = enumerate_terminations(crystal, miller_index=best.miller_index)
for ti in term_infos:
    print(f"[{ti.termination_index}] shift={ti.shift:.4f}  "
          f"type={ti.tasker_type}  preferred={ti.is_tasker_preferred}")

# Step 3: Build slabs (default returns Tasker-preferred terminations)
results = generate_slabs_with_terminations(
    crystal,
    miller_index=best.miller_index,
    min_slab_size=12.0,
    min_vacuum_size=15.0,
)
for slab, info in results:
    write_cif(slab, f"slab_term{info.termination_index}.cif",
              metadata={"termination_info": info})
```

For charge maps in salt-type crystals, see [tutorials.md](tutorials.md) Example 2.

### 5. QM Cluster Carving

```python
from molcrys_kit.operations import ClusterCarver

carver = ClusterCarver(crystal)

# bond_shells mode (production): topology-preserving BFS from seed atoms
cluster = carver.carve(
    seeds=[42],               # parent atom index (global)
    bond_shells=3,            # BFS shells from seed
    freeze_shell=1,           # 0=none, 1=cap+H, 2=cap+H+1layer
    max_atoms=500,            # hard safety cap
)

# rcut mode (diagnostic): radial cutoff in Å
cluster = carver.carve(seeds=[42], rcut=6.0)

# Output: CrystalCluster (non-periodic CrystalMolecule subclass) + JSON sidecar
cluster.write_xyz("cluster.xyz")
print(f"Cluster atoms: {len(cluster)}, frozen: {len(cluster.frozen_atoms)}")
print(f"Provenance: {cluster.provenance.convention_reference}")
```

See [tutorials.md](tutorials.md) for manual C-C bond cutting, `seed_merge_radius`, and freeze conventions.

### 6. Molecule Replacement with Clash Detection

```python
from molcrys_kit.operations.molecule_manipulation import (
    replace_molecule, MoleculeClashError,
)

crystal = mck.read_mol_crystal("host.cif")

try:
    new_crystal = replace_molecule(
        crystal,
        molecule_index=0,
        new_molecule="guest.xyz",   # or pass a CrystalMolecule
        clash_threshold=1.0,        # min acceptable distance (Å)
        max_rotation_attempts=100,
    )
except MoleculeClashError as e:
    print(f"Could not place molecule: {e}")
```

Also available: `translate_molecule(mc, idx, vector, fractional=False)` and `rotate_molecule(mc, idx, axis, angle, center="com")`.

### 7. Intermolecular Interaction Analysis

```python
from molcrys_kit.analysis import find_hydrogen_bonds, find_pi_stacking, find_halogen_bonds
from molcrys_kit.analysis.interactions import find_h_h_contacts, find_ch_pi_interactions

crystal = mck.read_mol_crystal("bulk.cif")

# Each finder returns a list of interaction records with geometry metadata
h_bonds = find_hydrogen_bonds(crystal)
pi_stacks = find_pi_stacking(crystal)
halogen = find_halogen_bonds(crystal)
hh = find_h_h_contacts(crystal)
ch_pi = find_ch_pi_interactions(crystal)

for hb in h_bonds:
    print(f"H-bond: {hb.donor} → {hb.acceptor}, "
          f"distance={hb.HA_distance:.2f} Å, angle={hb.angle_DHA:.1f}°")
for ps in pi_stacks:
    print(f"π-stack: {ps.ring_a} ↔ {ps.ring_b}, "
          f"centroid_dist={ps.centroid_distance:.2f} Å, type={ps.subtype}")
```

### 8. ExtXYZ Dataset Bundle for MLIP Training

```python
from molcrys_kit.io.extxyz import write_extxyz, read_extxyz

crystals = [mck.read_mol_crystal(f"{refcode}.cif") for refcode in refcodes]

frame_info = [
    {"refcode": refcode, "source": "CSD", "frame_index": i}
    for i, refcode in enumerate(refcodes)
]

write_extxyz(crystals, "dataset.extxyz", info=frame_info)

# Read back: use index=":" for dataset bundles
bundle = read_extxyz("dataset.extxyz", index=":")
print(len(bundle), bundle[0].metadata["refcode"])
```

## Module Index

### `mck.structures` — Core Types

| Symbol | Source | Description |
|---|---|---|
| `MolAtom` | `structures/atom.py` | Individual atom with symbol, position, occupancy |
| `CrystalMolecule` | `structures/molecule.py` | Molecule with graph connectivity |
| `MolecularCrystal` | `structures/crystal.py` | Crystal composed of molecules |
| `CrystalCluster` | `structures/cluster.py` | Non-periodic cluster (subclass of `CrystalMolecule`) |
| `ClusterProvenance` | `structures/cluster.py` | Provenance metadata for carved clusters |
| `CrystalTrajectory` | `structures/trajectory.py` | Trajectory of `MolecularCrystal` frames |
| `Molecule` | — | Backward-compat alias for `CrystalMolecule` |
| `all_ideal_polyhedra` | `structures/polyhedra.py` | Reference data for all ideal polyhedra |
| `ideal_polyhedra_for_cn` | `structures/polyhedra.py` | Ideal polyhedra keyed by coordination number |
| `convex_hull_payload` | `structures/polyhedra.py` | Convex hull payloads for shape measures |

### `mck.io` — Input / Output

| Symbol | Source | Description |
|---|---|---|
| `read_mol_crystal` | `io/cif.py` | Parse CIF into `MolecularCrystal` with molecule identification |
| `parse_cif_advanced` | `io/cif.py` | Advanced CIF parsing with disorder metadata |
| `identify_molecule_indices` | `io/cif.py` | Identify molecule membership indices in a crystal |
| `write_cif` | `io/output.py` | Write `MolecularCrystal` to CIF |
| `write_cif_sequence` | `io/output.py` | Write multiple crystals as a CIF sequence |
| `write_poscar` | `io/output.py` | Write `MolecularCrystal` to POSCAR (VASP format) |
| `write_poscar_sequence` | `io/output.py` | Write multiple crystals as POSCAR sequence |
| `write_xyz` | `io/output.py` | Write `MolecularCrystal` to XYZ |
| `write_xyz_with_freeze` | `io/output.py` | Write XYZ with freeze flags for QM input |
| `write_trajectory` | `io/output.py` | Write `CrystalTrajectory` to file |
| `read_xyz` | `io/xyz.py` | Read XYZ into `CrystalMolecule` |
| `read_poscar` | `io/poscar.py` | Read POSCAR into `MolecularCrystal` |
| `read_extxyz` | `io/extxyz.py` | Read ASE Extended XYZ frames |
| `write_extxyz` | `io/extxyz.py` | Write crystals as multi-frame Extended XYZ |

### `mck.operations` — Structural Operations

| Symbol | Source | Description |
|---|---|---|
| `add_hydrogens` | `hydrogen_completion.py` | Add hydrogens with heuristic placement + moiety correction |
| `HydrogenCompleter` | `hydrogen_completion.py` | Class-based hydrogen completion workflow |
| `generate_topological_slab` | `surface.py` | Generate slab preserving molecular topology |
| `TopologicalSlabGenerator` | `surface.py` | Class-based slab generation |
| `enumerate_terminations` | `surface.py` | Enumerate unique surface terminations with Tasker classification |
| `TerminationInfo` | `surface.py` | Dataclass for termination metadata |
| `generate_slabs_with_terminations` | `surface.py` | Build slabs for selected terminations |
| `ClusterCarver` | `cluster.py` | Carve finite QM clusters from periodic crystal |
| `carve_cluster` | `cluster.py` | Functional alias for `ClusterCarver.carve()` |
| `LigandTopologyOverflowError` | `cluster.py` | Raised when BFS exceeds `max_atoms` |
| `generate_vacancy` | `defects.py` | Remove molecular clusters to create vacancies |
| `VacancyGenerator` | `defects.py` | Class-based vacancy generation |
| `remove_solvents` | `desolvation.py` | Remove solvent molecules from crystal |
| `Desolvator` | `desolvation.py` | Class-based desolvation |
| `translate_molecule` | `molecule_manipulation.py` | Translate a molecule by Cartesian/fractional vector |
| `rotate_molecule` | `molecule_manipulation.py` | Rotate a molecule around COM or centroid |
| `replace_molecule` | `molecule_manipulation.py` | Replace molecule with clash detection & resolution |
| `MoleculeManipulator` | `molecule_manipulation.py` | Class-based molecule manipulation with species selection |
| `MoleculeClashError` | `molecule_manipulation.py` | Raised when clash cannot be resolved |
| `create_supercell` | `builders.py` | Create a supercell replica |
| `create_defect_structure` | `builders.py` | Create a defect structure from template |
| `apply_gaussian_displacement_molecule` | `perturbation.py` | Apply Gaussian noise to molecule positions |
| `apply_gaussian_displacement_crystal` | `perturbation.py` | Apply Gaussian noise to entire crystal |
| `apply_directional_displacement` | `perturbation.py` | Apply directed displacement |
| `apply_random_rotation` | `perturbation.py` | Apply random rotation to molecules |
| `rotate_molecule_at_center` | `rotation.py` | Rotate molecule around geometric center |
| `rotate_molecule_at_com` | `rotation.py` | Rotate molecule around center of mass |
| `interpolate_crystal` | `interpolation.py` | Interpolate between two `MolecularCrystal` states |
| `interpolate_molecule` | `interpolation.py` | Interpolate between two `CrystalMolecule` states |
| `interpolate_pose` | `interpolation.py` | Interpolate rigid-body pose (SE(3)) |
| `match_molecules` | `interpolation.py` | Match molecules between two crystals |
| `best_atom_mapping` | `interpolation.py` | Best atom-atom mapping between molecules |
| `find_flipping_molecules` | `interpolation.py` | Find molecules that flip orientation across frames |
| `InterpolationConfig` | `interpolation.py` | Configuration dataclass for interpolation |
| `InterpolationMethod` | `interpolation.py` | Enum for interpolation methods |
| `MoleculeMatch` | `interpolation.py` | Dataclass for molecule match results |

### `mck.analysis` — Analysis & Diagnostics

| Symbol | Source | Description |
|---|---|---|
| `BFDHFacetInfo` | `bfdh.py` | Dataclass for one BFDH facet candidate |
| `enumerate_bfdh_facets` | `bfdh.py` | Enumerate BFDH facet candidates with ranking |
| `enumerate_low_index_millers` | `bfdh.py` | Deterministic low-index Miller enumerator |
| `ChemicalEnvironment` | `chemical_env.py` | Chemical environment / hybridization analysis |
| `Fragment` | `formula_moiety.py` | Dataclass for one parsed moiety fragment |
| `parse_moiety_string` | `formula_moiety.py` | Parse `_chemical_formula_moiety` into fragments |
| `match_molecule_to_fragment` | `formula_moiety.py` | Match molecule to unique moiety fragment |
| `heavy_signature` | `formula_moiety.py` | H-free composition signature for matching |
| `MolChargeResult` | `charge.py` | Formal charge result for one topology type |
| `assign_mol_formal_charges` | `charge.py` | Assign formal charges with hybrid strategy |
| `compute_topo_signature` | `charge.py` | Unique topology fingerprint (formula + bond-graph) |
| `find_polyhedra` | `packing_shell.py` | Enumerate A–B coordination polyhedra |
| `detect_coordination_number` | `packing_shell.py` | Coordination number via gap+enclosure heuristic |
| `detect_prism_vs_antiprism` | `packing_shell.py` | Distinguish prismatic vs antiprismatic geometries |
| `classify_shell` | `shape.py` | Classify coordination shell geometry |
| `cshm` | `shape.py` | Continuous shape measures |
| `topology_signature` | `shape.py` | Topology signature for shell comparison |
| `angular_rmsd_vs_ideals` | `packing_shell.py` | Angular RMSD against ideal polyhedra |
| `compute_angular_signature` | `packing_shell.py` | Angular signature for shell classification |
| `hull_encloses_center` | `packing_shell.py` | Test if convex hull encloses center atom |
| `planarity_analysis` | `packing_shell.py` | Planarity analysis of coordination shell |
| `DEFAULT_POLYHEDRON_SEARCH_CUTOFF` | `packing_shell.py` | Default search cutoff for polyhedron detection (Å) |
| `DEFAULT_MOLECULAR_SEARCH_CUTOFF` | `packing_shell.py` | Default search cutoff for molecule-level polyhedra (Å) |
| `DEFAULT_CENTROID_OFFSET_FRAC` | `packing_shell.py` | Default centroid offset fraction for molecule polyhedra |

### `mck.analysis.disorder` — Disorder Resolution

| Symbol | Source | Description |
|---|---|---|
| `DisorderInfo` | `disorder/info.py` | Parsed disorder metadata from CIF |
| `DisorderSolver` | `disorder/solver.py` | MWIS-based disorder solver (greedy/random/enumerate) |
| `generate_ordered_replicas_from_disordered_sites` | `disorder/process.py` | Main entry point: resolve disorder → ordered replicas |
| `is_minor_site` | `disorder/predicates.py` | Test if a site is a minor-occupancy component |
| `DisorderProvenance` | `disorder/provenance.py` | Provenance record for disorder resolution |

Access these via: `from molcrys_kit.analysis.disorder import ...`

### `mck.analysis.interactions` — Intermolecular Interactions

| Symbol | Source | Description |
|---|---|---|
| `AtomRef` | `interactions/base.py` | Lightweight reference to an atom + periodic image |
| `RingRef` | `interactions/base.py` | Lightweight reference to a ring + periodic image |
| `BaseInteraction` | `interactions/base.py` | Abstract base class for all interactions |
| `HydrogenBond` | `interactions/hydrogen_bond.py` | H-bond record (D, H, A, distances, angle) |
| `HydrogenBondCriteria` | `interactions/hydrogen_bond.py` | Configurable H-bond detection criteria |
| `find_hydrogen_bonds` | `interactions/hydrogen_bond.py` | Find all H-bonds in a crystal |
| `HalogenBond` | `interactions/halogen_bond.py` | Halogen bond record |
| `HalogenBondCriteria` | `interactions/halogen_bond.py` | Configurable halogen bond criteria |
| `find_halogen_bonds` | `interactions/halogen_bond.py` | Find all halogen bonds |
| `PiStacking` | `interactions/pi_stacking.py` | π-stacking record |
| `PiStackingCriteria` | `interactions/pi_stacking.py` | Configurable π-stacking criteria |
| `PiStackingSubtype` | `interactions/pi_stacking.py` | Enum: sandwich, parallel-displaced, T-shaped |
| `find_pi_stacking` | `interactions/pi_stacking.py` | Find all π-stacking interactions |
| `find_pi_stacks` | `interactions/pi_stacking.py` | Backward-compat alias for `find_pi_stacking` |
| `CHPiInteraction` | `interactions/ch_pi.py` | CH–π interaction record |
| `CHPiInteractionCriteria` | `interactions/ch_pi.py` | Configurable CH–π criteria |
| `find_ch_pi` | `interactions/ch_pi.py` | Find all CH–π interactions |
| `find_ch_pi_interactions` | `interactions/ch_pi.py` | Backward-compat alias |
| `HHContact` | `interactions/h_h_contact.py` | H···H contact record |
| `HHContactCriteria` | `interactions/h_h_contact.py` | Configurable H···H criteria |
| `find_h_h_contacts` | `interactions/h_h_contact.py` | Find all H···H contacts |
| `get_bonding_threshold` | `interactions/bonding.py` | Element-pair bonding distance threshold |
| `build_crystal_atom_offsets` | `interactions/base.py` | Build periodic image offset table |
| `AtomLocalGeometry` | `interactions/local_geometry.py` | Per-atom local coordination geometry |
| `LocalGeometry` | `interactions/local_geometry.py` | Molecule-local neighbor/ring geometry |
| `LocalGeometryCache` | `interactions/local_geometry.py` | Cached local geometry for crystal |
| `RingGeometry` | `interactions/local_geometry.py` | Per-ring local geometry info |
| `ChemicalIdentity` | `analysis/molecular_identity.py` | Molecule/fragment/species identity annotation |
| `ChemicalIdentityCache` | `analysis/molecular_identity.py` | Cached identity lookup for crystal |

Access these via: `from molcrys_kit.analysis.interactions import ...`

### `mck.constants` — Element Data & Thresholds

| Symbol | Source | Description |
|---|---|---|
| `ATOMIC_MASSES` | `constants/__init__.py` | Dict of element symbol → atomic mass (amu) |
| `ATOMIC_RADII` | `constants/__init__.py` | Dict of element symbol → atomic radius (Å) |
| `METAL_ELEMENTS` | `constants/__init__.py` | Set of metal element symbols |
| `METAL_THRESHOLD_FACTOR` | `constants/__init__.py` | Bond threshold multiplier for metals |
| `NON_METAL_THRESHOLD_FACTOR` | `constants/__init__.py` | Bond threshold multiplier for non-metals |
| `METAL_NON_METAL_THRESHOLD_FACTOR` | `constants/__init__.py` | Bond threshold multiplier for metal–non-metal pairs |
| `DEFAULT_NEIGHBOR_CUTOFF` | `constants/__init__.py` | Default neighbor search cutoff (Å) |

### `mck.utils` — Geometry Utilities

| Symbol | Source | Description |
|---|---|---|
| `frac_to_cart` | `geometry.py` | Fractional → Cartesian coordinates |
| `cart_to_frac` | `geometry.py` | Cartesian → Fractional coordinates |
| `minimum_image_distance` | `geometry.py` | Minimum-image distance under PBC |
| `minimum_image_vector` | `geometry.py` | Minimum-image vector under PBC |
| `kabsch_align` | `geometry.py` | Kabsch alignment between point sets |
| `dihedral_angle` | `geometry.py` | Dihedral angle between four points |
| `angle_between_vectors` | `geometry.py` | Angle between two vectors |
| `distance_between_points` | `geometry.py` | Euclidean distance between two points |
| `normalize_vector` | `geometry.py` | Normalize a vector to unit length |
| `skew_matrix` | `geometry.py` | Skew-symmetric matrix from vector |
| `unskew_matrix` | `geometry.py` | Extract vector from skew-symmetric matrix |
| `rotation_to_axis_angle` | `geometry.py` | Rotation matrix → axis-angle representation |
| `rotation_log_vector` | `geometry.py` | SO(3) log map |
| `rotation_exp_vector` | `geometry.py` | SO(3) exp map |
| `se3_log` | `geometry.py` | SE(3) log map |
| `se3_exp` | `geometry.py` | SE(3) exp map |
| `rotation_matrix_to_quaternion` | `geometry.py` | Rotation matrix → quaternion |
| `quaternion_to_rotation_matrix` | `geometry.py` | Quaternion → rotation matrix |
| `quaternion_slerp` | `geometry.py` | Spherical linear interpolation of quaternions |
| `unwrap_positions_along_bonds` | `geometry.py` | Unwrap positions along bond connectivity |
| `volume_of_cell` | `geometry.py` | Compute cell volume from lattice vectors |
