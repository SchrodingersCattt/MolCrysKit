# MolCrysKit API & Capabilities

> Compact index for users and AI agents. This is not a full API reference.
> For exact parameters, read source docstrings. For worked examples, see
> [Tutorials](tutorials.md).

## Core Concepts

- `MolecularCrystal` is the central object for most workflows.
- Typical flow: CIF/ASE → `MolecularCrystal` → operation or analysis → writer.
- Operations return new objects; treat inputs as immutable unless a docstring says otherwise.
- Use `import molcrys_kit as mck`; common entry point: `mck.read_mol_crystal(path)`.
- Public symbols are grouped below by task. Details live in source docstrings.

## Capability Map

| Task | Entry point | Input | Output | More detail |
|---|---|---|---|---|
| Parse CIF | `mck.read_mol_crystal` | CIF path | `MolecularCrystal` | source docstring |
| Parse CIF (class) | `MolecularCrystal.from_cif` | CIF path, `use_asu_first=` | `MolecularCrystal` | source docstring |
| Identify molecules | `identify_molecule_indices` | ASE/CIF-derived structure | molecule indices | source docstring |
| List molecule inventory | `mck io molecules --json` | crystal file | JSON molecule records | `mck io molecules --help` |
| Extract molecule file | `mck io extract-molecule` | crystal file + selector | `.xyz` / `.cif` / `.extxyz` molecule file | `mck io extract-molecule --help` |
| Write structures | `write_cif`, `write_poscar`, `write_xyz`, `write_extxyz` | `MolecularCrystal` / frames | file | source docstring |
| Resolve disorder | `generate_ordered_replicas_from_disordered_sites` | `MolecularCrystal` | `list[MolecularCrystal]` | [Architecture](architecture.md) |
| Add hydrogens | `add_hydrogens` | `MolecularCrystal` | `MolecularCrystal` | [Tutorials](tutorials.md) |
| Generate slabs | `generate_topological_slab`, `generate_slabs_with_terminations` | crystal + Miller plane | slab(s) | [Tutorials](tutorials.md) |
| Reorient crystal | `reorient_crystal`, `get_surface_basis` | crystal + Miller direction | reoriented crystal + info | source docstring |
| Rank facets | `enumerate_bfdh_facets` | crystal/lattice | `list[BFDHFacetInfo]` | [Tutorials](tutorials.md) |
| Carve QM clusters | `ClusterCarver`, `carve_cluster` | crystal + seeds | `CrystalCluster` | [Tutorials](tutorials.md) |
| Edit molecules | `translate_molecule`, `rotate_molecule`, `replace_molecule` | crystal + molecule index | `MolecularCrystal` | [Tutorials](tutorials.md) |
| Defects/desolvation | `generate_vacancy`, `remove_solvents` | crystal | `MolecularCrystal` | source docstring |
| Interaction analysis | `find_hydrogen_bonds`, `find_pi_stacking`, `interaction_profile` | crystal | interaction records/profile | source docstring |
| Packing/polyhedra | `find_polyhedra`, `detect_coordination_number` | crystal/ASE atoms | records/CN | source docstring |
| Interpolation | `interpolate_crystal`, `interpolate_molecule`, `interpolate_pose` | two states | path/frames | source docstring |

## Module Index

### `mck.structures`
Core crystal data model.

- Core: `MolAtom`, `CrystalMolecule`, `MolecularCrystal`, `CrystalTrajectory`, `Molecule`
- Constructors: `MolecularCrystal.from_cif(path, use_asu_first=False)`, `MolecularCrystal.from_ase(atoms)`
  - `use_asu_first=True`: identify molecules on the asymmetric unit, then replicate via symmetry operations.  More efficient for high-symmetry crystals; falls back to the standard path on failure.
- Clusters: `CrystalCluster`, `ClusterProvenance`
- Polyhedra reference data: `all_ideal_polyhedra`, `ideal_polyhedra_for_cn`, `convex_hull_payload`

### `mck.io`
Read/write interfaces.

- Read: `read_mol_crystal`, `parse_cif_advanced`, `identify_molecule_indices`, `read_xyz`, `read_poscar`, `read_extxyz`
  - `read_mol_crystal` uses `scan_cif_disorder` as the sole authority for coordinates and disorder metadata.
- Write: `write_cif`, `write_cif_sequence`, `write_poscar`, `write_poscar_sequence`, `write_xyz`, `write_xyz_with_freeze`, `write_trajectory`, `write_extxyz`
- Disorder: `scan_cif_disorder`, `DisorderInfo`, `DisorderInfo.from_crystal`

### `mck.operations`
Structure-changing workflows. Prefer functional helpers for simple tasks and classes for repeated/custom workflows.

- Perturb/rotate: `apply_gaussian_displacement_molecule`, `apply_gaussian_displacement_crystal`, `apply_directional_displacement`, `apply_random_rotation`, `rotate_molecule_at_center`, `rotate_molecule_at_com`
- Build/edit: `create_supercell`, `create_defect_structure`, `translate_molecule`, `rotate_molecule`, `replace_molecule`, `MoleculeManipulator`, `MoleculeClashError`
- Surface: `generate_topological_slab`, `TopologicalSlabGenerator`, `TerminationInfo`, `enumerate_terminations`, `generate_slabs_with_terminations`, `get_surface_basis`
- Reorientation: `reorient_crystal`, `ReorientationInfo`
- H/solvent/defects: `HydrogenCompleter`, `add_hydrogens`, `Desolvator`, `remove_solvents`, `VacancyGenerator`, `generate_vacancy`
- Clusters: `ClusterCarver`, `LigandTopologyOverflowError`, `carve_cluster`
- Interpolation: `InterpolationConfig`, `InterpolationMethod`, `MoleculeMatch`, `VCMoleculeMatch`, `best_atom_mapping`, `find_flipping_molecules`, `interpolate_crystal`, `interpolate_crystal_vc`, `interpolate_molecule`, `interpolate_pose`, `match_molecules`, `match_molecules_vc`

### `mck.analysis`
Analysis workflows and selected re-exports. Interaction-specific exports are listed under `mck.analysis.interactions`.

- Facets/shape: `BFDHFacetInfo`, `enumerate_bfdh_facets`, `enumerate_low_index_millers`, `classify_shell`, `cshm`, `topology_signature`
- Chemistry/formula/charge: `ChemicalEnvironment`, `Fragment`, `parse_moiety_string`, `match_molecule_to_fragment`, `heavy_signature`, `MolChargeResult`, `assign_mol_formal_charges`, `compute_topo_signature`
- Packing/polyhedra: `find_polyhedra`, `detect_coordination_number`, `detect_prism_vs_antiprism`, `angular_rmsd_vs_ideals`, `compute_angular_signature`, `hull_encloses_center`, `planarity_analysis`, `DEFAULT_POLYHEDRON_SEARCH_CUTOFF`, `DEFAULT_MOLECULAR_SEARCH_CUTOFF`, `DEFAULT_CENTROID_OFFSET_FRAC`
- Volume/boundary: `calculate_atomic_volumes`, `calculate_total_volume`, `calculate_accessible_boundary`, `min_distance_to_boundary`
- Sanity checks: `sanity_check`, `SanityReport`, `CheckResult`, `check_hard_clash`, `check_intermolecular_clash`, `check_isolated_atoms`, `check_hydrogen_presence`, `check_formula_consistency`, `check_bond_distances`, `check_topology_preservation`

### `mck.analysis.volume`
Van der Waals volume estimation and solvent-accessible boundary computation.

- `calculate_atomic_volumes(atoms, radii_type="vdw")` — per-atom spherical volumes (4/3 π r³) using VdW or covalent radii.
- `calculate_total_volume(atoms, radii_type="vdw", overlap_correction=False)` — total volume as simple sum or overlap-corrected (3-D occupancy grid, voxel_size=0.2 Å).
- `calculate_accessible_boundary(atoms, probe_radius=1.4, radii_type="vdw", n_sphere_points=50)` — Shrake-Rupley style surface point generation; returns (M, 3) Cartesian array of solvent-accessible boundary points.
- `min_distance_to_boundary(new_positions, boundary_points, lattice=None, pbc=None)` — minimum distance from query positions to boundary; supports periodic boundary conditions via minimum image convention.

### `mck.analysis.disorder`
Disorder metadata, solving, and ordered-replica generation.

- `DisorderInfo`, `DisorderSolver`, `generate_ordered_replicas_from_disordered_sites`, `is_minor_site`, `DisorderProvenance`
- `DisorderInfo` carries CIF-derived metadata including the chemical formula moiety and Z value for stoichiometry validation during disorder resolution.

### `mck.analysis.interactions`
Weak-interaction detection plus continuous scoring. Raw detectors return records; `interaction_profile` summarizes counts and scores.

- `interaction_profile` aggregates three interaction families: hydrogen bonds, halogen bonds, and pi-stacking (parallel + T-shape). C-H···π is subsumed by T-shape pi-stacking; H···H close contacts are excluded as packing artifacts. The standalone detectors `find_ch_pi` and `find_h_h_contacts` remain importable.
- Pi-stacking uses subtype-specific geometry: parallel stacking filters by interplane distance h; T-shape uses centroid distance d with a wider cutoff and scoring center.  T-shape records include an approach distance field — the minimum distance from either ring's edge to the other ring's plane (negative = stem penetration, absent for parallel subtypes).
- Base/local identity: `AtomRef`, `RingRef`, `BaseInteraction`, `build_crystal_atom_offsets`, `AtomLocalGeometry`, `LocalGeometry`, `LocalGeometryCache`, `RingGeometry`, `ChemicalIdentity`, `ChemicalIdentityCache`
- Detectors: `HydrogenBond`, `HydrogenBondCriteria`, `find_hydrogen_bonds`, `HalogenBond`, `HalogenBondCriteria`, `find_halogen_bonds`, `PiStacking`, `PiStackingCriteria`, `PiStackingSubtype`, `find_pi_stacking`, `find_pi_stacks`, `CHPiInteraction`, `CHPiInteractionCriteria`, `find_ch_pi`, `find_ch_pi_interactions`, `HHContact`, `HHContactCriteria`, `find_h_h_contacts`, `get_bonding_threshold`
- Scoring/profile: `InteractionProfile`, `InteractionScoreSummary`, `interaction_profile`, `ScoringParams`, `DEFAULT_SCORING_PARAMS`, `composite_score`, `gaussian_kernel`, `lorentzian_kernel`, `normalized_vdw_distance`, `scaled_cutoff`, `vdw_radius_sum`

### `mck.constants`
Element data and bond-detection thresholds.

- Data: `ATOMIC_MASSES`, `ATOMIC_RADII`, `VDW_RADII`, `METAL_ELEMENTS`
- Thresholds: `METAL_THRESHOLD_FACTOR`, `NON_METAL_THRESHOLD_FACTOR`, `METAL_NON_METAL_THRESHOLD_FACTOR`, `DEFAULT_NEIGHBOR_CUTOFF`

### `mck.utils`
Geometry, rigid-body math helpers, and graph utilities.

- Graph: `graph_invariant`
- Coordinates/PBC: `frac_to_cart`, `cart_to_frac`, `minimum_image_distance`, `minimum_image_vector`, `unwrap_positions_along_bonds`, `volume_of_cell`
- Vector/angles: `normalize_vector`, `distance_between_points`, `angle_between_vectors`, `dihedral_angle`
- Rotations/alignment: `skew_matrix`, `unskew_matrix`, `kabsch_align`, `rotation_to_axis_angle`, `rotation_log_vector`, `rotation_exp_vector`, `rotation_matrix_to_quaternion`, `quaternion_to_rotation_matrix`, `quaternion_slerp`
- Lattice orientation: `orient_lattice`
- Lattice interpolation: `lattice_deformation_logm`, `lattice_at_lambda`
- SE(3): `se3_log`, `se3_exp`

## See Also

- Detailed workflows: [Tutorials](tutorials.md)
- Command-line interface: [CLI Reference](cli.md)
- Disorder architecture: [Architecture](architecture.md)
- Docker/cloud usage: [Docker Guide](docker.md)
- Exact function signatures: source docstrings
