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
| Convert pymatgen Structure | `to_pymatgen_structure`, `from_pymatgen_structure`, `MolecularCrystal.to_pymatgen`, `MolecularCrystal.from_pymatgen` | `MolecularCrystal` / `Structure` | `Structure` / `MolecularCrystal` | preserves occupancy; source docstring |
| Read CIF symmetry | `read_cif_symmetry` | CIF path/text | `CrystalSymmetry` | source docstring |
| Parse CIF (class) | `MolecularCrystal.from_cif` | CIF path, `use_asu_first=` | `MolecularCrystal` | source docstring |
| Identify molecules | `identify_molecule_indices`, `identify_molecules` | ASE/CIF-derived structure | molecule indices or molecules | `disorder_policy` defaults to `as_refined` (no atoms dropped). `major` and `exclusive` are explicit |
| Export renderer-ready structure | `MolecularCrystal.get_site_records`, `MolecularCrystal.get_bond_records` | `MolecularCrystal` | immutable site/bond records | source docstring |
| Bond role and connectivity regime | `MolecularCrystal.get_bond_records`, `MolecularCrystal.get_connectivity_report`, `classify_bond_role` | ASE/CIF `MolecularCrystal` | `BondRecord.role` (`covalent` or `coordination`) and molecular/extended regime | source docstring |
| Attach chemistry-domain records | `annotate_chemistry` | `MolecularCrystal` | `CrystalChemistry` | source docstring |
| Infer bond orders and formal charges | `infer_chemistry` | `MolecularCrystal` | `CrystalChemistry` with alternatives/warnings | source docstring |
| Infer bond orders for one entity | `infer_bond_orders` | `FiniteChemicalEntity` / `PeriodicChemicalEntity` | `BondOrderResult` with Hückel aromaticity, source locks, diagnostics, and alternatives | source docstring |
| Infer bond orders from arrays | `infer_bond_orders_array`, `infer_bond_orders_batch` | coordinates, elements, optional bond pairs | `BondOrderResult` / tuple of results | source docstring |
| Calibrate aromaticity geometry | `calibrate_aromaticity_geometry` | CSD-exported ring and bond observations | reviewable calibration snapshot with pair coverage and uncertainty bands | [Chemistry perception design](chemistry_perception_design.md) |
| Classify 0D--3D chemical topology | `analyze_periodic_topology` | atom ids + periodic bonds | `PeriodicTopology` | source docstring |
| Assign coordinate stereochemistry | `assign_stereochemistry` | chemical entity + embedding | `StereoReport` | source docstring |
| Aggregate crystal stereochemistry | `analyze_crystal_stereochemistry` | crystal + chemistry + entity stereo reports | `CrystalStereoReport` | source docstring |
| Convert chemical line notation | `to_line_notation`, `from_line_notation` | chemical entity / OpenSMILES or MCK-LN | `LineNotation` / chemical entity | source docstring |
| Compare notation equivalence | `notations_equivalent` | two line-notation strings | `True` / `False` / `None` | source docstring |
| Name an entity or crystal | `name_entity`, `name_crystal` | chemical entity / crystal | `NamingResult` | source docstring |
| Convert IUPAC and SMILES (bounded subset) | `smiles_to_iupac`, `iupac_to_smiles`, `from_iupac_name` | OpenSMILES / supported IUPAC name | `NamingResult` / `LineNotation` / `FiniteChemicalEntity` | source docstring |
| Select one compact formula unit | `StoichiometryAnalyzer.select_formula_unit` | `MolecularCrystal` | `FormulaUnitSelection` | source docstring |
| List molecule inventory | `mck io molecules --json` | crystal file | JSON molecule records | `mck io molecules --help` |
| Extract molecule file | `mck io extract-molecule` | crystal file + selector | `.xyz` / `.cif` / `.extxyz` molecule file | `mck io extract-molecule --help` |
| Summarize structure | `mck analyze summary --json` | structure file | JSON composition, cell, symmetry, and disorder report | `mck analyze summary --help` |
| Transform periodic cell | `MolecularCrystal.transform_cell` | target lattice or row-basis matrix | molecule-safe `MolecularCrystal` copy | source docstring |
| Audit periodic cell | `check_cell_integrity` | `MolecularCrystal` (+ optional reference) | integrity report with hashes and seam contacts | source docstring |
| Write structures | `write_cif`, `write_poscar`, `write_xyz`, `write_extxyz` | `MolecularCrystal` / frames | file | source docstring |
| Typed chemical edits / packing transfer | `apply_chemical_edit`, `transfer_packing`, `write_cf0`, `RelaxationProtocol`, `StoichiometryPolicy` | ordered full cell + arbitrary roles/charges + explicit mapping/placement; cross-stoichiometry needs `StoichiometryPolicy` | atomic CF0 result or typed refusal, normalized inventory provenance, external relaxation constraints | [Chemical editing](chemical_edit.md) |
| Forward crystal construction | `prepare_construction_context`, `analyze_molecular_pose`, `construct_molecular_crystal`, `construct_statistical_disorder`, `read_statistical_disorder_companion` | cell + complete symmetry + rigid pose(s), optional typed bonds and joint constraints | ordered crystal, true-ASU or P1 CIF, versioned disorder companion | Issue #172/#198 forward contract |
| Resolve disorder | `generate_ordered_replicas_from_disordered_sites` | `MolecularCrystal` | `list[MolecularCrystal]` | [Architecture](architecture.md) |
| Assemble disorder replicas | `assemble_replica_supercell` | ordered replicas + per-cell indices | `MolecularCrystal` supercell | source docstring |
| Add hydrogens | `add_hydrogens` | `MolecularCrystal` | `MolecularCrystal` | [Tutorials](tutorials.md) |
| Generate slabs | `generate_topological_slab`, `generate_slabs_with_terminations` | crystal + Miller plane | slab(s) | [Tutorials](tutorials.md) |
| Reorient crystal | `reorient_crystal`, `get_surface_basis` | crystal + Miller direction | reoriented crystal + info | source docstring |
| Rank facets | `enumerate_bfdh_facets` | crystal/lattice | `list[BFDHFacetInfo]` | [Tutorials](tutorials.md) |
| Carve QM clusters | `ClusterCarver`, `carve_cluster` | crystal + seeds | `CrystalCluster` | [Tutorials](tutorials.md) |
| Carve topology-preserving nanoclusters | `ImplicitShape`/`NanoShape`, `ImplicitShape.bfdh`, `NanoClusterCarver`, `carve_nanocluster` | crystal + implicit shape | non-periodic `MolecularCrystal` | [Tutorials](tutorials.md) |
| Carve topology-preserving voids | `ImplicitShape`, `VoidCarver`, `carve_void` | crystal + implicit shape | periodic remainder, optionally removed cluster | [Tutorials](tutorials.md) |
| Rotate bond fragments | `partition_at_bond`, `rotate_fragment_about_bond`, `rotate_fragment_in_crystal` | molecule/crystal + bond selection | new molecule/crystal | [Tutorials](tutorials.md) |
| Edit molecules | `translate_molecule`, `rotate_molecule`, `replace_molecule` | crystal + molecule index | `MolecularCrystal` | [Tutorials](tutorials.md) |
| Analyze ring puckering | `puckering_coordinates`, `reconstruct_z_from_modes`, `find_ring_systems` | molecule + ordered ring / modes / molecule | coordinates / normal displacements / `list[RingSystem]` | source docstring |
| Defects/desolvation | `generate_vacancy`, `remove_solvents` | crystal | `MolecularCrystal` | source docstring |
| Interaction analysis | `find_hydrogen_bonds`, `find_pi_stacking`, `interaction_profile` | crystal | interaction records/profile | source docstring |
| Packing/polyhedra | `find_polyhedra`, `detect_coordination_number` | crystal/ASE atoms | records/CN | `disorder_policy="major"` keeps the highest-occupancy orientation before a molecule-level shell is built |
| Collective symmetry path | `build_symmetry_path_plan`, `generate_collective_symmetry_path` | crystal + affine operation | validated rigid path | [Tutorials](tutorials.md) |
| Interpolation | `interpolate_crystal`, `interpolate_molecule`, `interpolate_pose` | two states | path/frames | source docstring |
| Composite conformational interpolation | `interpolate_molecule_with_internal_dofs` | two `CrystalMolecule` conformers | rigid + bridge-torsion/Cremer–Pople frames | source docstring |
| Reactive initial path | `interpolate_reactive_path` | atom-mapped endpoints + rigid groups | flat ASE frames + result metadata | [Tutorials](tutorials.md) |
| Build periodic fragment chains | `build_periodic_chains` | templates, connection rules, cell, and closure | `PeriodicBundle` | [Periodic chains](periodic-chains.md) |
| Validate periodic bundle | `validate_periodic_bundle` | bundle structure and sidecar metadata | integrity, graph, and periodic-distance report | [Periodic chains](periodic-chains.md) |

## Module Index

### `mck.structures`
Core crystal data model.

- Core: `MolAtom`, `CrystalMolecule`, `MolecularCrystal`, `CrystalTrajectory`, `Molecule`
- Cell transforms: `MolecularCrystal.transform_cell`
- Bonds: `BondPairs`, `BondCandidates`, `VerletBondTracker`, `build_bond_candidates`, `candidate_list_needs_rebuild`, `evaluate_bond_candidates`, `infer_bond_pairs`; see [Bond inference](bonds.md)
- Renderer contracts: `SiteRecord`, `BondRecord`, `ComponentConnectivity`, `ConnectivityReport`; use `MolecularCrystal.get_site_records()`, `MolecularCrystal.get_bond_records()`, and `MolecularCrystal.get_connectivity_report()` instead of private ASE metadata or element whitelists.
  - `SiteRecord` identifies one atom by global, molecule-local, and ASU-source indices and carries label/symmetry provenance, Cartesian and fractional coordinates, occupancy/disorder metadata, lattice image shift, uiso_A2, and Cartesian u_cart_A2.
  - `BondRecord` carries molecule-local and global endpoints, ASU-source endpoints when known, the right-end lattice image shift, Cartesian bond vector, distance, and role.
  - role is the plain string `BOND_ROLE_COVALENT` (`"covalent"`) or `BOND_ROLE_COORDINATION` (`"coordination"`). Every neighbor-list edge is labelled when it is accepted. Non-metal pairs are covalent. A metal–non-metal pair is covalent at or below the minimum of METAL_NONMETAL_COVALENT_MAX and (r_i + r_j) * NON_METAL_THRESHOLD_FACTOR, and coordination beyond that ceiling. Metal–metal pairs are coordination. `BOND_ROLES` is the allowed set. `classify_bond_role` applies the same rule. Renderers draw sticks for covalent edges and keep coordination edges for topology or polyhedra. Bond order is not on this record.
  - `get_bond_records()` returns all perceived periodic-image bonds, including
    multiple records for one atom pair and self-image bonds in one-site cells.
    Edge identity is (left_global_index, right_global_index, right_image_shift) with role on that same record.
  - `get_connectivity_report()` returns a `ConnectivityReport`. regime is `CONNECTIVITY_REGIME_MOLECULAR` when every component has translational rank 0, and `CONNECTIVITY_REGIME_EXTENDED` when any component has rank 1–3 (chain, layer, or framework). max_rank is the largest component rank. components is one `ComponentConnectivity` per molecule component (molecule_index, rank, regime, translation_generators). Rank is the lattice-winding dimension of the signed bond graph, including coordination contacts, and does not inspect element symbols. `CONNECTIVITY_REGIMES` is the allowed set. A cell-face crossing that only unwraps a finite molecule stays molecular.
- Symmetry: `FractionalAffineOperation`, `LatticeBasisChange`, `CrystalSymmetry`, `identity_operation`, `validate_subgroup`, `left_cosets`, `domain_representatives`
- Constructors: `MolecularCrystal.from_cif(path, use_asu_first=False)`, `MolecularCrystal.from_ase(atoms)`, `MolecularCrystal.from_pymatgen(structure)`
  - Conversions: `MolecularCrystal.to_pymatgen()` (and io helpers) preserve per-site occupancy, oxidation state, and H/D isotope identity; prefer them over ASE atoms adaptors that reject disordered structures.
  - `use_asu_first=True`: identify molecules on the asymmetric unit, then replicate via symmetry operations.  More efficient for high-symmetry crystals; falls back to the standard path on failure.
- Clusters: `CrystalCluster`, `ClusterProvenance`
- Polyhedra reference data: `all_ideal_polyhedra`, `ideal_polyhedra_for_cn`, `convex_hull_payload`
- Periodic geometry: `BoundaryPort`, `ChainSpec`, `ConnectionRule`, `FragmentInstance`, `FragmentTemplate`, `PeriodicBundle`, `PeriodicEdge`, `PeriodicGraph`, `ScrewSpec`

### `mck.chemistry`
Independent immutable chemistry records and crystal-to-chemistry mapping.
Implemented rule families and strict refusal boundaries are tracked in
[Chemistry coverage and release gates](chemistry_coverage.md).

- Models: `ChemicalAtom`, `ChemicalBond`, `ChemicalEntity`, `FiniteChemicalEntity`, `PeriodicChemicalEntity`, `PolymerChemicalEntity`, `MulticomponentEntity`, `Embedding`, `CrystalChemistry`, `PeriodicTopology`, `StereoDescriptor`, `StereoKind`, `StereoReport`, `CrystalStereoClass`, `CrystalStereoReport`, `EntityRelationship`, `EntityStereoSummary`, `EnantiomerCount`, `AbsoluteStructureParameter`, `LineNotation`, `NamingKind`, `NamingResult`, `BondOrderOptions`, `BondOrderResult`
- Calibration: `CalibrationError`, `calibrate_aromaticity_geometry`, `CalibrationPolicy`, `RING_CLASSES`, `runtime_overrides`, `read_records`, `write_snapshot`
- Provenance: `BondKind`, `Evidence`, `EvidenceSource`, `InferenceStatus`
- Analysis and conversion: `annotate_chemistry`, `infer_chemistry`, `infer_bond_orders`, `infer_bond_orders_array`, `infer_bond_orders_arrays`, `infer_bond_orders_from_arrays`, `infer_bond_orders_batch`, `analyze_periodic_topology`, `assign_stereochemistry`, `analyze_crystal_stereochemistry`, `classify_entity_relationship`, `to_line_notation`, `from_line_notation`, `notations_equivalent`, `name_entity`, `name_crystal`, `from_iupac_name`, `iupac_to_smiles`, `smiles_to_iupac`, `ChemistryIndeterminateError`, `CrystalStereoIndeterminateError`, `LineNotationError`, `NamingIndeterminateError`, `NamingParseError`

### `mck.io`
Read/write interfaces.

- Read: `read_mol_crystal`, `read_cif_symmetry`, `parse_cif_advanced`, `identify_molecule_indices`, `read_xyz`, `read_poscar`, `read_extxyz`
- pymatgen Structure bridge: `to_pymatgen_structure`, `from_pymatgen_structure` (occupancy-preserving; mixed sites expand to one atom per composition component; the reverse conversion does not re-collapse co-located atoms into a multi-species site)
- Periodic bundles: `read_periodic_bundle`, `write_periodic_bundle`
  - `read_mol_crystal` uses `scan_cif_disorder` as the sole authority for coordinates and disorder metadata.
  - It preserves CIF chemical names, published bond rows, isotopes/site charges, and absolute-structure values with standard uncertainties in `MolecularCrystal.metadata["cif_chemistry"]`, then attaches a provisional chemistry snapshot automatically. Geometry-only callers can pass `attach_chemistry=False` to skip the perception pass.
- Write: `write_cif`, `write_cif_p1`, `write_cif_asu`, `write_disorder_cif`, `write_disorder_cif_asu`, `write_forward_cif`, `write_cif_sequence`, `write_poscar`, `write_poscar_sequence`, `write_xyz`, `write_xyz_with_freeze`, `write_trajectory`, `write_extxyz`, `describe_structure_export`
  - Ordered P1 and true-ASU CIF keep known bond order/type, isotope (`2H` for deuterium), explicit formal charge, and atomic mass. Unknown bond type is `.`, and unknown charge is distinct from explicit zero.
  - `write_forward_cif(..., representation="asu")` on a statistical model writes one source-group ASU of the marginal sites. `representation="p1"` remains the materialized average and is not expanded a second time.
  - `describe_structure_export` states which of XYZ, POSCAR, ASE, ExtXYZ, CIF, and the JSON companion keep symmetry, occupancy, and whole-molecule alternatives. XYZ and POSCAR warn; they do not convert an average into a microstate.
- Disorder: `scan_cif_disorder`, `DisorderInfo`, `DisorderInfo.from_crystal`

### `mck.operations`
Structure-changing workflows. Prefer functional helpers for simple tasks and classes for repeated/custom workflows.

- Bond fragments: `BondPartition`, `BondRotationError`, `BondNotFoundError`, `BondRotationSelectionError`, `RingBondRotationError`, `partition_at_bond`, `rotate_fragment_about_bond`, `rotate_fragment_in_crystal`
- Perturb/rotate: `apply_gaussian_displacement_molecule`, `apply_gaussian_displacement_crystal`, `apply_directional_displacement`, `apply_random_rotation`, `rotate_molecule_at_center`, `rotate_molecule_at_com`
- Chemical editing: `ChemicalAtomMapping`, `ChemicalEdit`, `CrystalEditSpec`, `CrystalInventory`, `PlacementPolicy`, `MoleculeInsertion`, `StoichiometryPolicy`, `PackingTransferPlan`, `CounterfactualStatus`, `CounterfactualResult`, `RelaxationProtocol`, `apply_chemical_edit`, `transfer_packing`, `periodic_clash_diagnostics`, `write_cf0`; see [placement, inventories, refusals and protocols](chemical_edit.md). These symbols are also lazy attributes of the top-level package.
- Build/edit: `assemble_replica_supercell`, `create_supercell`, `translate_molecule`, `rotate_molecule`, `replace_molecule`, `MoleculeManipulator`, `MoleculeClashError`
- Surface: `generate_topological_slab`, `TopologicalSlabGenerator`, `TerminationInfo`, `enumerate_terminations`, `generate_slabs_with_terminations`, `get_surface_basis`
- Reorientation: `reorient_crystal`, `ReorientationInfo`
- H/solvent/defects: `HydrogenCompleter`, `add_hydrogens`, `Desolvator`, `remove_solvents`, `VacancyGenerator`, `generate_vacancy`, `VoidCarver`, `carve_void`
- Clusters: `ClusterCarver`, `LigandTopologyOverflowError`, `carve_cluster`
- Implicit shapes/nanoclusters: `ImplicitShape` (including `ImplicitShape.bfdh`), `NanoShape` (compatibility alias), `NanoClusterCarver`, `carve_nanocluster`, `DEFAULT_SHAPE_BATCH_SIZE`, `DEFAULT_NANOCLUSTER_BATCH_SIZE`
- Periodic chains: `build_periodic_chains`
- Symmetry paths: `RigidReachabilityTolerance`, `SymmetryPathConfig`, `AtomCorrespondence`, `SymmetryMoleculeMatch`, `CrystalCorrespondence`, `SymmetryPathProvenance`, `SymmetryPathPlan`, `RigidReachabilityError`, `transform_crystal_fractional`, `build_symmetry_path_plan`, `interpolate_symmetry_path`, `generate_collective_symmetry_path`
- Interpolation: `InterpolationConfig`, `InterpolationMethod`, `MoleculeMatch`, `VCMoleculeMatch`, `best_atom_mapping`, `find_flipping_molecules`, `interpolate_crystal`, `interpolate_crystal_vc`, `interpolate_molecule`, `interpolate_pose`, `match_molecules`, `match_molecules_vc`
- Internal DOF interpolation: `interpolate_molecule_with_internal_dofs`
- Reactive paths: `RigidGroup`, `BondChange`, `ReactivePathConfig`, `ReactivePathResult`, `interpolate_reactive_path`
- Forward construction: `MolecularTemplate`, `MolecularPose`, `ForwardConstructionContext`, `CrystalConstructionContext`, `prepare_construction_context`, `prepare_context_from_cif`, `SiteSymmetryAnalysis`, `analyze_molecular_pose`, `analyze_site_symmetry`, `construct_molecular_crystal`, `build_molecular_crystal`, `build_symmetry_complete_crystal`, `MolecularAlternative`, `AverageAtomSite`, `StatisticalDisorderModel`, `construct_statistical_disorder`, `build_orientational_disorder`, `select_ordered_realization`, `read_statistical_disorder_companion`, `intermolecular_contacts`, `PeriodicBondRecord`, `DISORDER_COMPANION_SCHEMA`, `DISORDER_CONSTRAINT_SCHEMA`, `ConstructionValidationError`, `InvalidOrientationError`, `InvalidSymmetryError`, `IncompatibleSiteError`, `IncompatibleChemistryError`, `DisorderCompanionLossError`
  - Declared space-group number, Hermann–Mauguin symbol, Hall symbol, setting, origin, and basis must match the complete operation set. An unlabelled closed group stays valid.
  - `chemistry_policy="enantiopure"` is checked against the full group. Allowed improper mates reassign known stereo from the typed graph and transformed coordinates; indeterminate mates stay unknown.
  - Stabilizer matching uses one coherent molecular image. The fractional near-special residual and its angstrom counterpart are reported on the site analysis and nothing is snapped.
  - Statistical companions use schema `molcrys.statistical_disorder.v1` (atoms, typed bonds, cell, operations, vacancies, and constraints). Older id-only payloads raise `DisorderCompanionLossError`. Joint constraints use `molcrys.disorder_constraints.v1`. Zero-weight states and unconditional selection of a partial occupation are rejected. `select_ordered_realization` reports realized symmetry separately from the source average.

Compatibility note: the former **create_defect_structure()** placeholder has been
removed because it returned its input unchanged. Use `generate_vacancy` for a
spatial molecular vacancy or `carve_void` for a shaped, stoichiometric void.

### `mck.analysis`
Analysis workflows and selected re-exports. Interaction-specific exports are listed under `mck.analysis.interactions`.

- Facets/shape: `BFDHFacetInfo`, `enumerate_bfdh_facets` (accepts explicit parent symmetry), `enumerate_low_index_millers`, `classify_shell`, `cshm`, `topology_signature`
- Chemistry/formula/charge: `ChemicalEnvironment`, `StoichiometryAnalyzer`, `FormulaUnitMember`, `FormulaUnitSelection`, `Fragment`, `parse_moiety_string`, `match_molecule_to_fragment`, `heavy_signature`, `MolChargeResult`, `assign_mol_formal_charges`, `compute_topo_signature`
- Packing/polyhedra: `find_polyhedra`, `detect_coordination_number`, `detect_prism_vs_antiprism`, `angular_rmsd_vs_ideals`, `compute_angular_signature`, `hull_encloses_center`, `planarity_analysis`, `DEFAULT_POLYHEDRON_SEARCH_CUTOFF`, `DEFAULT_MOLECULAR_SEARCH_CUTOFF`, `DEFAULT_CENTROID_OFFSET_FRAC`
  - Molecule-level shells default to the major disorder policy for the centre: alternative orientations of one site are not merged into the moiety. Shell vertices use the major ligand component under every policy. Pass `disorder_policy="exclusive"` for one shell per central orientation. `disorder_policy="as_refined"` keeps every refined central site. Ordered structures are unchanged. Ties prefer the greater component label, then the smaller atom index. The policy names live in `mck.analysis.disorder`.
- Ring conformation: `PuckeringCoordinates`, `RingSystem`, `RingConformationError`, `RingCycleLimitError`, `InvalidRingOrderError`, `DegenerateRingGeometryError`, `puckering_coordinates`, `reconstruct_z_from_modes`, `find_ring_systems`
- Volume/boundary: `calculate_atomic_volumes`, `calculate_total_volume`, `calculate_accessible_boundary`, `min_distance_to_boundary`
- Sanity checks: `sanity_check`, `SanityReport`, `CheckResult`, `check_hard_clash`, `check_intermolecular_clash`, `check_isolated_atoms`, `check_hydrogen_presence`, `check_formula_consistency`, `check_bond_distances`, `check_topology_preservation`; `sanity_check` accepts `profile="complete-crystal"` or `profile="fragment"`, and skipped checks are reported with `CheckResult.status == "skipped"`.
- Cell integrity: `CellIntegrityReport`, `check_cell_integrity`
- Periodic bundle validation: `validate_periodic_bundle`
- Site symmetry: `SiteSymmetryAnalysis`, `analyze_molecular_pose`, `analyze_site_symmetry`
- Structure summary: `summarize_structure`

### `mck.analysis.volume`
Van der Waals volume estimation and solvent-accessible boundary computation.

- `calculate_atomic_volumes(atoms, radii_type="vdw")` — per-atom spherical volumes (4/3 π r³) using VdW or covalent radii.
- `calculate_total_volume(atoms, radii_type="vdw", overlap_correction=False)` — total volume as simple sum or overlap-corrected (3-D occupancy grid, voxel_size=0.2 Å).
- `calculate_accessible_boundary(atoms, probe_radius=1.4, radii_type="vdw", n_sphere_points=50)` — Shrake-Rupley style surface point generation; returns (M, 3) Cartesian array of solvent-accessible boundary points.
- `min_distance_to_boundary(new_positions, boundary_points, lattice=None, pbc=None)` — minimum distance from query positions to boundary; supports periodic boundary conditions via minimum image convention.

### `mck.analysis.disorder`
Disorder metadata, solving, and ordered-replica generation.

- `DisorderInfo`, `DisorderSolver`, `generate_ordered_replicas_from_disordered_sites`, `is_minor_site`, `DisorderProvenance`
- `DEFAULT_DISORDER_POLICY`, `DISORDER_POLICIES`, `DisorderRealization`, `disorder_realizations`, `normalize_disorder_policy`
  - major keeps the highest-occupancy component of each mutually exclusive family and is the default for shell centres. exclusive returns one realization per alternative orientation (other families stay on their major component). as_refined keeps every refined site and is the default for identify_molecules, identify_molecule_indices, and structure readers, so identification does not drop atoms. Labelled assemblies are exclusive even when groups are far apart. Unlabelled PART labels are split into local clusters first; only clusters that approach each other compete, so a distant site keeps its own majority. Occupancy-only classes compete when their values sum to at most one plus a small tolerance and they cannot share one shell: a very short contact, an impossible bond angle at a shared ordered atom, or a union that would over-coordinate that atom. Equal occupancies are split into whole orientations before the tie-break (greater label, then smaller atom index). Shell vertices use that major component at both atom and molecule level, including an unlabelled minor ligand.
- Warning category: `UnresolvedDisorderWarning`
- `DisorderInfo` carries CIF-derived metadata including the chemical formula moiety and Z value for stoichiometry validation during disorder resolution.

### `mck.analysis.interactions`
Weak-interaction detection plus continuous scoring. Raw detectors return records; `interaction_profile` summarizes counts and scores.

- `interaction_profile` aggregates three interaction families: hydrogen bonds, halogen bonds, and pi-stacking (parallel + T-shape). C-H···π is subsumed by T-shape pi-stacking; H···H close contacts are excluded as packing artifacts. The standalone detectors `find_ch_pi` and `find_h_h_contacts` remain importable.
- Pi-stacking uses subtype-specific geometry: parallel stacking filters by interplane distance h; T-shape uses centroid distance d with a wider cutoff and scoring center.  T-shape records include an approach distance field — the minimum distance from either ring's edge to the other ring's plane (negative = stem penetration, absent for parallel subtypes).
- Base/local identity: `AtomRef`, `RingRef`, `BaseInteraction`, `build_crystal_atom_offsets`, `AtomLocalGeometry`, `LocalGeometry`, `LocalGeometryCache`, `RingGeometry`, `ChemicalIdentity`, `ChemicalIdentityCache`. `RingGeometry.cycle_atom_indices` preserves a deterministic edge-connected traversal for rendering.
- Detectors: `HydrogenBond`, `HydrogenBondCriteria`, `find_hydrogen_bonds`, `HalogenBond`, `HalogenBondCriteria`, `find_halogen_bonds`, `PiStacking`, `PiStackingCriteria`, `PiStackingSubtype`, `find_pi_stacking`, `find_pi_stacks`, `CHPiInteraction`, `CHPiInteractionCriteria`, `find_ch_pi`, `find_ch_pi_interactions`, `HHContact`, `HHContactCriteria`, `find_h_h_contacts`, `get_bonding_threshold`
- Scoring/profile: `InteractionProfile`, `InteractionScoreSummary`, `interaction_profile`, `ScoringParams`, `DEFAULT_SCORING_PARAMS`, `composite_score`, `gaussian_kernel`, `lorentzian_kernel`, `normalized_vdw_distance`, `scaled_cutoff`, `vdw_radius_sum`

### `mck.constants`
Element data and bond-detection thresholds.

- Data: `ATOMIC_MASSES`, `ATOMIC_RADII`, `VDW_RADII`, `METAL_ELEMENTS`
- Thresholds: `METAL_THRESHOLD_FACTOR`, `NON_METAL_THRESHOLD_FACTOR`, `METAL_NON_METAL_THRESHOLD_FACTOR`, `DEFAULT_NEIGHBOR_CUTOFF`
- Config: `BOND_ROTATION_AXIS_TOLERANCE`, `BFDH_GEOMETRY_TOLERANCE`, `PERIODIC_NEIGHBOR_TOLERANCE_A`, `PARTIAL_OCCUPANCY_TOLERANCE`, `MIN_PERIODIC_CELL_VOLUME_A3`

### `mck.utils`
Geometry, rigid-body math helpers, and graph utilities.

- Graph: `graph_invariant`
- Coordinates/PBC: `frac_to_cart`, `cart_to_frac`, `minimum_image_distance`, `minimum_image_vector`, `unwrap_positions_along_bonds`, `volume_of_cell`
- Fractional affine: `apply_fractional_affine`, `fractional_linear_to_cartesian`
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
