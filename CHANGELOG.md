# Changelog

## Unreleased

### Added

- Disorder policy for site-centred ligand shells and, when requested, for
  molecule identification. `disorder_realizations` exposes the same rules.
  `find_polyhedra` records `disorder_policy` and `disorder_component_ids`.
  `center_molecule_index` and `shell_molecule_indices` are indices into the
  parent molecule list (a realization does not invent a new molecule index).

### Changed

- `find_polyhedra` defaults to `major` for the central site. Alternative
  orientations are not fused into the moiety or the centre, so a disordered
  cation can match a real ligand shell. Ordered structures are unchanged.
  Callers who want the previous unfiltered centre pass
  `disorder_policy="as_refined"`.
- Shell vertices use the major ligand component under every policy, including
  `exclusive` and `as_refined`. A minor ligand position is not an extra vertex.
- `identify_molecules` and `identify_molecule_indices` accept `disorder_policy`
  but still default to `as_refined`. They do not drop refined atoms unless the
  caller passes `major` or `exclusive`. `read_mol_crystal`,
  `MolecularCrystal.from_cif`, and `MolecularCrystal.from_ase` keep that same
  default.

### Notes

- Labelled disorder assemblies are exclusive as a whole. Unlabelled PART labels
  are split into local clusters first, so a distant site that reuses a label
  keeps its own majority orientation.
- Occupancy-only atoms are alternatives when their values sum to at most 1.10
  and they cannot share one covalent shell. Metal–ligand contacts do not count
  toward that shell, and a tight angle does not split two partial atoms that
  are bonded to each other. Bonded chains and coordinated metals therefore
  stay intact.
- Equal occupancies are split into whole orientations. A shell is one radius
  (standard deviation at most 0.13 Å) whose angle spread stays in the best
  cluster (gap 6°, at most 10° above the best). The tie prefers the tighter
  shell, then the smaller radius, then geometry, so it does not follow atom
  order. A class that cannot be split is kept whole and raises
  `UnresolvedDisorderWarning`.
- Negative PART labels on a symmetry-generated substituent shell are split
  into orientations around the ordered centre.
- Atom-level distances use the periodic minimum image, and shell vertices
  follow the major-ligand rule for unlabelled as well as labelled disorder.
- Defaults in `DISORDER_CONFIG`: `OCCUPANCY_SUM_TOLERANCE` 0.10,
  `OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG` 80,
  `OCCUPANCY_ORIENTATION_STD_SLACK_DEG` 10,
  `OCCUPANCY_ORIENTATION_STD_GAP_DEG` 6,
  `OCCUPANCY_ORIENTATION_RADIUS_STD_A` 0.13,
  `OCCUPANCY_MAX_ORIENTATION_COMBINATIONS` 20000.
