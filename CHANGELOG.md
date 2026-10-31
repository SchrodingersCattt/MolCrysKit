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
  toward that shell. At or below that sum a tight angle splits the pair, even
  when the atoms are within a bond; above it they coexist. Bridging metals and
  a bonded ring whose occupancies sum to more than 1.10 stay intact. An
  ordinary split substituent (sum at most 1.10) keeps one position.
- Equal occupancies are split into whole orientations with one element
  composition. Radii are distance/bond-cutoff (standard deviation at most
  0.08), so a mixed hydrogen/chlorine shell is one orientation and an all-hydrogen
  subset of that shell is not. A 0.2 Å shifted copy stays within the 10° angle
  slack. A wider shift keeps the one regular orientation and warns. A class
  that cannot be split at all is kept whole and raises
  `UnresolvedDisorderWarning`. A translated duplicate with no ordered anchor is
  split; any other unanchored class warns.
- Negative PART labels, including one group inside a labelled assembly, are
  split into orientations around the ordered centre. Partial ligands of an
  ordered metal are split the same way. Several groups in one labelled
  assembly stay exact.
- Atom-level distances use the periodic minimum image, and shell vertices
  follow the major-ligand rule for unlabelled as well as labelled disorder.
- Bond cutoffs are cached per element pair and neighbour searches use one
  spatial query, so occupancy grouping on a long chain stays near-linear.
- Defaults in `DISORDER_CONFIG`: `OCCUPANCY_SUM_TOLERANCE` 0.10,
  `OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG` 80,
  `OCCUPANCY_ORIENTATION_STD_SLACK_DEG` 10,
  `OCCUPANCY_ORIENTATION_STD_GAP_DEG` 6,
  `OCCUPANCY_ORIENTATION_RADIUS_STD_FRACTION` 0.08,
  `OCCUPANCY_MAX_ORIENTATION_COMBINATIONS` 20000.
