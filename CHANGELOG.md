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
- Occupancy-only atoms are alternatives when their values sum to at most one
  plus a small refinement tolerance, and they cannot share one shell: the
  union would over-coordinate a shared ordered atom, a bond angle is tighter
  than a simultaneous pair, or a contact lies well below a covalent bond.
  Bonded chains of unequal partial occupancy stay intact.
- Equal occupancies are split into whole orientations before selection. Ties
  prefer the larger occupancy sum, then more atoms, then the greater component
  label, then the smaller atom index.
- Atom-level shell vertices follow the same major-ligand rule, so an
  unlabelled minor ligand atom is not an extra vertex.
