# Changelog

## Unreleased

### Added

- Disorder policy for molecule identification and site-centred ligand-shell
  searches. `major` (the default for `identify_molecules` and
  `find_polyhedra`) keeps the highest-occupancy orientation of each mutually
  exclusive disorder component instead of merging alternative orientations
  into one molecule. `exclusive` returns one realization, or one shell, per
  orientation. `as_refined` keeps every refined site and is the default for
  `read_mol_crystal`, `MolecularCrystal.from_cif`, and
  `MolecularCrystal.from_ase`. A shell search still applies `major` unless
  the caller overrides it, so a default structure read can be used directly.
  Ordered structures are unchanged.
