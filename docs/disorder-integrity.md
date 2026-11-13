# Disorder export and ordered-structure integrity

A CIF atom-site disorder group is scoped to its assembly. CCDC group
identifiers can be zero, absent, or reused, and an alternative site label may
appear with a trailing question mark. The repository and skill exporters assign
positive group identifiers within each assembly and preserve exact labels
before considering an alias. Conflicting exact labels or incomplete disorder
hierarchies are rejected; ungrouped partially occupied atoms are reported for
review because they can be vacancies rather than errors.

## Independent assemblies

The legacy disorder solver uses geometric heuristics that also accommodate CIFs
where assembly labels name alternative fragments. Equivalent numeric group
labels from different multi-group assemblies can therefore change a result in
legacy mode. If the source establishes that such assemblies are independent,
set `independent_assemblies=True` or pass `--independent-assemblies`. In
this mode, different multi-group assemblies are not made mutually exclusive
merely because their local group numbers differ; geometric overlaps remain
checked. This does not claim that single-group or ambiguous assemblies are
independent. Legacy mode warns when multiple multi-group assemblies are present,
but absence of that warning is not proof of correctness.

The flag is accepted by `mck operate disorder`, `mck operate disorder-supercell`,
and by `read_mol_crystal(..., resolve_disorder=True, independent_assemblies=True)`.
CLI loads that only set `--resolve-disorder` (`io info`, `io molecules`,
`io extract-molecule`, `io convert`, `operate nanocluster`, `operate void`,
`operate supercell`) accept the same flag and forward it. The default remains
false. Exporter group numbers stay local to each assembly (numbered from 1
within that assembly) and are not remapped onto one global sequence.

## Validate output

`--strict-formula` (or `strict_formula=True`) requires a parseable and
verified `_chemical_formula_moiety` and `_cell_formula_units_Z`; it fails
when generated candidates do not match. The default emits a warning and keeps
legacy best-effort behavior. The same default applies when
`read_mol_crystal` or a `--resolve-disorder` load omits `strict_formula`.
`disorder-supercell` forwards the flag into the replica resolver before
cells are assembled. A declared formula can itself be incomplete,
incorrect, or describe a framework, so strict mode must be used only with a
verified reference. Chemical formula alone does not establish molecular graph,
protonation, charge, or force-field suitability. Validate these independently.

The formula check reads the same metadata on every install profile. Both the
pymatgen reader and the lightweight reader used on constrained targets
(`molcrys_kit.compat.cifparse`, no pymatgen, SciPy, ASE or spglib) take
`_chemical_formula_moiety` and `_cell_formula_units_Z` from the first parsed
data block, so seeded `random` and `enumerate` candidates are identical on
phone and desktop. Without that metadata the filter is skipped, which is why
a stripped-down reader once returned fewer candidates. Rules shared by both
readers:

- CIF data names are case-insensitive: `_Cell_Formula_Units_z` is the same
  tag as `_cell_formula_units_Z`. An exact spelling wins over a differently
  cased duplicate.
- Quoted and semicolon (multi-line) values are accepted. Lines of a text field
  are joined with spaces.
- A missing moiety, `?` or `.` means "no moiety" (`formula_moiety is None`),
  not the literal text `?`.
- Z must be a positive integer, optionally written as `4.0` or with a
  standard uncertainty such as `4(1)`. Missing, `?`/`.`, non-numeric, zero,
  negative and non-integral values (for example the refinement artefact
  `0.99984`) give `z_value is None` and skip the filter; they are never
  truncated to a count. `strict_formula=True` then fails because the
  declaration is not usable.

Random sampling can return the same candidate more than once. That is part of
the contract and is preserved; use `enumerate` for distinct alternatives.

Regression cases in `tests/data/cif/synthetic_*assemblies.cif` and
`synthetic_water_orientation.cif` are synthetic. They check independent
assemblies, all four choices in a two-by-two synthetic model, equivalent
group-number mappings, molecular formula counts,
strict-mode failure, label aliases, and compatibility with legacy fixtures.
No database coordinates or private experimental metadata are needed.

## Symmetry operation loops

`read_cif_symmetry` and `scan_cif_disorder` keep an explicit
`_space_group_symop_operation_xyz` or `_symmetry_equiv_pos_as_xyz` loop when
that loop contains more than the identity. The Hermann–Mauguin name and the
IT number are labels on the loop. They supply operations only when a non-P1
CIF has no operation beyond identity, which emits
`SymmetryAutoExpandedWarning`. A longer loop is not rewritten when it is
unclosed, when its translations were edited by hand, or when it is not the
named space group. The operation list is the symmetry.

Disorder resolution expands those operations before it chooses alternatives.
A wrong loop therefore moves special-position molecules. The usual cage on a
threefold axis is the case to check: the asymmetric unit needs the axial
atoms once, at occupancy 1, and one bridge per orientation. The threefold
operation builds the other bridges. If the written operations do not fix that
axial site, each nitrogen is copied as a general position, the bridges do not
close, and the ordered replica is a P1 packing of overlapping fragments.
`write_cif` writes a resolved replica as P1 by default
(`representation="p1"`). That P1 cell is the expanded ordered model.

Build or repair the loop before resolving disorder:

- Copy the parent refinement's operation list unchanged, or omit the loop
  and keep `_space_group_IT_number` so identity-only input expands from the
  declared group.
- A site whose parent `_atom_site_site_symmetry_order` is 3 must be invariant
  under two non-identity operations of the loop. Stabilizer order 1 means
  the loop does not contain that axis.
- Atoms shared by every orientation stay at occupancy 1 and outside the
  disorder assembly. Atoms that differ between orientations are the partial
  sites, and the occupancies inside one assembly sum to 1. Do not also list
  the symmetry copies of those bridges in the asymmetric unit.
- `strict_formula=True` rejects candidates that do not match
  `_chemical_formula_moiety` times `_cell_formula_units_Z`. The default warns
  and still returns a best-effort crystal. Write the moiety as discrete
  residues, not as a copy of `_chemical_formula_sum`. The check compares
  element totals and, when every residue count is integral, per-molecule
  formulas. A connected framework is one molecule, so a residue list can
  still fail that second stage when the residues share a bonding graph.
  Inspect element totals and molecule formulas separately.
- Independent disorders that reuse local group numbers (an anion assembly
  and a molecular rotamer, both numbered 1 and 2) need
  `independent_assemblies=True`. Legacy mode can tie those assemblies
  together by group number and warns when several multi-group assemblies
  are present. `coupled=True` locks symmetry copies of one assembly to the
  same orientation, which is a single-domain replica. `coupled=False` lets
  each copy choose separately.