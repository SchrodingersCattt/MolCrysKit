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

Regression cases in `tests/data/cif/synthetic_*assemblies.cif` and
`synthetic_water_orientation.cif` are synthetic. They check independent
assemblies, all four choices in a two-by-two synthetic model, equivalent
group-number mappings, molecular formula counts,
strict-mode failure, label aliases, and compatibility with legacy fixtures.
No database coordinates or private experimental metadata are needed.