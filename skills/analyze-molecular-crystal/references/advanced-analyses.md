# Advanced Public API Analyses

Use only when the CLI page does not cover the requested result.

## Topology-aware stoichiometry

```python
from molcrys_kit.analysis import StoichiometryAnalyzer
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
analyzer = StoichiometryAnalyzer(crystal)
print(analyzer.species_map)
print(analyzer.get_simplest_unit())
```

Report unit-cell populations and the GCD-reduced ratio. Solvent matching is
heuristic.

## Formal charges

```python
from molcrys_kit.analysis import assign_mol_formal_charges
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
charges = assign_mol_formal_charges(crystal, mol_charge_map={"NH4": 1, "Cl": -1})
for signature, result in charges.items():
    print(signature, result.formula, result.formal_charge, result.source)
```

Report source as `user_map`, `auto_guess`, or `none`; verify cell
electroneutrality independently.

## Volume and accessible boundary

```python
from molcrys_kit.analysis import calculate_accessible_boundary, calculate_total_volume
from molcrys_kit.io import read_mol_crystal

atoms = read_mol_crystal("input.cif").to_ase()
volume = calculate_total_volume(
    atoms,
    radii_type="vdw",
    overlap_correction=True,
    voxel_size=0.2,
)
boundary = calculate_accessible_boundary(
    atoms,
    probe_radius=1.4,
    radii_type="vdw",
    n_sphere_points=200,
)
```

The boundary calculation is nonperiodic; use a finite model when periodic images
matter. Report radii, probe radius, voxel size, overlap correction, and sampling,
and converge numerical parameters for quantitative comparisons.

## Local environment

```python
from molcrys_kit.analysis import ChemicalEnvironment
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
environment = ChemicalEnvironment(crystal.molecules[0])
print(environment.rings())
print(environment.get_local_geometry_stats(0))
print(environment.compute_anion_protonation_groups())
```

Indices are molecule-local and zero-based. Aromaticity, planarity, rings, and
protonation groups are geometry-based heuristics.

## Detailed polyhedra and CShM

```python
from molcrys_kit.analysis import classify_shell, cshm, find_polyhedra
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
records = find_polyhedra(crystal, central="Zn", ligand="O", level="atom")
```

Use the selected record's centered ligand coordinates with `classify_shell` or
`cshm`. Report the reference shape, score, neighbors, cutoff, and ambiguity
between close candidates.
