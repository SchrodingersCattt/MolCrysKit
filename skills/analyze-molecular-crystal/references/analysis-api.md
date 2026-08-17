# Python Analysis API

Read only for analyses without a complete CLI front end or when detailed result objects are required.

## Topology-aware stoichiometry

```python
from molcrys_kit.analysis import StoichiometryAnalyzer
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
analyzer = StoichiometryAnalyzer(crystal)
print(analyzer.species_map)
print(analyzer.get_simplest_unit())
analyzer.print_species_summary()
```

Report unit-cell populations and the GCD-reduced ratio. Solvent identification is heuristic.

## Volume and accessible boundary

```python
from molcrys_kit.analysis import calculate_accessible_boundary, calculate_total_volume
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
atoms = crystal.to_ase()
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

Accessible-boundary calculation is non-periodic; extract a molecule or finite cluster first when periodic images matter. Converge voxel and sampling parameters.

## Molecular formal charge

```python
from molcrys_kit.analysis import assign_mol_formal_charges
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
charges = assign_mol_formal_charges(
    crystal,
    mol_charge_map={"NH4": 1, "Cl": -1},
)
for signature, result in charges.items():
    print(signature, result.formula, result.formal_charge, result.source)
```

Report `source`: `user_map`, `auto_guess`, or `none`. Check cell electroneutrality independently.

## Local chemical environment

```python
from molcrys_kit.analysis import ChemicalEnvironment
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
environment = ChemicalEnvironment(crystal.molecules[0])
print(environment.rings())
print(environment.get_local_geometry_stats(0))
print(environment.detect_ring_info(0))
print(environment.compute_anion_protonation_groups())
```

Indices are molecule-local and zero-based. Aromaticity, planarity, and anion-group results are geometry-based heuristics.

## Detailed polyhedra and CShM

```python
from molcrys_kit.analysis import classify_shell, cshm, find_polyhedra
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
records = find_polyhedra(crystal, central="Zn", ligand="O", level="atom")
# Extract the centered ligand-coordinate array from the selected record,
# then call classify_shell or compare against a chosen ideal with cshm.
```

Inspect the installed result schema before extracting coordinate arrays. Report coordination number, selected neighbors, cutoff, reference shape, CShM, and ambiguity between close candidates.
