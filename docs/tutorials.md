# MolCrysKit Tutorials

## Hydrogen Completion

MolCrysKit provides functionality to add hydrogen atoms to molecular crystals based on geometric rules, chemical constraints, and CIF formula metadata when available. This is particularly useful for generating complete structures from X-ray diffraction data, which often does not resolve hydrogen positions.

**Note:** `target_elements` is optional. When it is `None` (the default), all heavy atoms are processed. When provided, it acts as a whitelist (for example, `["O", "N", "C"]`).

### Key Features:
- Automatic determination of hydrogen atom positions based on coordination geometry
- Support for common coordination geometries (tetrahedral, trigonal pyramidal, bent, etc.)
- Customizable rules for specific atom types
- Configurable bond lengths for different atom pairs
- Preservation of molecular topology during hydrogen completion
- CIF `_chemical_formula_moiety` support: when a parseable moiety formula matches a molecular fragment, its H count is used to correct the heuristic plan before H atoms are placed

### Best Practices:
- Verify that the crystal structure is of sufficient quality for hydrogen addition
- Consider using custom rules for specific chemical environments
- Use `use_formula_moiety=False` only when you explicitly want the older heuristic-only behavior
- Validate the hydrogen-bonding network after hydrogen completion
- Use appropriate bond lengths for your specific system

### Basic Usage:
```python
from molcrys_kit import read_mol_crystal
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.io.output import write_cif

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Add hydrogens with default rules.
# If the CIF contains _chemical_formula_moiety, it is used to correct
# per-fragment H counts; geometry/placement still comes from heuristics.
hydrogenated_crystal = add_hydrogens(crystal)

print(f"Original crystal has {crystal.get_total_nodes()} atoms")
print(f"Hydrogenated crystal has {hydrogenated_crystal.get_total_nodes()} atoms")

# Add hydrogens with custom rules and an element whitelist
custom_rules = [
    {
        "symbol": "N",              # Nitrogen atoms
        "geometry": "trigonal_pyramidal",  # Geometry to use
        "target_coordination": 3,   # Target coordination number
    },
    {
        "symbol": "O",              # Oxygen atoms
        "geometry": "bent",         # Geometry to use
        "target_coordination": 2,   # Target coordination number
    }
]

# Custom bond lengths for specific atom pairs
custom_bond_lengths = {
    "O-H": 0.96,  # Bond length in Angstroms
    "N-H": 1.01,
    "C-H": 1.09,
}

hydrogenated_crystal = add_hydrogens(
    crystal,
    target_elements=["O", "N", "C"],
    rules=custom_rules,
    bond_lengths=custom_bond_lengths,
    use_formula_moiety=True,  # default; set False for heuristic-only mode
)

# Save the hydrogenated crystal to a CIF file
write_cif(hydrogenated_crystal, 'hydrogenated_bulk.cif')
```

## Surface Generation (Slab Creation)

MolCrysKit provides functionality for generating surface slabs from molecular crystals while preserving molecular topology. The surface generation ensures that intramolecular bonds are not broken during the cutting process, treating molecules as rigid units.

### Key Features:
- Miller index specification for surface orientation (h, k, l)
- Preservation of molecular topology during cutting
- Adjustable number of layers and vacuum spacing
- Automatic molecular centroid-based layer assignment

### Best Practices:
- Choose Miller indices that align with crystal symmetry for optimal results
- Use sufficient vacuum spacing (typically 10-20 Å) to avoid inter-slab interactions
- Consider the number of layers needed for your specific application
- Validate that molecules remain intact after slab generation

### Basic Usage:
```python
from molcrys_kit import read_mol_crystal
from molcrys_kit.operations import generate_topological_slab
from molcrys_kit.io.output import write_cif

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Generate a surface slab with specific Miller indices (layers or min_thickness required)
slab = generate_topological_slab(
    crystal=crystal,
    miller_indices=(1, 1, 0),  # Miller indices of the surface
    layers=3,                  # Number of layers in the slab
    vacuum=10.0                # Vacuum thickness in Angstroms
)

print(f"Generated slab with {len(slab.molecules)} molecules")

# Save the slab as a MolecularCrystal to a CIF file
write_cif(slab, 'slab.cif')

# To obtain an ASE Atoms object (e.g. for other workflows), use slab.to_ase()
slab_atoms_obj = slab.to_ase()
```

## Defect Engineering

Placeholder section for defect engineering functionality covering vacancy generation logic found in [molcrys_kit/operations/defects.py](../molcrys_kit/operations/defects.py). This includes the [VacancyGenerator](../molcrys_kit/operations/defects.py) class and the public API function [generate_vacancy](../molcrys_kit/operations/defects.py) which enable the systematic removal of specific molecular clusters based on spatial relationships. The implementation considers stoichiometric constraints and preserves the overall molecular crystal structure while introducing controlled defects.

## Surface Termination Enumeration and Tasker Analysis

MolCrysKit supports enumerating topologically distinct surface terminations for
any Miller plane, classifying each termination according to Tasker's theory
(adapted for molecular/organic-inorganic hybrid crystals), and building
topology-preserving slabs for the selected terminations.

### Key Concepts

- **Termination** – a specific way to cut the crystal perpendicular to a Miller
  plane, defined by a fractional *shift* along the stacking direction.  Different
  shifts expose different molecular layers at the surface.
- **Tasker Classification (molecular adaptation)**
  - `TypeI_like`: all surface layers are charge-neutral (e.g. pure organic
    molecular crystals).
  - `TypeII_like`: layers carry charge but the repeat unit has zero net dipole.
  - `TypeIII_like`: polar surface with a non-zero dipole moment per unit area.
- **Tasker-preferred**: `TypeI_like` and `TypeII_like` non-polar terminations are
  preferred (electrostatically stable).

### Charge Assignment Strategy (Hybrid)

Molecular formal charges are determined by a three-level fallback:

1. **user_map** – explicit `mol_charge_map={"formula": charge}` argument.
2. **auto_guess** – pymatgen `BVAnalyzer` bond-valence oxidation-state sum.
3. **none** – zero with a `UserWarning`; Tasker analysis degrades to
   topology-only ordering.

### Example 1: Neutral Organic Crystal (Fast Path)

```python
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.operations import enumerate_terminations, generate_slabs_with_terminations
from molcrys_kit.io.output import write_cif

crystal = read_mol_crystal("examples/Acetaminophen_HXACAN.cif")

# Enumerate unique terminations for (1, 0, 0)
term_infos = enumerate_terminations(crystal, miller_index=(1, 0, 0))
for ti in term_infos:
    print(
        f"[{ti.termination_index}] shift={ti.shift:.4f}  "
        f"type={ti.tasker_type}  preferred={ti.is_tasker_preferred}"
    )
# For a neutral organic crystal all terminations are TypeI_like (fast path).

# Build slabs – default returns Tasker-preferred terminations
results = generate_slabs_with_terminations(
    crystal,
    miller_index=(1, 0, 0),
    min_slab_size=12.0,
    min_vacuum_size=15.0,
)
for slab, info in results:
    write_cif(slab, f"slab_term{info.termination_index}.cif",
              metadata={"termination_info": info})
```

### Example 2: Salt-Type Crystal with mol_charge_map

```python
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.operations import generate_slabs_with_terminations
from molcrys_kit.io.output import write_cif

crystal = read_mol_crystal("examples/salt_crystal.cif")

# Provide explicit charges for cation and anion building blocks
results = generate_slabs_with_terminations(
    crystal,
    miller_index=(0, 0, 1),
    min_slab_size=15.0,
    min_vacuum_size=15.0,
    mol_charge_map={"C6H8N": +1, "Cl": -1},   # protonated amine + chloride
    term_selection="all",                        # inspect all terminations
)
for slab, info in results:
    print(
        f"Termination {info.termination_index}: {info.tasker_type}, "
        f"dipole/area={info.dipole_per_area:.4f} e·Å/Å², "
        f"preferred={info.is_tasker_preferred}"
    )
    write_cif(slab, f"slab_term{info.termination_index}.cif",
              metadata={"termination_info": info})
```

### CIF Metadata Fields

When `metadata={"termination_info": info}` is passed to `write_cif`, the
following custom fields are written to the CIF header:

| CIF field | Description |
|---|---|
| `_molcrys_termination_shift` | Fractional shift along the stacking direction |
| `_molcrys_termination_index` | Zero-based termination index |
| `_molcrys_tasker_type` | Tasker classification string |
| `_molcrys_tasker_polar` | Boolean polarity flag |
| `_molcrys_tasker_dipole_per_area` | Dipole per surface area (e·Å/Å²) |
| `_molcrys_charge_source` | Origin of charge data |

---

## Molecule Manipulation

MolCrysKit provides targeted operations on specific molecules within a molecular crystal, including **translation**, **rotation**, and **replacement** (with automatic clash detection and resolution).

All manipulation operations return a **new** `MolecularCrystal` — the original crystal is never mutated.

### Key Features:
- Select molecules by index or by species ID (via stoichiometry analysis)
- Translate molecules by Cartesian or fractional vectors
- Rotate molecules around their centre of mass or geometric centroid
- Replace molecules with new ones loaded from XYZ files
- Automatic clash detection: checks minimum atom–atom distance to host framework
- Automatic clash resolution: random rotations to find a clash-free orientation
- Raises `MoleculeClashError` when clashes cannot be resolved

### Basic Usage — Functional API:

```python
from molcrys_kit import read_mol_crystal
from molcrys_kit.operations.molecule_manipulation import (
    translate_molecule,
    rotate_molecule,
    replace_molecule,
    MoleculeClashError,
)

# Load a molecular crystal
crystal = read_mol_crystal("structure.cif")

# Translate molecule #0 by [1.0, 0.0, 0.0] Angstrom
new_crystal = translate_molecule(crystal, molecule_index=0, vector=[1.0, 0.0, 0.0])

# Translate using fractional coordinates
new_crystal = translate_molecule(crystal, molecule_index=0, vector=[0.1, 0.0, 0.0], fractional=True)

# Rotate molecule #0 by 45° around z-axis (pivoting at centre of mass)
new_crystal = rotate_molecule(crystal, molecule_index=0, axis=[0, 0, 1], angle=45.0)

# Rotate around centroid instead of COM
new_crystal = rotate_molecule(crystal, molecule_index=0, axis=[0, 0, 1], angle=45.0, center="centroid")

# Replace molecule #0 with a molecule from an XYZ file
try:
    new_crystal = replace_molecule(
        crystal,
        molecule_index=0,
        new_molecule="guest.xyz",     # path to XYZ file
        clash_threshold=1.0,           # minimum acceptable distance (Å)
        max_rotation_attempts=100,     # max random rotations to resolve clashes
    )
except MoleculeClashError as e:
    print(f"Could not place molecule: {e}")
```

### Class-based API with Species Selection:

```python
from molcrys_kit.operations.molecule_manipulation import MoleculeManipulator

crystal = read_mol_crystal("structure.cif")
manip = MoleculeManipulator(crystal)

# Select molecules by species ID (from stoichiometry analysis)
water_indices = manip.select_molecules(species_id="H2O_1")
print(f"Found {len(water_indices)} water molecules: {water_indices}")

# Operate on a specific molecule
new_crystal = manip.translate_molecule(water_indices[0], vector=[0.5, 0.0, 0.0])

# Chain operations (create a new manipulator for each step)
manip2 = MoleculeManipulator(new_crystal)
final_crystal = manip2.rotate_molecule(water_indices[0], axis=[0, 0, 1], angle=30.0)
```

### Reading XYZ Files:

```python
from molcrys_kit.io.xyz import read_xyz

# Load a molecule from an XYZ file
molecule = read_xyz("guest_molecule.xyz")
print(molecule.get_chemical_formula())

# Use it for replacement
new_crystal = replace_molecule(crystal, molecule_index=0, new_molecule=molecule)
```

### Replacement Clash Detection:

When replacing a molecule, the system automatically checks whether the replacement molecule's atoms are too close to the host framework (all other molecules in the crystal):

1. The replacement molecule's COM is aligned to the original molecule's COM
2. Minimum pairwise distance to all host atoms is computed
3. If any distance < `clash_threshold` (default 1.0 Å):
   - Random rotations around COM are attempted (up to `max_rotation_attempts`)
   - Each rotation preserves the COM position
   - If a clash-free orientation is found, it is used
4. If all attempts fail, `MoleculeClashError` is raised with diagnostic info

### Parameters Reference:

| Function | Parameter | Default | Description |
|---|---|---|---|
| `translate_molecule` | `vector` | required | Displacement vector (3D) |
| | `fractional` | `False` | Interpret vector as fractional coords |
| `rotate_molecule` | `axis` | required | Rotation axis (3D vector) |
| | `angle` | required | Rotation angle in degrees |
| | `center` | `"com"` | Pivot: `"com"` or `"centroid"` |
| `replace_molecule` | `new_molecule` | required | XYZ file path or `CrystalMolecule` |
| | `clash_threshold` | `1.0` | Min distance (Å) to host atoms |
| | `max_rotation_attempts` | `100` | Max rotation tries for clash resolution |
| | `align_method` | `"com"` | Alignment: `"com"` or `"centroid"` |

## Carving Finite Clusters for QM

MolCrysKit can carve finite, hydrogen-capped cluster models out of any
periodic `MolecularCrystal` so the cluster is ready to feed into
Gaussian / ORCA / Psi4. The carver lives in
`molcrys_kit.operations.cluster` and emits a `CrystalCluster` (a
non-periodic `CrystalMolecule` subclass) plus a JSON sidecar that
records exactly how the cluster was constructed. The algorithm is
framework-agnostic; the system-specific parameter choices (which seed,
how many shells, which freeze layer, what literature convention) live
in a project-side recipe.

### Two modes

* **`bond_shells`** (default, production): chemistry-aware BFS from one
  or more seed atoms. The carver only cuts **single C-C bonds that lie
  outside any small ring** (rings are detected on the BFS-local subgraph
  and rejected above 8 atoms so periodic macrocyclic "topological
  rings" do not mark every linker C-C as ring-bonded). Every cut is
  capped by an H atom placed along the original bond vector at the
  element-specific X-H length looked up from
  `molcrys_kit.constants.config.BOND_LENGTHS` -- the same table that
  powers `operations.add_hydrogens` (C-H 1.09, N-H 1.01, O-H 0.96,
  S-H 1.34, P-H 1.42). Pass `cap_distance=` to force a uniform value
  for every cap instead, or `cap_bond_lengths={"C-H": 1.10, ...}` to
  tweak individual entries.
* **`rcut`** (diagnostic): radial cutoff -- keep every atom whose
  minimum-image Cartesian distance to any seed is within `rcut`
  Angstrom. Any cut that is *not* a C-C bond raises a warning, since
  this is the classical red flag for accidentally severing C-O / C-N.

### Hop budget (`n_shells`)

`n_shells` is the **number of cut-boundary layers crossed** beyond the
seed, not raw bond hops:

* `n_shells=0` -- cluster stops at the very first cuttable C-C bond.
* `n_shells=1` -- include the first linker fragment up to the next
  cuttable C-C.
* `n_shells=2` -- continue one more layer.

### Freeze convention (`freeze_shell`)

The frozen-atom set written into the sidecar tells the downstream QM
input writer which atoms to hold fixed during geometry optimisation:

* `freeze_shell=0` -- nothing frozen.
* `freeze_shell=1` -- all cap H atoms plus the kept-side atom of every
  cut.
* `freeze_shell=2` -- as `freeze_shell=1` plus one additional layer of
  heavy atoms inward.

Cite the convention appropriate to your system in
`convention_reference` (see below) so the sidecar is self-documenting.

### Seed auto-grouping (`seed_merge_radius`)

By default (`seed_merge_radius=0.0`) each resolved seed produces its
own cluster. Set it to the diameter of a multi-atom node (paddle-wheel
~3.0 Å, M3 trimer ~3.5-3.8 Å, ...) when you want one cluster per node
group.

### Metal-boundary rule

Periodic frameworks are topologically closed: two metal nodes can be
linked through a non-C-C path (e.g. M-X-X-M through a heterocyclic
linker), so a naive BFS that only cuts at C-C bonds would walk past
every other node and the "cluster" would silently become the whole
framework. By default, `bond_shells` therefore treats any bond
reaching a metal atom **outside the current seed group** as an
implicit boundary: the bond is cut and capped on the kept (ligand)
side exactly like a regular C-C cut. The parent-side atom is still
recorded in `cut_bonds`, so downstream tools can distinguish C-C cuts
from metal-boundary cuts by inspecting the dropped element. Pass
`stop_at_non_seed_metals=False` (or `--no-stop-at-non-seed-metals` on
the CLI) for diagnostic carves that should sweep through every metal.

### Programmatic example

```python
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.io.output import write_xyz_with_freeze
from molcrys_kit.operations import carve_cluster

crystal = read_mol_crystal("structure.cif")
clusters = carve_cluster(
    crystal,
    seed=17,              # global atom index, or e.g. "Zn" / "Si" / "Cu"
    mode="bond_shells",
    n_shells=1,
    freeze_shell=1,
    seed_merge_radius=3.8,
    convention_reference="DOI: 10.xxxx/yyyy (your QM-cluster recipe)",
)
for k, cluster in enumerate(clusters):
    write_xyz_with_freeze(cluster, f"cluster_{k}.xyz")
    print(cluster.provenance.kept_global_indices)
```

### CLI

```bash
python scripts/carve_cluster.py \
    --cif structure.cif \
    --seed-index 17 \
    --mode bond_shells \
    --shells 1 \
    --freeze-shell 1 \
    --seed-merge-radius 3.8 \
    --convention-reference "DOI: 10.xxxx/yyyy" \
    --out outputs/cluster
```

The script emits `outputs/cluster__group<k>.xyz` (with an extra
per-atom flag column F/C/-) and `outputs/cluster__group<k>.xyz.cluster.json`
(the full `ClusterProvenance` payload) for each seed group.

### Sidecar JSON schema

The `.cluster.json` sidecar contains the full
`molcrys_kit.analysis.cluster_provenance.ClusterProvenance` payload,
which is the canonical record for any downstream QM input writer:

| Key | Type | Meaning |
|---|---|---|
| `mode` | str | `"bond_shells"` or `"rcut"` |
| `seed_global_indices` | list[int] | Parent-atom indices used as seeds |
| `n_shells` | int or null | Cut-boundary layers crossed (bond_shells) |
| `rcut_A` | float or null | Radial cutoff in Angstrom (rcut) |
| `kept_global_indices` | list[int] | Parent-atom indices retained |
| `cut_bonds` | list[[int, int]] | (kept, dropped) parent index per cut |
| `cap_local_indices` | list[int] | Local indices of cap H in the XYZ |
| `frozen_local_indices` | list[int] | Local indices to hold fixed |
| `freeze_shell` | int | 0, 1, or 2 |
| `cap_distance_A` | float or null | Uniform override; null = per-element |
| `cap_bond_lengths_A` | dict[str, float] | Element-keyed X-H table consulted |
| `cap_distances_used_A` | list[float] | Per-cap distance actually applied |
| `seed_merge_radius_A` | float | Auto-grouping threshold |
| `parent_label` | str or null | Free-text label of the parent structure |
| `convention_reference` | str | Caller-supplied citation of the QM-cluster recipe |

### Out of scope

* No domain typing (SBU / linker / paddle-wheel / pore / channel) --
  the carver works at the atom + bond level.
* No charge / spin inference -- supply these when writing the QM input.
* No Gaussian / ORCA / Psi4 writer; the sidecar JSON is the integration
  point.
