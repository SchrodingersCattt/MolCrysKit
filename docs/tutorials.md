# MolCrysKit Tutorials

## Hydrogen Completion

MolCrysKit provides functionality to add hydrogen atoms to molecular crystals based on geometric rules and chemical constraints. This is particularly useful for generating complete structures from X-ray diffraction data, which often does not resolve hydrogen positions.

**Note:** `add_hydrogens` uses a whitelist-only policy: you must pass `target_elements` (e.g. `["O","N","C"]`). If `target_elements` is `None` or empty, no hydrogens are added.

### Key Features:
- Automatic determination of hydrogen atom positions based on coordination geometry
- Support for common coordination geometries (tetrahedral, trigonal pyramidal, bent, etc.)
- Customizable rules for specific atom types
- Configurable bond lengths for different atom pairs
- Preservation of molecular topology during hydrogen completion

### Best Practices:
- Verify that the crystal structure is of sufficient quality for hydrogen addition
- Consider using custom rules for specific chemical environments
- Validate the hydrogen-bonding network after hydrogen completion
- Use appropriate bond lengths for your specific system

### Basic Usage:
```python
from molcrys_kit import read_mol_crystal
from molcrys_kit.operations import add_hydrogens
from molcrys_kit.io.output import write_cif

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Add hydrogens with default rules (target_elements is required: only these elements get hydrogens added)
hydrogenated_crystal = add_hydrogens(crystal, target_elements=["O", "N", "C"])

print(f"Original crystal has {crystal.get_total_nodes()} atoms")
print(f"Hydrogenated crystal has {hydrogenated_crystal.get_total_nodes()} atoms")

# Add hydrogens with custom rules and target elements
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
    bond_lengths=custom_bond_lengths
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
