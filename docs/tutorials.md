# MolCrysKit Tutorials

## Hydrogen Completion

MolCrysKit provides functionality to add hydrogen atoms to molecular crystals based on geometric rules and chemical constraints. This is particularly useful for generating complete structures from X-ray diffraction data, which often does not resolve hydrogen positions.

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
from molcrys_kit.operations import add_hydrogens

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Add hydrogens with default rules
hydrogenated_crystal = add_hydrogens(crystal)

print(f"Original crystal has {len(crystal)} atoms")
print(f"Hydrogenated crystal has {len(hydrogenated_crystal)} atoms")

# Add hydrogens with custom rules
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

# Add hydrogens with custom rules and bond lengths
hydrogenated_crystal = add_hydrogens(
    crystal,
    rules=custom_rules,
    bond_lengths=custom_bond_lengths
)

# Save the hydrogenated crystal to a CIF file
write_mol_crystal(hydrogenated_crystal, 'hydrogenated_bulk.cif')
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
from ase.io import write
from molcrys_kit.operations import generate_topological_slab

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Generate a surface slab with specific Miller indices
slab = generate_topological_slab(
    crystal=crystal,
    miller_indices=(1, 1, 0),  # Miller indices of the surface
    layers=3,                  # Number of layers in the slab
    vacuum=10.0                # Vacuum thickness in Angstroms
)

print(f"Generated slab with {len(slab.molecules)} molecules")

# Convert the slab to ASE Atoms object using the to_ase method
slab_atoms_obj = slab.to_ase()

# Save the generated slab to a CIF file
write_mol_crystal(slab_atoms_obj, 'slab.cif')
```

## Defect Engineering

Placeholder section for defect engineering functionality covering vacancy generation logic found in [molcrys_kit/operations/defects.py](molcrys_kit/operations/defects.py). This includes the [VacancyGenerator](molcrys_kit/operations/defects.py#L21-L178) class and the public API function [generate_vacancy](molcrys_kit/operations/defects.py#L181-L297) which enable the systematic removal of specific molecular clusters based on spatial relationships. The implementation considers stoichiometric constraints and preserves the overall molecular crystal structure while introducing controlled defects.