# MolCrysKit Tutorials

## Extended XYZ Dataset Bundles

MolCrysKit can write molecular-crystal datasets as multi-frame ASE Extended XYZ
files.  Each frame is flattened from a `MolecularCrystal` while preserving the
unit cell, PBC flags, and molecule membership through the `molecule_index`
array.  Use per-frame `info` for provenance such as refcodes or motif labels.

```python
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz

refcodes = ["ABAZIN", "ABAZOT"]
crystals = [read_mol_crystal(f"cache/{refcode}.cif") for refcode in refcodes]

frame_info = [
  {
    "dataset_id": "DeepMolCryst-26",
    "refcode": refcode,
    "source_family": "weak_interactions",
    "motif": "hydrogen_bond",
    "query_id": "Q26-WI-HYDROGEN-BOND-001",
    "frame_index": i,
  }
  for i, refcode in enumerate(refcodes)
]

write_extxyz(
  crystals,
  "hydrogen_bond.extxyz",
  info=frame_info,
)

# Use index=":" for dataset bundles; the default index=None returns only the
# last frame, following ASE convention.
bundle = read_extxyz("hydrogen_bond.extxyz", index=":")
print(len(bundle), bundle[0].metadata["refcode"])
```

The same file can be consumed directly by ASE-based MLIP tooling:

```python
import ase.io

frames = ase.io.read("hydrogen_bond.extxyz", index=":", format="extxyz")
print(frames[0].info["refcode"])
```

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

## BFDH Facet Candidate Enumeration

MolCrysKit can propose model-agnostic surface candidates using the
Bravais-Friedel-Donnay-Harker (BFDH) morphology rule.  Pure BFDH ranks facets
by interplanar spacing: planes with larger `d_hkl` are assigned lower relative
growth rates and therefore higher morphological importance.  This is a fast
empirical candidate generator, not a surface-energy calculation.

### Basic Usage

```python
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.analysis import enumerate_bfdh_facets
from molcrys_kit.operations import generate_slabs_with_terminations

crystal = read_mol_crystal("bulk.cif")

# Default: low-index Miller indices up to max_index=2, with
# Donnay-Harker-style systematic-absence filtering when structure symmetry is
# available.
facets = enumerate_bfdh_facets(crystal, top_n=5, include_equivalents=True)
for facet in facets:
  print(
    facet.rank,
    facet.miller_index,
    f"d={facet.d_hkl:.3f} Å",
    f"importance={facet.relative_morphological_importance:.3f}",
  )

# Downstream slab generation remains explicit and topology-aware.
best = facets[0]
slabs = generate_slabs_with_terminations(
  crystal,
  miller_index=best.miller_index,
  term_selection="tasker_preferred",
)
```

Use `miller_indices=[...]` to rank a manually curated list instead of the
default low-index search.  Pass `extinction_filter=False` for pure Friedel
`d_hkl` ranking without systematic-absence filtering.  If desired, pass the
returned candidates to external surface-energy or adsorption workflows for post
hoc re-ranking.

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
which explicit truncation bonds, which freeze layer, what literature
convention) live in a project-side recipe.

### Two modes

* **`bond_shells`** (default, production): topology-preserving BFS from
  one or more seed atoms. By default the carver keeps the full ligand
  topology and cuts only at metal-ligand boundaries introduced by
  `stop_at_non_seed_metals=True`. Single non-ring C-C bonds are
  truncated only when the caller explicitly provides
  `cut_cc_bonds=[(i, j), ...]` using parent global atom indices; each
  requested bond is validated before carving. Every cut is capped by an
  H atom placed along the original bond vector at the element-specific
  X-H length looked up from `molcrys_kit.constants.config.BOND_LENGTHS`
  -- the same table that powers `operations.add_hydrogens` (C-H 1.09,
  N-H 1.01, O-H 0.96, S-H 1.34, P-H 1.42). Pass `cap_distance=` to
  force a uniform value for every cap instead, or
  `cap_bond_lengths={"C-H": 1.10, ...}` to tweak individual entries.
* **`rcut`** (diagnostic): radial cutoff -- keep every atom whose
  minimum-image Cartesian distance to any seed is within `rcut`
  Angstrom. Any cut that is *not* a C-C bond raises a warning, since
  this is the classical red flag for accidentally severing C-O / C-N.

### Topology safety and manual C-C cuts

The default `bond_shells` policy is ligand-complete: it does not cut
single C-C bonds automatically. This is the intended production recipe
for compact bridging linkers where breaking the ligand topology would
change the local electronic structure.

`max_atoms` is a hard safety cap, not a soft size optimizer. If the BFS
exceeds this cap -- for example because the selected ligand is
periodically extended -- the carver raises `LigandTopologyOverflowError`
and lists cuttable C-C frontier candidates. Inspect those candidates,
choose chemically sensible cut points, and rerun with
`cut_cc_bonds=[(i, j), ...]`. The carver never auto-picks C-C truncation
sites.

### Freeze convention (`freeze_shell`)

The frozen-atom set written into the sidecar tells the downstream QM
input writer which atoms to hold fixed during geometry optimisation:

* `freeze_shell=0` -- nothing frozen.
* `freeze_shell=1` -- all cap H atoms plus the kept-side atom of every
  cut.  (Wu, Gagliardi, Truhlar, *PCCP* 2018,
  [10.1039/c7cp06751h](https://doi.org/10.1039/c7cp06751h); Vitillo,
  Bhan, Gagliardi, *JPCC* 2023,
  [10.1021/acs.jpcc.3c06423](https://doi.org/10.1021/acs.jpcc.3c06423).)
* `freeze_shell=2` -- as `freeze_shell=1` plus one additional layer of
  heavy atoms inward.  (Gaggioli, Bernales, Gagliardi, *Chem. Sci.*
  2020, [10.1039/d0sc02136a](https://doi.org/10.1039/d0sc02136a).)

The same citations are baked into the default
`ClusterProvenance.convention_reference` so every sidecar carries
provenance for the freeze rule; override the field with your own
system-specific citation when appropriate.

### Seed auto-grouping (`seed_merge_radius`)

By default (`seed_merge_radius=0.0`) each resolved seed produces its
own cluster. Set it to the diameter of a multi-atom node (paddle-wheel
~3.0 Å, M3 trimer ~3.5-3.8 Å, ...) when you want one cluster per node
group.

### Metal-boundary rule

Periodic frameworks are topologically closed: two metal nodes can be
linked through a non-C-C path (e.g. M-X-X-M through a heterocyclic
linker), so a naive BFS can walk past every other node and the
"cluster" can silently become the whole framework. By default,
`bond_shells` therefore treats any bond reaching a metal atom
**outside the current seed group** as an implicit boundary: the bond is
cut and capped on the kept (ligand) side. The parent-side atom is
recorded in both `cut_bonds` and `metal_boundary_cuts`, so downstream
tools can distinguish L-M cuts from user-requested C-C truncations.
Pass `stop_at_non_seed_metals=False` (or
`--no-stop-at-non-seed-metals` on the CLI) for diagnostic carves that
should sweep through every metal.

### Topologically nontrivial periodic loops

Multi-metal nodes (M3 trimers, paddle-wheels, ...) often sit on a
periodic framework whose cycles wind through the unit cell.  The carver
builds the bond graph with `pbc=True` and then constructs a
**maximum-weight spanning tree** of the kept connected component in
which ligand-internal bonds (non-metal/non-metal edges) are weighted
two orders of magnitude heavier than metal-ligand bonds.  BFS propagates
per-atom integer image offsets along the tree, so chemically connected
ligand rings remain in a single Cartesian frame.  Every non-tree (back)
edge is checked against the tree-induced offsets:

* if the image is consistent, it is a chemical ring closure and the
  bond is kept silently;
* if not, the back edge closes a topologically nontrivial periodic
  loop and is recorded as a `loop_cut`.

By construction the back edges are overwhelmingly metal-ligand bonds,
so loop cuts land at the metal boundary -- never inside a triazolate or
benzenedicarboxylate ring.  When a `loop_cut` does sever a metal-X
bond, only the non-metal (ligand) endpoint is capped with H: the metal
becomes an "open coordination site" representing the binding pocket
that, in the real material, would be saturated by a coordinated
solvent.  Adding a Zn-H / Cu-H hydride at the metal endpoint would be
chemically wrong and is forbidden.

Cap H placement on non-carbon keepers is also **chemistry-aware**: a
single bridging atom (e.g. a μ-N that loses two Zn contacts at once) is
protonated only ONCE, not once per cut.  This rule is what prevents
the over-capping NH2 pathology in azolate-based frameworks where a
ring N coordinates several metals through periodic images.

### Built-in invariant checker

`molcrys_kit.analysis.cluster_invariants.check_cluster_invariants_from_files`
runs the C1-C10 carve invariants on a `(parent CIF, cluster XYZ, sidecar
JSON)` triple and returns the list of violations.  These are the
conditions that hold by construction for every cluster the
`ClusterCarver` is supposed to emit; the checker certifies the produced
artefact independently of the carver internals:

* **C1** every seed atom retains all of its first-shell non-metal
  donors (no dropped Zn-N / Zn-O coordination);
* **C2** every parent-bonded pair of kept atoms is within the bond
  threshold in the cluster, except for the pairs listed in
  `loop_cuts`;
* **C3** the cluster (heavy atoms + cap H) is one connected component;
* **C4** every cut keeper appears in `cap_keeper_global_indices`;
  caps are paired against keepers, not against individual cuts, so the
  dedup of multiple-cuts-per-N is handled cleanly;
* **C5** each cap H sits at the recorded `cap_distances_used_A` from
  its keeper atom;
* **C6** every entry in `cut_bonds` is either a metal-boundary cut, a
  requested-and-applied C-C cut, or a loop cut;
* **C7** bonds between seed atoms (e.g. the metal-metal contacts inside
  an M3 SBU) survive the carve;
* **C8** chemistry-aware cap count: at most one cap H per **anion
  group** (carboxylate, sulfonate, phosphonate, hypercoordinate oxo
  anion, deprotonated aromatic N-heterocycle ring) and at most one
  cap H per non-carbon keeper atom; the total cap count equals
  (unique non-C anion groups) + (C-C cuts).  This is what prevents
  the two well-known pathologies on MOF carve outputs:
  *geminal-diol* `-C(OH)2` ends on bridging carboxylates and
  *dihydro-N-heterocycle* tautomers (e.g. 1,4-dihydrotriazole) on
  bridging triazolate/imidazolate rings.  The grouping is computed
  by
  [`ChemicalEnvironment.compute_anion_protonation_groups`](../molcrys_kit/analysis/chemical_env.py)
  so the carver and the checker share a single source of truth with
  `operations.add_hydrogens`;
* **C9** element conservation: the cluster's non-H element counts
  exactly equal the parent counts on `kept_global_indices`, and the
  cluster's H count equals (kept parent H) + `len(cap_local_indices)`;
* **C10** linker inventory: every connected non-metal fragment in the
  cluster has an identically-named (formula) counterpart in the
  parent's non-metal-fragment inventory -- the carver may not invent or
  destroy a ligand species.

Use it programmatically:

```python
from molcrys_kit.analysis.cluster_invariants import (
    check_cluster_invariants_from_files,
)

result = check_cluster_invariants_from_files("structure.cif", "cluster_0.xyz")
print(result.report())
```

or on the command line for a batch check:

```bash
python -m molcrys_kit.analysis.cluster_invariants \
    --parent-cif structure.cif \
    outputs/cluster__group*.xyz
```

### Handling periodically extended linkers

If topology-preserving BFS exceeds `max_atoms`, the error message lists
legal C-C frontier candidates:

```text
Topology-preserving cluster carving exceeded max_atoms=500 ...
Candidate cuttable C-C frontier bonds: (45, 87), (90, 92).
Suggested CLI retry: --cut-cc-bonds "45,87;90,92"
```

Do not blindly paste every candidate into a production calculation.
Inspect the parent structure, choose the chemically sensible truncation
site(s), then rerun with `cut_cc_bonds` / `--cut-cc-bonds`.

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
    max_atoms=500,
    cut_cc_bonds=None,    # default: keep ligand topology intact
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
    --max-atoms 500 \
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
`molcrys_kit.structures.cluster.ClusterProvenance` payload, which is
the canonical record for any downstream QM input writer:

| Key | Type | Meaning |
|---|---|---|
| `mode` | str | `"bond_shells"` or `"rcut"` |
| `seed_global_indices` | list[int] | Parent-atom indices used as seeds |
| `rcut_A` | float or null | Radial cutoff in Angstrom (rcut) |
| `max_atoms` | int or null | Hard topology-preserving BFS safety cap |
| `kept_global_indices` | list[int] | Parent-atom indices retained |
| `cut_bonds` | list[[int, int]] | (kept, dropped) parent index per cut |
| `cut_cc_bonds_requested` | list[[int, int]] | User-requested C-C truncation bonds |
| `cut_cc_bonds_applied` | list[[int, int]] | Requested C-C cuts that became boundaries |
| `metal_boundary_cuts` | list[[int, int]] | L-M cuts from `stop_at_non_seed_metals` |
| `loop_cuts` | list[[int, int]] | Edges broken because they close a topologically nontrivial periodic loop.  By design these are metal-ligand edges; only the non-metal endpoint receives a cap H, the metal endpoint becomes an open coordination site. |
| `cap_keeper_global_indices` | list[int] | For each entry in `cap_local_indices`, the parent-atom global index of the keeper atom the cap H is bonded to.  After chemistry-aware dedup, several cuts may share a single cap; use this list (not `cut_bonds`) to associate caps with their parent atoms. |
| `cap_local_indices` | list[int] | Local indices of cap H in the XYZ |
| `frozen_local_indices` | list[int] | Local indices to hold fixed |
| `freeze_shell` | int | 0, 1, or 2 |
| `cap_distance_A` | float or null | Uniform override; null = per-element |
| `cap_bond_lengths_A` | dict[str, float] | Element-keyed X-H table consulted |
| `cap_distances_used_A` | list[float] | Per-cap distance actually applied |
| `seed_merge_radius_A` | float | Auto-grouping threshold |
| `parent_label` | str or null | Free-text label of the parent structure |
| `kind` | str | Carve-strategy tag.  `ClusterCarver` always emits `"coordination"` (metal-seed + ligand-complete BFS + anion-group capping at the metal boundary); future packing- / pi-stack-style carvers would emit `"packing"`.  Sidecars produced before this field was introduced default to `"coordination"` on read. |
| `convention_reference` | str | Caller-supplied citation of the QM-cluster recipe |

### Scope: coordination clusters

`ClusterCarver` is a **coordination-cluster carver**: it seeds on
metal atoms, follows ligand-complete BFS, and caps at the metal
boundary using `ChemicalEnvironment.compute_anion_protonation_groups`
chemistry.  The clusters it emits are tagged
`provenance.kind == "coordination"` to mark this scope explicitly.
This is the "coordination cluster / SBU environment" of the MOF /
coordination-polymer literature, not the supramolecular
"weak-interaction packing cluster" of crystal engineering.  A future
packing- / pi-stack- / H-bond-driven carver would emit
`kind="packing"` and live next to this one.

### Out of scope

* No domain typing (SBU / linker / paddle-wheel / pore / channel) --
  the carver works at the atom + bond level.  Use the `kind` tag and
  layer your own ontology on top.
* No charge / spin inference -- supply these when writing the QM input.
* No Gaussian / ORCA / Psi4 writer; the sidecar JSON is the integration
  point.

### Convention references

These references ground the default freeze / cap / mode choices used
by the carver and the default `ClusterProvenance.convention_reference`
string.  Override the field with your own system-specific citation
when appropriate; the defaults are not project-specific endorsements.

* Beyzavi *et al.*, *J. Am. Chem. Soc.* **2014**, 136, 15861.
  DOI: [10.1021/ja508626n](https://doi.org/10.1021/ja508626n).
  Cap-vs-periodic energetic benchmark; foundational test of the
  cap-and-freeze convention.
* Wu, Gagliardi, Truhlar, *Phys. Chem. Chem. Phys.* **2018**, 20, 1953.
  DOI: [10.1039/c7cp06751h](https://doi.org/10.1039/c7cp06751h).
  Cap distance and freeze rule; methyl-vs-formate cap benchmark.
  Source of the shell-1 freeze convention.
* Vitillo, Bhan, Gagliardi, *J. Phys. Chem. C* **2023**.
  DOI: [10.1021/acs.jpcc.3c06423](https://doi.org/10.1021/acs.jpcc.3c06423).
  def2-SVP cluster opt + def2-TZVP single point, shell-1 freeze, for
  transition-metal cluster QM.
* Gaggioli, Bernales, Gagliardi, *Chem. Sci.* **2020**, 11.
  DOI: [10.1039/d0sc02136a](https://doi.org/10.1039/d0sc02136a).
  Shell-2 freeze convention.
* Migues, Auerbach, *J. Phys. Chem. C* **2018**, 122, 23230.
  DOI: [10.1021/acs.jpcc.8b08684](https://doi.org/10.1021/acs.jpcc.8b08684).
  Delta-cluster convergence test in zeolites; basis for the
  diagnostic `rcut` mode.
