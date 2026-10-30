# Typed chemical edits and packing transfer

`molcrys_kit.operations` provides an additive construction API for complete,
ordered full cells. It implements the construction contract in #204 and the
initial kernel of #201. A constructed CF0 is a hypothetical frozen geometry;
it establishes neither a local minimum nor experimental realizability.

The library uses the existing `CrystalMolecule` chemistry, graphs, arrays and
forward-builder anchor convention. Supply complete replacement geometry and
intended protonation, bonds, stereo and charge. This layer does not infer or
repair chemistry, regenerate hydrogen, or select a potential.

## Generic synthetic example

This example builds a synthetic ABX3-type cell with dication role A, monocation
role B and monoanion role X, then edits every B site in one transaction.
Coordinates are illustrative and are not an experimental structure.

```python
import numpy as np
from ase import Atoms
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal
from molcrys_kit.operations import (
    ChemicalAtomMapping, ChemicalEdit, CrystalEditSpec, CrystalInventory,
    CounterfactualStatus, PlacementPolicy, RelaxationProtocol,
    apply_chemical_edit, write_cf0,
)


def molecule(symbols, positions):
    return CrystalMolecule(Atoms(symbols, positions=positions), check_pbc=False)


tetrahedron = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
]) / np.sqrt(3)
ammonium = molecule("NH4", np.vstack([[0, 0, 0], 1.02 * tetrahedron]))
components = [molecule("Ca", [[3, 3, 3]]), ammonium.copy()]
components[1].positions += [10, 10, 10]
components += [molecule("Cl", [p]) for p in (
    [3, 10, 3], [10, 3, 10], [3, 3, 10],
)]
parent = MolecularCrystal(np.eye(3) * 20, components)
inventory = CrystalInventory(
    roles=("A", "B", "X", "X", "X"), charges=(2, 1, -1, -1, -1),
)
replacement = molecule("NH3OH", [
    [0, 0, 0], [-.34, .96, 0], [-.34, -.48, .83], [-.34, -.48, -.83],
    [1.42, 0, 0], [1.74, .91, 0],
])
edit = ChemicalEdit(
    replacement=replacement,
    source_formula=ammonium.get_chemical_formula(),
    source_charge=1, target_charge=1,
    placement=PlacementPolicy(source_anchor=0, target_anchor=0,
                              orientation="sample", samples=32),
    mapping=ChemicalAtomMapping(((0, 0),)),  # retain the nitrogen identity
)
result = apply_chemical_edit(
    parent, inventory, CrystalEditSpec(edit, role="B"), seed=42,
)
if result.status == CounterfactualStatus.CONSTRUCTED:
    paths = write_cf0(result, "chemical_edit_output")
    cf1 = RelaxationProtocol("CF1").constraints(result.crystal)
else:
    print(result.status.value, result.provenance["reason"])
```

Adding additional B molecules and corresponding inventory entries includes them
in the same role-wide edit and provenance record. `scope="sites", indices=(i,)`
requests a local defect. `scope="orbit", indices=(i,)` expands seed sites to all
members of their declared orbit IDs. Supply `CrystalInventory.orbit_ids` or use
the `forward_pose_index` evidence from forward construction. Missing evidence
is refused. Caller-declared orbit labels must describe the intended full cell;
the editor does not infer or certify their crystallographic equivalence.

## Placement and identity

Every edit requires an explicit `PlacementPolicy`. Atom anchors align the chosen
source and replacement atoms; `"com"` explicitly aligns mass-weighted centers.
The forward builder supplies the anchor convention. `orientation="sample"`
uses seeded proper rotations about the anchor and a finite attempt budget. A
non-collinear partial map supplies the initial orientation when available.
`orientation="mapped"` requires at least three non-collinear retained pairs and
uses a proper least-squares rotation without internal distortion. Sampling may
change the mapped frame; the map retains chemical identity, not exact geometric
coincidence of every mapped atom.

`ChemicalAtomMapping` is an injective partial map of local indices. It is separate
from the existing geometric `symmetry_path.AtomCorrespondence`. Retained pairs
must preserve elements and the supplied isotope/stereo arrays. Missing identity
metadata on one side is refused. Bonds may change through the prebuilt target;
there is no hidden graph edit or charge/stereo reassignment. Existing target
chemistry records are copied with remapped stable IDs and a refreshed coordinate
embedding; bonds, charges, stereo labels and evidence are preserved. Finite
molecular bond images are rebased to zero in the contiguous output geometry. The result records
source/target indices, retained stable IDs, deleted IDs, created IDs and rotations.

NH4+ to NH3OH+ is supported with a shared N anchor and sampled orientation.
Ag+ to NH4+ and the reverse are supported as whole-ion replacements with an empty
retained map and an explicit anchor/COM sampling policy. Ag and N cannot be
claimed as the same retained atom. These placements are checked for clashes;
a small or crowded cell can still be refused.

Contacts use the repository covalent radii multiplied by `clash_scale`, with
ASE periodic neighbor enumeration. Checks include skewed cells, all images
within the cutoff, contacts between replacements, and contacts with a molecule's
own periodic copies. Intramolecular pairs in the supplied contiguous image are
excluded. Mixed-role policies use the largest requested scale conservatively.
`CLASH` means the bounded search failed; it is not a proof of incompatibility.

The parent cell, all input coordinates, arrays and nested molecule metadata stay
unchanged. Edited output is an ordered full-cell P1 model, permitting symmetry
breaking. Orbit-complete selection does not impose molecular site symmetry or
claim the old group survives. For preservation of a specified space group, use
the separate forward-construction site validation and expansion API.

## Composition and packing transfer

`CrystalInventory` assigns one role and integral molecular charge to every
component. Source formula/charge guards are checked at every selected site.
Known per-atom formal charges and a known chemical-entity net charge must agree
with declarations. Unknown molecular charge is an explicit caller assertion,
recorded as a validation limit. The result reports the exact element inventory,
charge sum and role counts. Isotope arrays remain on the molecules; the formula
inventory counts elements. Use `expected_formula={"N": ..., "H": ...}` for an
independent inventory assertion. Charge-changing edits require an explicit
`expected_charge`; otherwise replacement must preserve the parent charge sum.

To put composition P onto template Q's lattice and molecular sites:

```python
from molcrys_kit.operations import PackingTransferPlan, transfer_packing

# P and Q are independently prepared ordered MolecularCrystals.
# p_inventory and q_inventory each describe every molecule in their own order.
plan = PackingTransferPlan(
    composition_inventory=p_inventory,
    template_inventory=q_inventory,
    placements={role: PlacementPolicy("com", "com", samples=32)
                for role in set(q_inventory.roles)},
)
transferred = transfer_packing(P, Q, plan, seed=42)
```

Each role must have the same count in P and Q. This supports, for example,
transfers between two distinct synthetic 2:1:5 cells. The template supplies the
cell and anchors; composition supplies complete molecules and charges. Pairing
uses role then input order. Supply a complete `site_mapping=((p_index, q_index),
...)` bijection to choose another role-preserving assignment, and per-role
`mappings` for retained atom identities. Missing placement policies or incomplete
or role-changing assignments return `INCOMPATIBLE`.

Different element inventories or role counts set `composition_change=true`.
Different role counts require a vacancy/insertion policy and currently return
`INCOMPATIBLE`, even if an incomplete mapping was supplied. There is no implicit
supercell multiplication, deletion or insertion. Different chemistry at equal
role counts is supported; output charge and formula default to P's inventory.
Do not compare raw energies across unlike compositions without a common
thermodynamic reference.

## Results, export and refusal semantics

A successful `CounterfactualResult` contains a new crystal, output inventory and
one provenance object. Refusals contain no candidate crystal or output inventory.
`INCOMPATIBLE` covers source guards, charge/formula mismatches, occupancy,
undefined placement/mapping and unsupported role stoichiometry changes. Partial
occupancy and declared periodic bonded frameworks require another representation
and are refused. Inputs must already be finite contiguous molecular components;
the editor does not discover or unwrap periodic networks. Invalid typed object
construction and invalid seeds raise `ValueError` as programming errors.

`write_cf0` writes `cf0.cif`, optional `cf0.extxyz`, and `provenance.json` into a
directory without existing edit artifacts. Refusals write only provenance.
The sidecar is authoritative for roles, declared charges and mapping semantics
that plain CIF cannot represent. It includes parent hash, edit hash and complete
specification, seed, status/reason, composition-change flag, identities, actual
rotation matrices, inventory and clash diagnostics. Packing transfers also hash
P and record the site assignment. Hashes include exact coordinates, atom arrays,
bonds and declared inventories; they are not symmetry-equivalence fingerprints.
Unknown numeric CIF metadata is encoded with explicit nonfinite-value tokens
in strict JSON.
A result geometry changed after construction is rejected by the writer. CIF
round-trips preserve full-cell geometry and molecule formulas; retain the JSON
alongside CIF for the full edit record. ExtXYZ additionally carries frame metadata
and molecule partitioning. No source experimental metadata or calculator results
are silently reused on the changed crystal.

## External relaxation hooks

`RelaxationProtocol(stage).constraints(crystal)` returns molecule-to-atom groups,
allowed molecular degrees of freedom, an ASE-Voigt-order cell mask and a P1
symmetry policy:

| Stage | Internal geometry | Translation / orientation | Cell |
| --- | --- | --- | --- |
| CF0 | frozen | frozen | frozen |
| CF1 | free | frozen | frozen |
| CF2 | frozen | free (rigid-body motion) | frozen |
| CF3 | free | free | free |

CF1 uses a mass-weighted COM and proper Eckart frame to define the fixed pose.
The external driver must implement these collective constraints; they cannot be
replaced by Cartesian `FixAtoms` masks. Protocol dictionaries explicitly mark
this requirement for CF1/CF2. The library selects no calculator, optimizer,
convergence tolerance or force field and runs no relaxation. Consumers should
record those choices and outcomes with their resulting calculation artifacts.
