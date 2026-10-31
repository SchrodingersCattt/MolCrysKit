# Typed chemical edits and packing transfer

`molcrys_kit.operations` provides an additive construction API for complete,
ordered full cells. The same symbols are lazy exports of `molcrys_kit`, in the
same way as the forward-construction helpers. Related API discussions: #204 and
#201. A constructed CF0 is a hypothetical frozen geometry; it establishes neither
a local minimum nor experimental realizability.

Issue #204 also lists pilot acceptance boxes that name unpublished template
CIFs and swap identifiers. Those boxes stay in the research project. The library
contract is this page's synthetic example and the synthetic unit tests. This
package does not vendor those structures.

The library uses the existing `CrystalMolecule` chemistry, graphs, arrays and
forward-builder anchor convention. Supply complete replacement geometry and
intended protonation, bonds, stereo and charge. This layer does not infer or
repair chemistry, regenerate hydrogen, or select a potential.

## Generic synthetic example

This example builds a synthetic cell containing an ion pair and neutral guests,
then edits every molecule labelled `counterion` in one transaction. Role labels,
role counts and the number of distinct roles are supplied by the caller.
Coordinates are illustrative.

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
components = [ammonium.copy(), molecule("Cl", [[3, 3, 3]])]
components[0].positions += [10, 10, 10]
components += [molecule("He", [p]) for p in ([3, 10, 3], [10, 3, 10])]
parent = MolecularCrystal(np.eye(3) * 20, components)
inventory = CrystalInventory(
    roles=("counterion", "anion", "guest", "guest"), charges=(1, -1, 0, 0),
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
    parent, inventory, CrystalEditSpec(edit, role="counterion"), seed=42,
)
if result.status == CounterfactualStatus.CONSTRUCTED:
    paths = write_cf0(result, "chemical_edit_output")
    cf1 = RelaxationProtocol("CF1").constraints(result.crystal)
else:
    print(result.status.value, result.provenance["reason"])
```

Additional molecules with the selected role and corresponding inventory entries
join the same role-wide edit and provenance record. `scope="sites", indices=(i,)`
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
source/target indices, the output molecule index, retained stable IDs, deleted IDs, created IDs and rotations.

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

Role names, role counts and the number of roles are unrestricted. Matching
reduced inventories require a complete role-preserving bijection between the
supplied full cells. Changed reduced inventories use `StoichiometryPolicy`
instead of that full-cell bijection. The template supplies the cell and anchors; composition
supplies complete molecules and charges. Pairing uses role then input order.
Supply `site_mapping=((p_index, q_index), ...)` to choose another complete
bijection, and per-role `mappings` for retained atom identities. Missing placement
policies or incomplete or role-changing assignments return `INCOMPATIBLE`.

Provenance separates reduced stoichiometry from full-cell multiplicity. Each
cell's role counts are reduced by their greatest common divisor; element and
charge inventories are compared at that same scale. `inventory_comparison`
records raw counts, reduced counts, reduction factors, `site_count_change`,
`stoichiometry_change` and `composition_change`. Reduction factors describe the
declared role inventory; they are not inferred crystallographic Z values.

Proportionally scaling an otherwise identical full cell does not by itself set
`composition_change=true`. If only the full-cell multiplicities differ, transfer
returns `INCOMPATIBLE` with `CELL_MULTIPLICITY_MAPPING_REQUIRED`: prepare
commensurate input cells explicitly so every molecule has one destination.
The API does not build supercells or discard molecules implicitly. Matching
expanded cells remain supported for any role inventory.

Changed reduced role counts set `stoichiometry_change=true` and
`composition_change=true`. Without `PackingTransferPlan.stoichiometry_policy`
the transfer returns `INCOMPATIBLE` / `STOICHIOMETRY_POLICY_REQUIRED` and no
geometry. Supply a `StoichiometryPolicy` to construct that what-if cell:

- `vacant_sites` are template molecule indices omitted from the output. A
  vacancy removes the molecule. It is not a partial-occupancy site, and the
  library does not choose which sites to drop.
- `MoleculeInsertion` places a composition molecule that has no template
  counterpart. `fractional_anchor` is the template-cell fractional coordinate
  of that molecule's placement anchor (`anchor @ lattice`). Orientation uses
  the role policy. `mapped` orientation is refused because an insertion has no
  template frame; the first sample keeps the supplied orientation.
- Every remaining composition molecule must map onto every remaining template
  site. The default pairing is by role, then input order. `site_mapping` may
  name that occupied bijection explicitly. Role `mappings` apply only to those
  template-site replacements; insertions receive new atom IDs.
- Output order is ascending non-vacant template indices, then insertions in
  the order declared. Formula and charge follow the composition. The template
  lattice is kept. Orbit IDs are kept only when every output molecule is a
  retained template site.

A policy that does not produce that bijection, names an out-of-range index, or
is supplied when the reduced role counts and the full-cell counts already match
returns `STOICHIOMETRY_POLICY_INVALID`. A multiplicity-only mismatch is checked
first. It is not a vacancy or an insertion, and it returns
`CELL_MULTIPLICITY_MAPPING_REQUIRED` even if a policy object is supplied.
Different chemistry or charge per reduced inventory also sets
`composition_change=true`, even at unchanged role stoichiometry. At
matching full-cell counts, output formula and charge default to the supplied
composition's inventory. Do not compare raw energies across unlike compositions
without a common thermodynamic reference.

```python
from molcrys_kit import (
    MoleculeInsertion, PackingTransferPlan, PlacementPolicy, StoichiometryPolicy,
)

# Composition A:B:X = 1:1:3 placed on a 2:1:5 template by omitting the extra
# template sites. The indices below match one synthetic ordering; name the
# actual extra sites in the template you pass in.
plan = PackingTransferPlan(
    composition_inventory=p_inventory,
    template_inventory=q_inventory,
    placements={role: PlacementPolicy("com", "com", samples=32)
                for role in set(p_inventory.roles)},
    stoichiometry_policy=StoichiometryPolicy(
        vacant_sites=(1, 6, 7),  # extra A and two extra X sites on Q
    ),
)
```

The reverse direction inserts the extra composition molecules at declared
anchors instead of inventing coordinates:

```python
StoichiometryPolicy(insertions=(
    MoleculeInsertion(1, (0.85, 0.85, 0.75)),
    MoleculeInsertion(6, (0.15, 0.85, 0.75)),
    MoleculeInsertion(7, (0.50, 0.85, 0.75)),
))
```

## Results, export and refusal semantics

A successful `CounterfactualResult` contains a new crystal, output inventory and
one provenance object. Refusals contain no candidate crystal or output inventory.
`INCOMPATIBLE` covers source guards, charge/formula mismatches, occupancy,
undefined placement/mapping, and a missing or incomplete stoichiometry policy.
A multiplicity-only mismatch, including one that also carries a stoichiometry
policy, uses `CELL_MULTIPLICITY_MAPPING_REQUIRED`. Partial
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
