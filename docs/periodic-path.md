# Periodic geometric paths

`interpolate_periodic_paths` constructs geometric candidates from two
`MolecularCrystal` objects and an explicit one-to-one atom mapping. It reads
the existing endpoint bond records, aligns equivalent periodic cell bases,
and retains the endpoint quotient graphs. The construction is experimental:
its geometric scores do not establish a mechanism or a physical energy.
The original endpoint bond-record snapshots, including their roles, are also
returned. Geometry-dependent endpoint bond records are not copied onto the
intermediate images as though they described those images.

```python
from molcrys_kit.operations import PeriodicPathConfig, interpolate_periodic_paths

result = interpolate_periodic_paths(
    initial,
    final,
    product_index_by_reactant=atom_mapping,
    config=PeriodicPathConfig(n_images=9, max_candidates=3),
)
candidate = result.candidates[0]
```

The example assumes the two structure objects and their atom mapping already
exist. Candidate images follow the reactant atom order. The endpoint graph
snapshots use each endpoint's original cell basis; the product graph's node
identities follow the reactant mapping. The endpoint bond-record snapshots keep
their respective original indexing. `end_basis_transform` records the integer
matrix used to align the product cell with the reactant representation, and
candidate image shifts are expressed in the aligned product basis.

`PeriodicPathConfig` controls image count, candidate count, numerical weights,
and screening tolerances. Candidates include a lowest-displacement assignment
and bounded single-component one-cell winding alternatives. The returned
diagnostics describe evaluated and rejected candidates, ambiguous common
edges, convergence, collisions, bond deviations, and path spacing. When no
candidate passes screening, `PeriodicPathError` exposes the diagnostics.

The method preserves explicit image shifts on periodic edges. It does not
assume that an infinite periodic network can be unfolded into a finite
molecule. Finite components use a rotation-aware geometric initial guess;
non-rigid endpoint differences remain free to change. Atom pairs that overlap
exactly are separated deterministically before optimization and are checked
again afterward. Non-periodic cell vectors may be zero, provided the active
periodic vectors are linearly independent.

When a converged candidate misses the short-contact tolerance, the optimizer
can increase the contact penalty and retry a bounded number of times. Every
stage and its iteration count is included in the diagnostics; persistent
failures are rejected rather than silently accepted.

Candidate enumeration is finite and does not cover arbitrary multiple
windings or coupled component translations. Equivalent-basis normalization
uses a bounded search after lattice reduction, and degenerate symmetric cases
can have more than one equally valid geometric representation. Endpoints must
describe the same atoms and periodic directions; preparing an atom mapping or
making incompatible supercells commensurate is outside this operation.

The geometric objective combines endpoint-derived edge distances, a short
contact penalty, and optional path regularity. These are adjustable numerical criteria,
not universal chemical constraints. Unchanged edge identities are screened
against interpolated lengths; changed edge identities are not required to
remain intact. Always inspect the diagnostics and compare independent methods
for the target application.


When specified, `fixed_atom_indices` holds the listed reactant-order atoms
fixed in Cartesian coordinates during initial construction and optimization.
Their product positions must match the initial positions modulo the product
lattice. The output frames carry the same fixed-atom constraint. Structure
conversion does not infer these indices; callers must provide them explicitly.

For independent-image pair-distance refinement of an explicit warm path, see [IDPP refinement](idpp-refinement.md).


Cartesian second-difference smoothing is disabled by default because it
penalizes an exact rigid rotational arc and can distort a rotating component.
Set `smoothness_weight` explicitly to a positive value only when this tradeoff
is intended. Zero smoothing does not guarantee shape preservation, adequate
image spacing, or physical plausibility: collision and bond penalties can
still move atoms, and screening diagnostics must be inspected.

`interpolate_periodic_paths(..., restraints=...)` accepts caller-specified
soft geometric coordinates and per-image schedules. The same restraints can
be used by `refine_idpp_path`; see [path coordinates](path-coordinates.md).
Targets are not inferred from topology, bonding changes, or electronic state.
The objective never changes its endpoints or configured fixed atoms; the
reported restraint values do not imply that a soft target was reached.
