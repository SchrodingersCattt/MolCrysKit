# Composable geometry for path construction

`PathCoordinate` is a protocol for a scalar coordinate and its Cartesian
position gradient for one image. `PathRestraint` adds a caller-specified
quadratic penalty to a path refiner. These are numerical geometry tools: the
caller chooses atom identities, targets, weights, schedules, and their physical
interpretation. Nothing infers a reaction, electronic state, or universally
preferred conformation.

The built-in coordinates are:

- `DistanceCoordinate(i, j, image_shift=None)`, in Å.
- `AngleCoordinate(i, center, k, image_shifts=(None, None))`, in radians.
- `SignedVolumeCoordinate(center, i, j, k, image_shifts=(None, None, None))`,
  an oriented triple product in Å³. Its sign changes if two neighbors are
  swapped; zero indicates coplanarity or a degenerate arrangement.
- `SignedPlaneDistanceCoordinate(center, i, j, k, image_shifts=(None, None, None))`,
  the signed distance in Å from `center` to the plane through `i`, `j`, `k`.
  The sign depends on plane-atom order; collinear plane atoms are rejected.
- `LinearCombinationCoordinate(((coefficient, coordinate), ...))`, a scalar
  linear combination. Coefficients must supply any needed unit conversions.

Users may implement `value_and_gradient(positions, cell, pbc)` for other
coordinates, including nonlinear combinations. It must return a finite scalar
and an `(n_atoms, 3)` finite Cartesian gradient at fixed cell and must not
modify its inputs. Weights scale squared residuals and are not automatically
physical force constants.

For example, given a caller-provided sequence of at least three atom-mapped ASE
`Atoms` images, each with at least four atoms:

```python
import numpy as np
from molcrys_kit.operations import (
    DistanceCoordinate,
    LinearCombinationCoordinate,
    PathRestraint,
    SignedVolumeCoordinate,
    refine_idpp_path,
)

# The values and weights are illustrative numerical inputs, not chemistry rules.
contrast = LinearCombinationCoordinate((
    (1.0, DistanceCoordinate(0, 1)),
    (-1.0, DistanceCoordinate(2, 3)),
))
shape = SignedVolumeCoordinate(0, 1, 2, 3)
count = len(images)
restraints = [
    PathRestraint(contrast, targets=np.linspace(-0.5, 0.5, count),
                  weights=2.0, name="distance contrast"),
    PathRestraint(shape, targets=0.0, weights=[0.0] + [0.1] * (count - 2) + [0.0],
                  name="shape"),
]
result = refine_idpp_path(
    images, prescribed_atom_indices=(0,), restraints=restraints
)
print(result.converged, result.restraint_diagnostics)
```

A scalar target or weight is repeated for all images. A sequence must have
exactly one entry per image including endpoints; weights must be finite and
non-negative. Zero disables a term for that image. Endpoints are retained and
never optimized: the diagnostics distinguish endpoint observations from
interior penalties. `prescribed_atom_indices` holds each listed atom at its
own input position in every image; the atom may move between images. Existing
`FixAtoms` constraints are also honored.

The same `restraints` argument is accepted by `interpolate_periodic_paths`,
which reports each target, value, residual and weighted penalty in candidate
diagnostics. Restraints are soft. A geometrically accepted candidate may have
large residuals, so callers must inspect them and apply their own acceptance
criteria. Neither refiner uses an electronic energy or ensures a mechanism.

By default every relative vector uses a minimum image, including for skew or
partially periodic cells. Minimum-image selection is non-differentiable where
two images are equally close. Supply explicit integer image shifts when the
coordinate must follow a chosen periodic branch, using one shift per vector
from its center. Shifts are relative to the frame's current lattice basis and
must be zero on non-periodic axes. If coordinates are rewrapped or the basis is
changed, transform the shifts accordingly. For the periodic path constructor,
shifts refer to the aligned image-cell basis returned by that constructor.

Distinct atom indices are required by the built-in coordinates. An angle is
undefined when either arm vanishes or the arms are collinear. A distance is
undefined when its two atoms overlap. The signed volume remains defined for a
coplanar arrangement, but its gradient may vanish for degenerate inputs.


## Combine explicitly mapped path layers

`compose_path_images` overlays disjoint atom subsets from caller-provided ASE
frames onto a base path. This allows rigid, internal-coordinate, or other
image sequences to be combined without reaction-specific constructors. For
example, given `base_images` and a same-length `fragment_images` sequence:

```python
from molcrys_kit.operations import PathLayer, compose_path_images, refine_idpp_path

# Local row 0 maps to global atom 4; local row 1 maps to global atom 7.
layer = PathLayer(
    atom_indices=(4, 7), images=fragment_images, anchor_index=4,
    image_policy="nearest", name="fragment",
)
composed = compose_path_images(base_images, [layer])
# These two atoms follow the composed positions at each image during refinement.
refined = refine_idpp_path(composed.images, prescribed_atom_indices=(4, 7))
```

The layer is translated to the base anchor independently at each image, with
no inferred rotation. Its coordinates must already use the base frame's
Cartesian orientation and units. `image_policy="preserve"` is the default;
`"nearest"` explicitly chooses the image nearest each base atom and may alter
a winding. Layers cannot overlap. If atom IDs exist they must be present and
match on both sides, with one scalar ID per atom; otherwise the explicit row
mapping remains an assumption.
By default layer endpoints must match the base modulo PBC and the exact base
endpoint coordinates are retained. `preserve_endpoints=False` explicitly
allows changing them. A layer cannot move an ASE `FixAtoms` atom; other ASE
constraints are rejected. The operation removes the known coordinate-derived
`_molcrys_bond_records` cache on changed images, but does not infer connectivity,
update other derived metadata, or detect clashes. Callers must review those
separately.


## Rigid groups and explicit image choices

`interpolate_reactive_path` supports one or more `RigidGroup` objects and
optional `FreeAtomArc` coordinates. `RigidGroup.unwrap_mode` controls input
embedding: `"auto"` uses a periodic bond walk when both endpoint groups are
connected, otherwise it preserves their supplied Cartesian embeddings;
`"bond_walk"` requires connectivity; `"as_provided"` always preserves the
provided embeddings. The effective mode and per-atom shifts are recorded in
result metadata. A fallback does not determine a physically correct image of
a disconnected cluster: users must supply a consistent embedding and inspect
the result. Optional group lattice shifts must be zero on nonperiodic axes.
Atom mappings and indices must be integers, and declared bond-state flags must
be booleans. No reaction-specific endpoint constructor is required.

A runnable, project-neutral example composing a moving subset and coupled
coordinates is available at `examples/composed_geometry_path.py`.
