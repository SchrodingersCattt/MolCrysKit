# Geometry-only IDPP path refinement

Use refine_idpp_path to refine an explicitly initialized sequence of ASE
Atoms images. The caller supplies corresponding atoms in the same order
throughout, including atoms involved in bond formation or breaking.

    from molcrys_kit.operations import refine_idpp_path

    # warm_images is an existing atom-mapped path, including both endpoints.
    result = refine_idpp_path(
        warm_images,
        prescribed_atom_indices=(0, 1, 2),
        gradient_tolerance=1.0e-3,
    )
    if not result.converged:
        print(result.diagnostics)

The function never constructs an implicit Cartesian-linear path. A caller can
prepare a rotation-aware initial path, including an explicitly placed
transferred atom, and then prescribe the positions of selected atoms in each
image. Prescribed indices are fixed at their respective input coordinates for
this refinement only; they may move from image to image and are not implicitly
added as downstream ASE constraints. Existing FixAtoms constraints remain
attached and are also honored. Other ASE constraint types are rejected.

The minimum-image convention uses a reduced periodic basis, including skewed
cells and partial periodicity. Each frame uses its own cell and the common PBC
mask; the caller is responsible for expressing changing cells in a consistent
basis. Optional fractions specify where each frame lies between the endpoint
distance targets. Atom IDs, when provided, must agree across images. A
coincident pair is rejected because the IDPP objective is singular there.

The pair-distance objective is that of the original IDPP formulation. Interior
images are minimized independently and diagnostics include the numerical
gradient and convergence status. There is no inter-image spring or physical
potential, and no guarantee of path spacing, bond preservation, physical
mechanism, or transition-state quality. This operation is not the same
optimizer as ASE's IDPP band initialization. The objective considers one
minimum-image distance per distinct atom pair; it does not include interactions
of an atom with its own periodic images.

See Smidstrup et al., J. Chem. Phys. 140, 214106 (2014),
[DOI: 10.1063/1.4878664](https://doi.org/10.1063/1.4878664).


Optional caller-defined soft restraints can be supplied through `restraints`;
the coordinate primitives, schedules, coupling and image-shift rules are
documented in [path coordinates](path-coordinates.md). The output includes
`restraint_diagnostics` with targets, weights, values, residuals, and an
indicator of which images contributed to the minimized objective. A soft
restraint does not guarantee satisfaction; inspect its residuals. Endpoints
and prescribed atoms remain fixed by the refinement.
