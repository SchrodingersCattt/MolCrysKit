"""Numerical defaults for crystallographic symmetry-path construction.

All tolerances used by the symmetry algebra and strict rigid-path planner live
here so that structures and operations contain no hidden scientific thresholds.
Public configuration objects may override path-planning defaults.
"""

# Fractional affine-operation validation.
AFFINE_INTEGER_TOLERANCE = 1.0e-8
AFFINE_DETERMINANT_TOLERANCE = 1.0e-8
AFFINE_METRIC_TOLERANCE = 1.0e-8
AFFINE_ORTHOGONALITY_TOLERANCE = 1.0e-8
AFFINE_EQUIVALENCE_TOLERANCE = 1.0e-8
BASIS_UNIMODULAR_TOLERANCE = 1.0e-8

# Correspondence and strict rigid reachability.
CORRESPONDENCE_DISTANCE_TOLERANCE_ANGSTROM = 1.0e-4
RIGID_MASS_WEIGHTED_RMSD_TOLERANCE_ANGSTROM = 5.0e-2
RIGID_MAX_BOND_RELATIVE_ERROR = 2.0e-2

# Generated-path validation.
HARD_CLASH_RADIUS_SCALE = 0.60
PBC_IMAGE_CONTINUITY_TOLERANCE_ANGSTROM = 1.0e-6
SYMMETRY_EQUIVARIANCE_TOLERANCE_ANGSTROM = 1.0e-6

# Global assignment.  The sentinel must dominate every finite physical cost.
ASSIGNMENT_INFEASIBLE_COST = 1.0e12
ASSIGNMENT_PROVENANCE_WEIGHT = 1.0e6
ASSIGNMENT_ATOM_RMSD_WEIGHT = 1.0e3
ASSIGNMENT_COM_DISTANCE_WEIGHT = 1.0
