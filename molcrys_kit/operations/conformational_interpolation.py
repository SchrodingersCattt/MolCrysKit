"""Composite rigid-body and internal-coordinate molecular interpolation."""

from __future__ import annotations

from math import pi
from typing import Sequence

import networkx as nx
import numpy as np

from ..analysis.ring_conformation import (
    find_ring_systems,
    puckering_coordinates,
    reconstruct_z_from_modes,
)
from ..constants.config import BOND_ROTATION_AXIS_TOLERANCE
from ..structures.molecule import (
    CrystalMolecule,
    _refresh_contiguous_bond_geometry,
    _strip_stale_frac_arrays,
)
from ..utils.geometry import dihedral_angle, kabsch_align
from ._path_core import (
    coerce_interpolation_method,
    interpolate_rigid_positions,
    path_lambda_values,
)

__all__ = ["interpolate_molecule_with_internal_dofs"]


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the shortest signed interval."""
    return (float(angle) + pi) % (2.0 * pi) - pi


def _torsion_descriptor(
    molecule: CrystalMolecule,
    positions_a: np.ndarray,
    positions_b_in_a: np.ndarray,
    bond: tuple[int, int],
) -> tuple[int, int, int, int, float] | None:
    """Return a bridge bond and its shortest endpoint torsion change."""
    from .bond_rotation import partition_at_bond

    atom_i, atom_j = (int(bond[0]), int(bond[1]))
    partition = partition_at_bond(molecule, atom_i, atom_j)
    if partition.is_ring_bond:
        return None
    graph = molecule.graph
    left_neighbors = sorted(
        int(index) for index in graph.neighbors(atom_i) if index != atom_j
    )
    right_neighbors = sorted(
        int(index) for index in graph.neighbors(atom_j) if index != atom_i
    )
    if not left_neighbors or not right_neighbors:
        return None
    atom_k, atom_l = left_neighbors[0], right_neighbors[0]
    normal_a = np.cross(
        positions_a[atom_i] - positions_a[atom_k],
        positions_a[atom_j] - positions_a[atom_i],
    )
    normal_b = np.cross(
        positions_b_in_a[atom_i] - positions_b_in_a[atom_k],
        positions_b_in_a[atom_j] - positions_b_in_a[atom_i],
    )
    if (
        np.linalg.norm(normal_a) <= BOND_ROTATION_AXIS_TOLERANCE
        or np.linalg.norm(normal_b) <= BOND_ROTATION_AXIS_TOLERANCE
    ):
        raise ValueError(
            f"Cannot interpolate torsion {atom_k}-{atom_i}-{atom_j}-{atom_l}: "
            "endpoint geometry is collinear"
        )
    angle_a = dihedral_angle(
        positions_a[atom_k],
        positions_a[atom_i],
        positions_a[atom_j],
        positions_a[atom_l],
    )
    angle_b = dihedral_angle(
        positions_b_in_a[atom_k],
        positions_b_in_a[atom_i],
        positions_b_in_a[atom_j],
        positions_b_in_a[atom_l],
    )
    return atom_k, atom_i, atom_j, atom_l, _wrap_angle(angle_b - angle_a)


def _auto_torsion_bonds(molecule: CrystalMolecule) -> list[tuple[int, int]]:
    """Return deterministic acyclic bridge bonds with two torsion neighbours."""
    bonds = []
    positions = molecule.get_positions()
    for atom_i, atom_j in sorted(nx.bridges(molecule.graph)):
        try:
            descriptor = _torsion_descriptor(
                molecule, positions, positions, (atom_i, atom_j)
            )
        except ValueError:
            # A collinear bridge cannot define a torsion; leave it to the
            # residual Cartesian interpolation instead of inventing an angle.
            continue
        if descriptor is not None:
            bonds.append((int(atom_i), int(atom_j)))
    return bonds


def _interpolate_ring_positions(
    molecule_a: CrystalMolecule,
    base_positions: np.ndarray,
    positions_b_in_a: np.ndarray,
    ring_atoms: Sequence[int],
    lam: float,
) -> np.ndarray:
    """Interpolate in-plane coordinates and CP modes for one explicit ring."""
    ring = tuple(int(index) for index in ring_atoms)
    b_molecule = molecule_a.copy()
    b_molecule.set_positions(positions_b_in_a)
    cp_a = puckering_coordinates(molecule_a, ring)
    cp_b = puckering_coordinates(b_molecule, ring)
    amplitudes = (1.0 - lam) * cp_a.amplitudes + lam * cp_b.amplitudes
    phases = cp_a.phases.copy()
    paired = np.isfinite(cp_a.phases) & np.isfinite(cp_b.phases)
    phases[paired] = cp_a.phases[paired] + lam * np.array(
        [_wrap_angle(b - a) for a, b in zip(cp_a.phases[paired], cp_b.phases[paired])]
    )
    z = reconstruct_z_from_modes(len(ring), amplitudes, phases)
    positions_a = np.asarray(molecule_a.get_positions(), dtype=float)
    ring_a = positions_a[list(ring)]
    ring_b = np.asarray(positions_b_in_a, dtype=float)[list(ring)]
    normal = cp_a.mean_plane_normal
    center = cp_a.mean_plane_center
    ring_a_in_plane = ring_a - ((ring_a - center) @ normal)[:, None] * normal
    ring_b_in_plane = ring_b - ((ring_b - center) @ normal)[:, None] * normal
    ring_in_plane = (1.0 - lam) * ring_a_in_plane + lam * ring_b_in_plane
    result = ring_in_plane + z[:, None] * normal
    positions = np.asarray(base_positions, dtype=float).copy()
    positions[list(ring)] = result
    return positions


def _center_positions(
    positions: np.ndarray,
    molecule: CrystalMolecule,
    center: np.ndarray,
) -> np.ndarray:
    """Translate coordinates so their mass centre is ``center``."""
    coords = np.asarray(positions, dtype=float)
    masses = np.asarray(molecule.get_masses(), dtype=float)
    current = np.average(coords, axis=0, weights=masses)
    return coords - current + np.asarray(center, dtype=float)


def interpolate_molecule_with_internal_dofs(
    mol_a: CrystalMolecule,
    mol_b: CrystalMolecule,
    n_images: int = 11,
    *,
    rigid_method: str = "se3_screw",
    ring_atoms: Sequence[int] | None = None,
    torsion_bonds: Sequence[tuple[int, int]] | None = None,
) -> list[CrystalMolecule]:
    """Interpolate two conformers using rigid and internal coordinates.

    The endpoints are atom-mapped with the existing graph-aware matcher.  The
    rigid component follows the selected SE(3)/SO(3)/SLERP path, while bridge
    bonds use shortest signed dihedral changes.  When ``ring_atoms`` is given,
    its Cremer-Pople amplitudes and phases are interpolated for the ring
    coordinates.  If omitted, the first simple ring is used when one exists.

    Ring closure is deliberately not optimized here; callers needing exact
    constrained closure can pass the intermediate frames to the solver tracked
    in issue #118.
    """
    if not isinstance(mol_a, CrystalMolecule) or not isinstance(mol_b, CrystalMolecule):
        raise TypeError("mol_a and mol_b must be CrystalMolecule instances.")
    method = coerce_interpolation_method(rigid_method)
    if isinstance(n_images, bool) or int(n_images) != n_images or n_images < 1:
        raise ValueError("n_images must be an integer >= 1")

    from .interpolation import best_atom_mapping

    order_b = best_atom_mapping(mol_a, mol_b)
    positions_a = np.asarray(mol_a.get_positions(), dtype=float)
    positions_b = np.asarray(mol_b.get_positions(), dtype=float)[order_b]
    com_a = np.asarray(mol_a.get_center_of_mass(), dtype=float)
    com_b = np.asarray(mol_b.get_center_of_mass(), dtype=float)
    centered_a = positions_a - com_a
    centered_b = positions_b - com_b
    # This provisional alignment is used only to express ring-plane coordinates
    # in A's frame.  The actual rigid pose is fitted *after* internal motion is
    # constructed below; fitting the complete A→B pair first double-counts
    # internal torsions.
    provisional_rotation, _ = kabsch_align(centered_a, centered_b)
    positions_b_in_a = centered_b @ provisional_rotation + com_a

    if torsion_bonds is None:
        torsion_bonds = _auto_torsion_bonds(mol_a)
    torsion_descriptors = []
    for bond in torsion_bonds:
        # Dihedrals are invariant under the endpoint's rigid pose, so compare
        # directly with mapped B coordinates.  This keeps the internal path
        # independent of the provisional Kabsch orientation.
        descriptor = _torsion_descriptor(mol_a, positions_a, positions_b, bond)
        if descriptor is not None:
            torsion_descriptors.append(
                (tuple(int(value) for value in bond), descriptor[-1])
            )

    if ring_atoms is None:
        ring_systems = find_ring_systems(mol_a)
        simple = [system for system in ring_systems if system.is_simple]
        ring_atoms = simple[0].ring_atoms if simple else None
    if ring_atoms is not None:
        ring_atoms = tuple(int(index) for index in ring_atoms)
        puckering_coordinates(mol_a, ring_atoms)

    def _internal_without_rigid(fraction: float) -> np.ndarray:
        """Build internal coordinates in the A reference frame.

        Bridge torsions and ring modes provide the requested internal path;
        any remaining endpoint displacement is added as a residual below.
        Evaluating this function at one is intentional: it gives the residual
        endpoint of the selected internal-coordinate model without snapping.
        """
        if fraction <= 0.0:
            return positions_a.copy()

        if torsion_descriptors:
            working = mol_a.copy()
            from .bond_rotation import rotate_fragment_about_bond

            for (atom_i, atom_j), delta in torsion_descriptors:
                working = rotate_fragment_about_bond(
                    working,
                    atom_i,
                    atom_j,
                    np.degrees(fraction * delta),
                )
            raw = np.asarray(working.get_positions(), dtype=float)
        else:
            raw = (1.0 - fraction) * positions_a + fraction * positions_b_in_a

        if ring_atoms is not None and fraction <= 1.0:
            # Replace only the selected ring indices; preserve side-chain and
            # other internal motion already present in ``raw``.
            raw = _interpolate_ring_positions(
                mol_a, raw, positions_b_in_a, ring_atoms, fraction
            )
        return _center_positions(raw, mol_a, com_a)

    # A torsion/ring path can leave residual bond-length or bond-angle changes
    # at B.  Carry that residual continuously instead of forcing B only on the
    # final frame (which creates an endpoint jump).
    internal_endpoint = _internal_without_rigid(1.0)
    internal_center = com_a
    centered_internal_endpoint = internal_endpoint - internal_center
    residual_rotation, _ = kabsch_align(centered_internal_endpoint, centered_b)
    residual_translation = com_b - internal_center
    # Express B in the endpoint of the internal path's frame.  The difference
    # is the residual bond-length/angle (or ring) motion to add continuously.
    positions_b_in_internal = centered_b @ residual_rotation + internal_center
    internal_residual = positions_b_in_internal - internal_endpoint

    frames: list[CrystalMolecule] = []
    for lam in path_lambda_values(int(n_images), True):
        fraction = float(lam)
        if fraction >= 1.0:
            # Use the residual-frame target so the endpoint is reached through
            # the same decomposition as interior frames; no final-frame snap
            # is needed to hide a discontinuity.
            internal = positions_b_in_internal.copy()
        else:
            internal = _internal_without_rigid(fraction) + fraction * internal_residual
            internal = _center_positions(internal, mol_a, com_a)

        internal_com = np.average(internal, axis=0, weights=mol_a.get_masses())
        internal_centered = internal - internal_com + com_a
        pose = interpolate_rigid_positions(
            internal_centered,
            center=internal_center,
            rotation=residual_rotation,
            translation=residual_translation,
            lam=fraction,
            method=method,
        )
        # Keep the public endpoint bit-for-bit identical to mapped B.  The
        # residual-frame construction above already makes this a continuous
        # limit; this assignment only removes floating-point Kabsch noise.
        if fraction >= 1.0:
            pose = positions_b.copy()
        elif fraction <= 0.0:
            pose = positions_a.copy()
        frame = mol_a.copy()
        frame.set_positions(pose)
        _strip_stale_frac_arrays(frame)
        frame._calc_results = None
        # ``CrystalMolecule.copy()`` also copies a materialized graph.  Updating
        # coordinates alone therefore leaves cached edge vectors/distances at
        # the source geometry; refresh the cache for every emitted frame.
        _refresh_contiguous_bond_geometry(frame)
        frames.append(frame)
    return frames
