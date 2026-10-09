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
    left_neighbors = sorted(int(index) for index in graph.neighbors(atom_i) if index != atom_j)
    right_neighbors = sorted(int(index) for index in graph.neighbors(atom_j) if index != atom_i)
    if not left_neighbors or not right_neighbors:
        return None
    atom_k, atom_l = left_neighbors[0], right_neighbors[0]
    angle_a = dihedral_angle(
        positions_a[atom_k], positions_a[atom_i], positions_a[atom_j], positions_a[atom_l]
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
        if _torsion_descriptor(molecule, positions, positions, (atom_i, atom_j)) is not None:
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
    its Cremer–Pople amplitudes and phases are interpolated for the ring
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
    rotation, _ = kabsch_align(centered_a, centered_b)
    translation = com_b - com_a
    positions_b_in_a = centered_b @ rotation + com_a

    if torsion_bonds is None:
        torsion_bonds = _auto_torsion_bonds(mol_a)
    torsion_descriptors = []
    for bond in torsion_bonds:
        descriptor = _torsion_descriptor(mol_a, positions_a, positions_b_in_a, bond)
        if descriptor is not None:
            torsion_descriptors.append((tuple(int(value) for value in bond), descriptor[-1]))

    if ring_atoms is None:
        ring_systems = find_ring_systems(mol_a)
        simple = [system for system in ring_systems if system.is_simple]
        ring_atoms = simple[0].ring_atoms if simple else None
    if ring_atoms is not None:
        ring_atoms = tuple(int(index) for index in ring_atoms)
        puckering_coordinates(mol_a, ring_atoms)

    frames: list[CrystalMolecule] = []
    for lam in path_lambda_values(int(n_images), True):
        fraction = float(lam)
        if fraction <= 0.0:
            internal = positions_a.copy()
        elif fraction >= 1.0:
            internal = positions_b_in_a.copy()
        elif torsion_descriptors:
            working = mol_a.copy()
            for (atom_i, atom_j), delta in torsion_descriptors:
                from .bond_rotation import rotate_fragment_about_bond

                working = rotate_fragment_about_bond(
                    working,
                    atom_i,
                    atom_j,
                    np.degrees(fraction * delta),
                )
            internal = np.asarray(working.get_positions(), dtype=float)
            internal_com = np.asarray(working.get_center_of_mass(), dtype=float)
            internal += com_a - internal_com
        else:
            internal = positions_a + fraction * (positions_b_in_a - positions_a)

        if ring_atoms is not None and 0.0 < fraction < 1.0:
            internal = _interpolate_ring_positions(
                mol_a, internal, positions_b_in_a, ring_atoms, fraction
            )

        internal_com = np.average(internal, axis=0, weights=mol_a.get_masses())
        internal_centered = internal - internal_com + com_a
        pose = interpolate_rigid_positions(
            internal_centered,
            center=com_a,
            rotation=rotation,
            translation=translation,
            lam=fraction,
            method=method,
        )
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
