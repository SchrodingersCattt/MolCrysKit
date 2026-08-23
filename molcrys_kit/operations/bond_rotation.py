"""Graph-based rotation of molecular fragments about covalent bonds.

The operations in this module treat each side of an acyclic bond as a rigid
fragment.  They are intended as the fundamental geometry primitive for torsion
scans and conformational-path construction.

Ring bonds are detected but deliberately rejected: rotating one side of a ring
bond independently would break ring closure.  Cyclic conformational changes
require a constrained ring solver rather than a bond-cut operation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Sequence

import networkx as nx
import numpy as np

from ..constants.config import (
    BOND_ROTATION_AXIS_TOLERANCE,
    KEY_FRAC_X,
    KEY_FRAC_Y,
    KEY_FRAC_Z,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import (
    CrystalMolecule,
    _refresh_contiguous_bond_geometry,
    _strip_stale_frac_arrays,
)
from ..utils.geometry import get_rotation_matrix


class BondRotationError(ValueError):
    """Base class for bond/fragment rotation errors."""


class BondNotFoundError(BondRotationError):
    """Raised when the requested atom pair is not a molecular-graph edge."""


class BondRotationSelectionError(BondRotationError):
    """Raised when the requested moving fragment is ambiguous or inconsistent."""


class RingBondRotationError(BondRotationError):
    """Raised when ordinary bond rotation is requested for a ring bond."""


@dataclass(frozen=True)
class BondPartition:
    """Graph partition produced by removing a molecular bond.

    Parameters
    ----------
    atom_i, atom_j
        Directed bond-axis atoms.  Positive rotation follows the right-hand
        rule around the Cartesian vector from ``atom_i`` to ``atom_j``.
    fixed_atoms, moving_atoms
        Default components when the side containing ``atom_j`` moves.
    is_ring_bond
        True when removing the edge does not disconnect the molecular graph.
    component_sizes
        Sizes of the two components for a bridge; ``(n_atoms, n_atoms)`` for a
        ring edge because no graph partition exists.
    cycle_atoms
        Deterministically selected cycle containing the edge, empty for a
        bridge.
    """

    atom_i: int
    atom_j: int
    fixed_atoms: tuple[int, ...]
    moving_atoms: tuple[int, ...]
    is_ring_bond: bool
    component_sizes: tuple[int, int]
    cycle_atoms: tuple[int, ...] = ()


def _validate_atom_index(molecule: CrystalMolecule, atom_index: int) -> None:
    if atom_index < 0 or atom_index >= len(molecule):
        raise IndexError(
            f"Atom index {atom_index} out of range for molecule with "
            f"{len(molecule)} atoms."
        )


def _cycles_containing_edge(
    graph: nx.Graph, atom_i: int, atom_j: int
) -> list[tuple[int, ...]]:
    cycles = []
    for cycle in nx.cycle_basis(graph):
        edges = {
            frozenset((cycle[k], cycle[(k + 1) % len(cycle)]))
            for k in range(len(cycle))
        }
        if frozenset((atom_i, atom_j)) in edges:
            cycles.append(tuple(sorted(int(index) for index in cycle)))
    return sorted(cycles, key=lambda cycle: (len(cycle), cycle))


def partition_at_bond(
    molecule: CrystalMolecule,
    atom_i: int,
    atom_j: int,
) -> BondPartition:
    """Partition a molecule by removing the directed bond ``atom_i``–``atom_j``.

    For a bridge, the default moving component contains ``atom_j``.  For a ring
    edge no valid rigid-fragment partition exists; the returned partition has
    ``is_ring_bond=True`` and records one cycle containing the edge. Atom
    indices are local to ``molecule``. A disconnected molecular graph is
    rejected because unrelated components cannot be assigned to either side.
    """
    _validate_atom_index(molecule, atom_i)
    _validate_atom_index(molecule, atom_j)
    if atom_i == atom_j:
        raise BondNotFoundError("A bond requires two different atom indices.")

    graph = molecule.get_graph()
    if not nx.is_connected(graph):
        raise BondRotationSelectionError(
            "Bond rotation requires a connected molecular graph; found "
            f"{nx.number_connected_components(graph)} components."
        )
    if not graph.has_edge(atom_i, atom_j):
        raise BondNotFoundError(
            f"Atoms {atom_i} and {atom_j} are not bonded in the molecular graph."
        )

    cut_graph = graph.copy()
    cut_graph.remove_edge(atom_i, atom_j)
    components = [set(component) for component in nx.connected_components(cut_graph)]

    if len(components) == 1:
        cycle_atoms = _cycles_containing_edge(graph, atom_i, atom_j)
        cycle = cycle_atoms[0] if cycle_atoms else ()
        all_atoms = tuple(sorted(int(index) for index in graph.nodes))
        return BondPartition(
            atom_i=atom_i,
            atom_j=atom_j,
            fixed_atoms=all_atoms,
            moving_atoms=all_atoms,
            is_ring_bond=True,
            component_sizes=(len(all_atoms), len(all_atoms)),
            cycle_atoms=cycle,
        )

    if len(components) != 2:
        raise BondRotationSelectionError(
            f"Removing bond {atom_i}-{atom_j} produced {len(components)} "
            "components; expected exactly two."
        )

    component_i = next(component for component in components if atom_i in component)
    component_j = next(component for component in components if atom_j in component)
    fixed = tuple(sorted(int(index) for index in component_i))
    moving = tuple(sorted(int(index) for index in component_j))
    return BondPartition(
        atom_i=atom_i,
        atom_j=atom_j,
        fixed_atoms=fixed,
        moving_atoms=moving,
        is_ring_bond=False,
        component_sizes=(len(fixed), len(moving)),
    )


def _resolved_moving_atoms(
    partition: BondPartition,
    moving_side: Literal["i", "j"],
    moving_atoms: Sequence[int] | None,
) -> tuple[int, ...]:
    if moving_side not in {"i", "j"}:
        raise BondRotationSelectionError(
            f"Unknown moving_side {moving_side!r}; expected 'i' or 'j'."
        )

    component_i = set(partition.fixed_atoms)
    component_j = set(partition.moving_atoms)
    default = component_j if moving_side == "j" else component_i

    if moving_atoms is None:
        return tuple(sorted(default))

    requested = {int(index) for index in moving_atoms}
    if requested not in {frozenset(component_i), frozenset(component_j)}:
        raise BondRotationSelectionError(
            "moving_atoms must equal exactly one complete graph component "
            "produced by cutting the bond."
        )
    return tuple(sorted(requested))


def _set_positions_clean(molecule: CrystalMolecule, positions: np.ndarray) -> None:
    molecule.set_positions(np.asarray(positions, dtype=float))
    _strip_stale_frac_arrays(molecule)
    _refresh_contiguous_bond_geometry(molecule)


def rotate_fragment_about_bond(
    molecule: CrystalMolecule,
    atom_i: int,
    atom_j: int,
    angle: float,
    *,
    moving_side: Literal["i", "j"] = "j",
    moving_atoms: Sequence[int] | None = None,
) -> CrystalMolecule:
    """Return a copy with one rigid fragment rotated about a molecular bond.

    Parameters
    ----------
    molecule
        Contiguous (unwrapped) molecular coordinates.
    atom_i, atom_j
        Molecule-local indices defining the directed bond axis. Positive angles
        use the right-hand rule around the vector from ``atom_i`` to ``atom_j``.
    angle
        Rotation angle in degrees.
    moving_side
        ``"j"`` (default) rotates the graph component containing ``atom_j``;
        ``"i"`` rotates the opposite component.
    moving_atoms
        Optional molecule-local explicit selection. It must equal exactly one
        complete graph component produced by cutting the bond.

    Notes
    -----
    Ring bonds and disconnected molecular graphs are rejected. The input
    chemical graph is copied unchanged; connectivity is not inferred again
    from the rotated geometry. Use a constrained ring-conformation operation
    for cyclic changes.
    """
    partition = partition_at_bond(molecule, atom_i, atom_j)
    if partition.is_ring_bond:
        cycle_text = (
            f"; cycle atoms={list(partition.cycle_atoms)}"
            if partition.cycle_atoms
            else ""
        )
        raise RingBondRotationError(
            f"Bond {atom_i}-{atom_j} is a ring bond{cycle_text}. Ordinary "
            "fragment rotation would break ring closure."
        )

    selected = _resolved_moving_atoms(partition, moving_side, moving_atoms)
    positions = np.asarray(molecule.get_positions(), dtype=float).copy()
    pivot = positions[atom_i]
    axis = positions[atom_j] - positions[atom_i]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < BOND_ROTATION_AXIS_TOLERANCE:
        raise BondRotationError(
            f"Bond axis {atom_i}-{atom_j} has zero length; cannot rotate."
        )

    rotation = get_rotation_matrix(axis, np.radians(float(angle)))
    selected_array = np.asarray(selected, dtype=int)
    relative = positions[selected_array] - pivot
    positions[selected_array] = relative @ rotation.T + pivot

    rotated = molecule.copy()
    _set_positions_clean(rotated, positions)
    return rotated


def rotate_fragment_in_crystal(
    crystal: MolecularCrystal,
    molecule_index: int,
    atom_i: int,
    atom_j: int,
    angle: float,
    *,
    moving_side: Literal["i", "j"] = "j",
    moving_atoms: Sequence[int] | None = None,
) -> MolecularCrystal:
    """Return a crystal copy with one molecular fragment rotated about a bond.

    ``atom_i``, ``atom_j``, and ``moving_atoms`` are local indices in the
    selected molecule. For molecules read from a crystal,
    ``molecule.info["atom_indices"]`` maps local indices to original crystal
    indices. Frame metadata and non-coordinate extra arrays are preserved;
    stale fractional-coordinate arrays and calculator results are invalidated.
    """
    if molecule_index < 0 or molecule_index >= len(crystal.molecules):
        raise IndexError(
            f"Molecule index {molecule_index} out of range for crystal with "
            f"{len(crystal.molecules)} molecules."
        )

    molecules = [molecule.copy() for molecule in crystal.molecules]
    molecules[molecule_index] = rotate_fragment_about_bond(
        crystal.molecules[molecule_index],
        atom_i,
        atom_j,
        angle,
        moving_side=moving_side,
        moving_atoms=moving_atoms,
    )
    preserved_extra_arrays = {
        key: value
        for key, value in crystal.extra_arrays.items()
        if key not in {KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z}
    }
    return MolecularCrystal(
        lattice=np.asarray(crystal.lattice, dtype=float).copy(),
        molecules=molecules,
        pbc=crystal.pbc,
        formula_moiety=crystal.formula_moiety,
        disorder_provenance=copy.deepcopy(crystal.disorder_provenance),
        calc_results=None,
        metadata=copy.deepcopy(crystal.metadata),
        extra_arrays=preserved_extra_arrays,
    )
