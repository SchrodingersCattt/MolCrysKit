"""Stable, renderer-ready records derived from molecular crystals.

The records in this module are immutable snapshots.  They deliberately use
plain Python scalars and tuples so downstream tools can serialise them without
depending on ASE's mutable array model or on private ``Atoms.info`` payloads.
"""

from __future__ import annotations

from dataclasses import dataclass


Vector3 = tuple[float, float, float]
ImageShift = tuple[int, int, int]
Tensor3 = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class SiteRecord:
    """One crystallographic site in a :class:`MolecularCrystal` snapshot.

    ``global_index`` follows :meth:`MolecularCrystal.to_ase` ordering, while
    ``molecule_index`` and ``local_index`` locate the same atom in the
    molecule partition.  ``asym_index`` is the source asymmetric-unit index
    when known.  ``image_shift`` records the integer lattice translation used
    to make the molecule contiguous.
    """

    site_id: str
    global_index: int
    molecule_index: int
    local_index: int
    symbol: str
    label: str
    cartesian_position_A: Vector3
    fractional_position: Vector3
    occupancy: float
    disorder_group: int
    disorder_assembly: str | None
    asym_index: int | None
    sym_op_index: int | None
    site_symmetry_order: int
    image_shift: ImageShift
    uiso_A2: float | None
    u_cart_A2: Tensor3 | None


@dataclass(frozen=True)
class BondRecord:
    """One canonical intramolecular bond with periodic-image provenance.

    The bond is oriented from ``left`` to ``right``.  ``right_image_shift``
    is the lattice translation applied to the right source site relative to
    the left source site, and ``vector_A`` is the corresponding Cartesian
    bond vector.
    """

    molecule_index: int
    left_local_index: int
    right_local_index: int
    left_global_index: int
    right_global_index: int
    left_asym_index: int | None
    right_asym_index: int | None
    right_image_shift: ImageShift
    vector_A: Vector3
    distance_A: float


__all__ = ["BondRecord", "SiteRecord"]
