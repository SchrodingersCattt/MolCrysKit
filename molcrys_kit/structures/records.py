"""Stable, renderer-ready records derived from molecular crystals.

The records in this module are immutable snapshots.  They deliberately use
plain Python scalars and tuples so downstream tools can serialise them without
depending on ASE's mutable array model or on private ``Atoms.info`` payloads.

``BondRecord.role`` and :class:`ConnectivityReport` are the chemistry signals
a renderer needs in order to choose stick edges and a molecular versus
extended presentation.  They are plain strings.  Visual style (display radii,
ball versus stick) is intentionally not represented here.
"""

from __future__ import annotations

from dataclasses import dataclass


Vector3 = tuple[float, float, float]
ImageShift = tuple[int, int, int]
Tensor3 = tuple[Vector3, Vector3, Vector3]

# Renderer-facing edge roles.  Values are the serialised form.
BOND_ROLE_COVALENT = "covalent"
BOND_ROLE_COORDINATION = "coordination"
BOND_ROLES = frozenset({BOND_ROLE_COVALENT, BOND_ROLE_COORDINATION})

# Crystal-level bond-graph regimes.  Rank 0 is molecular; rank >= 1 is extended.
CONNECTIVITY_REGIME_MOLECULAR = "molecular"
CONNECTIVITY_REGIME_EXTENDED = "extended"
CONNECTIVITY_REGIMES = frozenset(
    {CONNECTIVITY_REGIME_MOLECULAR, CONNECTIVITY_REGIME_EXTENDED}
)


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
    isotope: int | None
    formal_charge: int | None
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

    ``role`` is a plain string, ``"covalent"`` or ``"coordination"``
    (:data:`BOND_ROLE_COVALENT`, :data:`BOND_ROLE_COORDINATION`).  Neighbor-list
    edges emitted by MolCrysKit always set it.  The default is ``"covalent"``
    so existing positional constructions keep working.  Renderers draw sticks
    for ``"covalent"`` and keep ``"coordination"`` for topology or polyhedra.
    The same edge identity is ``(left_global_index, right_global_index,
    right_image_shift)``.  Bond order is not part of this record.
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
    role: str = BOND_ROLE_COVALENT

    def __post_init__(self) -> None:
        if self.role not in BOND_ROLES:
            raise ValueError(
                f"role must be 'covalent' or 'coordination', got {self.role!r}"
            )


@dataclass(frozen=True)
class ComponentConnectivity:
    """Translational rank of one bond-graph component.

    ``rank`` is the dimension of the lattice-winding cycle space on the
    signed bond graph (0 finite, 1 chain, 2 layer, 3 framework).
    ``regime`` is ``"molecular"`` when ``rank`` is 0 and ``"extended"``
    otherwise.  ``translation_generators`` is an independent basis of the
    nonzero winding vectors.  Element symbols are not part of this record.
    """

    molecule_index: int
    rank: int
    regime: str
    translation_generators: tuple[ImageShift, ...]

    def __post_init__(self) -> None:
        if self.rank not in {0, 1, 2, 3}:
            raise ValueError("translational rank must be between 0 and 3")
        expected = (
            CONNECTIVITY_REGIME_MOLECULAR
            if self.rank == 0
            else CONNECTIVITY_REGIME_EXTENDED
        )
        if self.regime != expected:
            raise ValueError("regime does not match translational rank")
        if len(self.translation_generators) != self.rank:
            raise ValueError("generator count must equal translational rank")


@dataclass(frozen=True)
class ConnectivityReport:
    """Crystal-level connectivity regime of the signed bond graph.

    ``regime`` is ``"extended"`` when any component has translational rank
    at least 1, and ``"molecular"`` otherwise (including an empty crystal).
    ``max_rank`` is the largest component rank.  Coordination contacts stay
    in the graph: an ionic framework is extended because those edges wind
    around the lattice, not because of an element whitelist.
    """

    regime: str
    max_rank: int
    components: tuple[ComponentConnectivity, ...]

    def __post_init__(self) -> None:
        if self.components:
            expected_max = max(component.rank for component in self.components)
        else:
            expected_max = 0
        expected = (
            CONNECTIVITY_REGIME_EXTENDED
            if expected_max
            else CONNECTIVITY_REGIME_MOLECULAR
        )
        if self.max_rank != expected_max or self.regime != expected:
            raise ValueError("crystal regime does not match component ranks")
        indices = [component.molecule_index for component in self.components]
        if indices != sorted(indices):
            raise ValueError("components must be ordered by molecule_index")


__all__ = [
    "BOND_ROLE_COORDINATION",
    "BOND_ROLE_COVALENT",
    "BOND_ROLES",
    "BondRecord",
    "CONNECTIVITY_REGIME_EXTENDED",
    "CONNECTIVITY_REGIME_MOLECULAR",
    "CONNECTIVITY_REGIMES",
    "ComponentConnectivity",
    "ConnectivityReport",
    "SiteRecord",
]
