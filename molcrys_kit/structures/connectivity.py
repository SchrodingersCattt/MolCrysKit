"""Bond-edge roles and the crystal connectivity regime.

Chemistry heuristics live here so a renderer can stay free of element
whitelists.  A stick-eligible edge has role ``"covalent"``.  A
metal–ligand or metal–metal contact kept for topology and polyhedra has
role ``"coordination"``.  The crystal is ``"molecular"`` when the signed
bond graph has translational rank 0 and ``"extended"`` when any component
winds around the lattice.  Display radii and ball-versus-stick styling are
out of scope.
"""

from __future__ import annotations

from typing import Sequence

from ..constants import (
    NON_METAL_THRESHOLD_FACTOR,
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..constants.config import BONDING_CONFIG, BONDING_THRESHOLDS
from .records import (
    BOND_ROLE_COORDINATION,
    BOND_ROLE_COVALENT,
    BOND_ROLES,
    CONNECTIVITY_REGIME_EXTENDED,
    CONNECTIVITY_REGIME_MOLECULAR,
    ComponentConnectivity,
    ConnectivityReport,
)

_DEFAULT_ATOMIC_RADIUS = float(BONDING_CONFIG["DEFAULT_ATOMIC_RADIUS"])
_METAL_NONMETAL_COVALENT_MAX = float(BONDING_THRESHOLDS["METAL_NONMETAL_COVALENT_MAX"])


def _atomic_radius(symbol: str) -> float:
    if has_atomic_radius(symbol):
        return float(get_atomic_radius(symbol))
    return _DEFAULT_ATOMIC_RADIUS


def covalent_metal_nonmetal_limit(symbol_i: str, symbol_j: str) -> float:
    """Return the metal–non-metal distance ceiling for a covalent role.

    The ceiling is the tighter of the absolute metal–organic cutoff
    ``METAL_NONMETAL_COVALENT_MAX`` and the ordinary non-metal radius sum
    ``(r_i + r_j) * NON_METAL_THRESHOLD_FACTOR``.  Contacts accepted only
    because ``METAL_NON_METAL_THRESHOLD_FACTOR`` reaches farther than this
    ceiling are coordination edges.
    """
    radius_sum = _atomic_radius(symbol_i) + _atomic_radius(symbol_j)
    return min(
        _METAL_NONMETAL_COVALENT_MAX,
        radius_sum * float(NON_METAL_THRESHOLD_FACTOR),
    )


def classify_bond_role(symbol_i: str, symbol_j: str, distance_A: float) -> str:
    """Classify one accepted neighbor-list edge.

    Non-metal pairs are ``"covalent"``.  A metal–non-metal pair is
    ``"covalent"`` at or below :func:`covalent_metal_nonmetal_limit` and
    ``"coordination"`` beyond it.  Metal–metal pairs are ``"coordination"``:
    the bond graph only keeps those that pass the short metal–metal cutoff,
    and they are not covalent sticks.

    Parameters
    ----------
    symbol_i, symbol_j : str
        Element symbols of the two endpoints.
    distance_A : float
        Contact distance in angstroms.

    Returns
    -------
    str
        ``"covalent"`` or ``"coordination"``.
    """
    metal_i = is_metal_element(symbol_i)
    metal_j = is_metal_element(symbol_j)
    if not metal_i and not metal_j:
        return BOND_ROLE_COVALENT
    if metal_i and metal_j:
        return BOND_ROLE_COORDINATION
    if float(distance_A) <= covalent_metal_nonmetal_limit(symbol_i, symbol_j):
        return BOND_ROLE_COVALENT
    return BOND_ROLE_COORDINATION


def resolve_bond_role(
    stored,
    symbol_i: str,
    symbol_j: str,
    distance_A: float,
) -> str:
    """Return a stored role when it is valid, otherwise classify the edge."""
    if isinstance(stored, str) and stored in BOND_ROLES:
        return stored
    return classify_bond_role(symbol_i, symbol_j, distance_A)


def _translational_rank(
    atom_ids: Sequence[str],
    edges: Sequence[tuple[str, str, tuple[int, int, int]]],
    periodic_axes: tuple[bool, bool, bool],
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    """Rank of lattice windings on one quotient-graph component.

    Delegates to :func:`molcrys_kit.chemistry.topology.analyze_periodic_topology`,
    which treats a cell-face crossing as a finite unwrap unless a cycle
    accumulates a nonzero lattice translation.
    """
    from ..chemistry.models import ChemicalBond
    from ..chemistry.topology import analyze_periodic_topology

    bonds = tuple(
        ChemicalBond(
            atom1_id=left,
            atom2_id=right,
            atom2_image_shift=shift,
        )
        for left, right, shift in edges
    )
    topology = analyze_periodic_topology(
        tuple(atom_ids),
        bonds,
        periodic_axes=periodic_axes,
    )
    return topology.rank, tuple(topology.translation_generators)


def build_connectivity_report(
    bond_records,
    n_components: int,
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> ConnectivityReport:
    """Build a crystal connectivity report from canonical bond records.

    Parameters
    ----------
    bond_records
        Records from :meth:`MolecularCrystal.get_bond_records`.  Every
        accepted neighbor-list edge participates, including coordination
        contacts.  Edge roles are ignored: regime is topology only.
    n_components : int
        Number of molecule components, including isolated atoms.
    periodic_axes : tuple of bool
        Which lattice axes are periodic.  Windings on a non-periodic axis
        do not raise the rank.
    """
    if len(periodic_axes) != 3:
        raise ValueError("periodic_axes must contain three booleans")
    if n_components < 0:
        raise ValueError("n_components must not be negative")

    grouped: list[list] = [[] for _ in range(n_components)]
    for record in bond_records:
        index = int(record.molecule_index)
        if index < 0 or index >= n_components:
            raise ValueError("bond record molecule_index is outside the crystal")
        grouped[index].append(record)

    axes = tuple(bool(flag) for flag in periodic_axes)
    components: list[ComponentConnectivity] = []
    for molecule_index, records in enumerate(grouped):
        atom_ids = tuple(
            sorted(
                {str(int(record.left_global_index)) for record in records}
                | {str(int(record.right_global_index)) for record in records}
            )
        )
        edges = tuple(
            (
                str(int(record.left_global_index)),
                str(int(record.right_global_index)),
                tuple(int(value) for value in record.right_image_shift),
            )
            for record in records
        )
        rank, generators = _translational_rank(atom_ids, edges, axes)
        regime = (
            CONNECTIVITY_REGIME_MOLECULAR if rank == 0 else CONNECTIVITY_REGIME_EXTENDED
        )
        components.append(
            ComponentConnectivity(
                molecule_index=molecule_index,
                rank=rank,
                regime=regime,
                translation_generators=generators,
            )
        )

    max_rank = max((component.rank for component in components), default=0)
    regime = CONNECTIVITY_REGIME_EXTENDED if max_rank else CONNECTIVITY_REGIME_MOLECULAR
    return ConnectivityReport(
        regime=regime,
        max_rank=max_rank,
        components=tuple(components),
    )


__all__ = [
    "build_connectivity_report",
    "classify_bond_role",
    "covalent_metal_nonmetal_limit",
    "resolve_bond_role",
]
