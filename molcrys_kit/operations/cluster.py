"""Carve finite, H-capped clusters out of a periodic crystal.

This module turns a periodic :class:`MolecularCrystal` into one or more
finite, hydrogen-capped :class:`CrystalCluster` objects suitable for
finite-cluster QM work (Gaussian / ORCA / Psi4 etc.).  It is a generic
framework tool -- the algorithm does not assume MOFs, COFs, zeolites or
any specific chemistry; system-specific parameter choices live with the
caller.

Two carving modes are supported:

* ``bond_shells`` (default) -- chemistry-aware BFS from one or more seed
  atoms.  The only cuttable bonds are **single C-C bonds that are not
  part of any small ring** (the BFS leaves M-X, C=O, C-N, C=C and ring
  bonds intact).  Each cut is capped with an H atom placed along the
  original bond vector at the element-keyed X-H length from
  :data:`molcrys_kit.constants.config.BOND_LENGTHS`.  Hop budget is
  expressed in cut-boundary layers (``n_shells``), *not* raw bond hops.
  An optional rule (``stop_at_non_seed_metals``, on by default) treats
  any bond reaching a metal not in the current seed group as an
  implicit cut, so frameworks that loop back via M-X-X-M paths do not
  silently sweep the whole structure.
* ``rcut`` -- diagnostic radial cutoff: keep every atom within
  ``rcut`` Angstrom of any seed (minimum-image distance).  Same H-cap
  placement, but with a warning when a cut bond is not C-C (the
  classical red flag for accidentally severing a C-O / C-N bond).

Output is XYZ + a JSON sidecar carrying the full
:class:`ClusterProvenance` payload (kept-atom map, cut bonds, cap /
frozen indices, mode, shells / rcut, per-cap distances actually used,
the X-H table consulted).  The sidecar is the canonical handoff to any
downstream QM input writer (out of scope for MCK).

Out of scope
------------
* No domain typing (SBU / linker / paddle-wheel / pore / channel).  Such
  classification belongs in a downstream layer that builds on top of
  :class:`molcrys_kit.analysis.ChemicalEnvironment`.
* No charge / spin inference; the caller supplies these when writing
  the QM input from the XYZ + sidecar.
* No Gaussian / ORCA / Psi4 writer; the sidecar JSON is the integration
  point.

References
----------
The algorithm follows the standard "BFS from a chosen seed, cut only at
single non-ring C-C bonds, cap with H along the cut bond, freeze a
configurable shell" recipe shared across the QM-cluster literature.
Specific defaults (``seed_merge_radius``, ``n_shells``, ``freeze_shell``)
should be picked per system; see the project-specific cookbook (e.g. a
``RECIPE.md`` next to your CIF inputs) for citations and parameter
choices.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import networkx as nx

from ase import Atoms
from ase.neighborlist import neighbor_list

from ..analysis.cluster_provenance import ClusterProvenance
from ..analysis.interactions import get_bonding_threshold
from ..constants import (
    DEFAULT_NEIGHBOR_CUTOFF,
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..constants.config import BOND_LENGTHS
from ..structures.cluster import CrystalCluster
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import minimum_image_distance


# Default seed-merge radius (Angstrom).  ``0.0`` means "do not auto-group
# seeds" -- each resolved seed is its own cluster.  Callers working with
# clustered metal nodes (paddle-wheels ~3.0 A, M3/M6 nodes ~3.5-3.8 A,
# etc.) should set this to their node diameter; callers working with
# isolated centres or non-metal seeds should leave it at zero.
DEFAULT_SEED_MERGE_RADIUS = 0.0

# Fallback cap distance (Angstrom) when neither the explicit override nor
# the per-element BOND_LENGTHS lookup yields a value.  Equal to the C-H
# standard since the BOND_LENGTHS table also lists 1.09 A for "C-H".
_FALLBACK_CAP_DISTANCE = 1.09


def _resolve_cap_bond_lengths(
    overrides: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Build a {element: X-H bond length} dict from BOND_LENGTHS + overrides.

    The shared :data:`molcrys_kit.constants.config.BOND_LENGTHS` table is
    keyed as ``"C-H"``, ``"N-H"``, ... -- the same convention the
    ``HydrogenCompleter`` already accepts via its ``bond_lengths`` kwarg.
    The carver consumes the table after stripping the ``"-H"`` suffix so
    cap placement can do a quick ``table[symbol]`` lookup.
    """
    table: Dict[str, float] = {}
    for key, value in BOND_LENGTHS.items():
        if "-H" in key:
            element = key.split("-", 1)[0]
            table[element] = float(value)
    if overrides:
        for key, value in overrides.items():
            element = key.split("-", 1)[0] if "-H" in key else key
            table[element] = float(value)
    return table

# Default cut-boundary budget.  ``n_shells = 1`` corresponds to "include
# the seed plus the first linker fragment up to the first cuttable
# C-C bond"; ``0`` cuts at the very first cuttable bond.  Tune per system.
DEFAULT_N_SHELLS = 1


# Sentinel for "use the BOND_LENGTHS lookup keyed by the kept-side element"
# in the cap-distance argument.  Passing a positive float overrides the
# lookup with a uniform value (the original v0 behaviour).
DEFAULT_CAP_DISTANCE: Optional[float] = None


SeedSpec = Union[str, int, Sequence[int], Callable[[Atoms], Sequence[int]]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_seed_atoms(atoms: Atoms, seed: SeedSpec) -> List[int]:
    """Translate a flexible seed specification into a sorted list of global indices."""
    symbols = atoms.get_chemical_symbols()
    if isinstance(seed, str):
        target = seed.strip()
        indices = [i for i, sym in enumerate(symbols) if sym == target]
        if not indices:
            raise ValueError(
                f"No atoms with element '{target}' found in the parent crystal."
            )
        return indices
    if isinstance(seed, int):
        if seed < 0 or seed >= len(atoms):
            raise ValueError(f"Seed index {seed} out of range [0, {len(atoms)}).")
        return [seed]
    if callable(seed):
        return sorted(int(i) for i in seed(atoms))
    indices = sorted({int(i) for i in seed})
    for i in indices:
        if i < 0 or i >= len(atoms):
            raise ValueError(f"Seed index {i} out of range [0, {len(atoms)}).")
    return indices


def _partition_seeds_by_distance(
    atoms: Atoms,
    seed_indices: Sequence[int],
    lattice: np.ndarray,
    seed_merge_radius: float,
) -> List[List[int]]:
    """Group seeds into clusters using union-find under minimum-image distance.

    Two seeds belong to the same group iff their PBC minimum-image distance
    is at most ``seed_merge_radius``.  Returns a list of seed-index groups.
    """
    if len(seed_indices) <= 1:
        return [list(seed_indices)]

    frac = atoms.get_scaled_positions(wrap=True)
    parent = list(range(len(seed_indices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(len(seed_indices)):
        for b in range(a + 1, len(seed_indices)):
            dist = minimum_image_distance(
                frac[seed_indices[a]], frac[seed_indices[b]], lattice
            )
            if dist <= seed_merge_radius:
                union(a, b)

    groups: Dict[int, List[int]] = {}
    for idx, seed_global in enumerate(seed_indices):
        root = find(idx)
        groups.setdefault(root, []).append(int(seed_global))
    return [sorted(group) for group in groups.values()]


def _build_offset_bond_graph(
    atoms: Atoms,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
) -> nx.Graph:
    """Build a periodic bond graph with image-offset displacement vectors on edges.

    Follows the same pattern as :func:`molcrys_kit.io.cif._build_molecule_graph`:
    each edge stores ``vector`` (the displacement ``r_j + S @ lattice - r_i``)
    so the carver can place caps along the *actual* bond direction even when
    the bond crosses a periodic image.
    """
    symbols = atoms.get_chemical_symbols()
    i_list, j_list, d_list, D_vectors = neighbor_list(
        "ijdD", atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF
    )

    graph = nx.Graph()
    graph.add_nodes_from((i, {"symbol": s}) for i, s in enumerate(symbols))

    bond_thresholds = bond_thresholds or {}
    for i, j, distance, D_vec in zip(i_list, j_list, d_list, D_vectors):
        if i >= j:
            continue
        pair_key1, pair_key2 = (symbols[i], symbols[j]), (symbols[j], symbols[i])
        if pair_key1 in bond_thresholds or pair_key2 in bond_thresholds:
            threshold = bond_thresholds.get(
                pair_key1, bond_thresholds.get(pair_key2)
            )
        else:
            r_i = get_atomic_radius(symbols[i]) if has_atomic_radius(symbols[i]) else 0.5
            r_j = get_atomic_radius(symbols[j]) if has_atomic_radius(symbols[j]) else 0.5
            threshold = get_bonding_threshold(
                r_i,
                r_j,
                is_metal_element(symbols[i]),
                is_metal_element(symbols[j]),
            )
        if distance < threshold:
            graph.add_edge(int(i), int(j), vector=np.array(D_vec, dtype=float),
                           distance=float(distance))

    return graph


_MAX_CHEMICAL_RING_SIZE = 8  # covers furanose/pyranose/triazole/imidazole/benzene/cyclohexane


def _local_rings(
    subgraph: nx.Graph, max_ring_size: int = _MAX_CHEMICAL_RING_SIZE
) -> Dict[int, frozenset]:
    """Compute small-ring memberships only inside a local subgraph.

    Returns a dict mapping ``node -> frozenset(ring_id, ...)`` for every
    ring of size ``<= max_ring_size`` found by ``nx.minimum_cycle_basis``.
    The size cap is critical on periodic framework graphs: a fully-built
    bond graph of a 3D framework contains macrocyclic "topological rings"
    (closed paths that wind through the unit cell / channels) whose
    member bonds are not chemical ring bonds in any meaningful sense.
    Without the cap, the chemistry-aware carver would mark every linker
    C-C as ring-bonded and never find a cuttable bond.
    """
    if subgraph.number_of_edges() == 0:
        return {n: frozenset() for n in subgraph.nodes}

    relabel = {n: i for i, n in enumerate(subgraph.nodes)}
    inv = {i: n for n, i in relabel.items()}
    relabelled = nx.relabel_nodes(subgraph, relabel)
    try:
        rings = nx.minimum_cycle_basis(relabelled, weight=None)
    except nx.NetworkXError:  # pragma: no cover - defensive
        rings = []

    membership: Dict[int, Set[int]] = {n: set() for n in subgraph.nodes}
    ring_id = 0
    for ring_nodes in rings:
        if len(ring_nodes) > max_ring_size:
            continue
        for n in ring_nodes:
            membership[inv[n]].add(ring_id)
        ring_id += 1
    return {n: frozenset(r) for n, r in membership.items()}


def _is_cuttable_cc(
    symbols: Sequence[str],
    graph: nx.Graph,
    i: int,
    j: int,
    rings_of: Dict[int, frozenset],
) -> bool:
    """Return True iff the bond i-j is a single C-C bond outside any shared ring.

    A bond is *not* cuttable if either:

    * any endpoint is not carbon (we never cut M-X, C-O, C-N, C-H),
    * the bond participates in a small ring (we never break aromatic
      rings or saturated 5-/6-membered carbocyclic / heterocyclic rings;
      "small" means at most ``_MAX_CHEMICAL_RING_SIZE`` atoms so that
      topological macrocycles in the periodic graph do not count),
    * the bond looks multi-order (currently detected only via the bond
      length being clearly shorter than a single C-C: < 1.42 A signals
      C=C / aromatic; this is a conservative heuristic given ASE's
      neighbor_list has no order information).
    """
    if symbols[i] != "C" or symbols[j] != "C":
        return False

    rings_i = rings_of.get(i, frozenset())
    rings_j = rings_of.get(j, frozenset())
    if rings_i & rings_j:
        # Same ring: ring bond, never cut.
        return False

    # Conservative double-bond / aromatic detection by distance.
    # Single C-C ~ 1.50-1.55 A; aromatic C-C ~ 1.39 A; C=C ~ 1.33 A.
    # A 1.42 A threshold places the cut squarely in the single-bond regime.
    edge_data = graph.get_edge_data(i, j) or {}
    distance = edge_data.get("distance")
    if distance is not None and distance < 1.42:
        return False

    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


class ClusterCarver:
    """Carve finite molecular clusters out of a periodic crystal.

    Parameters
    ----------
    crystal : MolecularCrystal
        The parent (ideally already disorder-resolved) periodic crystal.
    seed_merge_radius : float, default 0.0
        Distance (Angstrom) under which adjacent seeds are auto-grouped
        into one cluster.  ``0.0`` (the default) means "no auto-grouping
        -- each resolved seed produces its own cluster".  Set this to
        the diameter of a multi-atom node (paddle-wheel ~3.0 A, trimer
        ~3.8 A, etc.) if you want one cluster per node group.
    bond_thresholds : dict, optional
        Per-element-pair distance overrides, in the same format that
        ``molcrys_kit.io.read_mol_crystal`` accepts.

    Notes
    -----
    The carver builds its own atom-level periodic bond graph with image
    offsets from scratch (via the same ``neighbor_list("ijdD", ...)``
    recipe used by :mod:`molcrys_kit.io.cif`); ``CrystalMolecule._build_graph``
    does *not* carry offsets and therefore cannot be reused.
    """

    def __init__(
        self,
        crystal: MolecularCrystal,
        seed_merge_radius: float = DEFAULT_SEED_MERGE_RADIUS,
        bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.crystal = crystal
        self.seed_merge_radius = float(seed_merge_radius)
        self._atoms: Atoms = crystal.to_ase()
        # Important: to_ase reorders atoms by molecule; we keep that
        # convention for all global indices throughout the carver.
        self._lattice: np.ndarray = np.array(crystal.lattice)
        self._graph: nx.Graph = _build_offset_bond_graph(
            self._atoms, bond_thresholds
        )

    # ----- public API -----------------------------------------------------

    def carve_bond_shells(
        self,
        seed: SeedSpec,
        n_shells: int = DEFAULT_N_SHELLS,
        freeze_shell: int = 1,
        cap_distance: Optional[float] = DEFAULT_CAP_DISTANCE,
        cap_bond_lengths: Optional[Dict[str, float]] = None,
        parent_label: Optional[str] = None,
        convention_reference: str = "",
        stop_at_non_seed_metals: bool = True,
    ) -> List[CrystalCluster]:
        """Chemistry-aware cluster carving from one or more seed atoms.

        See module docstring for the literature convention this follows.

        Parameters
        ----------
        cap_distance : float, optional
            If ``None`` (default), each cap H is placed at the
            element-specific X-H length from
            :data:`molcrys_kit.constants.config.BOND_LENGTHS` (the same
            table that powers ``operations.add_hydrogens``: ``C-H=1.09``,
            ``N-H=1.01``, ``O-H=0.96``, ``S-H=1.34``, ``P-H=1.42``).  This
            matters whenever the kept-side atom is not C -- for example
            a metal-boundary cut leaving an O or N exposed should cap at
            0.96 A or 1.01 A, not at 1.09 A.  Pass a positive float to
            force a uniform cap length regardless of element.
        cap_bond_lengths : dict, optional
            Per-element overrides to the X-H table, in either the
            ``HydrogenCompleter``-style form (``{"C-H": 1.09, ...}``) or
            the bare-element form (``{"C": 1.09, ...}``).  Combined with
            the default :data:`BOND_LENGTHS` table; user keys win.
        stop_at_non_seed_metals : bool, default True
            When ``True``, BFS treats bonds reaching a metal atom that is
            not in the current seed group as an implicit boundary: the
            bond is cut and capped with H on the kept (ligand) side.
            This is needed for frameworks whose connectivity loops back
            through ``M-X-X-M`` paths (any framework where two metal
            nodes are bridged through non-C-C bonds); without it BFS
            would walk past every other node via the bridging ligands
            and the "cluster" would be the whole framework.

        Returns
        -------
        list[CrystalCluster]
            One cluster per seed group (after multi-metal SBU
            auto-grouping under ``seed_merge_radius``).  Logs a warning
            when more than one group is found.
        """
        seed_indices = _resolve_seed_atoms(self._atoms, seed)
        seed_groups = _partition_seeds_by_distance(
            self._atoms, seed_indices, self._lattice, self.seed_merge_radius
        )
        if len(seed_groups) > 1:
            warnings.warn(
                f"ClusterCarver: seed resolved to {len(seed_groups)} disjoint "
                f"groups under seed_merge_radius={self.seed_merge_radius} A; "
                f"returning one cluster per group.",
                stacklevel=2,
            )

        cap_lengths_table = _resolve_cap_bond_lengths(cap_bond_lengths)
        cap_override = float(cap_distance) if cap_distance is not None else None
        clusters: List[CrystalCluster] = []
        for group in seed_groups:
            cluster = self._carve_one_group_bond_shells(
                group,
                n_shells=int(n_shells),
                freeze_shell=int(freeze_shell),
                cap_distance=cap_override,
                cap_lengths_table=cap_lengths_table,
                parent_label=parent_label,
                convention_reference=str(convention_reference),
                stop_at_non_seed_metals=bool(stop_at_non_seed_metals),
            )
            clusters.append(cluster)
        return clusters

    def carve_rcut(
        self,
        seed: SeedSpec,
        rcut: float,
        freeze_shell: int = 1,
        cap_distance: Optional[float] = DEFAULT_CAP_DISTANCE,
        cap_bond_lengths: Optional[Dict[str, float]] = None,
        parent_label: Optional[str] = None,
        convention_reference: str = "",
    ) -> List[CrystalCluster]:
        """Diagnostic radial cluster carving.

        Returns
        -------
        list[CrystalCluster]
            See :meth:`carve_bond_shells`.  Mode-A warnings for non-C-C
            cut bonds are emitted from inside this method.
        """
        seed_indices = _resolve_seed_atoms(self._atoms, seed)
        seed_groups = _partition_seeds_by_distance(
            self._atoms, seed_indices, self._lattice, self.seed_merge_radius
        )
        if len(seed_groups) > 1:
            warnings.warn(
                f"ClusterCarver.carve_rcut: seed resolved to {len(seed_groups)} "
                f"disjoint groups under seed_merge_radius={self.seed_merge_radius} A.",
                stacklevel=2,
            )

        cap_lengths_table = _resolve_cap_bond_lengths(cap_bond_lengths)
        cap_override = float(cap_distance) if cap_distance is not None else None
        clusters: List[CrystalCluster] = []
        for group in seed_groups:
            cluster = self._carve_one_group_rcut(
                group,
                rcut=float(rcut),
                freeze_shell=int(freeze_shell),
                cap_distance=cap_override,
                cap_lengths_table=cap_lengths_table,
                parent_label=parent_label,
                convention_reference=str(convention_reference),
            )
            clusters.append(cluster)
        return clusters

    # ----- bond_shells implementation ------------------------------------

    def _carve_one_group_bond_shells(
        self,
        seed_indices: Sequence[int],
        n_shells: int,
        freeze_shell: int,
        cap_distance: Optional[float],
        cap_lengths_table: Dict[str, float],
        parent_label: Optional[str],
        convention_reference: str = "",
        stop_at_non_seed_metals: bool = True,
    ) -> CrystalCluster:
        symbols = self._atoms.get_chemical_symbols()
        seed_set = set(int(i) for i in seed_indices)

        # ----- Pre-pass: BFS bond-hop envelope + local ring detection -----
        # Run a wide, ring-agnostic BFS to pick up enough atoms that the
        # subsequent ring detection captures every ring inside the
        # eventual cluster.  Envelope = (n_shells + 2) * max_hops_per_shell
        # where one shell is conservatively bounded by 6 bond hops (M -> O
        # -> C(carboxylate) -> C(aryl) -> C(aryl) -> C(alpha)).
        envelope_hops = max(6 * (n_shells + 1) + 2, 8)
        envelope_nodes = self._bfs_envelope(seed_indices, envelope_hops)
        local_subgraph = self._graph.subgraph(envelope_nodes).copy()
        rings_of = _local_rings(local_subgraph)

        # ----- Main pass: ring-aware BFS with cut rule -------------------
        kept_global: Set[int] = set(seed_indices)
        # Per-node depth in "cut boundaries crossed".  Seed atoms are at
        # depth 0; depth increments by 1 the moment BFS traverses through
        # what would have been a cuttable C-C bond.  (But we only choose
        # not to cut and to traverse when n_shells_remaining > 0.)
        depth: Dict[int, int] = {int(i): 0 for i in seed_indices}
        # Cumulative position offsets (in cell-vector units) for each
        # kept atom relative to the original seed-anchored frame.  Seeds
        # start at the unwrapped position of the seed itself.
        offsets: Dict[int, np.ndarray] = {
            int(i): np.zeros(3, dtype=float) for i in seed_indices
        }
        queue: deque = deque(seed_indices)
        cut_bonds: List[Tuple[int, int]] = []
        cut_keeper_offsets: List[np.ndarray] = []  # absolute Cartesian, kept side
        cut_dropped_directions: List[np.ndarray] = []  # unit vector kept->dropped

        cell = self._lattice

        while queue:
            i = queue.popleft()
            for j, edge_data in self._graph[i].items():
                # The stored vector points from min(i,j) to max(i,j);
                # flip the sign when BFS traverses in the reverse order.
                vec_stored = edge_data["vector"]
                vec_ij = vec_stored if i < j else -vec_stored
                if j in kept_global:
                    continue
                # Metal-boundary rule: never expand into a metal atom
                # that is not part of the current seed group; record an
                # implicit cut and cap on the kept (i) side.  This is
                # what prevents a multi-node framework from sweeping
                # past every other node via non-C-C (M-X-X-M) bridges.
                if (
                    stop_at_non_seed_metals
                    and is_metal_element(symbols[j])
                    and int(j) not in seed_set
                ):
                    cut_bonds.append((int(i), int(j)))
                    cut_keeper_offsets.append(
                        self._atoms.positions[i] + offsets[i]
                    )
                    cut_dropped_directions.append(
                        vec_ij / float(np.linalg.norm(vec_ij))
                    )
                    continue
                # Decide if this bond is a cut boundary.
                is_cuttable = _is_cuttable_cc(symbols, self._graph, i, j, rings_of)
                if is_cuttable:
                    # If we have shells remaining, *cross* the cut (consume
                    # one shell) and continue BFS through j; otherwise
                    # record the cut, place a cap, do not enqueue j.
                    if depth[i] >= n_shells:
                        # Cut here.  Record and skip.
                        # Direction from kept i toward dropped j (already
                        # offset-aware via vec_ij).
                        cut_bonds.append((int(i), int(j)))
                        cut_keeper_offsets.append(
                            self._atoms.positions[i] + offsets[i]
                        )
                        cut_dropped_directions.append(
                            vec_ij / float(np.linalg.norm(vec_ij))
                        )
                        continue
                    else:
                        new_depth = depth[i] + 1
                else:
                    new_depth = depth[i]

                kept_global.add(int(j))
                depth[int(j)] = new_depth
                # Image offset of atom j relative to its parent frame is
                # determined by extending the kept-side cumulative offset
                # with the displacement that brings j into the same
                # contiguous frame as i.
                # vec_ij = r_j_raw - r_i_raw + S @ cell, where r_i_raw
                # is the unwrapped position we already have.  The S we
                # need is the integer shift that fits this constraint.
                # We invert: S = round( (vec_ij - (r_j_raw - r_i_raw)) @ inv(cell) ).
                raw_disp = self._atoms.positions[j] - self._atoms.positions[i]
                shift = np.linalg.solve(cell.T, (vec_ij - raw_disp))
                offsets[int(j)] = offsets[i] + shift @ cell
                queue.append(int(j))

        # ----- Emit the cluster --------------------------------------------
        return self._finalize_cluster(
            kept_global=kept_global,
            offsets=offsets,
            cut_bonds=cut_bonds,
            cut_keeper_offsets=cut_keeper_offsets,
            cut_dropped_directions=cut_dropped_directions,
            seed_indices=seed_indices,
            mode="bond_shells",
            n_shells=n_shells,
            rcut_A=None,
            freeze_shell=freeze_shell,
            cap_distance=cap_distance,
            cap_lengths_table=cap_lengths_table,
            parent_label=parent_label,
            convention_reference=convention_reference,
            audit_non_cc_cuts=False,
        )

    # ----- rcut implementation -------------------------------------------

    def _carve_one_group_rcut(
        self,
        seed_indices: Sequence[int],
        rcut: float,
        freeze_shell: int,
        cap_distance: Optional[float],
        cap_lengths_table: Dict[str, float],
        parent_label: Optional[str],
        convention_reference: str = "",
    ) -> CrystalCluster:
        # Strategy: BFS the bond graph; an atom is kept iff its
        # *minimum-image* Cartesian distance to any seed atom is <= rcut.
        # Using BFS (rather than a flat distance scan) guarantees
        # contiguity in the cluster -- isolated guest molecules within
        # rcut of the seed are not pulled in.

        # Seed cartesians as reference points.
        seed_carts = np.array(
            [self._atoms.positions[i] for i in seed_indices], dtype=float
        )

        kept_global: Set[int] = set(seed_indices)
        offsets: Dict[int, np.ndarray] = {
            int(i): np.zeros(3, dtype=float) for i in seed_indices
        }
        queue: deque = deque(seed_indices)
        cut_bonds: List[Tuple[int, int]] = []
        cut_keeper_offsets: List[np.ndarray] = []
        cut_dropped_directions: List[np.ndarray] = []

        cell = self._lattice

        while queue:
            i = queue.popleft()
            for j, edge_data in self._graph[i].items():
                vec_stored = edge_data["vector"]
                vec_ij = vec_stored if i < j else -vec_stored
                if j in kept_global:
                    continue
                # Candidate position of j in the seed-anchored frame.
                raw_disp = self._atoms.positions[j] - self._atoms.positions[i]
                shift = np.linalg.solve(cell.T, (vec_ij - raw_disp))
                offset_j = offsets[i] + shift @ cell
                pos_j = self._atoms.positions[j] + offset_j

                # Minimum image distance to any seed.
                dists = np.linalg.norm(seed_carts - pos_j, axis=1)
                if float(dists.min()) > rcut:
                    # j is outside the radius: record a cut at i, do not
                    # traverse.
                    cut_bonds.append((int(i), int(j)))
                    cut_keeper_offsets.append(
                        self._atoms.positions[i] + offsets[i]
                    )
                    cut_dropped_directions.append(
                        vec_ij / float(np.linalg.norm(vec_ij))
                    )
                    continue

                kept_global.add(int(j))
                offsets[int(j)] = offset_j
                queue.append(int(j))

        return self._finalize_cluster(
            kept_global=kept_global,
            offsets=offsets,
            cut_bonds=cut_bonds,
            cut_keeper_offsets=cut_keeper_offsets,
            cut_dropped_directions=cut_dropped_directions,
            seed_indices=seed_indices,
            mode="rcut",
            n_shells=None,
            rcut_A=rcut,
            freeze_shell=freeze_shell,
            cap_distance=cap_distance,
            cap_lengths_table=cap_lengths_table,
            parent_label=parent_label,
            convention_reference=convention_reference,
            audit_non_cc_cuts=True,
        )

    # ----- shared assembly + cap placement -------------------------------

    def _bfs_envelope(
        self, seed_indices: Sequence[int], max_hops: int
    ) -> Set[int]:
        seen: Set[int] = {int(i) for i in seed_indices}
        layer = {int(i) for i in seed_indices}
        for _ in range(max_hops):
            next_layer: Set[int] = set()
            for node in layer:
                for nb in self._graph[node]:
                    if nb not in seen:
                        next_layer.add(int(nb))
            if not next_layer:
                break
            seen.update(next_layer)
            layer = next_layer
        return seen

    def _finalize_cluster(
        self,
        kept_global: Set[int],
        offsets: Dict[int, np.ndarray],
        cut_bonds: List[Tuple[int, int]],
        cut_keeper_offsets: List[np.ndarray],
        cut_dropped_directions: List[np.ndarray],
        seed_indices: Sequence[int],
        mode: str,
        n_shells: Optional[int],
        rcut_A: Optional[float],
        freeze_shell: int,
        cap_distance: Optional[float],
        cap_lengths_table: Dict[str, float],
        parent_label: Optional[str],
        convention_reference: str,
        audit_non_cc_cuts: bool,
    ) -> CrystalCluster:
        symbols = self._atoms.get_chemical_symbols()

        # Optional audit (rcut mode only).
        if audit_non_cc_cuts:
            for kept_i, dropped_j in cut_bonds:
                if symbols[kept_i] != "C" or symbols[dropped_j] != "C":
                    warnings.warn(
                        f"ClusterCarver.carve_rcut: cut bond "
                        f"{symbols[kept_i]}{kept_i}-{symbols[dropped_j]}{dropped_j} "
                        f"is not a C-C bond; H-capping it is likely "
                        f"chemically inappropriate (literature flags this "
                        f"as the canonical red flag).",
                        stacklevel=4,
                    )

        # ----- Build the emitted cluster atom list ------------------------
        # Preserve a stable ordering: kept atoms sorted by global index,
        # then cap H atoms (one per cut bond, in cut-bond order).
        kept_sorted = sorted(kept_global)
        global_to_local: Dict[int, int] = {g: lo for lo, g in enumerate(kept_sorted)}

        positions: List[np.ndarray] = []
        new_symbols: List[str] = []
        for g in kept_sorted:
            positions.append(self._atoms.positions[g] + offsets[g])
            new_symbols.append(symbols[g])

        # Place cap H atoms.  When the user did not pass an explicit
        # ``cap_distance`` override, the X-H length is looked up per
        # kept-side element from the shared BOND_LENGTHS table
        # (1.09 for C-H, 1.01 for N-H, 0.96 for O-H, ...).  This matters
        # in particular for metal-boundary cuts where the kept side is
        # an N or O.
        cap_local_indices: List[int] = []
        cap_distances_used: List[float] = []
        for k, ((kept_i, _dropped_j), keeper_pos, direction) in enumerate(
            zip(cut_bonds, cut_keeper_offsets, cut_dropped_directions)
        ):
            if cap_distance is not None:
                dist = float(cap_distance)
            else:
                dist = float(
                    cap_lengths_table.get(symbols[kept_i], _FALLBACK_CAP_DISTANCE)
                )
            cap_pos = keeper_pos + dist * direction
            positions.append(cap_pos)
            new_symbols.append("H")
            cap_local_indices.append(len(positions) - 1)
            cap_distances_used.append(dist)

        cluster_atoms = Atoms(
            symbols=new_symbols,
            positions=np.array(positions, dtype=float),
            pbc=False,
        )

        # ----- Determine frozen indices ----------------------------------
        kept_global_indices_list = [int(g) for g in kept_sorted]
        frozen_local_indices = self._collect_frozen_indices(
            freeze_shell=freeze_shell,
            kept_sorted=kept_sorted,
            global_to_local=global_to_local,
            cut_bonds=cut_bonds,
            cap_local_indices=cap_local_indices,
        )

        provenance_kwargs: Dict[str, object] = dict(
            mode=mode,
            seed_global_indices=list(seed_indices),
            n_shells=n_shells,
            rcut_A=rcut_A,
            kept_global_indices=kept_global_indices_list,
            cut_bonds=[(int(a), int(b)) for a, b in cut_bonds],
            cap_local_indices=cap_local_indices,
            frozen_local_indices=frozen_local_indices,
            freeze_shell=freeze_shell,
            cap_distance_A=cap_distance,
            cap_bond_lengths_A=dict(cap_lengths_table),
            cap_distances_used_A=list(cap_distances_used),
            seed_merge_radius_A=self.seed_merge_radius,
            parent_label=parent_label,
        )
        if convention_reference:
            provenance_kwargs["convention_reference"] = convention_reference
        provenance = ClusterProvenance(**provenance_kwargs)

        return CrystalCluster(
            cluster_atoms,
            provenance=provenance,
            frozen_local_indices=frozen_local_indices,
            cap_local_indices=cap_local_indices,
            parent_crystal=self.crystal,
        )

    def _collect_frozen_indices(
        self,
        freeze_shell: int,
        kept_sorted: Sequence[int],
        global_to_local: Dict[int, int],
        cut_bonds: Sequence[Tuple[int, int]],
        cap_local_indices: Sequence[int],
    ) -> List[int]:
        """Compute frozen local indices for ``freeze_shell in {0, 1, 2}``.

        * ``freeze_shell == 0`` -- nothing frozen.
        * ``freeze_shell == 1`` -- all cap H + each kept-side atom of every cut.
        * ``freeze_shell == 2`` -- shell-1 plus every kept atom that is one
          bond inward from a kept-side atom of a cut.

        Callers should stamp the system-specific literature that motivated
        their freeze choice into ``convention_reference`` so the sidecar
        is self-documenting.
        """
        if freeze_shell <= 0:
            return []

        frozen: Set[int] = set(int(i) for i in cap_local_indices)
        kept_set = set(int(g) for g in kept_sorted)
        keeper_globals: Set[int] = {int(a) for a, _b in cut_bonds}
        for g in keeper_globals:
            frozen.add(global_to_local[g])

        if freeze_shell >= 2:
            for g in keeper_globals:
                for nb in self._graph[g]:
                    if int(nb) in kept_set:
                        frozen.add(global_to_local[int(nb)])
        return sorted(frozen)


# ---------------------------------------------------------------------------
# Convenience function (mirrors the module-level shortcuts elsewhere in
# operations/, e.g. operations/desolvation.py::remove_solvents).
# ---------------------------------------------------------------------------


def carve_cluster(
    crystal: MolecularCrystal,
    seed: SeedSpec,
    mode: str = "bond_shells",
    n_shells: int = DEFAULT_N_SHELLS,
    rcut: Optional[float] = None,
    freeze_shell: int = 1,
    cap_distance: Optional[float] = DEFAULT_CAP_DISTANCE,
    cap_bond_lengths: Optional[Dict[str, float]] = None,
    seed_merge_radius: float = DEFAULT_SEED_MERGE_RADIUS,
    parent_label: Optional[str] = None,
    convention_reference: str = "",
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    stop_at_non_seed_metals: bool = True,
) -> List[CrystalCluster]:
    """Module-level entry point.

    See :class:`ClusterCarver` for the full API.
    """
    carver = ClusterCarver(
        crystal,
        seed_merge_radius=seed_merge_radius,
        bond_thresholds=bond_thresholds,
    )
    if mode == "bond_shells":
        return carver.carve_bond_shells(
            seed,
            n_shells=n_shells,
            freeze_shell=freeze_shell,
            cap_distance=cap_distance,
            cap_bond_lengths=cap_bond_lengths,
            parent_label=parent_label,
            convention_reference=convention_reference,
            stop_at_non_seed_metals=stop_at_non_seed_metals,
        )
    if mode == "rcut":
        if rcut is None:
            raise ValueError("`rcut` must be provided when mode='rcut'.")
        return carver.carve_rcut(
            seed,
            rcut=rcut,
            freeze_shell=freeze_shell,
            cap_distance=cap_distance,
            cap_bond_lengths=cap_bond_lengths,
            parent_label=parent_label,
            convention_reference=convention_reference,
        )
    raise ValueError(f"Unknown mode '{mode}'. Use 'bond_shells' or 'rcut'.")


__all__ = ["ClusterCarver", "carve_cluster"]
