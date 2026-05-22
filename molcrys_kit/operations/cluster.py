"""Carve finite, H-capped clusters out of a periodic crystal.

This module turns a periodic :class:`MolecularCrystal` into one or more
finite, hydrogen-capped :class:`CrystalCluster` objects suitable for
finite-cluster QM work (Gaussian / ORCA / Psi4 etc.).  It is a generic
framework tool -- the algorithm does not assume MOFs, COFs, zeolites or
any specific chemistry; system-specific parameter choices live with the
caller.

Two carving modes are supported:

* ``bond_shells`` (default) -- topology-preserving BFS from one or more
  seed atoms.  By default the BFS keeps the full ligand topology and
  cuts only at metal-boundary edges introduced by
  ``stop_at_non_seed_metals`` (on by default).  Single non-ring C-C bonds
  are cut only when the caller explicitly lists them in ``cut_cc_bonds``;
  these user cuts are validated before carving and capped with H just
  like metal-boundary cuts.  ``max_atoms`` is a hard safety limit for
  periodically extended components: if the topology-preserving BFS grows
  past it, the carver raises an error listing candidate C-C cut points
  instead of silently choosing one.
* ``rcut`` -- diagnostic radial cutoff: keep every atom within
  ``rcut`` Angstrom of any seed (minimum-image distance).  Same H-cap
  placement, but with a warning when a cut bond is not C-C (the
  classical red flag for accidentally severing a C-O / C-N bond).

Output is XYZ + a JSON sidecar carrying the full
:class:`ClusterProvenance` payload (kept-atom map, cut bonds, cap /
frozen indices, mode, ``max_atoms`` / rcut, per-cap distances actually used,
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

Convention references
---------------------
The algorithm follows the standard "BFS from a chosen seed, keep the
local ligand topology intact unless a user explicitly chooses a
truncation bond, cap with H along the cut bond, freeze a configurable
shell" recipe shared across the QM-cluster literature.
These references ground the default freeze / cap / mode choices and
are the basis of the default :class:`ClusterProvenance.convention_reference`
string; callers should still override that field with the citation
appropriate to their own system.

* Beyzavi et al., *J. Am. Chem. Soc.* **2014**, 136, 15861.
  DOI: 10.1021/ja508626n.  (Cap-vs-periodic energetic benchmark;
  foundational test of the cap-and-freeze convention.)
* Wu, Gagliardi, Truhlar, *Phys. Chem. Chem. Phys.* **2018**, 20, 1953.
  DOI: 10.1039/c7cp06751h.  (Cap distance and freeze rule; methyl-
  vs-formate cap benchmark.  Source of the shell-1 freeze convention.)
* Vitillo, Bhan, Gagliardi, *J. Phys. Chem. C* **2023**.
  DOI: 10.1021/acs.jpcc.3c06423.  (def2-SVP cluster opt + def2-TZVP
  single point, shell-1 freeze, for transition-metal cluster QM.)
* Gaggioli, Bernales, Gagliardi, *Chem. Sci.* **2020**, 11.
  DOI: 10.1039/d0sc02136a.  (Shell-2 freeze convention.)
* Migues, Auerbach, *J. Phys. Chem. C* **2018**, 122, 23230.
  DOI: 10.1021/acs.jpcc.8b08684.  (Delta-cluster convergence test in
  zeolites; basis for the diagnostic ``rcut`` mode.)

System-specific defaults (which seed, what ``seed_merge_radius``, where
to place any explicit C-C truncation, which ``freeze_shell``, whether to deviate from
the per-element X-H cap table) should still be picked by the caller
and recorded in :class:`ClusterProvenance.convention_reference` so
the sidecar JSON is self-documenting per project.
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

# Hard safety cap for topology-preserving BFS.  The default should be
# comfortably above normal QM-cluster sizes while still catching periodic
# linkers or accidental whole-framework sweeps before they become
# pathological.
DEFAULT_MAX_ATOMS = 500

# Fallback cap distance (Angstrom) when neither the explicit override nor
# the per-element BOND_LENGTHS lookup yields a value.  Equal to the C-H
# standard since the BOND_LENGTHS table also lists 1.09 A for "C-H".
_FALLBACK_CAP_DISTANCE = 1.09


def _build_anion_group_map(
    parent_atoms: Atoms,
    bond_graph: nx.Graph,
) -> Dict[int, int]:
    """Thin shim around
    :meth:`molcrys_kit.analysis.chemical_env.ChemicalEnvironment.compute_anion_protonation_groups`.

    The chemistry of "what counts as one anionic site" lives entirely
    in :mod:`molcrys_kit.analysis.chemical_env` so the carver shares a
    single source of truth with ``operations.add_hydrogens``: both
    consult the same ``_is_carboxylate_like_C`` / ``_is_sulfonate_like_S``
    / ``_is_phosphonate_like_P`` / ``_is_hypercoordinate_oxo_center``
    detectors plus the geometry-validated aromatic-ring inventory.

    Two implementation details matter for correctness:

    * **Strip metal-ligand edges before calling chem_env.**  The
      anion-site detectors classify an O as "terminal" by checking
      that it has exactly one heavy neighbour; in the parent crystal
      a carboxylate O is bonded to both its C and its coordinating
      Zn, so the raw PBC bond graph would make the O *non*-terminal
      and silently miss the carboxylate.  We therefore feed
      :class:`ChemicalEnvironment` the *organic skeleton* (the bond
      graph with metal-non-metal edges removed) so the detectors see
      the same picture they do on a fully protonated crystal where
      no metals are present.
    * **Keep metal atoms as nodes** so the returned dict still maps
      every parent index, with metals defaulting to a group of one.
    """
    from ..analysis.chemical_env import ChemicalEnvironment

    syms = parent_atoms.get_chemical_symbols()
    pos = parent_atoms.get_positions()
    g = nx.Graph()
    for i, s in enumerate(syms):
        g.add_node(i, symbol=s)
    for u, v in bond_graph.edges():
        if is_metal_element(syms[int(u)]) or is_metal_element(syms[int(v)]):
            # Metal-ligand edges are dropped from the organic-skeleton
            # view used for carboxylate / sulfonate / ring detection;
            # see the docstring above for the rationale.
            continue
        g.add_edge(int(u), int(v))
    return ChemicalEnvironment((g, pos)).compute_anion_protonation_groups()


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

# Sentinel for "use the BOND_LENGTHS lookup keyed by the kept-side element"
# in the cap-distance argument.  Passing a positive float overrides the
# lookup with a uniform value (the original v0 behaviour).
DEFAULT_CAP_DISTANCE: Optional[float] = None


SeedSpec = Union[str, int, Sequence[int], Callable[[Atoms], Sequence[int]]]


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    """Return a stable, order-independent representation of an edge."""
    a, b = int(i), int(j)
    return (a, b) if a <= b else (b, a)


class LigandTopologyOverflowError(ValueError):
    """Raised when topology-preserving carving exceeds ``max_atoms``.

    The error carries cuttable C-C candidates on the current frontier so a
    caller can choose explicit ``cut_cc_bonds`` and rerun deterministically.
    """

    def __init__(
        self,
        seed_indices: Sequence[int],
        actual_atom_count: int,
        max_atoms: int,
        candidates: Sequence[Tuple[int, int]],
    ):
        self.seed_indices = [int(i) for i in seed_indices]
        self.actual_atom_count = int(actual_atom_count)
        self.max_atoms = int(max_atoms)
        self.candidates = [_edge_key(a, b) for a, b in candidates]
        super().__init__(self.__str__())

    def __str__(self) -> str:
        candidate_text = (
            ", ".join(f"({a}, {b})" for a, b in self.candidates)
            if self.candidates
            else "none found on the current frontier"
        )
        suggestion = ""
        if self.candidates:
            first = self.candidates[:4]
            joined = ";".join(f"{a},{b}" for a, b in first)
            suggestion = (
                f" Suggested CLI retry: --cut-cc-bonds \"{joined}\" "
                f"(inspect the candidates before choosing)."
            )
        return (
            "Topology-preserving cluster carving exceeded max_atoms="
            f"{self.max_atoms} for seed group {self.seed_indices}; "
            f"current kept atom count is {self.actual_atom_count}. "
            "This usually means the selected component is periodically "
            "extended or the safety cap is too small. Candidate cuttable "
            f"C-C frontier bonds: {candidate_text}.{suggestion}"
        )


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
    """Build a PBC-aware bond graph with integer image offsets per edge.

    The carver lives in a non-periodic Cartesian frame, but the parent
    bond graph is genuinely periodic: a Zn-N coordination bond may
    connect atoms whose wrapped positions sit on opposite faces of the
    unit cell.  We therefore identify bonds using ASE's
    ``neighbor_list("ijdS", atoms, ...)`` on the wrapped Atoms with
    ``pbc=True``, and store both:

    * ``image`` -- the integer image triple ``S`` such that the bonded
      pair ``(min(i, j), max(i, j))`` is realized as ``pos[max] + S @ cell
      - pos[min]``.  ``S`` is direction-sensitive and is signed for the
      ``min -> max`` traversal direction.
    * ``vector`` -- the Cartesian displacement from ``min`` to ``max`` in
      that bonded frame, i.e. ``pos[max] + S @ cell - pos[min]``.
    * ``distance`` -- the Euclidean length of ``vector``.

    BFS in :class:`ClusterCarver` consumes these attributes to keep
    every kept atom in a single consistent image frame around the seed
    group.  Edges that close a periodic loop with an inconsistent image
    are explicitly broken as "loop cuts" rather than left as phantom
    bonds.
    """
    symbols = atoms.get_chemical_symbols()
    i_list, j_list, d_list, S_list = neighbor_list(
        "ijdS", atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF
    )

    graph = nx.Graph()
    graph.add_nodes_from((i, {"symbol": s}) for i, s in enumerate(symbols))

    cell = np.asarray(atoms.get_cell())
    wrapped = atoms.get_positions()
    bond_thresholds = bond_thresholds or {}
    for i, j, distance, S in zip(i_list, j_list, d_list, S_list):
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
            image = tuple(int(x) for x in S)
            vec = wrapped[int(j)] + np.asarray(S, dtype=float) @ cell - wrapped[int(i)]
            graph.add_edge(
                int(i),
                int(j),
                image=image,
                vector=np.array(vec, dtype=float),
                distance=float(distance),
            )

    return graph


def _seed_group_offsets(
    atoms: Atoms,
    seed_group: Sequence[int],
    lattice: np.ndarray,
) -> Dict[int, Tuple[int, int, int]]:
    """Per-seed integer image offset that co-locates the seed group.

    Returns ``{seed_global_index: (sx, sy, sz)}`` such that
    ``wrapped[seed] + S @ cell`` puts every seed in one geometric
    minimum-image neighbourhood (anchored at the first seed at the
    origin image ``(0, 0, 0)``).  This is the starting frame for the
    offset-tracking BFS in the carver.
    """
    if not seed_group:
        return {}
    wrapped = atoms.get_positions()
    cell = np.asarray(lattice, dtype=float)
    inv = np.linalg.inv(cell)
    anchor = int(seed_group[0])
    offsets: Dict[int, Tuple[int, int, int]] = {anchor: (0, 0, 0)}
    anchor_pos = wrapped[anchor]
    for s in seed_group:
        s_int = int(s)
        if s_int == anchor:
            continue
        disp = wrapped[s_int] - anchor_pos
        frac = disp @ inv
        S = -np.round(frac).astype(int)
        offsets[s_int] = (int(S[0]), int(S[1]), int(S[2]))
    return offsets


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
    The carver builds a *PBC-aware* bond graph (atoms with ``pbc=True``)
    where each edge carries the integer image triple ``S`` such that the
    bonded pair sits at ``pos[max] + S @ cell - pos[min]``.  BFS from the
    seed group then propagates per-atom image offsets so that every
    kept atom lives in one consistent Cartesian frame around the seeds.
    Whenever a topologically nontrivial periodic loop is encountered --
    a kept atom is re-reached with an incompatible offset -- the
    closing bond is recorded as a ``loop_cut`` and capped with H on
    *both* sides; no phantom (algorithm-says-bonded but
    several-Angstrom-apart) edge can survive.

    Seed grouping itself uses the *wrapped* parent positions plus
    minimum-image distance, so a metal trimer that wraps across a
    periodic face is identified correctly; the per-seed image offsets
    needed to co-locate the group geometrically are computed at the
    start of each carve.
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
        self._wrapped_positions: np.ndarray = self._atoms.get_positions()
        # PBC-aware bond graph; edges carry integer ``image`` triples
        # and the corresponding Cartesian ``vector``.
        self._graph: nx.Graph = _build_offset_bond_graph(
            self._atoms, bond_thresholds
        )

    # ----- public API -----------------------------------------------------

    def carve_bond_shells(
        self,
        seed: SeedSpec,
        max_atoms: Optional[int] = DEFAULT_MAX_ATOMS,
        cut_cc_bonds: Optional[Sequence[Tuple[int, int]]] = None,
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
        max_atoms : int, optional
            Hard safety limit for the topology-preserving BFS.  If the
            retained parent atoms exceed this count, the carver raises
            :class:`LigandTopologyOverflowError` and lists cuttable C-C
            frontier bonds for an explicit retry.  ``None`` disables the
            guard.
        cut_cc_bonds : sequence[tuple[int, int]], optional
            Parent global atom-index pairs where a single non-ring C-C
            bond should be treated as a manual truncation boundary.  Each
            pair is validated before carving; invalid bonds raise
            ``ValueError`` rather than being silently ignored.
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
        validated_cut_cc_bonds = self._validate_user_cuts(cut_cc_bonds)
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
                max_atoms=None if max_atoms is None else int(max_atoms),
                cut_cc_bonds=validated_cut_cc_bonds,
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
        max_atoms: Optional[int],
        cut_cc_bonds: Sequence[Tuple[int, int]],
        freeze_shell: int,
        cap_distance: Optional[float],
        cap_lengths_table: Dict[str, float],
        parent_label: Optional[str],
        convention_reference: str = "",
        stop_at_non_seed_metals: bool = True,
    ) -> CrystalCluster:
        """Topology-preserving carve with chemistry-aware loop-cut placement.

        Algorithm
        ---------
        1. Find the kept connected component starting from the seed
           group: BFS through every edge **except** those that hit a
           non-seed metal (when ``stop_at_non_seed_metals=True``) or
           that the user explicitly listed in ``cut_cc_bonds``.  The
           skipped edges become ``metal_boundary_cuts`` and
           ``cut_cc_bonds_applied`` respectively.
        2. Build a **maximum-weight spanning tree** of that component
           with edge weights chosen so non-metal/non-metal (intra-
           ligand) edges are vastly preferred over metal/non-metal
           (coordination) edges.  This guarantees that every back edge
           -- the one in each cycle that the tree does not contain --
           is preferentially a metal-ligand bond, not an internal ring
           bond.
        3. BFS through the spanning tree from ``seed_indices[0]`` to
           propagate per-atom integer image offsets consistently.
        4. Sweep every non-tree (back) edge.  If its stored image
           matches the offsets propagated through the tree, it is a
           chemical ring closure and is kept silently.  If not, it
           closes a topologically nontrivial periodic loop and is
           recorded as a ``loop_cut`` -- capped with H on both sides.
           By design the back edges are metal-ligand bonds, so the
           ligand topology stays intact and the chemistry is correct.
        """
        symbols = self._atoms.get_chemical_symbols()
        seed_set = set(int(i) for i in seed_indices)
        cut_cc_keys = {_edge_key(a, b) for a, b in cut_cc_bonds}

        # ---- Step 1: identify the kept connected component ----------
        kept_global: Set[int] = set(int(i) for i in seed_indices)
        queue: deque = deque(int(i) for i in seed_indices)
        metal_boundary_cuts: List[Tuple[int, int]] = []
        seen_metal_boundary: Set[Tuple[int, int]] = set()
        while queue:
            i = queue.popleft()
            for j in self._graph[i]:
                pair = _edge_key(i, j)
                if pair in cut_cc_keys:
                    # User-requested C-C cut; do not traverse.
                    continue
                if (
                    stop_at_non_seed_metals
                    and is_metal_element(symbols[j])
                    and int(j) not in seed_set
                ):
                    # Non-seed metal: record a metal-boundary cut and
                    # do not enter.
                    if pair not in seen_metal_boundary:
                        seen_metal_boundary.add(pair)
                        metal_boundary_cuts.append((int(i), int(j)))
                    continue
                if int(j) in kept_global:
                    continue
                kept_global.add(int(j))
                queue.append(int(j))
                if max_atoms is not None and len(kept_global) > max_atoms:
                    raise LigandTopologyOverflowError(
                        seed_indices=seed_indices,
                        actual_atom_count=len(kept_global),
                        max_atoms=max_atoms,
                        candidates=self._collect_frontier_cuttables(kept_global),
                    )

        # ---- Step 2: build the maximum-weight spanning tree ----------
        # Edge weight rule: a ligand-internal (non-metal/non-metal)
        # edge is two orders of magnitude heavier than a metal/non-
        # metal edge, so the maximum-weight spanning tree keeps every
        # available ligand-internal edge as a tree edge whenever doing
        # so does not disconnect the cluster.  Back edges (the ones
        # the tree omits) are therefore overwhelmingly metal-X edges,
        # which is exactly where a topologically forced cut belongs.
        # We add a tiny preference for shorter bonds inside each weight
        # tier so the tree is deterministic regardless of insertion
        # order.
        subgraph_view = self._graph.subgraph(kept_global)
        weighted = nx.Graph()
        weighted.add_nodes_from(subgraph_view.nodes(data=True))
        for u, v, data in subgraph_view.edges(data=True):
            pair = _edge_key(u, v)
            if pair in cut_cc_keys:
                # User-requested C-C cut: do not include in the cluster
                # bond graph used for offset propagation.  The cut is
                # processed separately below.
                continue
            metal_endpoint = (
                is_metal_element(symbols[int(u)])
                or is_metal_element(symbols[int(v)])
            )
            base_weight = 1.0 if metal_endpoint else 100.0
            # Smaller bond distance => slightly higher weight (break
            # weight-tier ties in a deterministic, geometrically
            # reasonable way).
            tiebreak = -0.001 * float(data.get("distance", 0.0))
            weighted.add_edge(
                int(u), int(v), weight=base_weight + tiebreak, _data=data
            )

        if weighted.number_of_nodes() == 0:
            mst_edges: Set[Tuple[int, int]] = set()
        else:
            mst = nx.maximum_spanning_tree(weighted, weight="weight")
            mst_edges = {_edge_key(u, v) for u, v in mst.edges()}

        # ---- Step 3: BFS along the spanning tree to set offsets -----
        anchor = int(seed_indices[0])
        kept_offset: Dict[int, Tuple[int, int, int]] = {anchor: (0, 0, 0)}
        tree_queue: deque = deque([anchor])
        while tree_queue:
            i = tree_queue.popleft()
            offset_i = np.asarray(kept_offset[i], dtype=int)
            for j in weighted.neighbors(i):
                if _edge_key(i, j) not in mst_edges:
                    continue
                if int(j) in kept_offset:
                    continue
                edge_data = self._graph[i][int(j)]
                image_stored = np.asarray(edge_data["image"], dtype=int)
                if i < j:
                    image_ij = image_stored
                else:
                    image_ij = -image_stored
                kept_offset[int(j)] = tuple(
                    int(x) for x in (offset_i + image_ij).tolist()
                )
                tree_queue.append(int(j))

        # Any atom not visited by the tree BFS (defensive guard --
        # should only happen for atoms isolated by an entirely-
        # excluded edge, e.g. an orphan inside the cut_cc_bonds
        # boundary) keeps its wrapped position.
        for g in kept_global:
            kept_offset.setdefault(int(g), (0, 0, 0))

        def _cluster_pos(g: int) -> np.ndarray:
            offset = np.asarray(kept_offset[g], dtype=float)
            return self._wrapped_positions[g] + offset @ self._lattice

        # ---- Step 4: sweep back edges, mark inconsistent as loop_cut --
        cut_bonds: List[Tuple[int, int]] = []
        loop_cuts: List[Tuple[int, int]] = []
        loop_cut_keys: Set[Tuple[int, int]] = set()
        cut_keeper_positions: List[np.ndarray] = []
        cut_dropped_directions: List[np.ndarray] = []

        for u, v, edge_data in subgraph_view.edges(data=True):
            pair = _edge_key(u, v)
            if pair in cut_cc_keys:
                continue
            if pair in mst_edges:
                continue
            # Back edge: check offset consistency.
            image_stored = np.asarray(edge_data["image"], dtype=int)
            vec_stored = edge_data["vector"]
            if u < v:
                image_uv = image_stored
                vec_uv = vec_stored
            else:
                image_uv = -image_stored
                vec_uv = -vec_stored
            expected = tuple(
                int(x) for x in (
                    np.asarray(kept_offset[int(u)], dtype=int) + image_uv
                ).tolist()
            )
            if kept_offset[int(v)] == expected:
                # Consistent chemical ring closure -- the bond is
                # geometrically realized in the cluster.
                continue
            # Topologically nontrivial loop crossing PBC: sever this
            # bond.  Chemistry rule for capping: only protonate the
            # non-metal endpoint(s).  A metal endpoint becomes an
            # "open coordination site" that, in practice, would bind a
            # solvent in the real material -- adding a Zn-H or Cu-H
            # hydride here would be chemically wrong.  When both sides
            # are non-metals (a real ligand-polymeric PBC loop, e.g.
            # graphene-like covalent net) we cap both sides as before.
            loop_cut_keys.add(pair)
            loop_cuts.append(pair)
            unit_uv = np.asarray(vec_uv) / float(np.linalg.norm(vec_uv))
            u_is_metal = is_metal_element(symbols[int(u)])
            v_is_metal = is_metal_element(symbols[int(v)])
            if not u_is_metal:
                cut_bonds.append((int(u), int(v)))
                cut_keeper_positions.append(_cluster_pos(int(u)))
                cut_dropped_directions.append(unit_uv)
            if not v_is_metal:
                cut_bonds.append((int(v), int(u)))
                cut_keeper_positions.append(_cluster_pos(int(v)))
                cut_dropped_directions.append(-unit_uv)

        # ---- Metal-boundary cuts: emit cap from the kept side --------
        for kept_i, dropped_j in metal_boundary_cuts:
            edge_data = self._graph[int(kept_i)][int(dropped_j)]
            vec_stored = edge_data["vector"]
            vec_ij = vec_stored if int(kept_i) < int(dropped_j) else -vec_stored
            cut_bonds.append((int(kept_i), int(dropped_j)))
            cut_keeper_positions.append(_cluster_pos(int(kept_i)))
            cut_dropped_directions.append(
                np.asarray(vec_ij) / float(np.linalg.norm(vec_ij))
            )

        # ---- User-requested C-C cuts ---------------------------------
        cut_cc_bonds_applied: List[Tuple[int, int]] = []
        for a, b in cut_cc_bonds:
            a_kept = int(a) in kept_global
            b_kept = int(b) in kept_global
            if a_kept == b_kept:
                continue
            kept_i, dropped_j = (int(a), int(b)) if a_kept else (int(b), int(a))
            edge_data = self._graph[kept_i][dropped_j]
            vec_stored = edge_data["vector"]
            vec_ij = vec_stored if kept_i < dropped_j else -vec_stored
            cut = (kept_i, dropped_j)
            cut_bonds.append(cut)
            cut_cc_bonds_applied.append(cut)
            cut_keeper_positions.append(_cluster_pos(kept_i))
            cut_dropped_directions.append(
                np.asarray(vec_ij) / float(np.linalg.norm(vec_ij))
            )

        return self._finalize_cluster(
            kept_global=kept_global,
            kept_offset=kept_offset,
            cut_bonds=cut_bonds,
            cut_keeper_positions=cut_keeper_positions,
            cut_dropped_directions=cut_dropped_directions,
            seed_indices=seed_indices,
            mode="bond_shells",
            rcut_A=None,
            max_atoms=max_atoms,
            cut_cc_bonds_requested=cut_cc_bonds,
            cut_cc_bonds_applied=cut_cc_bonds_applied,
            metal_boundary_cuts=metal_boundary_cuts,
            loop_cuts=loop_cuts,
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
        # Strategy: BFS the PBC bond graph while tracking each kept
        # atom's integer image offset.  An atom is kept iff its
        # cluster-frame distance to any seed (computed from the
        # propagated offset) is <= ``rcut``.  Loop edges with
        # inconsistent offsets are recorded as ``loop_cuts`` exactly as
        # in the bond_shells mode.
        kept_offset: Dict[int, Tuple[int, int, int]] = _seed_group_offsets(
            self._atoms, seed_indices, self._lattice
        )
        kept_global: Set[int] = set(kept_offset.keys())
        queue: deque = deque(kept_offset.keys())

        cut_bonds: List[Tuple[int, int]] = []
        loop_cuts: List[Tuple[int, int]] = []
        loop_cut_keys: Set[Tuple[int, int]] = set()
        cut_keeper_positions: List[np.ndarray] = []
        cut_dropped_directions: List[np.ndarray] = []

        def _cluster_pos(g: int) -> np.ndarray:
            offset = np.asarray(kept_offset[g], dtype=float)
            return self._wrapped_positions[g] + offset @ self._lattice

        seed_carts = np.array(
            [_cluster_pos(int(i)) for i in seed_indices], dtype=float
        )

        while queue:
            i = queue.popleft()
            offset_i = np.asarray(kept_offset[i], dtype=int)
            for j, edge_data in self._graph[i].items():
                image_stored = np.asarray(edge_data["image"], dtype=int)
                vec_stored = edge_data["vector"]
                if i < j:
                    image_ij = image_stored
                    vec_ij = vec_stored
                else:
                    image_ij = -image_stored
                    vec_ij = -vec_stored
                new_offset = tuple(int(x) for x in (offset_i + image_ij).tolist())

                if int(j) in kept_global:
                    if kept_offset[int(j)] != new_offset:
                        pair = _edge_key(i, j)
                        if pair in loop_cut_keys:
                            continue
                        loop_cut_keys.add(pair)
                        loop_cuts.append(pair)
                        symbols = self._atoms.get_chemical_symbols()
                        i_is_metal = is_metal_element(symbols[int(i)])
                        j_is_metal = is_metal_element(symbols[int(j)])
                        unit_ij = np.asarray(vec_ij) / float(np.linalg.norm(vec_ij))
                        if not i_is_metal:
                            cut_bonds.append((int(i), int(j)))
                            cut_keeper_positions.append(_cluster_pos(int(i)))
                            cut_dropped_directions.append(unit_ij)
                        if not j_is_metal:
                            cut_bonds.append((int(j), int(i)))
                            cut_keeper_positions.append(_cluster_pos(int(j)))
                            cut_dropped_directions.append(-unit_ij)
                    continue

                pos_j = self._wrapped_positions[int(j)] + np.asarray(new_offset, dtype=float) @ self._lattice
                if float(np.linalg.norm(seed_carts - pos_j, axis=1).min()) > rcut:
                    cut_bonds.append((int(i), int(j)))
                    cut_keeper_positions.append(_cluster_pos(int(i)))
                    cut_dropped_directions.append(
                        np.asarray(vec_ij) / float(np.linalg.norm(vec_ij))
                    )
                    continue

                kept_global.add(int(j))
                kept_offset[int(j)] = new_offset
                queue.append(int(j))

        return self._finalize_cluster(
            kept_global=kept_global,
            kept_offset=kept_offset,
            cut_bonds=cut_bonds,
            cut_keeper_positions=cut_keeper_positions,
            cut_dropped_directions=cut_dropped_directions,
            seed_indices=seed_indices,
            mode="rcut",
            rcut_A=rcut,
            max_atoms=None,
            cut_cc_bonds_requested=[],
            cut_cc_bonds_applied=[],
            metal_boundary_cuts=[],
            loop_cuts=loop_cuts,
            freeze_shell=freeze_shell,
            cap_distance=cap_distance,
            cap_lengths_table=cap_lengths_table,
            parent_label=parent_label,
            convention_reference=convention_reference,
            audit_non_cc_cuts=True,
        )

    # ----- shared assembly + cap placement -------------------------------

    def _ring_membership_near(self, nodes: Sequence[int]) -> Dict[int, frozenset]:
        """Return small-ring membership in a local neighbourhood.

        Expanding by ``_MAX_CHEMICAL_RING_SIZE`` hops is enough to identify
        whether a queried edge belongs to any chemical ring up to that size,
        without running cycle-basis detection on the full framework graph.
        """
        seed_nodes = {int(n) for n in nodes if int(n) in self._graph}
        expanded: Set[int] = set(seed_nodes)
        layer: Set[int] = set(seed_nodes)
        for _ in range(_MAX_CHEMICAL_RING_SIZE):
            next_layer: Set[int] = set()
            for node in layer:
                for nb in self._graph[node]:
                    if int(nb) not in expanded:
                        next_layer.add(int(nb))
            if not next_layer:
                break
            expanded.update(next_layer)
            layer = next_layer
        return _local_rings(self._graph.subgraph(expanded).copy())

    def _uncuttable_reason(
        self, i: int, j: int, rings_of: Dict[int, frozenset]
    ) -> str:
        """Explain why edge ``i-j`` is not a legal manual C-C truncation."""
        symbols = self._atoms.get_chemical_symbols()
        if i < 0 or i >= len(symbols) or j < 0 or j >= len(symbols):
            return f"atom index out of range [0, {len(symbols)})"
        if not self._graph.has_edge(i, j):
            return "no bond exists between these parent atom indices"
        if symbols[i] != "C" or symbols[j] != "C":
            return f"expected C-C, got {symbols[i]}{i}-{symbols[j]}{j}"
        if rings_of.get(i, frozenset()) & rings_of.get(j, frozenset()):
            return "bond belongs to a small ring"
        distance = (self._graph.get_edge_data(i, j) or {}).get("distance")
        if distance is not None and distance < 1.42:
            return (
                f"C-C distance {float(distance):.3f} A is shorter than the "
                "single-bond cutoff 1.42 A"
            )
        return ""

    def _validate_user_cuts(
        self, cut_cc_bonds: Optional[Sequence[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """Validate and normalize user-specified C-C truncation bonds."""
        if not cut_cc_bonds:
            return []

        normalized: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        nodes: Set[int] = set()
        for pair in cut_cc_bonds:
            if len(pair) != 2:
                raise ValueError(
                    "cut_cc_bonds entries must be (i, j) parent-index pairs; "
                    f"got {pair!r}."
                )
            key = _edge_key(pair[0], pair[1])
            if key[0] == key[1]:
                raise ValueError(f"cut_cc_bonds cannot contain self-edge {key}.")
            if key not in seen:
                normalized.append(key)
                seen.add(key)
                nodes.update(key)

        rings_of = self._ring_membership_near(sorted(nodes))
        errors: List[str] = []
        for i, j in normalized:
            reason = self._uncuttable_reason(i, j, rings_of)
            if reason:
                errors.append(f"({i}, {j}): {reason}")
        if errors:
            raise ValueError(
                "Invalid cut_cc_bonds; only single non-ring C-C bonds may be "
                "manually truncated. " + "; ".join(errors)
            )
        return normalized

    def _collect_frontier_cuttables(
        self, kept_set: Sequence[int]
    ) -> List[Tuple[int, int]]:
        """List legal C-C truncation candidates crossing the current frontier."""
        kept = {int(i) for i in kept_set}
        frontier: List[Tuple[int, int]] = []
        local_nodes: Set[int] = set(kept)
        for u in sorted(kept):
            for v in self._graph[u]:
                if int(v) in kept:
                    continue
                edge = _edge_key(u, v)
                frontier.append(edge)
                local_nodes.update(edge)

        if not frontier:
            return []

        rings_of = self._ring_membership_near(sorted(local_nodes))
        symbols = self._atoms.get_chemical_symbols()
        candidates = {
            edge
            for edge in frontier
            if _is_cuttable_cc(symbols, self._graph, edge[0], edge[1], rings_of)
        }
        return sorted(candidates)

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
        kept_offset: Dict[int, Tuple[int, int, int]],
        cut_bonds: List[Tuple[int, int]],
        cut_keeper_positions: List[np.ndarray],
        cut_dropped_directions: List[np.ndarray],
        seed_indices: Sequence[int],
        mode: str,
        rcut_A: Optional[float],
        max_atoms: Optional[int],
        cut_cc_bonds_requested: Sequence[Tuple[int, int]],
        cut_cc_bonds_applied: Sequence[Tuple[int, int]],
        metal_boundary_cuts: Sequence[Tuple[int, int]],
        loop_cuts: Sequence[Tuple[int, int]],
        freeze_shell: int,
        cap_distance: Optional[float],
        cap_lengths_table: Dict[str, float],
        parent_label: Optional[str],
        convention_reference: str,
        audit_non_cc_cuts: bool,
    ) -> CrystalCluster:
        symbols = self._atoms.get_chemical_symbols()
        loop_cut_keys: Set[Tuple[int, int]] = {
            _edge_key(a, b) for a, b in loop_cuts
        }

        # Optional audit (rcut mode only).
        if audit_non_cc_cuts:
            for kept_i, dropped_j in cut_bonds:
                if _edge_key(kept_i, dropped_j) in loop_cut_keys:
                    # Loop cuts are not user-controlled; they are
                    # required by the topology and are reported via the
                    # loop_cuts list rather than as a "red-flag" cut.
                    continue
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
        # then cap H atoms (one per cut bond, in cut-bond order).  Each
        # kept atom's cluster position is its wrapped parent position
        # offset by ``kept_offset[g] @ lattice`` so the entire cluster
        # lives in one consistent image frame around the seeds.
        kept_sorted = sorted(kept_global)
        global_to_local: Dict[int, int] = {g: lo for lo, g in enumerate(kept_sorted)}

        positions: List[np.ndarray] = []
        new_symbols: List[str] = []
        for g in kept_sorted:
            offset = np.asarray(kept_offset[g], dtype=float)
            positions.append(self._wrapped_positions[g] + offset @ self._lattice)
            new_symbols.append(symbols[g])

        # Place cap H atoms.  When the user did not pass an explicit
        # ``cap_distance`` override, the X-H length is looked up per
        # kept-side element from the shared BOND_LENGTHS table
        # (1.09 for C-H, 1.01 for N-H, 0.96 for O-H, ...).
        #
        # Chemistry-aware deduplication has two layers, both designed
        # to match the local-environment heuristics the
        # ``add_hydrogens`` HydrogenCompleter uses on full molecular
        # crystals so the carver does not invent chemistry that the
        # hydrogenation pipeline would reject:
        #
        # 1.  *Per-atom dedup* -- a single non-carbon keeper that is
        #     the keeper side of several cuts gets at most one cap H
        #     (a μ-N bridging two non-seed Zn becomes N-H, not NH2).
        # 2.  *Per-anion-group dedup* -- a chemical group (a -COO^-
        #     carboxylate, a sulfonate -SO3^-, a phosphonate -PO3^-,
        #     a hypercoordinate oxo anion such as NO3^- / ClO4^-, or
        #     a fully-deprotonated aromatic N-heterocycle ring like
        #     triazolate / imidazolate / tetrazolate) collectively
        #     gets at most one cap H even when several keeper atoms
        #     in the same group were cut from their metals.  Without
        #     this rule a μ-carboxylate losing both Zn-O bonds would
        #     become a chemically impossible ``-C(OH)2`` geminal diol
        #     instead of the correct ``-COOH``; the rule is what
        #     turns the geminal diol failure mode into the right
        #     neutral acid form upon QM relaxation.
        #
        # Carbon keepers (which only show up for explicit C-C
        # truncations, where every cut C-X bond is one missing valence
        # on that carbon) keep the one-cap-per-cut behaviour and are
        # not subject to the anion-group dedup.
        anion_groups = _build_anion_group_map(self._atoms, self._graph)

        cap_local_indices: List[int] = []
        cap_distances_used: List[float] = []
        cap_keeper_globals: List[int] = []

        keeper_directions: Dict[int, List[np.ndarray]] = {}
        keeper_first_pos: Dict[int, np.ndarray] = {}
        for (kept_i, _dropped_j), keeper_pos, direction in zip(
            cut_bonds, cut_keeper_positions, cut_dropped_directions
        ):
            keeper_directions.setdefault(int(kept_i), []).append(np.asarray(direction))
            keeper_first_pos.setdefault(int(kept_i), np.asarray(keeper_pos))

        emitted_non_c_keepers: Set[int] = set()
        emitted_anion_groups: Set[int] = set()
        for (kept_i, _dropped_j), keeper_pos, direction in zip(
            cut_bonds, cut_keeper_positions, cut_dropped_directions
        ):
            keeper_sym = symbols[int(kept_i)]
            if keeper_sym != "C":
                if int(kept_i) in emitted_non_c_keepers:
                    continue
                gid = anion_groups.get(int(kept_i), int(kept_i))
                if gid in emitted_anion_groups:
                    # The anionic site this keeper belongs to has
                    # already been protonated via another keeper in
                    # the same group (e.g., the other O of the same
                    # carboxylate, or another N of the same
                    # triazolate ring).  Skip to avoid double-
                    # protonation.
                    emitted_non_c_keepers.add(int(kept_i))
                    continue
                emitted_non_c_keepers.add(int(kept_i))
                emitted_anion_groups.add(gid)
                # Average of unit directions; for a symmetric μ2
                # bridge this points roughly along the bisector
                # between the two cut directions, which is the
                # chemically natural place for the lone-pair proton.
                directions = keeper_directions[int(kept_i)]
                stacked = np.stack(directions, axis=0)
                avg = stacked.sum(axis=0)
                norm = float(np.linalg.norm(avg))
                if norm < 1e-6:
                    # Diametrically opposed cuts: fall back to the
                    # first cut's direction so we still get a
                    # defined cap position.
                    avg_dir = directions[0]
                else:
                    avg_dir = avg / norm
                keeper_pos_used = keeper_first_pos[int(kept_i)]
                direction_used = avg_dir
            else:
                # C keeper: one cap per cut, along the cut's own
                # direction.  Each cut C-X bond is restored as a
                # distinct C-H.
                keeper_pos_used = np.asarray(keeper_pos)
                direction_used = np.asarray(direction)

            if cap_distance is not None:
                dist = float(cap_distance)
            else:
                dist = float(
                    cap_lengths_table.get(keeper_sym, _FALLBACK_CAP_DISTANCE)
                )
            cap_pos = keeper_pos_used + dist * direction_used
            positions.append(cap_pos)
            new_symbols.append("H")
            cap_local_indices.append(len(positions) - 1)
            cap_distances_used.append(dist)
            cap_keeper_globals.append(int(kept_i))

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
            rcut_A=rcut_A,
            max_atoms=max_atoms,
            kept_global_indices=kept_global_indices_list,
            cut_bonds=[(int(a), int(b)) for a, b in cut_bonds],
            cut_cc_bonds_requested=[
                (int(a), int(b)) for a, b in cut_cc_bonds_requested
            ],
            cut_cc_bonds_applied=[
                (int(a), int(b)) for a, b in cut_cc_bonds_applied
            ],
            metal_boundary_cuts=[
                (int(a), int(b)) for a, b in metal_boundary_cuts
            ],
            loop_cuts=[(int(a), int(b)) for a, b in loop_cuts],
            cap_keeper_global_indices=list(cap_keeper_globals),
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
        * ``freeze_shell == 1`` -- all cap H + each kept-side atom of every
          cut.  (Wu, Gagliardi, Truhlar, PCCP 2018, 10.1039/c7cp06751h;
          Vitillo, Bhan, Gagliardi, JPCC 2023, 10.1021/acs.jpcc.3c06423.)
        * ``freeze_shell == 2`` -- shell-1 plus every kept atom that is one
          bond inward from a kept-side atom of a cut.  (Gaggioli,
          Bernales, Gagliardi, Chem. Sci. 2020, 10.1039/d0sc02136a.)

        These citations ground the default freeze convention; callers may
        stamp a system-specific override into ``convention_reference`` so
        the sidecar JSON records exactly which literature justifies the
        carve for the system at hand.
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
    rcut: Optional[float] = None,
    max_atoms: Optional[int] = DEFAULT_MAX_ATOMS,
    cut_cc_bonds: Optional[Sequence[Tuple[int, int]]] = None,
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
            max_atoms=max_atoms,
            cut_cc_bonds=cut_cc_bonds,
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


__all__ = ["ClusterCarver", "LigandTopologyOverflowError", "carve_cluster"]
