"""
Structure Generation for Disorder Handling.

This module implements the DisorderSolver that collapses the exclusion graph
into valid, ordered MolecularCrystal objects by solving the Maximum Independent Set problem.
"""

import logging
import numpy as np
import networkx as nx
import random
import re
from typing import List

logger = logging.getLogger(__name__)

from ase import Atoms
from ase.geometry import get_distances

from ...structures.crystal import MolecularCrystal
from .info import DisorderInfo
from ...io.cif import identify_molecules
from ...constants import get_atomic_radius, has_atomic_radius, is_metal_element
from ...analysis.interactions import get_bonding_threshold


class DisorderSolver:
    """
    Solves the disorder problem by finding independent sets in the exclusion graph.
    """

    _SP_COMPLETION_DISTANCE = 1.5
    _SP_TOO_CLOSE_H_DISTANCE = 0.65
    _SP_RELOCATED_C_H_DISTANCE = 1.09
    _SP_UNDERCOORD_H_SEARCH = 1.7

    def __init__(self, info: DisorderInfo, graph: nx.Graph, lattice: np.ndarray):
        """
        Initialize the solver.

        Parameters:
        -----------
        info : DisorderInfo
            Raw disorder data from Phase 1
        graph : networkx.Graph
            Exclusion graph from Phase 2
        lattice : np.ndarray
            3x3 matrix representing the lattice vectors
        """
        self.info = info
        self.graph = graph
        self.lattice = lattice
        self.atom_groups = []  # Will store groups of atoms by (disorder_group, assembly)

    def _identify_atom_groups(self):
        """
        Group atoms by key:
        - For PART >= 0: (disorder_group, assembly)
        - For PART < 0:  (disorder_group, assembly, sym_op_index)

        This separation ensures that symmetry-generated copies (which share the same PART ID
        but have different symmetry origins) are treated as separate Rigid Bodies.
        """
        # Dictionary to map group key to list of atom indices
        groups_map = {}

        # Check availability of symmetry info
        has_sym_info = hasattr(self.info, "sym_op_indices") and self.info.sym_op_indices

        for i in range(len(self.info.labels)):
            # Get disorder group and assembly for this atom
            disorder_group = self.info.disorder_groups[i]
            assembly = self.info.assemblies[i] if i < len(self.info.assemblies) else ""

            # Logic to handle PART -1 separation
            if disorder_group < 0 and has_sym_info:
                sym_op = self.info.sym_op_indices[i] if i < len(self.info.sym_op_indices) else 0
                # Include sym_op in the key to separate ghosts
                group_key = (disorder_group, assembly, sym_op)
            else:
                # Normal behavior for PART 1, 2 or PART -1 without sym info
                group_key = (disorder_group, assembly)

            if group_key not in groups_map:
                groups_map[group_key] = []

            groups_map[group_key].append(i)

        # Process each initial group with spatial clustering
        for group_atoms in groups_map.values():
            if len(group_atoms) == 1:
                # Single atom group - add directly
                self.atom_groups.append(group_atoms)
            else:
                # First check: does the ENTIRE group have any internal conflicts?
                # If not, keep it as one rigid body without spatial clustering.
                # This handles cases like NH4+ where 4 H atoms belong to the same
                # disorder alternative but are not bonded to each other (H-H distance
                # exceeds bonding threshold), so spatial clustering would incorrectly
                # split them into singletons.
                group_set = set(group_atoms)
                whole_group_has_conflict = False
                for idx1 in group_atoms:
                    for idx2 in group_atoms:
                        if idx1 != idx2 and self.graph.has_edge(idx1, idx2):
                            whole_group_has_conflict = True
                            break
                    if whole_group_has_conflict:
                        break

                if not whole_group_has_conflict:
                    # No internal conflicts at all — keep the whole group as one rigid body
                    self.atom_groups.append(group_atoms)
                    continue

                # There are internal conflicts: use spatial clustering to split into
                # sub-groups (connected components), then check each sub-group.
                # Build a temporary distance graph using ASE's get_distances
                group_coords = self.info.frac_coords[group_atoms]
                cart_coords = np.dot(group_coords, self.lattice)

                # Calculate pairwise distances between atoms in the group
                # Using ASE's get_distances which handles PBC
                dist_matrix = get_distances(
                    cart_coords, cart_coords, cell=self.lattice, pbc=True
                )[1]

                # Create a temporary graph for clustering
                temp_graph = nx.Graph()
                temp_graph.add_nodes_from(range(len(group_atoms)))

                # Connect atoms based on bonding thresholds calculated from atomic radii
                # This keeps molecules (like ClO4 or Methylammonium) together as rigid bodies
                for i in range(len(group_atoms)):
                    for j in range(i + 1, len(group_atoms)):
                        distance = dist_matrix[i, j]
                        
                        # Get symbols for both atoms
                        symbol_i = self.info.symbols[group_atoms[i]]
                        symbol_j = self.info.symbols[group_atoms[j]]
                        
                        # Get atomic radii
                        radius_i = get_atomic_radius(symbol_i) if has_atomic_radius(symbol_i) else 0.5
                        radius_j = get_atomic_radius(symbol_j) if has_atomic_radius(symbol_j) else 0.5
                        
                        # Check if atoms are metals
                        is_metal_i = is_metal_element(symbol_i)
                        is_metal_j = is_metal_element(symbol_j)
                        
                        # Calculate bonding threshold
                        threshold = get_bonding_threshold(radius_i, radius_j, is_metal_i, is_metal_j)
                        
                        if distance < threshold:
                            temp_graph.add_edge(i, j)

                # Find connected components (potential "Rigid Bodies")
                components = list(nx.connected_components(temp_graph))

                # For each component, check if it contains internal conflicts
                for comp in components:
                    comp_list = list(comp)
                    comp_atoms = [group_atoms[i] for i in comp_list]

                    # Check if this component has any internal conflicts in the main graph
                    has_internal_conflict = False

                    # Check all pairs of atoms in this component for conflicts in the main graph
                    if len(comp_atoms) > 1:
                        for idx1 in comp_atoms:
                            for idx2 in comp_atoms:
                                if idx1 != idx2 and self.graph.has_edge(idx1, idx2):
                                    has_internal_conflict = True
                                    break
                            if has_internal_conflict:
                                break

                    if has_internal_conflict:
                        # Explode into single-atom groups if the rigid body is self-conflicting
                        for atom_idx in comp_atoms:
                            self.atom_groups.append([atom_idx])
                    else:
                        # Keep as a rigid body group
                        self.atom_groups.append(comp_atoms)

        # Step 2: Merge cross-asym-id chemical motifs (e.g., water O+H)
        self._merge_chemical_motifs()

    # Per-element settings for chemical-motif reconstruction
    # (water O + 2 H; ammonium N + 4 H).  Subclasses can extend.
    _MOTIF_BOND_CUTOFF = {"O": 1.4, "N": 1.5}        # Å, X-H upper bound
    _MOTIF_BOND_MIN = {"O": 0.6, "N": 0.7}           # Å, X-H lower bound
    _MOTIF_MAX_H = {"O": 2, "N": 4}                  # H atoms per X
    _MOTIF_ANGLE_RANGE = {"O": (90.0, 120.0),        # H-X-H bounds
                          "N": (95.0, 125.0)}
    _MOTIF_HEAVY_CUTOFF = 1.8                        # Å, isolation check
    _MOTIF_HARD_CONFLICTS = frozenset({
        "logical_alternative", "symmetry_clash", "explicit", "valence",
    })
    # Soft-conflict types and per-element rejection policy.  Water (O)
    # re-uses the historic pre-fix behaviour of rejecting the whole motif
    # if any pair of picked atoms has a soft conflict edge — this filters
    # out chemically-questionable waters whose H atoms only fall inside
    # the bond cutoff because of overlapping split positions.  Ammonium
    # (N) keeps the permissive policy because its 24-orientation SP
    # disorder produces soft conflict edges between every H pair, so any
    # rejection there drops the whole NH4+ motif (the original PAP-4 bug).
    _MOTIF_SOFT_CONFLICTS = frozenset({
        "geometric", "valence_geometry", "implicit_sp",
    })
    _MOTIF_REJECT_SOFT = {"O": True, "N": False}

    def _merge_chemical_motifs(self):
        """
        Merge cross-asym-id chemical motifs (water, ammonium) into composite
        atom groups.

        After Step 1 explodes self-conflicting spatial clusters into
        singletons, isolated centers (water O, ammonium N) and their
        partial-occ H atoms end up as independent singleton groups.  This
        pass reconstructs each X(H)_n motif by:

        1. Identifying isolated centers — atoms of supported elements (O, N)
           in singleton groups with no heavy non-H neighbor within
           ``_MOTIF_HEAVY_CUTOFF``.  Framework O and ligand N fail this
           check and are not considered.

        2. Greedily selecting up to ``_MOTIF_MAX_H[element]`` H atoms within
           the X-H bond cutoff, sorted by distance.  An H is accepted iff:
             * it has no *hard* conflict (logical_alternative, symmetry_clash,
               explicit, valence) with an already-picked H, AND
             * the H-X-H angle to every already-picked H lies inside
               ``_MOTIF_ANGLE_RANGE[element]``.

           Soft conflicts (valence_geometry, geometric, implicit_sp) are
           ignored *within* the motif.  Those passes mark every H that
           looks geometrically suspect as exclusive of its neighbours,
           which is over-cautious for highly-disordered SP centres
           (e.g. PAP-4's NH4+ at a 24-orientation special position) where
           any chemically valid tetrahedron of H atoms necessarily uses
           split positions that the SP pass treats as exclusive.

        3. Merging the selected ``{X, H, ...}`` into a single rigid-body
           group, replacing the per-atom singletons.

        Typical cases:
          * MAF-4 / ZIF-8 water — O + 2 H from different asym_ids.
          * PAP-4 NH4+ — N + 4 H from one tetrahedral orientation, picked
            out of ~30 split positions.
        """
        n_atoms = len(self.info.labels)

        atom_to_group = {}
        for gi, group in enumerate(self.atom_groups):
            for a in group:
                atom_to_group[a] = gi

        centers = self._find_motif_centers(atom_to_group)
        if not centers:
            return

        h_singletons = [
            i for i in range(n_atoms)
            if self.info.symbols[i] in ("H", "D")
            and 0 < self.info.occupancies[i] < 1.0
            and atom_to_group.get(i) is not None
            and len(self.atom_groups[atom_to_group[i]]) == 1
        ]
        if not h_singletons:
            return

        c_frac = self.info.frac_coords[centers]
        h_frac = self.info.frac_coords[h_singletons]
        diff = c_frac[:, None, :] - h_frac[None, :, :]
        diff = diff - np.round(diff)
        cart_vecs = np.einsum("nij,jk->nik", diff, self.lattice)  # (n_c, n_h, 3)
        dists_c_h = np.linalg.norm(cart_vecs, axis=2)             # (n_c, n_h)

        # Each H atom can only be a real X-H bond for at most one center.
        # Pre-assign every H to the *closest* center within that center's
        # chemical bond range so a stretched O-H (e.g. 1.1 Å) on an
        # early-iterated center can't steal an H that has a stronger 0.86 Å
        # bond to a later-iterated center.  The lower bound rejects ghost
        # positions: split-occupancy refinements often place O and H within
        # ~0.4 Å of each other, which would otherwise win the closest-center
        # contest despite being below any chemically valid X-H bond length.
        cutoffs_max = np.array([
            self._MOTIF_BOND_CUTOFF[self.info.symbols[centers[ci]]]
            for ci in range(len(centers))
        ])
        cutoffs_min = np.array([
            self._MOTIF_BOND_MIN[self.info.symbols[centers[ci]]]
            for ci in range(len(centers))
        ])
        within = (dists_c_h < cutoffs_max[:, None]) & (dists_c_h >= cutoffs_min[:, None])
        masked = np.where(within, dists_c_h, np.inf)
        best_ci_per_h = np.argmin(masked, axis=0) if len(centers) else np.array([], int)
        any_within = np.any(within, axis=0)

        new_groups = []
        merged_group_indices = set()

        for ci, c_idx in enumerate(centers):
            allowed_h_mask = any_within & (best_ci_per_h == ci)
            picked_atoms = self._select_motif_hydrogens(
                c_idx, ci, dists_c_h, cart_vecs, h_singletons,
                allowed_h_mask=allowed_h_mask,
            )
            if not picked_atoms:
                continue

            motif_atoms = [c_idx] + picked_atoms
            motif_group_indices = {atom_to_group[a] for a in motif_atoms}

            if len(motif_group_indices) < 2:
                continue
            if motif_group_indices & merged_group_indices:
                continue

            new_groups.append(motif_atoms)
            merged_group_indices.update(motif_group_indices)

        if not merged_group_indices:
            return

        rebuilt = [
            g for gi, g in enumerate(self.atom_groups)
            if gi not in merged_group_indices
        ]
        rebuilt.extend(new_groups)
        self.atom_groups = rebuilt

    def _find_motif_centers(self, atom_to_group):
        """Return atom indices of isolated motif centers (water O, ammonium N).

        A center qualifies iff it (1) is in a singleton group from Step 1 and
        (2) has no heavy non-H neighbour of a different element within
        ``_MOTIF_HEAVY_CUTOFF``.  The same-element exclusion lets alternative
        positions of the same chemical site (e.g. two split positions of a
        single water O) coexist without disqualifying each other.
        """
        n_atoms = len(self.info.labels)
        centers = []

        for sym in self._MOTIF_BOND_CUTOFF.keys():
            cands = [
                i for i in range(n_atoms)
                if self.info.symbols[i] == sym
                and 0 < self.info.occupancies[i] <= 1.0
                and atom_to_group.get(i) is not None
                and len(self.atom_groups[atom_to_group[i]]) == 1
            ]
            if not cands:
                continue

            disq_mask = np.array(
                [s not in ("H", "D", sym) for s in self.info.symbols]
            )
            disq_indices = np.where(disq_mask)[0]
            if len(disq_indices) == 0:
                centers.extend(cands)
                continue

            c_frac = self.info.frac_coords[cands]
            d_frac = self.info.frac_coords[disq_indices]
            diff = c_frac[:, None, :] - d_frac[None, :, :]
            diff = diff - np.round(diff)
            cart = np.einsum("nij,jk->nik", diff, self.lattice)
            dists = np.linalg.norm(cart, axis=2)
            is_isolated = ~np.any(dists < self._MOTIF_HEAVY_CUTOFF, axis=1)
            centers.extend(
                cands[i] for i in range(len(cands)) if is_isolated[i]
            )

        return centers

    def _select_motif_hydrogens(self, c_idx, ci, dists_c_h, cart_vecs,
                                 h_singletons, allowed_h_mask=None):
        """Greedy distance-sorted selection of H atoms around one motif center.

        Returns the picked H indices (subset of ``h_singletons``).  Empty
        list means no acceptable H was found.

        ``allowed_h_mask``, when given, restricts the candidate pool to the
        H atoms whose closest motif center is this one.  This prevents a
        center with a stretched X-H distance from claiming an H that has a
        shorter bond to a different center.

        For elements where ``_MOTIF_REJECT_SOFT[element]`` is True, a final
        check rejects the whole motif if any picked-pair has a soft
        conflict edge in the disorder graph.  This keeps water (O) close
        to the historic pre-fix behaviour of "any internal conflict drops
        the merge".
        """
        c_sym = self.info.symbols[c_idx]
        cutoff = self._MOTIF_BOND_CUTOFF[c_sym]
        cutoff_min = self._MOTIF_BOND_MIN[c_sym]
        max_h = self._MOTIF_MAX_H[c_sym]
        ang_min, ang_max = self._MOTIF_ANGLE_RANGE[c_sym]
        reject_soft = self._MOTIF_REJECT_SOFT.get(c_sym, False)

        in_range = (dists_c_h[ci] < cutoff) & (dists_c_h[ci] >= cutoff_min)
        if allowed_h_mask is None:
            near_local = np.where(in_range)[0]
        else:
            near_local = np.where(in_range & allowed_h_mask)[0]
        if len(near_local) == 0:
            return []

        sorted_local = sorted(near_local, key=lambda j: dists_c_h[ci, j])

        picked_atoms = []
        picked_unit = []

        # Guard: when there are at least `max_h` distinct crystallographic H
        # sites around this centre (distinct asym_ids), each valid orientation
        # of the XHn motif uses exactly one H per site.  Enforce "one pick per
        # asym_id" so that the greedy never selects two copies of the same
        # disorder position (e.g. two H1D atoms pointing in slightly different
        # but near-parallel directions) before exhausting the other sites.
        #
        # When fewer distinct asym_ids are present (e.g. DAP-4 NH4+ where only
        # H3A and H3B label two sites but a valid tetrahedron uses 3×H3A +
        # 1×H3B), enforcing the guard would incorrectly cap the count at 2.
        # In that case we fall back to the original angle-only heuristic.
        has_asym = bool(self.info.asym_id)
        if has_asym:
            candidate_asym_ids = {
                self.info.asym_id[h_singletons[j]]
                for j in near_local
                if h_singletons[j] < len(self.info.asym_id)
            }
            enforce_one_per_asym = len(candidate_asym_ids) >= max_h
        else:
            enforce_one_per_asym = False
        picked_asym_ids: set = set()

        for j in sorted_local:
            if len(picked_atoms) >= max_h:
                break
            h = h_singletons[j]

            # One H per crystallographic site (when enough distinct sites exist).
            if enforce_one_per_asym and h < len(self.info.asym_id):
                h_asym = self.info.asym_id[h]
                if h_asym in picked_asym_ids:
                    continue

            has_hard = False
            for p in picked_atoms:
                if self.graph.has_edge(h, p):
                    ct = self.graph[h][p].get("conflict_type", "")
                    if ct in self._MOTIF_HARD_CONFLICTS:
                        has_hard = True
                        break
            if has_hard:
                continue

            uv = -cart_vecs[ci, j] / dists_c_h[ci, j]
            angle_ok = True
            for puv in picked_unit:
                cos_a = float(np.clip(np.dot(uv, puv), -1.0, 1.0))
                ang = np.degrees(np.arccos(cos_a))
                if not (ang_min <= ang <= ang_max):
                    angle_ok = False
                    break
            if not angle_ok:
                continue

            picked_atoms.append(h)
            picked_unit.append(uv)
            if enforce_one_per_asym and h < len(self.info.asym_id):
                picked_asym_ids.add(self.info.asym_id[h])

        if reject_soft and picked_atoms:
            motif_atoms = [c_idx] + picked_atoms
            for i, a in enumerate(motif_atoms):
                for b in motif_atoms[i + 1:]:
                    if self.graph.has_edge(a, b):
                        ct = self.graph[a][b].get("conflict_type", "")
                        if ct in self._MOTIF_SOFT_CONFLICTS:
                            return []

        return picked_atoms

    def _max_weight_independent_set_by_groups(
        self, graph=None, weight_attr="occupancy"
    ):
        """
        Find an independent set using Group-Based solving approach (Greedy MWIS).
        This mimics the robust logic of the main branch but applied to molecular groups.

        Parameters:
        -----------
        graph : networkx.Graph, optional
            Graph to work with. If None, uses self.graph
        weight_attr : str
            The node attribute to use as weight (default: occupancy)

        Returns:
        --------
        List[int]
            List of node indices forming the independent set
        """
        # Use provided graph or fallback to self.graph
        working_graph = graph.copy() if graph is not None else self.graph.copy()

        # Calculate weight for each Group (sum of atom weights)
        # Also factor in degree (number of conflicts) to mimic main branch's heuristic:
        # Score = Weight / (Degree + 1) -> Prefer High Weight, Low Conflict
        group_scores = []

        for group in self.atom_groups:
            # Check if group is valid in current graph
            if not all(working_graph.has_node(node) for node in group):
                group_scores.append(-1.0)  # Mark as invalid
                continue

            # [FIX] Skip groups where ALL atoms have zero base occupancy.
            # These are dummy/placeholder atoms (e.g., N01 in caffeine2) that
            # should never appear in a physical structure.  In random mode,
            # tiny noise would give them positive randomized_weight, letting
            # them slip past the score>0 check.  Using the canonical occupancy
            # from DisorderInfo prevents this.
            base_occs = [self.info.occupancies[node] for node in group
                         if node < len(self.info.occupancies)]
            if base_occs and all(occ <= 0.0 for occ in base_occs):
                group_scores.append(-1.0)  # Mark as invalid — zero-occ group
                continue

            # Total Weight of the group
            weight = sum(
                working_graph.nodes[node].get(weight_attr, 1.0) for node in group
            )

            # Total external degree (conflicts with nodes OUTSIDE the group)
            degree = 0
            for node in group:
                for neighbor in working_graph.neighbors(node):
                    if (
                        neighbor not in group
                    ):  # Ignore internal edges (though there shouldn't be any)
                        degree += 1

            # Heuristic score: Weight × GroupSize / (Degree + 1)
            # The GroupSize multiplier ensures that multi-atom rigid bodies
            # (e.g., ClO3 perchlorate) are preferred over their individual
            # fragment atoms that appear as singletons at other symmetry
            # operations.  Without this, a lone O singleton with low degree
            # can outscore a complete ClO3 group, causing Cl to vanish from
            # the resolved structure.
            score = weight * len(group) / (degree + 1.0)
            group_scores.append(score)

        # Sort Groups by score (descending)
        sorted_group_indices = sorted(
            range(len(self.atom_groups)), key=lambda i: group_scores[i], reverse=True
        )

        independent_set = []

        # Iterate Groups in descending order of score
        for group_idx in sorted_group_indices:
            if group_scores[group_idx] <= 0:
                continue

            group = self.atom_groups[group_idx]

            # Check if ALL atoms in the Group are currently available in the graph
            all_available = all(working_graph.has_node(node) for node in group)

            if not all_available:
                continue

            # Since we removed neighbors immediately upon selection,
            # if the nodes are present, they are guaranteed to be valid candidates
            # (unless internal conflicts exist, but we checked that).

            # Double check for conflicts with already selected set (Defensive coding)
            no_conflicts = True
            for node in group:
                for selected_node in independent_set:
                    if working_graph.has_edge(node, selected_node):
                        no_conflicts = False
                        break
                if not no_conflicts:
                    break

            # If the group is valid, add all its atoms to the independent set
            if all_available and no_conflicts:
                # Add all atoms in this group to the independent set
                independent_set.extend(group)

                # Remove all group members AND their neighbors from the graph
                nodes_to_remove = []
                for node in group:
                    nodes_to_remove.append(node)
                    # Add neighbors of this node
                    nodes_to_remove.extend(list(working_graph.neighbors(node)))

                # Remove duplicates
                nodes_to_remove = list(set(nodes_to_remove))

                # Remove from working graph
                working_graph.remove_nodes_from(nodes_to_remove)

        return independent_set

    def _root_label(self, atom_idx: int) -> str:
        label = self.info.labels[atom_idx]
        match = re.match(r"([A-Za-z]+[0-9]*)", label)
        return match.group(1) if match else label

    def _minimum_image_distance(self, atom_a: int, atom_b: int) -> float:
        diff = self.info.frac_coords[atom_a] - self.info.frac_coords[atom_b]
        diff = diff - np.round(diff)
        return float(np.linalg.norm(np.dot(diff, self.lattice)))

    def _bond_threshold(self, atom_a: int, atom_b: int) -> float:
        symbol_a = self.info.symbols[atom_a]
        symbol_b = self.info.symbols[atom_b]
        radius_a = get_atomic_radius(symbol_a) if has_atomic_radius(symbol_a) else 0.5
        radius_b = get_atomic_radius(symbol_b) if has_atomic_radius(symbol_b) else 0.5
        return get_bonding_threshold(
            radius_a,
            radius_b,
            is_metal_element(symbol_a),
            is_metal_element(symbol_b),
        )

    def _bonded_neighbors(self, atom_idx: int, selected: set[int]) -> list[int]:
        neighbors = []
        for other in selected:
            if other == atom_idx:
                continue
            if self._minimum_image_distance(atom_idx, other) < self._bond_threshold(atom_idx, other):
                neighbors.append(other)
        return neighbors

    def _snap_atom_to_partner(self, kept_atom: int, partner_atom: int) -> None:
        kept = self.info.frac_coords[kept_atom]
        partner = self.info.frac_coords[partner_atom]
        diff = partner - kept
        diff = diff - np.round(diff)
        self.info.frac_coords[kept_atom] = (kept + 0.5 * diff) % 1.0

    def _place_h_away_from_anchor_neighbors(
        self, h_atom: int, anchor_atom: int, selected: set[int]
    ) -> None:
        anchor_frac = self.info.frac_coords[anchor_atom]
        anchor_cart = np.dot(anchor_frac, self.lattice)
        seed_direction = np.zeros(3)

        for neighbor in self._bonded_neighbors(anchor_atom, selected):
            if neighbor == h_atom:
                continue
            neigh_frac = self.info.frac_coords[neighbor]
            diff = neigh_frac - anchor_frac
            diff = diff - np.round(diff)
            vec = np.dot(diff, self.lattice)
            norm = np.linalg.norm(vec)
            if norm > 0:
                seed_direction -= vec / norm

        directions = []
        norm = np.linalg.norm(seed_direction)
        if norm > 0:
            directions.append(seed_direction / norm)

        # Deterministic low-cost sphere search.  This is only used for rare
        # SP-completion H cleanup, so a few dozen trial directions are enough.
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                for z in (-1.0, 0.0, 1.0):
                    vec = np.array([x, y, z], dtype=float)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        directions.append(vec / norm)

        inv_lattice = np.linalg.inv(self.lattice)
        best_frac = None
        best_score = -np.inf
        original_frac = self.info.frac_coords[h_atom].copy()

        for direction in directions:
            new_cart = anchor_cart + direction * self._SP_RELOCATED_C_H_DISTANCE
            new_frac = np.dot(new_cart, inv_lattice) % 1.0
            self.info.frac_coords[h_atom] = new_frac

            min_margin = np.inf
            for other in selected:
                if other in (h_atom, anchor_atom):
                    continue
                dist = self._minimum_image_distance(h_atom, other)
                threshold = self._bond_threshold(h_atom, other)
                min_margin = min(min_margin, dist - threshold)

            if min_margin > best_score:
                best_score = min_margin
                best_frac = new_frac.copy()
            if min_margin > 0.05:
                break

        self.info.frac_coords[h_atom] = best_frac if best_frac is not None else original_frac

    def _apply_sp_completion(self, independent_set: List[int]) -> List[int]:
        """
        Complete negative-PART fragments that straddle a special position.

        The graph builder records conformer pairs where most same-label atoms
        are nearly overlapping symmetry mates.  MWIS either keeps one half or
        both halves; in both cases we collapse the overlapping sites onto their
        average position and retain only the non-overlapping partner atoms that
        complete the molecule.
        """
        pairs = self.graph.graph.get("sp_completion_pairs", [])
        if not pairs:
            return independent_set

        selected = set(independent_set)
        ordered = list(independent_set)

        for raw_a, raw_b in pairs:
            conf_a = set(raw_a)
            conf_b = set(raw_b)
            hit_a = selected & conf_a
            hit_b = selected & conf_b
            if not hit_a and not hit_b:
                continue

            if len(hit_b) > len(hit_a):
                kept, partner = conf_b, conf_a
            else:
                kept, partner = conf_a, conf_b

            snapped_partners = set()
            for kept_atom in sorted(kept):
                if kept_atom not in selected:
                    continue
                best_atom = None
                best_dist = np.inf
                kept_root = self._root_label(kept_atom)
                for partner_atom in partner:
                    if self._root_label(partner_atom) != kept_root:
                        continue
                    dist = self._minimum_image_distance(kept_atom, partner_atom)
                    if dist < best_dist:
                        best_dist = dist
                        best_atom = partner_atom
                if best_atom is None or best_dist >= self._SP_COMPLETION_DISTANCE:
                    continue

                self._snap_atom_to_partner(kept_atom, best_atom)
                snapped_partners.add(best_atom)

            for partner_atom in sorted(snapped_partners):
                selected.discard(partner_atom)

            for partner_atom in sorted(partner - snapped_partners):
                if partner_atom not in selected:
                    selected.add(partner_atom)
                    ordered.append(partner_atom)

        return [atom for atom in ordered if atom in selected]

    def _remove_too_close_sp_hydrogens(self, independent_set: List[int]) -> List[int]:
        if not self.graph.graph.get("sp_completion_pairs"):
            return independent_set

        selected = set(independent_set)
        to_remove = set()
        heavy_atoms = [a for a in selected if self.info.symbols[a] not in ("H", "D")]
        for h_atom in [a for a in selected if self.info.symbols[a] in ("H", "D")]:
            for heavy_atom in heavy_atoms:
                if self._minimum_image_distance(h_atom, heavy_atom) < self._SP_TOO_CLOSE_H_DISTANCE:
                    to_remove.add(h_atom)
                    break

        if to_remove:
            logger.debug(
                "Removing %d SP-completion H atom(s) too close to heavy atoms: %s",
                len(to_remove),
                sorted(to_remove),
            )
        return [atom for atom in independent_set if atom not in to_remove]

    def _relocate_overcoord_sp_hydrogens(self, independent_set: List[int]) -> List[int]:
        if not self.graph.graph.get("sp_completion_pairs"):
            return independent_set

        selected = set(independent_set)
        max_coord = {"C": 4, "N": 4, "O": 3}

        for carbon in sorted(a for a in selected if self.info.symbols[a] == "C"):
            neighbors = self._bonded_neighbors(carbon, selected)
            if len(neighbors) <= max_coord["C"]:
                continue

            h_neighbors = [
                n for n in neighbors if self.info.symbols[n] in ("H", "D")
            ]
            if not h_neighbors:
                continue

            undercoord_carbons = []
            for candidate in selected:
                if candidate == carbon or self.info.symbols[candidate] != "C":
                    continue
                cand_neighbors = self._bonded_neighbors(candidate, selected)
                if len(cand_neighbors) >= max_coord["C"]:
                    continue
                undercoord_carbons.append(candidate)

            moved = False
            for h_atom in sorted(
                h_neighbors,
                key=lambda h: self._minimum_image_distance(carbon, h),
                reverse=True,
            ):
                targets = [
                    c for c in undercoord_carbons
                    if self._minimum_image_distance(h_atom, c) < self._SP_UNDERCOORD_H_SEARCH
                ]
                if not targets:
                    continue
                target = min(
                    targets,
                    key=lambda c: self._minimum_image_distance(h_atom, c),
                )
                self._place_h_away_from_anchor_neighbors(h_atom, target, selected)
                moved = True
                break

            if moved:
                logger.debug(
                    "Relocated an SP-completion H from over-coordinated C%d",
                    carbon,
                )

        return independent_set

    def solve(
        self, num_structures: int = 1, method: str = "optimal"
    ) -> List[MolecularCrystal]:
        """
        Solve the disorder problem and generate ordered structures using Group-Based approach.

        Parameters:
        -----------
        num_structures : int
            Number of structures to generate (for 'random' method)
        method : str
            'optimal' for single best structure, 'random' for ensemble

        Returns:
        --------
        List[MolecularCrystal]
            List of ordered molecular crystal structures
        """
        # Initialize atom groups (Identify Rigid Bodies)
        self._identify_atom_groups()

        # Ensure graph has occupancy weights
        for node in self.graph.nodes():
            if "occupancy" not in self.graph.nodes[node]:
                atom_idx = node
                if atom_idx < len(self.info.occupancies):
                    self.graph.nodes[node]["occupancy"] = self.info.occupancies[
                        atom_idx
                    ]

        if method == "optimal":
            # [FIX] Do NOT apply stoichiometry constraints.
            # Rely strictly on the Exclusion Graph + MWIS (Max Weight Independent Set).
            # This matches 'main' branch logic but with added benefit of Rigid Body groups.

            independent_sets = [
                self._max_weight_independent_set_by_groups(
                    graph=self.graph, weight_attr="occupancy"
                )
            ]

        elif method == "random":
            independent_sets = []
            seen_structures = set()

            for _ in range(num_structures):
                # Create temp graph with randomized weights
                temp_graph = self.graph.copy()

                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                try:
                    # Solve using Group-Based MWIS
                    solution = self._max_weight_independent_set_by_groups(
                        graph=temp_graph, weight_attr="randomized_weight"
                    )
                except:
                    solution = self._random_independent_set()

                solution_tuple = tuple(sorted(solution))
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)

            # Fill remaining with pure random if needed
            attempts = 0
            max_attempts = num_structures * 10
            while len(independent_sets) < num_structures and attempts < max_attempts:
                temp_graph = self.graph.copy()
                for node in temp_graph.nodes():
                    base_weight = temp_graph.nodes[node].get("occupancy", 1.0)
                    random_noise = random.uniform(0, 1e-5)
                    temp_graph.nodes[node]["randomized_weight"] = (
                        base_weight + random_noise
                    )

                solution = self._max_weight_independent_set_by_groups(
                    graph=temp_graph, weight_attr="randomized_weight"
                )

                solution_tuple = tuple(sorted(solution))
                if solution_tuple not in seen_structures:
                    seen_structures.add(solution_tuple)
                    independent_sets.append(solution)
                attempts += 1
        else:
            raise ValueError(f"Unknown method: {method}. Use 'optimal' or 'random'")

        # Post-process: remove orphan H/D atoms that lack bonded heavy-atom
        # partners in the survived set.  This fixes cross-asym-id water
        # disorder (e.g., MAF-4) where O and H are clustered independently.
        cleaned_sets = []
        for independent_set in independent_sets:
            completed_set = self._apply_sp_completion(independent_set)
            completed_set = self._remove_too_close_sp_hydrogens(completed_set)
            cleaned_set = self._remove_orphan_hydrogens(completed_set)
            cleaned_sets.append(self._relocate_overcoord_sp_hydrogens(cleaned_set))

        # Reconstruct crystals and run valence-completeness diagnostics
        from .diagnostics import check_valence_completeness

        crystals = []
        for independent_set in cleaned_sets:
            crystal = self._reconstruct_crystal(independent_set)
            issues = check_valence_completeness(crystal, self.info)
            if issues:
                for issue in issues:
                    logger.warning(
                        "Valence-completeness check: %s", issue
                    )
            crystals.append(crystal)

        return crystals

    def _random_independent_set(self) -> List[int]:
        """
        Fallback: Generate a random independent set using a randomized greedy algorithm.
        """
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)
        independent_set = []
        for node in nodes:
            connected_to_set = False
            for selected_node in independent_set:
                if self.graph.has_edge(node, selected_node):
                    connected_to_set = True
                    break
            if not connected_to_set:
                independent_set.append(node)
        return independent_set

    def _remove_orphan_hydrogens(self, independent_set: List[int]) -> List[int]:
        """
        Remove chemically impossible fragments from the independent set.

        Pass 1 — Orphan H: Remove H/D atoms that lack a bonded heavy-atom
        partner within the survived set.  In structures like MAF-4, water O
        and H have different asym_ids and are resolved independently by the
        SP conflict clustering.  An isolated H atom is chemically impossible.

        Pass 2 — Incomplete water: Remove partial-occupancy O atoms (water O)
        that, after Pass 1, still have fewer than 2 bonded H in the survived
        set.  A water molecule by definition has 2 H atoms; an O with only 1
        bonded H (HO fragment) is an artefact of independent O/H resolution.
        We identify "water O" as: partial occupancy AND no bonded C/N/S/P
        heavy atom in the full structure (connectivity-based, not just
        survived set).  When a water O is incomplete we remove the O AND all
        H within O_H_SWEEP_CUTOFF of it that are in the survived set — those
        H belong to this water site and have nowhere valid to go.

        The check uses a generous bond-length cutoff (1.4 Å) to catch
        both standard O-H (~0.96 Å) and the short crystallographic
        H positions (~0.75 Å from X-ray refinement).  The sweep cutoff
        (2.0 Å) is wider to also catch H copies that the random solver
        picked from a nearby but slightly different position.
        """
        from ...constants.config import DISORDER_CONFIG
        H_BOND_CUTOFF     = DISORDER_CONFIG["ORPHAN_H_BOND_CUTOFF"]
        O_H_CUTOFF        = DISORDER_CONFIG["WATER_O_H_CUTOFF"]
        O_H_SWEEP_CUTOFF  = DISORDER_CONFIG["WATER_O_H_SWEEP_CUTOFF"]
        HEAVY_BOND_CUTOFF = DISORDER_CONFIG["WATER_O_HEAVY_BOND_CUTOFF"]

        # Pre-compute O-coordination in the FULL structure (not just survived),
        # to identify "water O": partial-occ O bonded ONLY to H (and other O).
        # Any O bonded to C, N, S, P, Cl, or any metal is NOT water O.
        n_atoms_total = len(self.info.labels)

        # Symbols that, if bonded to O, mean the O is NOT water
        non_water_O_neighbors = {s for s in set(self.info.symbols)
                                  if s not in ("H", "D", "O")}

        # Build boolean mask of "potentially disqualifying" atoms
        disq_mask = np.array([s in non_water_O_neighbors
                              for s in self.info.symbols], dtype=bool)
        disq_indices = np.where(disq_mask)[0]

        # Candidate partial-occ O atoms
        partial_O_indices = [i for i in range(n_atoms_total)
                             if (self.info.symbols[i] == "O"
                                 and self.info.occupancies[i] < 1.0)]

        is_water_O = {}
        if partial_O_indices and len(disq_indices) > 0:
            # Vectorized PBC distance: partial_O × disqualifying_atoms
            o_frac = self.info.frac_coords[partial_O_indices]   # (n_o, 3)
            d_frac = self.info.frac_coords[disq_indices]         # (n_d, 3)
            diff = o_frac[:, None, :] - d_frac[None, :, :]      # (n_o, n_d, 3)
            diff = diff - np.round(diff)
            cart = np.einsum("nij,jk->nik", diff, self.lattice)  # (n_o, n_d, 3)
            dists_o_d = np.linalg.norm(cart, axis=2)            # (n_o, n_d)

            for oi, i in enumerate(partial_O_indices):
                bonded_non_water = bool(np.any(dists_o_d[oi] < HEAVY_BOND_CUTOFF))
                is_water_O[i] = not bonded_non_water
        else:
            for i in partial_O_indices:
                is_water_O[i] = True  # no disqualifying atoms → all are water O

        def _do_pass(ind_set):
            """One round of orphan-H removal + incomplete-water removal."""
            survived = list(ind_set)
            if not survived:
                return survived

            surv_set = set(survived)

            # --- Pass 1: orphan H ---
            h_atoms  = [a for a in survived if self.info.symbols[a] in ("H", "D")]
            hvy_atoms = [a for a in survived if self.info.symbols[a] not in ("H", "D")]

            if h_atoms and hvy_atoms:
                h_frac   = self.info.frac_coords[h_atoms]
                hvy_frac = self.info.frac_coords[hvy_atoms]
                diff = h_frac[:, None, :] - hvy_frac[None, :, :]
                diff = diff - np.round(diff)
                cart_diff = np.einsum("nij,jk->nik", diff, self.lattice)
                dists_h_hvy = np.linalg.norm(cart_diff, axis=2)

                orphan_H = set()
                for hi, h_idx in enumerate(h_atoms):
                    if np.min(dists_h_hvy[hi]) > H_BOND_CUTOFF:
                        orphan_H.add(h_idx)
                if orphan_H:
                    logger.debug(
                        "Removing %d orphan H/D atom(s) with no heavy-atom "
                        "partner within %.2f Å: indices %s",
                        len(orphan_H), H_BOND_CUTOFF, sorted(orphan_H),
                    )
                surv_set -= orphan_H

            # --- Pass 2: incomplete water O ---
            # For each surviving water O, count bonded H in survived set.
            # Water O needs exactly 2 H.  If < 2, remove it.  We also remove
            # any H within O_H_SWEEP_CUTOFF that have NO other surviving heavy
            # atom within H_BOND_CUTOFF — i.e., H that are exclusively bonded
            # to this incomplete O and would become orphans anyway.
            o_atoms = [a for a in surv_set if self.info.symbols[a] == "O"
                       and is_water_O.get(a, False)]
            h_atoms2 = [a for a in surv_set if self.info.symbols[a] in ("H", "D")]

            if o_atoms and h_atoms2:
                o_frac = self.info.frac_coords[o_atoms]
                h_frac2 = self.info.frac_coords[h_atoms2]
                diff2 = o_frac[:, None, :] - h_frac2[None, :, :]
                diff2 = diff2 - np.round(diff2)
                cart_diff2 = np.einsum("nij,jk->nik", diff2, self.lattice)
                dists_o_h = np.linalg.norm(cart_diff2, axis=2)  # (n_o, n_h)

                # Also compute H-to-all-heavy distances to check if H has
                # another heavy anchor besides the incomplete O
                hvy_in_surv = [a for a in surv_set if self.info.symbols[a] not in ("H", "D")]
                if hvy_in_surv:
                    h_frac3 = self.info.frac_coords[h_atoms2]
                    hvy_frac3 = self.info.frac_coords[hvy_in_surv]
                    diff3 = h_frac3[:, None, :] - hvy_frac3[None, :, :]
                    diff3 = diff3 - np.round(diff3)
                    cart_diff3 = np.einsum("nij,jk->nik", diff3, self.lattice)
                    dists_h_hvy3 = np.linalg.norm(cart_diff3, axis=2)  # (n_h, n_hvy)
                else:
                    dists_h_hvy3 = None

                # Map h_atoms2 index → position in hvy_in_surv for "other heavy" check
                hvy_set = set(hvy_in_surv)

                incomplete_O = set()
                h_to_remove = set()
                for oi, o_idx in enumerate(o_atoms):
                    n_bonded_h = int(np.sum(dists_o_h[oi] < O_H_CUTOFF))
                    if n_bonded_h < 2:
                        incomplete_O.add(o_idx)
                        # Remove H within sweep range that have no other heavy anchor
                        if dists_h_hvy3 is not None:
                            for hj, h_idx in enumerate(h_atoms2):
                                if dists_o_h[oi, hj] < O_H_SWEEP_CUTOFF:
                                    # Check if H has ANY other surviving heavy atom
                                    # within H_BOND_CUTOFF (i.e., not just this O)
                                    near_hvy_count = 0
                                    for ki, hvy_idx in enumerate(hvy_in_surv):
                                        if hvy_idx != o_idx and dists_h_hvy3[hj, ki] < H_BOND_CUTOFF:
                                            near_hvy_count += 1
                                            break
                                    if near_hvy_count == 0:
                                        h_to_remove.add(h_idx)
                if incomplete_O:
                    logger.debug(
                        "Removing %d incomplete water O atom(s) with < 2 bonded H "
                        "within %.2f Å: indices %s",
                        len(incomplete_O), O_H_CUTOFF, sorted(incomplete_O),
                    )
                if h_to_remove:
                    logger.debug(
                        "Removing %d H/D atom(s) near incomplete water O "
                        "(sweep cutoff %.2f Å): indices %s",
                        len(h_to_remove), O_H_SWEEP_CUTOFF, sorted(h_to_remove),
                    )
                surv_set -= incomplete_O
                surv_set -= h_to_remove

            return [a for a in survived if a in surv_set]

        # Run until convergence: removing incomplete-O can expose new orphan-H,
        # which in turn may expose more incomplete-O, etc.
        result = list(independent_set)
        for _iteration in range(10):  # hard cap to prevent infinite loops
            next_result = _do_pass(result)
            if next_result == result:
                break
            result = next_result
        return result

    def _reconstruct_crystal(self, independent_set: List[int]) -> MolecularCrystal:
        """
        Reconstruct a MolecularCrystal from an independent set of atoms.
        """
        selected_symbols = [self.info.symbols[i] for i in independent_set]
        selected_frac_coords = self.info.frac_coords[independent_set]

        atoms = Atoms(
            symbols=selected_symbols,
            scaled_positions=selected_frac_coords,
            cell=self.lattice,
            pbc=True,
        )
        molecules = identify_molecules(atoms)
        pbc = (True, True, True)
        crystal = MolecularCrystal(self.lattice, molecules, pbc)

        return crystal
