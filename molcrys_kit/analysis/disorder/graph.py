"""
Exclusion Graph Construction for Disorder Handling.

This module implements the "Referee" logic that determines which atoms cannot
coexist in the same physical structure based on raw disorder data.
"""

import numpy as np
import networkx as nx
import re
from typing import List
from itertools import combinations
from .info import DisorderInfo
from ...constants.config import (
    DISORDER_CONFIG,
    BONDING_THRESHOLDS,
    MAX_COORDINATION_NUMBERS,
    DEFAULT_MAX_COORDINATION,
    TRANSITION_METALS,
)
from ...utils.geometry import (
    angle_between_vectors,
    frac_to_cart,
)


class DisorderGraphBuilder:
    """
    Builds an exclusion graph from DisorderInfo data.
    """

    def __init__(self, info: DisorderInfo, lattice: np.ndarray):
        self.info = info
        self.lattice = lattice
        self.graph = nx.Graph()
        self.conformers = []

        # Add nodes
        for i in range(len(info.labels)):
            self.graph.add_node(
                i,
                label=info.labels[i],
                symbol=info.symbols[i],
                frac_coord=info.frac_coords[i],
                occupancy=info.occupancies[i],
                disorder_group=info.disorder_groups[i],
                assembly=info.assemblies[i] if i < len(info.assemblies) else "",
            )

        self._precompute_metrics()

    def _precompute_metrics(self):
        coords = self.info.frac_coords
        # Precompute distance matrix with PBC
        coord_diffs = coords[:, None, :] - coords[None, :, :]
        coord_diffs = coord_diffs - np.round(coord_diffs)
        cart_diffs = np.einsum("nij,jk->nik", coord_diffs, self.lattice)
        self.dist_matrix = np.linalg.norm(cart_diffs, axis=2)

        self.root_labels = []
        for label in self.info.labels:
            match = re.match(r"([A-Za-z]+[0-9]*)", label)
            root_label = match.group(1) if match else label
            self.root_labels.append(root_label)

    def build(self) -> nx.Graph:
        """
        Build the exclusion graph using the Conformer-Centric Architecture.
        """
        self._identify_conformers()
        self._add_conformer_conflicts()
        self._add_explicit_conflicts()
        self._add_geometric_conflicts()
        self._add_implicit_sp_conflicts()
        self._resolve_valence_conflicts()
        return self.graph

    def _identify_conformers(self):
        """
        Identify discrete conformers (clusters) using connectivity, PART rules, AND Symmetry.
        Prevents 'Frankenstein' merging of symmetry images.

        A conformer is a group of atoms that belong to the same disorder alternative
        and the same symmetry-operation copy. The key is (part_id, sym_op).
        """
        n_atoms = len(self.info.labels)
        has_sym_info = hasattr(self.info, "sym_op_indices") and self.info.sym_op_indices

        bond_graph = nx.Graph()
        bond_graph.add_nodes_from(range(n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                group_i = self.info.disorder_groups[i]
                group_j = self.info.disorder_groups[j]

                # Check bonding validity
                if (group_i == group_j and group_i != 0) or (
                    group_i == 0 or group_j == 0
                ):
                    # Anti-Frankenstein: Do not bond two disordered atoms if they come from different SymOps
                    if group_i != 0 and group_j != 0 and has_sym_info:
                        idx_i = (
                            self.info.sym_op_indices[i]
                            if i < len(self.info.sym_op_indices)
                            else 0
                        )
                        idx_j = (
                            self.info.sym_op_indices[j]
                            if j < len(self.info.sym_op_indices)
                            else 0
                        )
                        if idx_i != idx_j:
                            continue

                    dist = self.dist_matrix[i, j]
                    symbol_i = self.info.symbols[i]
                    symbol_j = self.info.symbols[j]

                    if self._are_bonded(symbol_i, symbol_j, dist, group_i, group_j):
                        if not (group_i < 0 and group_j < 0 and group_i != group_j):
                            bond_graph.add_edge(i, j)

        molecule_components = list(nx.connected_components(bond_graph))
        self.conformers = []

        for component in molecule_components:
            atoms_by_key = {}
            for atom_idx in component:
                part_id = self.info.disorder_groups[atom_idx]
                if part_id != 0:
                    sym_op = 0
                    if has_sym_info and atom_idx < len(self.info.sym_op_indices):
                        sym_op = self.info.sym_op_indices[atom_idx]

                    key = (part_id, sym_op)

                    if key not in atoms_by_key:
                        atoms_by_key[key] = set()
                    atoms_by_key[key].add(atom_idx)

            for key, atom_set in atoms_by_key.items():
                if atom_set:
                    self.conformers.append(atom_set)

    def _get_robust_centroid(self, atom_indices: List[int]) -> np.ndarray:
        """
        Calculate centroid handling PBC by unwrapping molecule around the first atom.
        """
        if not atom_indices:
            return np.array([0.0, 0.0, 0.0])

        coords = self.info.frac_coords[list(atom_indices)]
        if len(coords) == 1:
            return coords[0]

        # Unwrap coordinates relative to the first atom
        ref = coords[0]
        diffs = coords - ref
        # Minimum image for diffs
        diffs = diffs - np.round(diffs)
        unwrapped_coords = ref + diffs

        # Calculate mean
        mean_coord = np.mean(unwrapped_coords, axis=0)

        # Wrap back to unit cell
        return mean_coord - np.floor(mean_coord)

    def _add_conformer_conflicts(self):
        """
        Add conflicts using Dual-Track Logic with Bonded Immunity for Framework.
        """
        SITE_RADIUS = 3.0
        GHOST_CLASH_THRESHOLD = 2.0

        for i, conf_a in enumerate(self.conformers):
            for j, conf_b in enumerate(self.conformers):
                if i >= j:
                    continue

                atoms_a = list(conf_a)
                atoms_b = list(conf_b)

                part_a = self.info.disorder_groups[atoms_a[0]]
                part_b = self.info.disorder_groups[atoms_b[0]]

                is_bonded_framework_interaction = False

                if part_a == 0 or part_b == 0:
                    for aa in atoms_a:
                        for bb in atoms_b:
                            dist = self.dist_matrix[aa, bb]
                            if dist < 2.5:
                                symbol_a = self.info.symbols[aa]
                                symbol_b = self.info.symbols[bb]
                                if self._are_bonded(symbol_a, symbol_b, dist, part_a, part_b):
                                    is_bonded_framework_interaction = True
                                    break
                        if is_bonded_framework_interaction:
                            break
                
                if is_bonded_framework_interaction:
                    continue 
                centroid_a = self._get_robust_centroid(atoms_a)
                centroid_b = self._get_robust_centroid(atoms_b)

                diff_vec = centroid_a - centroid_b
                diff_vec = diff_vec - np.round(diff_vec)
                cart_dist_vec = np.dot(diff_vec, self.lattice)
                centroid_dist = np.linalg.norm(cart_dist_vec)

                if part_a != part_b:
                    if centroid_dist < SITE_RADIUS:
                        self._add_conflict_edge(atoms_a, atoms_b, "logical_alternative")
                        continue

                is_diff_sym = self._has_different_symmetry_provenance(atoms_a, atoms_b)

                if part_a == part_b and is_diff_sym:
                    if centroid_dist < SITE_RADIUS:
                        has_clash = False
                        for aa in atoms_a:
                            for bb in atoms_b:
                                if self.dist_matrix[aa, bb] < GHOST_CLASH_THRESHOLD:
                                    has_clash = True
                                    break
                            if has_clash:
                                break

                        if has_clash:
                            self._add_conflict_edge(atoms_a, atoms_b, "symmetry_clash")

    def _add_conflict_edge(self, atoms_a, atoms_b, type_str):
        for u in atoms_a:
            for v in atoms_b:
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v, conflict_type=type_str)

    def _has_different_symmetry_provenance(
        self, atoms_a: List[int], atoms_b: List[int]
    ) -> bool:
        if not hasattr(self.info, "sym_op_indices") or not self.info.sym_op_indices:
            return False
        for a in atoms_a:
            for b in atoms_b:
                if a < len(self.info.sym_op_indices) and b < len(
                    self.info.sym_op_indices
                ):
                    if self.info.sym_op_indices[a] != self.info.sym_op_indices[b]:
                        return True
        return False

    def _add_explicit_conflicts(self):
        n_atoms = len(self.info.labels)
        for i in range(n_atoms):
            if self.info.disorder_groups[i] == 0:
                continue
            for j in range(i + 1, n_atoms):
                if self.info.disorder_groups[j] == 0:
                    continue
                if self.info.disorder_groups[i] == self.info.disorder_groups[j]:
                    continue

                assembly_i = self.graph.nodes[i]["assembly"]
                assembly_j = self.graph.nodes[j]["assembly"]

                has_conflict = False
                if assembly_i and assembly_j and assembly_i == assembly_j:
                    has_conflict = True
                elif not assembly_i and not assembly_j:
                    if (
                        self.dist_matrix[i, j]
                        < DISORDER_CONFIG["ASSEMBLY_CONFLICT_THRESHOLD"]
                    ):
                        has_conflict = True

                if has_conflict and not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j, conflict_type="explicit")

    def _add_geometric_conflicts(self):
        n_atoms = len(self.info.labels)
        has_asym_info = hasattr(self.info, "asym_id") and self.info.asym_id

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = self.dist_matrix[i, j]
                g_i = self.info.disorder_groups[i]
                g_j = self.info.disorder_groups[j]

                # Skip same-parent pairs that are true periodic copies (NOT
                # competing disorder alternatives).  Full-occupancy same-parent
                # atoms and explicit-dg same-parent atoms are always periodic
                # copies.  But partial-occ dg=0 same-parent atoms are genuinely
                # overlapping copies that need geometric conflict detection.
                if has_asym_info:
                    ai_i = self.info.asym_id[i] if i < len(self.info.asym_id) else -1
                    ai_j = self.info.asym_id[j] if j < len(self.info.asym_id) else -1
                    if ai_i == ai_j and g_i == g_j:
                        # Same parent, same disorder group — skip UNLESS both
                        # are partial-occupancy with dg=0 (special-position
                        # disorder without explicit labels).
                        # Require strictly positive occupancy: zero-occ atoms
                        # are dummy/placeholder positions, not disorder copies.
                        occ_i = self.info.occupancies[i]
                        occ_j = self.info.occupancies[j]
                        if not (g_i == 0
                                and 0 < occ_i < 1.0
                                and 0 < occ_j < 1.0):
                            continue

                # Check if this is an implicit special-position disorder pair:
                # same parent, same dg=0, both partial occupancy.
                is_implicit_sp_disorder = False
                if has_asym_info and g_i == 0 and g_j == 0:
                    ai_i = self.info.asym_id[i] if i < len(self.info.asym_id) else -1
                    ai_j = self.info.asym_id[j] if j < len(self.info.asym_id) else -1
                    if (ai_i == ai_j
                            and 0 < self.info.occupancies[i] < 1.0
                            and 0 < self.info.occupancies[j] < 1.0):
                        is_implicit_sp_disorder = True

                symbol_i = self.info.symbols[i]
                symbol_j = self.info.symbols[j]

                # Threshold selection:
                # - Explicit disorder pairs (different dg): use DISORDER_CLASH_THRESHOLD
                # - Everything else (including implicit SP disorder): HARD_SPHERE_THRESHOLD
                # NOTE: Non-H implicit SP disorder is handled separately by
                # _add_implicit_sp_conflicts() using proximity clustering, which is
                # more robust than a fixed threshold.  H implicit SP disorder is
                # handled by valence/tetrahedral decomposition from bonded centers.
                if g_i != 0 and g_j != 0 and g_i != g_j:
                    threshold = DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]
                else:
                    threshold = DISORDER_CONFIG["HARD_SPHERE_THRESHOLD"]

                if dist < threshold:
                    # For implicit SP disorder, SKIP the _are_bonded check.
                    # These are overlapping disorder copies of the same atom,
                    # not genuinely bonded pairs.  The _are_bonded heuristic
                    # would incorrectly classify close S-S or Cd-Cd copies
                    # as "bonded" and prevent conflict edge creation.
                    if is_implicit_sp_disorder or not self._are_bonded(symbol_i, symbol_j, dist, g_i, g_j):
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(
                                i, j, conflict_type="geometric", distance=dist
                            )
                        else:
                            # Fixed: Check against high-priority conflict types
                            high_priority_conflicts = [
                                "logical_alternative",
                                "symmetry_clash",
                                "explicit",
                                "valence",
                                "valence_geometry",
                            ]
                            if (
                                self.graph[i][j]["conflict_type"]
                                not in high_priority_conflicts
                            ):
                                self.graph[i][j]["conflict_type"] = "geometric"
                                self.graph[i][j]["distance"] = dist

    def _add_implicit_sp_conflicts(self):
        """
        Add mutual-exclusion edges for implicit special-position disorder.

        Atoms with dg=0, 0 < occ < 1, and non-H element are copies of the same
        atom placed at symmetry-equivalent positions on a special position.
        Instead of using a fixed distance threshold (which fails when intra-site
        and inter-site distances overlap), we use proximity clustering:

        1. Group atoms by asym_id (all copies of same asymmetric-unit atom).
        2. Compute expected site multiplicity = round(1/occ).
        3. Cluster the N copies into N/multiplicity groups using hierarchical
           clustering on the pairwise distance matrix.
        4. Add all-pairs conflict edges within each cluster.

        This correctly identifies same-site copies regardless of absolute distances.
        H atoms are excluded — they are handled by valence/tetrahedral decomposition.
        """
        has_asym_info = hasattr(self.info, "asym_id") and self.info.asym_id
        if not has_asym_info:
            return

        from collections import defaultdict

        # Collect non-H implicit SP disorder atoms grouped by asym_id
        sp_groups = defaultdict(list)
        for i in range(len(self.info.labels)):
            if (self.info.disorder_groups[i] == 0
                    and 0 < self.info.occupancies[i] < 1.0
                    and self.info.symbols[i] not in ("H", "D")
                    and i < len(self.info.asym_id)):
                sp_groups[self.info.asym_id[i]].append(i)

        for asym_id, indices in sp_groups.items():
            n_copies = len(indices)
            if n_copies < 2:
                continue

            occ = self.info.occupancies[indices[0]]
            multiplicity = max(1, round(1.0 / occ))

            if n_copies % multiplicity != 0:
                # Fallback: can't cleanly partition; skip clustering,
                # rely on geometric threshold edges already added.
                continue

            n_sites = n_copies // multiplicity

            if n_sites < 1 or multiplicity < 2:
                continue

            # Build sub-distance-matrix for this asym_id group
            sub_dists = np.zeros((n_copies, n_copies))
            for ii in range(n_copies):
                for jj in range(ii + 1, n_copies):
                    d = self.dist_matrix[indices[ii], indices[jj]]
                    sub_dists[ii, jj] = d
                    sub_dists[jj, ii] = d

            # Hierarchical clustering: partition n_copies atoms into n_sites
            # clusters of `multiplicity` atoms each, based on proximity.
            # Use a simple greedy approach: repeatedly find the closest pair
            # and merge, stopping when we have n_sites clusters.
            # Each cluster must have exactly `multiplicity` atoms.

            # Convert to condensed distance matrix for scipy
            from scipy.cluster.hierarchy import linkage, fcluster
            condensed = []
            for ii in range(n_copies):
                for jj in range(ii + 1, n_copies):
                    condensed.append(sub_dists[ii, jj])
            condensed = np.array(condensed)

            if len(condensed) == 0:
                continue

            Z = linkage(condensed, method='complete')
            # Cut the dendrogram to get n_sites clusters
            labels = fcluster(Z, t=n_sites, criterion='maxclust')

            # Add mutual-exclusion edges within each cluster
            from collections import Counter
            cluster_counts = Counter(labels)

            for cluster_id in set(labels):
                cluster_members = [indices[k] for k in range(n_copies)
                                   if labels[k] == cluster_id]
                # Add all-pairs conflict edges
                for ii in range(len(cluster_members)):
                    for jj in range(ii + 1, len(cluster_members)):
                        u, v = cluster_members[ii], cluster_members[jj]
                        if not self.graph.has_edge(u, v):
                            self.graph.add_edge(
                                u, v,
                                conflict_type="implicit_sp",
                                distance=self.dist_matrix[u, v]
                            )

    def _are_bonded(self, s1, s2, dist, group1=0, group2=0):
        if group1 != 0 and group2 != 0 and group1 != group2:
            return False

        is_metal1 = s1 in TRANSITION_METALS
        is_metal2 = s2 in TRANSITION_METALS
        is_nonmetal1 = not is_metal1
        is_nonmetal2 = not is_metal2
        
        if (is_metal1 and is_nonmetal2) or (is_metal2 and is_nonmetal1):
            if dist > BONDING_THRESHOLDS["METAL_NONMETAL_COVALENT_MAX"]: # 2.05
                return False
            else:
                return True

        if bool({"H", "D"}.intersection({s1, s2})):
            if bool({"C", "N", "O", "S", "P"}.intersection({s1, s2})):
                return (
                    BONDING_THRESHOLDS["H_CNO_THRESHOLD_MIN"]
                    < dist
                    < BONDING_THRESHOLDS["H_CNO_THRESHOLD_MAX"]
                )
            elif bool({"H", "D"}.intersection({s1, s2})):
                return BONDING_THRESHOLDS["HH_BOND_POSSIBLE"]
            else:
                return (
                    BONDING_THRESHOLDS["H_OTHER_THRESHOLD_MIN"]
                    < dist
                    < BONDING_THRESHOLDS["H_OTHER_THRESHOLD_MAX"]
                )
        elif bool({"C", "N", "O"}.intersection({s1, s2})):
            return (
                BONDING_THRESHOLDS["CNO_THRESHOLD_MIN"]
                < dist
                < BONDING_THRESHOLDS["CNO_THRESHOLD_MAX"]
            )
        elif bool(
            {"C", "N", "O"}.intersection({s1}) and {"C", "N", "O"}.intersection({s2})
        ):
            return (
                BONDING_THRESHOLDS["CNO_PAIR_THRESHOLD_MIN"]
                < dist
                < BONDING_THRESHOLDS["CNO_PAIR_THRESHOLD_MAX"]
            )
        else:
            return (
                BONDING_THRESHOLDS["GENERAL_THRESHOLD_MIN"]
                < dist
                < BONDING_THRESHOLDS["GENERAL_THRESHOLD_MAX"]
            )

    def _resolve_valence_conflicts(self):
        n_atoms = len(self.info.labels)
        has_asym_info = hasattr(self.info, "asym_id") and self.info.asym_id
        connectivity_graph = nx.Graph()
        for i in range(n_atoms):
            connectivity_graph.add_node(i)

        # Simple connectivity for valence check
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if self.dist_matrix[i, j] < 2.0:
                    if self._are_bonded(
                        self.info.symbols[i],
                        self.info.symbols[j],
                        self.dist_matrix[i, j],
                        self.info.disorder_groups[i],
                        self.info.disorder_groups[j],
                    ):
                        connectivity_graph.add_edge(i, j)

        for center_idx in range(n_atoms):
            neighbors = list(connectivity_graph.neighbors(center_idx))
            if not neighbors:
                continue
            sym = self.info.symbols[center_idx]
            max_c = MAX_COORDINATION_NUMBERS.get(sym, DEFAULT_MAX_COORDINATION)

            # Change C2: When asym_id is available, filter out neighbors that are
            # symmetry copies of the center atom itself (same asym_id AND same dg).
            # This prevents metals on special positions from seeing their own
            # periodic copies as spurious "too many neighbors".
            if has_asym_info:
                center_asym = (
                    self.info.asym_id[center_idx]
                    if center_idx < len(self.info.asym_id)
                    else -1
                )
                neighbors = [
                    n for n in neighbors
                    if not (
                        n < len(self.info.asym_id)
                        and self.info.asym_id[n] == center_asym
                        and self.info.disorder_groups[n] == self.info.disorder_groups[center_idx]
                    )
                ]

            if len(neighbors) > max_c:
                # C2b: Skip _decompose_cliques entirely for dg=0 centers
                # with occupancy < 1.0.  These are special-position framework
                # atoms whose partial occupancy arises from sso > 1, NOT from
                # genuine disorder.  Their expanded neighbours are all legitimate
                # framework bonds; generating mutual-exclusion edges between them
                # would incorrectly forbid co-existing framework atoms (e.g.
                # S1/N1/C1/Cd1 in NatComm-1).
                center_dg = self.info.disorder_groups[center_idx]
                center_occ = self.info.occupancies[center_idx]
                if center_dg == 0 and center_occ < 1.0:
                    continue

                # Skip valence decomposition for centers with explicit disorder
                # groups (dg != 0).  These are already handled by conformer and
                # explicit conflict logic; adding valence conflicts would create
                # spurious exclusion edges (e.g. Cl3-O in ClO4- of PAP-HM4).
                if center_dg != 0:
                    continue

                low_occ = [n for n in neighbors if self.info.occupancies[n] < 1.0]

                # When center has dg=0, only consider neighbors that ALSO have
                # dg=0.  Neighbors with dg!=0 are already handled by conformer/
                # explicit conflict logic and should not be double-processed.
                if center_dg == 0:
                    low_occ = [n for n in low_occ
                               if self.info.disorder_groups[n] == 0]

                if len(low_occ) > 1:
                    self._decompose_cliques(low_occ, center_idx)

    def _is_same_parent_pair(self, i_idx: int, j_idx: int) -> bool:
        """
        Return True if i_idx and j_idx are symmetry copies of the same
        asymmetric-unit atom (same asym_id AND same disorder_group) AND
        both have full occupancy.

        Full-occupancy same-parent pairs must never receive conflict edges —
        they are the same physical atom at different unit-cell positions and
        must always coexist in the ordered structure.

        Partial-occupancy (occ < 1.0) same-parent pairs with dg=0 are
        genuinely competing copies on special positions — they MUST be
        allowed to have conflict edges.
        """
        has_asym_info = hasattr(self.info, "asym_id") and self.info.asym_id
        if not has_asym_info:
            return False
        if i_idx >= len(self.info.asym_id) or j_idx >= len(self.info.asym_id):
            return False
        same_parent = (
            self.info.asym_id[i_idx] == self.info.asym_id[j_idx]
            and self.info.disorder_groups[i_idx] == self.info.disorder_groups[j_idx]
        )
        if not same_parent:
            return False
        # Partial-occupancy atoms with dg=0 are genuinely competing copies,
        # NOT identical atoms at different unit-cell positions.
        # Require strictly positive occupancy: zero-occ atoms are dummy/
        # placeholder positions, not disorder copies.
        occ_i = self.info.occupancies[i_idx]
        occ_j = self.info.occupancies[j_idx]
        if (self.info.disorder_groups[i_idx] == 0
                and 0 < occ_i < 1.0
                and 0 < occ_j < 1.0):
            return False
        return True

    def _decompose_cliques(self, atom_indices: List[int], center_idx: int):
        center_frac = self.info.frac_coords[center_idx]
        relative_positions = []
        for idx in atom_indices:
            atom_frac = self.info.frac_coords[idx]
            delta = atom_frac - center_frac
            delta = delta - np.round(delta)
            relative_positions.append(delta)

        cart_positions = [frac_to_cart(rel, self.lattice) for rel in relative_positions]

        center_sym = self.info.symbols[center_idx]
        n_low = len(atom_indices)

        # Tetrahedral decomposition: NH4+, PO4, SO4, etc.
        # Generalized to handle any neighbor count >= 8 (not just exactly 8),
        # because special-position atoms can produce variable copy counts.
        # Safety cap at 30 to avoid combinatorial explosion (C(30,4) = 27405).
        if n_low >= 8 and n_low <= 30 and center_sym in ["N", "P", "S"]:
            self._find_tetrahedral_groups(atom_indices, cart_positions)

        # Trigonal decomposition: NH3, CH3 (methyl rotation disorder)
        # Generalized to handle any neighbor count >= 6.
        elif n_low >= 6 and n_low <= 30 and center_sym in ["N", "C"]:
            self._find_trigonal_groups(atom_indices, cart_positions)

        else:
            # Fallback: mutually exclusive
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
                        # Skip symmetry copies of the same asymmetric-unit atom.
                        # These are never genuine disorder alternatives.
                        if self._is_same_parent_pair(i_idx, j_idx):
                            continue
                        g_i = self.info.disorder_groups[i_idx]
                        g_j = self.info.disorder_groups[j_idx]
                        if g_i != 0 and g_j != 0 and g_i == g_j:
                            continue
                        # Check if there's already a geometric edge and upgrade it to valence
                        if self.graph.has_edge(i_idx, j_idx):
                            # If the existing conflict is geometric, upgrade it to valence
                            if self.graph[i_idx][j_idx]["conflict_type"] == "geometric":
                                self.graph[i_idx][j_idx]["conflict_type"] = "valence"
                        else:
                            self.graph.add_edge(i_idx, j_idx, conflict_type="valence")

    def _find_tetrahedral_groups(self, atom_indices, cart_positions):
        n = len(atom_indices)
        candidates = []
        for combo in combinations(range(n), 4):
            combo_pos = [cart_positions[i] for i in combo]
            angles = []
            for i in range(4):
                for j in range(i + 1, 4):
                    v1 = combo_pos[i]
                    v2 = combo_pos[j]
                    if np.allclose(v1, 0) or np.allclose(v2, 0):
                        continue
                    angles.append(np.degrees(angle_between_vectors(v1, v2)))
            if angles:
                score = sum(abs(a - 109.5) for a in angles)
                candidates.append((score, [atom_indices[i] for i in combo]))

        candidates.sort(key=lambda x: x[0])
        self._apply_disjoint_groups(candidates, atom_indices)

    def _find_trigonal_groups(self, atom_indices, cart_positions):
        """Find 2 sets of 3 atoms (NH3 geometry)"""
        n = len(atom_indices)
        candidates = []
        for combo in combinations(range(n), 3):
            combo_pos = [cart_positions[i] for i in combo]
            angles = []
            for i in range(3):
                for j in range(i + 1, 3):
                    v1 = combo_pos[i]
                    v2 = combo_pos[j]
                    if np.allclose(v1, 0) or np.allclose(v2, 0):
                        continue
                    angles.append(np.degrees(angle_between_vectors(v1, v2)))
            if angles:
                # Target angle 107-109 for NH3
                score = sum(abs(a - 109.5) for a in angles)
                candidates.append((score, [atom_indices[i] for i in combo]))

        candidates.sort(key=lambda x: x[0])
        self._apply_disjoint_groups(candidates, atom_indices)

    def _apply_disjoint_groups(self, candidates, all_atoms):
        """Helper to apply exclusions based on best disjoint candidate groups."""
        all_atoms_set = set(all_atoms)
        for i in range(len(candidates)):
            part_a = candidates[i][1]
            for j in range(i + 1, len(candidates)):
                part_b = candidates[j][1]
                if set(part_a).isdisjoint(set(part_b)):
                    # Mutually exclude Part A and Part B
                    for u in part_a:
                        for v in part_b:
                            # Skip symmetry copies of the same asymmetric-unit atom
                            if self._is_same_parent_pair(u, v):
                                continue
                            # Check if there's already a geometric edge and upgrade it to valence_geometry
                            if self.graph.has_edge(u, v):
                                if self.graph[u][v]["conflict_type"] == "geometric":
                                    self.graph[u][v][
                                        "conflict_type"
                                    ] = "valence_geometry"
                            else:
                                self.graph.add_edge(
                                    u, v, conflict_type="valence_geometry"
                                )

                    # Exclude Rogue atoms from both
                    rogues = all_atoms_set - set(part_a) - set(part_b)
                    for r in rogues:
                        for target in list(part_a) + list(part_b):
                            # Skip symmetry copies of the same asymmetric-unit atom
                            if self._is_same_parent_pair(r, target):
                                continue
                            # Check if there's already a geometric edge and upgrade it to valence_geometry
                            if self.graph.has_edge(r, target):
                                if (
                                    self.graph[r][target]["conflict_type"]
                                    == "geometric"
                                ):
                                    self.graph[r][target][
                                        "conflict_type"
                                    ] = "valence_geometry"
                            else:
                                self.graph.add_edge(
                                    r, target, conflict_type="valence_geometry"
                                )
                    return
