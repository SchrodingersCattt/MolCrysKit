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
        self._resolve_valence_conflicts()
        return self.graph

    def _identify_conformers(self):
        """
        Identify discrete conformers (clusters) using connectivity, PART rules, AND Symmetry.
        Prevents 'Frankenstein' merging of symmetry images.
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
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = self.dist_matrix[i, j]
                g_i = self.info.disorder_groups[i]
                g_j = self.info.disorder_groups[j]

                threshold = (
                    DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]
                    if (g_i != 0 and g_j != 0 and g_i != g_j)
                    else DISORDER_CONFIG["HARD_SPHERE_THRESHOLD"]
                )

                symbol_i = self.info.symbols[i]
                symbol_j = self.info.symbols[j]

                if dist < threshold:
                    if not self._are_bonded(symbol_i, symbol_j, dist, g_i, g_j):
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
            if len(neighbors) > max_c:
                low_occ = [n for n in neighbors if self.info.occupancies[n] < 1.0]
                if len(low_occ) > 1:
                    self._decompose_cliques(low_occ, center_idx)

    def _decompose_cliques(self, atom_indices: List[int], center_idx: int):
        center_frac = self.info.frac_coords[center_idx]
        relative_positions = []
        for idx in atom_indices:
            atom_frac = self.info.frac_coords[idx]
            delta = atom_frac - center_frac
            delta = delta - np.round(delta)
            relative_positions.append(delta)

        cart_positions = [frac_to_cart(rel, self.lattice) for rel in relative_positions]

        # TODO: USE HybridizedSite instead of hardcoded geometries
        # Tetrahedral: PO4, SO4, etc. (8 neighbors -> 2 sets of 4)
        if len(atom_indices) == 8 and self.info.symbols[center_idx] in ["N", "P", "S"]:
            self._find_tetrahedral_groups(atom_indices, cart_positions)
            
        # Trigonal: NH3, CH3 (Methyl) (6 neighbors -> 2 sets of 3)
        # Added "C" to handle methyl group rotation disorder
        elif len(atom_indices) == 6 and self.info.symbols[center_idx] in ["N", "C"]:
            self._find_trigonal_groups(atom_indices, cart_positions)
            
        else:
            # Fallback: mutually exclusive
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
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
