"""
Exclusion Graph Construction for Disorder Handling.

This module implements the "Referee" logic that determines which atoms cannot
coexist in the same physical structure based on raw disorder data.
"""

import logging
import numpy as np
import networkx as nx
import re
from collections import deque as _deque
from typing import List
from itertools import combinations
from .info import DisorderInfo
from .edge_priority import add_or_promote_edge
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

logger = logging.getLogger(__name__)


class DisorderGraphBuilder:
    """
    Builds an exclusion graph from DisorderInfo data.
    """

    def __init__(
        self,
        info: DisorderInfo,
        lattice: np.ndarray,
        coupled: bool = False,
    ):
        """Initialize the graph builder.

        Parameters
        ----------
        coupled:
            When True, preserve the legacy behaviour where symmetry copies
            sharing one disorder assembly are locked to the same PART choice.
            The default treats each symmetry copy as an independent disorder
            decision when symmetry provenance is available.
        """
        self.info = info
        self.lattice = lattice
        self._coupled = coupled
        _pbc = getattr(info, 'pbc', None)
        self._pbc = _pbc if _pbc is not None else (True, True, True)
        self.graph = nx.Graph()
        self.conformers = []
        self.sp_completion_pairs = []
        self._sp_completion_pair_keys = set()

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
        n = len(coords)
        # Warn for large structures: N×N distance matrix can be slow/memory-heavy
        # (e.g. 5000×5000 = 200 MB for float64).
        if n > 2000:
            logger.warning(
                "Large structure: %d atoms after symmetry expansion. "
                "Distance matrix will be %.0f MB. This may be slow.",
                n, n * n * 8 / 1e6
            )
        # Precompute distance matrix with PBC (respecting actual periodicity)
        coord_diffs = coords[:, None, :] - coords[None, :, :]
        # Apply minimum-image convention only along periodic axes.
        # For non-periodic axes (e.g. vacuum direction of a slab with
        # pbc=(True,True,False)), raw fractional differences are used so
        # that atoms separated by the vacuum gap are not folded on top of
        # each other.
        pbc_mask = np.array(self._pbc, dtype=bool)
        if pbc_mask.all():
            # Fast path — fully periodic (common case)
            coord_diffs = coord_diffs - np.round(coord_diffs)
        else:
            coord_diffs[..., pbc_mask] -= np.round(coord_diffs[..., pbc_mask])
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
        # Pre-processing: infer synthetic disorder groups for partial-occ
        # atoms that CIF authors left at group=0 (common in CCDC data).
        self._promote_partial_framework_atoms()
        self._split_shared_groups_across_assemblies()
        self._infer_groups_from_occupancy_complement()
        self._identify_conformers()
        self._identify_sp_completion_pairs()
        self._add_conformer_conflicts()
        self._add_explicit_conflicts()
        self._add_geometric_conflicts()
        self._add_implicit_sp_conflicts()
        self._resolve_valence_conflicts()
        self.graph.graph["sp_completion_pairs"] = self.sp_completion_pairs
        return self.graph

        self._add_explicit_conflicts()
        self._add_geometric_conflicts()
        self._add_implicit_sp_conflicts()
        self._resolve_valence_conflicts()
        self.graph.graph["sp_completion_pairs"] = self.sp_completion_pairs
        return self.graph

    # ------------------------------------------------------------------
    # Synthetic PART inference for partial-occupancy atoms left at group=0
    # ------------------------------------------------------------------

    def _promote_partial_framework_atoms(self):
        """
        Category A fix: promote group=0, occ<1 atoms that share an assembly
        with a group!=0 atom into the complementary group.

        CIF convention allows the "major component" to be marked group=0
        while only the minor alternative carries a proper disorder_group.
        This method detects such pairs (same assembly, occupancy-complement,
        spatial overlap) and assigns the major component a synthetic group so
        that _add_explicit_conflicts can wire them correctly.
        """
        n = len(self.info.labels)
        tolerance = DISORDER_CONFIG["OCCUPANCY_COMPLEMENT_TOLERANCE"]
        radius = DISORDER_CONFIG["SP_COMPLETION_SITE_RADIUS"]

        # Collect existing non-zero groups per assembly (asymmetric-unit level)
        asm_groups: dict[str, set[int]] = {}
        for i in range(n):
            asm = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
            g = self.info.disorder_groups[i]
            if asm and g != 0:
                asm_groups.setdefault(asm, set()).add(g)

        if not asm_groups:
            return  # No assemblies with disorder annotations at all

        # Determine the synthetic group number to assign to promoted atoms.
        # Strategy: for a 2-group assembly {g}, the complement is max+1.
        # For multi-group assemblies, we use pair matching below.
        max_group = max(
            (abs(g) for g in self.info.disorder_groups if g != 0),
            default=0,
        )
        next_synthetic = max_group + 1

        promoted_count = 0
        for i in range(n):
            if self.info.disorder_groups[i] != 0:
                continue
            occ_i = self.info.occupancies[i]
            if occ_i >= 1.0:
                continue
            asm_i = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
            if not asm_i:
                continue
            if asm_i not in asm_groups:
                continue

            # Find a complementary partner in same assembly with group!=0
            # that is at the same spatial site.
            best_partner = None
            best_dist = radius
            for j in range(n):
                if i == j:
                    continue
                if self.info.disorder_groups[j] == 0:
                    continue
                asm_j = self.info.assemblies[j] if j < len(self.info.assemblies) else ""
                if asm_j != asm_i:
                    continue
                # Check spatial overlap
                dist = self.dist_matrix[i, j]
                if dist >= radius:
                    continue
                # Check occupancy complementarity
                occ_sum = occ_i + self.info.occupancies[j]
                if abs(occ_sum - 1.0) > tolerance:
                    continue
                # Prefer closest
                if dist < best_dist:
                    best_dist = dist
                    best_partner = j

            if best_partner is not None:
                # Assign a synthetic group that is different from the partner's
                partner_group = self.info.disorder_groups[best_partner]
                existing_in_asm = asm_groups[asm_i]
                # If the assembly only has one group, assign next_synthetic
                if len(existing_in_asm) == 1:
                    synthetic_group = next_synthetic
                    # Only increment once per assembly
                    if synthetic_group not in asm_groups.get(asm_i, set()):
                        asm_groups[asm_i].add(synthetic_group)
                        next_synthetic += 1
                else:
                    # Multi-group assembly: find the group that ISN'T the partner
                    others = existing_in_asm - {partner_group}
                    synthetic_group = min(others) if others else next_synthetic

                self.info.disorder_groups[i] = synthetic_group
                # Update the graph node attribute too
                if self.graph.has_node(i):
                    self.graph.nodes[i]["disorder_group"] = synthetic_group
                promoted_count += 1

        if promoted_count:
            logger.info(
                "Promoted %d partial-occ framework atoms to synthetic "
                "disorder groups (Category A inference).",
                promoted_count,
            )

    def _split_shared_groups_across_assemblies(self):
        """
        Split a single disorder_group that spans multiple assemblies into
        separate groups — one per assembly — but ONLY when atoms from
        different assemblies are spatially close (< site radius), meaning
        they would be incorrectly merged into one conformer.

        Handles cases where CIF marks all minor components with the same
        group number across multiple assemblies, causing conformer
        identification to merge bonded-neighbor atoms from different sites.

        Does NOT split when atoms in different assemblies are far apart
        (legitimate independent disorder sites sharing a group number).
        """
        n = len(self.info.labels)
        radius = DISORDER_CONFIG["ASSEMBLY_CONFLICT_THRESHOLD"]  # 2.2 Å — bonded neighbor distance

        # Collect (group, assembly) → atom indices, only for positive group!=0
        group_asm: dict[int, dict[str, list[int]]] = {}
        for i in range(n):
            g = self.info.disorder_groups[i]
            if g <= 0:
                continue
            asm = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
            if not asm:
                continue
            group_asm.setdefault(g, {}).setdefault(asm, []).append(i)

        # Find groups that span multiple assemblies AND have spatially close
        # atoms across assemblies (indicating they'd be merged in conformers).
        max_group = max(
            (abs(g) for g in self.info.disorder_groups if g != 0),
            default=0,
        )
        next_group = max_group + 1
        split_count = 0

        for g, asm_dict in group_asm.items():
            if len(asm_dict) <= 1:
                continue

            # Check if any atom pair across assemblies AND within the same
            # symop is spatially close.  Cross-symop proximity is just crystal
            # packing — not an indication that atoms belong to the same site.
            assemblies_sorted = sorted(asm_dict.keys())
            has_close_pair = False
            for idx_a, asm_a in enumerate(assemblies_sorted):
                if has_close_pair:
                    break
                for asm_b in assemblies_sorted[idx_a + 1:]:
                    for i in asm_dict[asm_a]:
                        if has_close_pair:
                            break
                        sop_i = (
                            self.info.sym_op_indices[i]
                            if i < len(self.info.sym_op_indices)
                            else 0
                        )
                        for j in asm_dict[asm_b]:
                            sop_j = (
                                self.info.sym_op_indices[j]
                                if j < len(self.info.sym_op_indices)
                                else 0
                            )
                            if sop_i != sop_j:
                                continue
                            if self.dist_matrix[i, j] < radius:
                                has_close_pair = True
                                break
                    if has_close_pair:
                        break

            if not has_close_pair:
                continue

            # Spatially close atoms in different assemblies — split.
            # Keep the first assembly with the original group, re-assign others.
            for asm in assemblies_sorted[1:]:
                new_g = next_group
                next_group += 1
                for atom_idx in asm_dict[asm]:
                    self.info.disorder_groups[atom_idx] = new_g
                    if self.graph.has_node(atom_idx):
                        self.graph.nodes[atom_idx]["disorder_group"] = new_g
                    split_count += 1

        if split_count:
            logger.info(
                "Split %d atoms from shared disorder groups spanning "
                "multiple assemblies into unique per-assembly groups.",
                split_count,
            )

    def _infer_groups_from_occupancy_complement(self):
        """
        Category B fix: for structures where ALL disordered atoms are group=0,
        infer synthetic groups from occupancy-complement pairs at overlapping
        sites, then expand via bond connectivity to form complete fragments.

        Detects whole-molecule orientational disorder and site-occupancy disorder
        that CIF authors annotated only via occupancy (no PART/group tags).

        All symmetry copies of the same asymmetric-unit fragment get the SAME
        assembly label and group number, so that --coupled mode works correctly.
        """
        n = len(self.info.labels)
        tolerance = DISORDER_CONFIG["OCCUPANCY_COMPLEMENT_TOLERANCE"]
        radius = DISORDER_CONFIG["SP_COMPLETION_SITE_RADIUS"]

        # Only activate if there are group=0 partial-occ atoms with no partner.
        # We need at least SOME atoms with a non-empty assembly label to
        # anchor the pairing.  Structures where ALL partial-occ atoms have
        # empty assembly are implicit SP disorder (orientational copies on
        # special positions) — those are handled by _add_implicit_sp_conflicts.
        partial_g0 = [
            i for i in range(n)
            if self.info.disorder_groups[i] == 0
            and self.info.occupancies[i] < 1.0
        ]
        if not partial_g0:
            return

        # Check that we still have unresolved atoms (Phase 1 might have fixed them)
        partial_g0 = [
            i for i in partial_g0
            if self.info.disorder_groups[i] == 0
        ]
        if not partial_g0:
            return

        # If ALL remaining partial-occ atoms have empty assembly, this is pure
        # implicit SP disorder (orientational H on special positions) — skip.
        has_any_assembly = any(
            bool(self.info.assemblies[i] if i < len(self.info.assemblies) else "")
            for i in partial_g0
        )
        if not has_any_assembly:
            return

        # Strategy: work on the ASYMMETRIC UNIT only (sym_op=0), infer groups
        # there, then propagate to all symmetry copies.
        has_sym = hasattr(self.info, "sym_op_indices") and self.info.sym_op_indices
        if has_sym:
            asym_partial = [i for i in partial_g0 if self.info.sym_op_indices[i] == 0]
        else:
            asym_partial = partial_g0

        if not asym_partial:
            return

        # Group partial-occ atoms by their occupancy value (rounded)
        occ_clusters: dict[float, list[int]] = {}
        for i in asym_partial:
            occ_key = round(self.info.occupancies[i], 3)
            occ_clusters.setdefault(occ_key, []).append(i)

        # Find occupancy-complement pairs (occ_A + occ_B ≈ 1.0)
        occ_values = sorted(occ_clusters.keys())
        complement_pairs: list[tuple[float, float]] = []
        for idx_a, occ_a in enumerate(occ_values):
            for occ_b in occ_values[idx_a:]:
                if abs(occ_a + occ_b - 1.0) <= tolerance:
                    complement_pairs.append((occ_a, occ_b))

        if not complement_pairs:
            return

        # For each complement pair, find seed atoms that overlap spatially
        max_group = max(
            (abs(g) for g in self.info.disorder_groups if g != 0),
            default=0,
        )
        synthetic_base = max_group + 100  # Large offset for clarity
        assigned_asym = set()
        asm_counter = 0

        # Build asym_id → [atom indices across all sym_ops] mapping
        asym_to_symcopies: dict[int, list[int]] = {}
        if has_sym:
            for i in partial_g0:
                asym_id = self.info.asym_id[i] if i < len(self.info.asym_id) else i
                asym_to_symcopies.setdefault(asym_id, []).append(i)

        for occ_a, occ_b in complement_pairs:
            atoms_a = [i for i in occ_clusters.get(occ_a, []) if i not in assigned_asym]
            atoms_b = [i for i in occ_clusters.get(occ_b, []) if i not in assigned_asym]
            if occ_a == occ_b:
                atoms_b = atoms_a

            # Find spatial overlap seeds within asymmetric unit
            for i in atoms_a:
                if i in assigned_asym:
                    continue
                for j in atoms_b:
                    if j in assigned_asym or j == i:
                        continue
                    if self.dist_matrix[i, j] >= radius:
                        continue
                    # Found a seed pair in the asymmetric unit!
                    group_a = synthetic_base
                    group_b = synthetic_base + 1
                    synthetic_base += 2
                    asm_label = f"syn_{asm_counter}"
                    asm_counter += 1

                    # Expand fragments within the asymmetric unit
                    fragment_a = self._expand_fragment(i, occ_a, assigned_asym, asym_partial)
                    fragment_b = self._expand_fragment(j, occ_b, assigned_asym, asym_partial)

                    # Assign groups to the asymmetric unit fragments
                    for atom_idx in fragment_a:
                        assigned_asym.add(atom_idx)
                    for atom_idx in fragment_b:
                        assigned_asym.add(atom_idx)

                    # Now propagate to ALL symmetry copies:
                    # For each atom in the asym fragment, find all its sym copies
                    # and assign them the same group + assembly.
                    all_assigned = set()

                    if has_sym:
                        for atom_idx in fragment_a:
                            asym_id = self.info.asym_id[atom_idx] if atom_idx < len(self.info.asym_id) else atom_idx
                            copies = asym_to_symcopies.get(asym_id, [atom_idx])
                            for copy_idx in copies:
                                self.info.disorder_groups[copy_idx] = group_a
                                if copy_idx < len(self.info.assemblies):
                                    self.info.assemblies[copy_idx] = asm_label
                                if self.graph.has_node(copy_idx):
                                    self.graph.nodes[copy_idx]["disorder_group"] = group_a
                                    self.graph.nodes[copy_idx]["assembly"] = asm_label
                                all_assigned.add(copy_idx)

                        for atom_idx in fragment_b:
                            asym_id = self.info.asym_id[atom_idx] if atom_idx < len(self.info.asym_id) else atom_idx
                            copies = asym_to_symcopies.get(asym_id, [atom_idx])
                            for copy_idx in copies:
                                self.info.disorder_groups[copy_idx] = group_b
                                if copy_idx < len(self.info.assemblies):
                                    self.info.assemblies[copy_idx] = asm_label
                                if self.graph.has_node(copy_idx):
                                    self.graph.nodes[copy_idx]["disorder_group"] = group_b
                                    self.graph.nodes[copy_idx]["assembly"] = asm_label
                                all_assigned.add(copy_idx)
                    else:
                        # No symmetry info: just assign directly
                        for atom_idx in fragment_a:
                            self.info.disorder_groups[atom_idx] = group_a
                            if atom_idx < len(self.info.assemblies):
                                self.info.assemblies[atom_idx] = asm_label
                            if self.graph.has_node(atom_idx):
                                self.graph.nodes[atom_idx]["disorder_group"] = group_a
                                self.graph.nodes[atom_idx]["assembly"] = asm_label
                            all_assigned.add(atom_idx)

                        for atom_idx in fragment_b:
                            self.info.disorder_groups[atom_idx] = group_b
                            if atom_idx < len(self.info.assemblies):
                                self.info.assemblies[atom_idx] = asm_label
                            if self.graph.has_node(atom_idx):
                                self.graph.nodes[atom_idx]["disorder_group"] = group_b
                                self.graph.nodes[atom_idx]["assembly"] = asm_label
                            all_assigned.add(atom_idx)

        total_assigned = sum(
            1 for i in range(n)
            if self.info.disorder_groups[i] != 0
            and i < len(self.info.assemblies)
            and self.info.assemblies[i].startswith("syn_")
        )
        if total_assigned:
            logger.info(
                "Inferred synthetic disorder groups for %d atoms "
                "from occupancy-complement pairing (Category B inference).",
                total_assigned,
            )

    def _expand_fragment(
        self, seed: int, target_occ: float, already_assigned: set, candidates: list
    ) -> list[int]:
        """
        Expand from a seed atom along bonded neighbors that share the same
        occupancy, forming a complete molecular fragment.
        """
        tolerance = DISORDER_CONFIG["OCCUPANCY_COMPLEMENT_TOLERANCE"]
        fragment = [seed]
        visited = {seed}
        queue = _deque([seed])

        while queue:
            current = queue.popleft()
            for other in candidates:
                if other in visited or other in already_assigned:
                    continue
                if abs(self.info.occupancies[other] - target_occ) > tolerance:
                    continue
                # Check bonding distance
                dist = self.dist_matrix[current, other]
                sym_i = self.info.symbols[current]
                sym_j = self.info.symbols[other]
                if self._are_bonded(sym_i, sym_j, dist, 0, 0):
                    visited.add(other)
                    fragment.append(other)
                    queue.append(other)

        return fragment

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

    def _conformer_pair_key(self, atoms_a, atoms_b):
        return frozenset((frozenset(atoms_a), frozenset(atoms_b)))

    def _is_sp_completion_pair(self, atoms_a, atoms_b) -> bool:
        return self._conformer_pair_key(atoms_a, atoms_b) in self._sp_completion_pair_keys

    def _sp_completion_match_fraction(self, atoms_a, atoms_b) -> float:
        """
        Fraction of atoms in ``atoms_a`` that have a same-label near partner in
        ``atoms_b``.  Whole-molecule PART -n disorder on a special position has
        many nearly overlapping symmetry mates plus a few off-mirror atoms that
        complete the molecule; ordinary ghost clashes do not.
        """
        if not atoms_a or not atoms_b:
            return 0.0

        matches = 0
        for aa in atoms_a:
            best = np.inf
            root = self.root_labels[aa]
            for bb in atoms_b:
                if self.root_labels[bb] != root:
                    continue
                dist = self.dist_matrix[aa, bb]
                if dist < best:
                    best = dist
            if best < DISORDER_CONFIG["SP_COMPLETION_MATCH_DISTANCE"]:
                matches += 1

        return matches / len(atoms_a)

    def _identify_sp_completion_pairs(self):
        """
        Mark symmetry-related negative-PART conformers that represent a single
        molecule straddling a special position, rather than mutually-exclusive
        ghost copies.
        """
        self.sp_completion_pairs = []
        self._sp_completion_pair_keys = set()

        for i, conf_a in enumerate(self.conformers):
            for j, conf_b in enumerate(self.conformers):
                if i >= j:
                    continue

                atoms_a = list(conf_a)
                atoms_b = list(conf_b)
                if not atoms_a or not atoms_b:
                    continue
                if (
                    min(len(atoms_a), len(atoms_b))
                    < DISORDER_CONFIG["SP_COMPLETION_MIN_ATOMS"]
                ):
                    continue
                min_occ = min(
                    self.info.occupancies[atom]
                    for atom in atoms_a + atoms_b
                    if atom < len(self.info.occupancies)
                )
                if min_occ < DISORDER_CONFIG["SP_COMPLETION_MIN_OCCUPANCY"]:
                    continue

                part_a = self.info.disorder_groups[atoms_a[0]]
                part_b = self.info.disorder_groups[atoms_b[0]]
                if part_a != part_b or part_a >= 0:
                    continue
                if not self._has_different_symmetry_provenance(atoms_a, atoms_b):
                    continue

                centroid_a = self._get_robust_centroid(atoms_a)
                centroid_b = self._get_robust_centroid(atoms_b)
                diff_vec = centroid_a - centroid_b
                diff_vec = diff_vec - np.round(diff_vec)
                centroid_dist = np.linalg.norm(np.dot(diff_vec, self.lattice))
                if centroid_dist >= DISORDER_CONFIG["SP_COMPLETION_SITE_RADIUS"]:
                    continue

                match_fraction = max(
                    self._sp_completion_match_fraction(atoms_a, atoms_b),
                    self._sp_completion_match_fraction(atoms_b, atoms_a),
                )
                if not (
                    DISORDER_CONFIG["SP_COMPLETION_MIN_MATCH_FRAC"]
                    <= match_fraction
                    <= DISORDER_CONFIG["SP_COMPLETION_MAX_MATCH_FRAC"]
                ):
                    continue

                key = self._conformer_pair_key(atoms_a, atoms_b)
                if key in self._sp_completion_pair_keys:
                    continue
                self._sp_completion_pair_keys.add(key)
                self.sp_completion_pairs.append((tuple(sorted(atoms_a)), tuple(sorted(atoms_b))))

    def _add_conformer_conflicts(self):
        """
        Add conflicts using Dual-Track Logic with Bonded Immunity for Framework.
        """
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
                    if centroid_dist < DISORDER_CONFIG["SP_COMPLETION_SITE_RADIUS"]:
                        # For NEGATIVE PART groups (e.g. -1 vs -2 in same
                        # assembly), the assembly label is shared across ALL
                        # symmetry copies of the disordered fragment, so
                        # centroid-proximity alone over-fires.  Two negative
                        # PARTs are logical alternatives only when they
                        # describe the SAME crystallographic site, i.e. share
                        # at least one sym_op_index.  When the sym_op sets
                        # are disjoint they're distinct sites and must
                        # coexist (SHELXL convention where negative PARTs
                        # jointly fill symmetry copies of one ligand).
                        if part_a < 0 and part_b < 0:
                            if self.info.sym_op_indices:
                                sop_a = {
                                    self.info.sym_op_indices[aa]
                                    for aa in atoms_a
                                    if aa < len(self.info.sym_op_indices)
                                }
                                sop_b = {
                                    self.info.sym_op_indices[bb]
                                    for bb in atoms_b
                                    if bb < len(self.info.sym_op_indices)
                                }
                                if sop_a.isdisjoint(sop_b):
                                    continue
                        # For POSITIVE PART groups in different assemblies,
                        # proximity does NOT mean they're alternatives — they
                        # are independent disorder sites that happen to be
                        # bonded neighbors (e.g., S/Se in a chelating ligand).
                        # EXCEPTION: if centroids nearly overlap (< 1.5 Å),
                        # they're at the same physical site and ARE alternatives
                        # (e.g., water O in assembly A vs O' in assembly B).
                        if part_a > 0 and part_b > 0:
                            asm_a = set(
                                self.info.assemblies[aa]
                                for aa in atoms_a
                                if aa < len(self.info.assemblies)
                            )
                            asm_b = set(
                                self.info.assemblies[bb]
                                for bb in atoms_b
                                if bb < len(self.info.assemblies)
                            )
                            if asm_a and asm_b and asm_a.isdisjoint(asm_b):
                                # Only skip if NOT overlapping at same site
                                if centroid_dist > DISORDER_CONFIG["SP_COMPLETION_MATCH_DISTANCE"]:
                                    continue
                        self._add_conflict_edge(atoms_a, atoms_b, "logical_alternative")
                        continue

                is_diff_sym = self._has_different_symmetry_provenance(atoms_a, atoms_b)

                if part_a == part_b and is_diff_sym:
                    if centroid_dist < DISORDER_CONFIG["SP_COMPLETION_SITE_RADIUS"]:
                        if self._is_sp_completion_pair(atoms_a, atoms_b):
                            continue
                        has_clash = False
                        for aa in atoms_a:
                            for bb in atoms_b:
                                if (
                                    self.dist_matrix[aa, bb]
                                    < DISORDER_CONFIG["SYMMETRY_GHOST_CLASH_THRESHOLD"]
                                ):
                                    has_clash = True
                                    break
                            if has_clash:
                                break

                        if has_clash:
                            self._add_conflict_edge(atoms_a, atoms_b, "symmetry_clash")

    def _add_conflict_edge(self, atoms_a, atoms_b, type_str):
        for u in atoms_a:
            for v in atoms_b:
                add_or_promote_edge(self.graph, u, v, type_str)

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
                # Belt-and-suspenders: even if synthetic inference missed this
                # atom, still check if it's partial-occ and shares an assembly
                # with a group!=0 atom.  Only activate for atoms WITH an
                # assembly label — empty-assembly partial-occ atoms are
                # implicit SP disorder handled by _add_implicit_sp_conflicts.
                asm_i = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
                if self.info.occupancies[i] >= 1.0 or not asm_i:
                    continue
                # Partial-occ group=0 with assembly: fall through to check for
                # same-assembly conflicts with group!=0 atoms below.
            for j in range(i + 1, n_atoms):
                if self.info.disorder_groups[j] == 0:
                    asm_j = self.info.assemblies[j] if j < len(self.info.assemblies) else ""
                    if self.info.occupancies[j] >= 1.0 or not asm_j:
                        continue
                if self.info.disorder_groups[i] == self.info.disorder_groups[j]:
                    continue

                assembly_i = self.graph.nodes[i]["assembly"]
                assembly_j = self.graph.nodes[j]["assembly"]

                has_conflict = False
                if assembly_i and assembly_j and assembly_i == assembly_j:
                    g_i = self.info.disorder_groups[i]
                    g_j = self.info.disorder_groups[j]
                    # In decoupled mode, one assembly label describes the
                    # asymmetric-unit disorder model; expanded symmetry copies
                    # should make independent PART choices.  Only atoms from
                    # the same symmetry provenance compete.
                    # Negative PARTs keep the historical guard in both modes:
                    # their assembly label is shared across symmetry copies.
                    if (
                        self.info.sym_op_indices
                        and (not self._coupled or (g_i < 0 and g_j < 0))
                    ):
                        sop_i = (
                            self.info.sym_op_indices[i]
                            if i < len(self.info.sym_op_indices)
                            else None
                        )
                        sop_j = (
                            self.info.sym_op_indices[j]
                            if j < len(self.info.sym_op_indices)
                            else None
                        )
                        if (
                            sop_i is not None
                            and sop_j is not None
                            and sop_i != sop_j
                        ):
                            continue
                    has_conflict = True
                elif not assembly_i and not assembly_j:
                    if (
                        self.dist_matrix[i, j]
                        < DISORDER_CONFIG["ASSEMBLY_CONFLICT_THRESHOLD"]
                    ):
                        has_conflict = True

                if has_conflict:
                    add_or_promote_edge(self.graph, i, j, "explicit")

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
                # NOTE: Implicit SP disorder is handled separately by
                # _add_implicit_sp_conflicts() using proximity clustering, which is
                # more robust than a fixed threshold for both heavy atoms and H.
                # Only apply the wide DISORDER_CLASH_THRESHOLD (2.2 Å) when
                # BOTH atoms belong to *positive* explicit disorder groups at
                # the same crystallographic site.  In decoupled mode, symmetry
                # copies of one assembly are independent and should only be
                # screened by the conservative hard-sphere threshold.
                # Different *negative* PART groups
                # (e.g. PART -1 vs PART -2) are distinct chemical species that
                # coexist in the structure (e.g. perchlorate O vs NH4+ H) and
                # must NOT be treated as disorder alternatives — use the
                # conservative HARD_SPHERE_THRESHOLD (0.85 Å) instead to avoid
                # false conflict edges that would exclude one species entirely.
                same_sym_site = True
                if (
                    not self._coupled
                    and self.info.sym_op_indices
                    and i < len(self.info.sym_op_indices)
                    and j < len(self.info.sym_op_indices)
                ):
                    same_sym_site = (
                        self.info.sym_op_indices[i] == self.info.sym_op_indices[j]
                    )
                if g_i > 0 and g_j > 0 and g_i != g_j and same_sym_site:
                    # Only use the wide clash threshold if atoms are in the
                    # SAME assembly (true alternatives at the same site).
                    # Cross-assembly atoms (e.g., neighboring S/Se in a
                    # chelating ligand) are NOT alternatives — they coexist.
                    asm_i = self.info.assemblies[i] if i < len(self.info.assemblies) else ""
                    asm_j = self.info.assemblies[j] if j < len(self.info.assemblies) else ""
                    if asm_i and asm_j and asm_i == asm_j:
                        threshold = DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]
                    elif not asm_i and not asm_j:
                        threshold = DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]
                    else:
                        threshold = DISORDER_CONFIG["HARD_SPHERE_THRESHOLD"]
                else:
                    threshold = DISORDER_CONFIG["HARD_SPHERE_THRESHOLD"]

                if dist < threshold:
                    # For implicit SP disorder, SKIP the _are_bonded check.
                    # These are overlapping disorder copies of the same atom,
                    # not genuinely bonded pairs.  The _are_bonded heuristic
                    # would incorrectly classify close S-S or Cd-Cd copies
                    # as "bonded" and prevent conflict edge creation.
                    if is_implicit_sp_disorder or not self._are_bonded(symbol_i, symbol_j, dist, g_i, g_j):
                        add_or_promote_edge(
                            self.graph, i, j, "geometric", distance=dist
                        )

    def _add_implicit_sp_conflicts(self):
        """
        Add mutual-exclusion edges for implicit special-position disorder.

        Atoms with dg=0 and 0 < occ < 1 are copies of the same atom placed at
        symmetry-equivalent positions on a special position.  Instead of using a
        fixed distance threshold (which fails when intra-site and inter-site
        distances overlap), we use proximity clustering:

        1. Group atoms by asym_id (all copies of same asymmetric-unit atom).
        2. Compute expected site multiplicity = round(1/occ).
        3. Cluster the N copies into N/multiplicity groups using hierarchical
           clustering on the pairwise distance matrix.
        4. Add all-pairs conflict edges within each cluster.

        This correctly identifies same-site copies regardless of absolute
        distances.

        H atoms are included ONLY when they lack a full-occupancy bonded center
        that would handle them via valence/tetrahedral decomposition.  Specifically,
        H atoms whose nearest bonded non-H neighbor is partial-occupancy (e.g.,
        water H bonded to O with occ<1) are included because C2b skips such
        centers.  H atoms bonded to full-occ centres are excluded — their
        disorder is orientational and handled by tetrahedral decomposition
        from that full-occ center.
        """
        has_asym_info = hasattr(self.info, "asym_id") and self.info.asym_id
        if not has_asym_info:
            return

        has_site_sym = (
            hasattr(self.info, "site_symmetry_order")
            and self.info.site_symmetry_order
        )

        from collections import defaultdict

        # Pre-compute: for each H atom with dg=0, occ<1, find if it has a
        # full-occupancy bonded non-H center.  If so, valence decomposition
        # from that center will handle it; exclude from SP clustering.
        #
        # Exception: "multi-site" SP disorder.  When a group of symmetry-
        # equivalent H copies (same asym_id) has round(n_copies * occ) > 1,
        # the H is NOT pure orientational disorder about a single center
        # (e.g., round(4 * 0.25) = 1 is orientational).  It is genuine
        # multi-site SP disorder where only a subset of the copies coexist
        # (e.g., round(4 * 0.5) = 2).  Valence decomposition cannot resolve
        # this because it only sees one center at a time; include such H
        # atoms in SP clustering so pairwise conflicts are added correctly.
        h_needs_sp_clustering = set()
        n_atoms = len(self.info.labels)

        h_asym_copies: dict = defaultdict(list)
        for i in range(n_atoms):
            if (self.info.symbols[i] in ("H", "D")
                    and self.info.disorder_groups[i] == 0
                    and 0 < self.info.occupancies[i] < 1.0
                    and i < len(self.info.asym_id)):
                h_asym_copies[self.info.asym_id[i]].append(i)

        multi_site_h: set = set()
        for asym_id, h_indices in h_asym_copies.items():
            n_copies = len(h_indices)
            if n_copies < 2:
                continue
            occ = self.info.occupancies[h_indices[0]]
            n_sites = max(1, round(n_copies * occ))
            if n_sites > 1:
                multi_site_h.update(h_indices)

        for i in range(n_atoms):
            if (self.info.symbols[i] in ("H", "D")
                    and self.info.disorder_groups[i] == 0
                    and 0 < self.info.occupancies[i] < 1.0):
                if i in multi_site_h:
                    h_needs_sp_clustering.add(i)
                    continue
                # Check if any bonded non-H neighbor has full occupancy
                has_full_occ_center = False
                for j in range(n_atoms):
                    if i == j:
                        continue
                    if self.info.symbols[j] in ("H", "D"):
                        continue
                    dist = self.dist_matrix[i, j]
                    if dist < DISORDER_CONFIG["SP_H_BOND_DETECTION_CUTOFF"]:
                        if self.info.occupancies[j] >= 1.0:
                            has_full_occ_center = True
                            break
                if not has_full_occ_center:
                    h_needs_sp_clustering.add(i)

        # Collect implicit SP disorder atoms grouped by asym_id:
        # - Non-H atoms: always included
        # - H atoms: only if they lack a full-occ bonded center
        #
        # When site_symmetry_order data is available, use it as a positive
        # confirmation that the atom is on a special position (order > 1).
        # Atoms with order == 1 (general position) are excluded even if they
        # happen to be partial-occupancy — they are not SP disorder.
        # When site_symmetry_order is absent (older CIFs), fall back to the
        # occupancy-only heuristic (backward compatible).
        sp_groups = defaultdict(list)
        for i in range(n_atoms):
            if (self.info.disorder_groups[i] == 0
                    and 0 < self.info.occupancies[i] < 1.0
                    and i < len(self.info.asym_id)):
                # Gate on site_symmetry_order when available.
                # Only skip atoms that are on a strict general position (order=1)
                # AND whose occupancy implies all copies can coexist (occ≈1/n_symops
                # but n_copies*occ ≈ n_copies).  We cannot compute n_copies here,
                # so we defer that check to the downstream n_sites >= n_copies guard
                # (line ~486).  We only use site_sym_order as a positive signal:
                # order > 1 guarantees SP disorder; order == 1 is ambiguous and
                # must be allowed through so the downstream check can decide.
                # (Removing the order<=1 skip fixes atoms on general positions
                # that still have fractional occupancy because many symops
                # generate copies of which only a subset can coexist.)
                if self.info.symbols[i] in ("H", "D"):
                    if i not in h_needs_sp_clustering:
                        continue
                sp_groups[self.info.asym_id[i]].append(i)

        # Pre-compute connected-component fragment ids over full-occupancy
        # heavy atoms.  This is only needed when there are multi-site H
        # atoms whose proximity-only clustering can be misled by lattice
        # coincidences (copies anchored to different molecular cations may
        # lie within bonding distance, making them look like SP equivalents
        # even though they belong to chemically distinct fragments).
        # Bucketing H copies by their nearest full-occ anchor's fragment
        # restores the correct semantics: only H atoms anchored to the
        # same chemical fragment compete for the same proton site.
        heavy_anchor_fragment: dict = {}
        if multi_site_h:
            heavy_graph = nx.Graph()
            for a in range(n_atoms):
                if self.info.symbols[a] in ("H", "D"):
                    continue
                if self.info.occupancies[a] < 1.0:
                    continue
                heavy_graph.add_node(a)
            for a in heavy_graph.nodes():
                for b in heavy_graph.nodes():
                    if a >= b:
                        continue
                    d = self.dist_matrix[a, b]
                    if d >= 2.0:
                        continue
                    if self._are_bonded(
                        self.info.symbols[a], self.info.symbols[b], d,
                        self.info.disorder_groups[a],
                        self.info.disorder_groups[b],
                    ):
                        heavy_graph.add_edge(a, b)
            for fid, comp in enumerate(nx.connected_components(heavy_graph)):
                for atom in comp:
                    heavy_anchor_fragment[atom] = fid

        def _bucket_indices_by_anchor(indices_list):
            """Partition multi-site H copies by full-occ anchor fragment.

            Returns a list of buckets (each a list of atom indices) ready for
            independent proximity clustering.  Non-H atoms and H atoms that
            have no full-occ heavy anchor fall through as a single bucket so
            existing behaviour is preserved.
            """
            if not heavy_anchor_fragment:
                return [list(indices_list)]
            # Only re-bucket multi-site H copies; everything else stays whole.
            if not all(idx in multi_site_h for idx in indices_list):
                return [list(indices_list)]

            cutoff = DISORDER_CONFIG["SP_H_BOND_DETECTION_CUTOFF"]
            buckets: dict = defaultdict(list)
            unbucketed: list = []
            for h_idx in indices_list:
                best_anchor = None
                best_dist = np.inf
                for j in range(n_atoms):
                    if j == h_idx:
                        continue
                    if self.info.symbols[j] in ("H", "D"):
                        continue
                    if j not in heavy_anchor_fragment:
                        continue
                    d = self.dist_matrix[h_idx, j]
                    if d < cutoff and d < best_dist:
                        best_dist = d
                        best_anchor = j
                if best_anchor is None:
                    unbucketed.append(h_idx)
                else:
                    buckets[heavy_anchor_fragment[best_anchor]].append(h_idx)
            if not buckets:
                return [list(indices_list)]
            result = list(buckets.values())
            if unbucketed:
                # Unbucketed H copies (no full-occ anchor) keep the legacy
                # all-pairs proximity behaviour amongst themselves.
                result.append(unbucketed)
            return result

        for asym_id, indices in sp_groups.items():
            for bucket in _bucket_indices_by_anchor(indices):
                self._cluster_and_add_sp_edges(bucket)

    def _cluster_and_add_sp_edges(self, indices):
        """Run hierarchical proximity clustering over a single SP bucket and
        add mutual-exclusion edges within each resulting cluster."""
        n_copies = len(indices)
        if n_copies < 2:
            return

        occ = self.info.occupancies[indices[0]]

        # Compute expected number of physical sites.
        # Primary method: n_sites = round(n_copies * occ)
        # This is more robust than round(1/occ) for non-standard
        # occupancies like 0.06 or 0.04.
        n_sites = max(1, round(n_copies * occ))
        multiplicity = n_copies // n_sites if n_sites > 0 else n_copies

        # If division isn't clean, try round(1/occ) as fallback
        if n_copies % n_sites != 0:
            multiplicity_alt = max(1, round(1.0 / occ))
            if n_copies % multiplicity_alt == 0:
                multiplicity = multiplicity_alt
                n_sites = n_copies // multiplicity

        if n_sites < 1 or n_sites >= n_copies:
            return

        sub_dists = np.zeros((n_copies, n_copies))
        for ii in range(n_copies):
            for jj in range(ii + 1, n_copies):
                d = self.dist_matrix[indices[ii], indices[jj]]
                sub_dists[ii, jj] = d
                sub_dists[jj, ii] = d

        from scipy.cluster.hierarchy import linkage, fcluster
        condensed = []
        for ii in range(n_copies):
            for jj in range(ii + 1, n_copies):
                condensed.append(sub_dists[ii, jj])
        condensed = np.array(condensed)
        if len(condensed) == 0:
            return

        Z = linkage(condensed, method='complete')
        labels = fcluster(Z, t=n_sites, criterion='maxclust')

        for cluster_id in set(labels):
            cluster_members = [indices[k] for k in range(n_copies)
                               if labels[k] == cluster_id]
            for ii in range(len(cluster_members)):
                for jj in range(ii + 1, len(cluster_members)):
                    u, v = cluster_members[ii], cluster_members[jj]
                    add_or_promote_edge(
                        self.graph,
                        u,
                        v,
                        "implicit_sp",
                        distance=self.dist_matrix[u, v],
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
        has_site_sym = (
            hasattr(self.info, "site_symmetry_order")
            and self.info.site_symmetry_order
        )
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
                # would incorrectly forbid co-existing framework atoms.
                center_dg = self.info.disorder_groups[center_idx]
                center_occ = self.info.occupancies[center_idx]
                if center_dg == 0 and center_occ < 1.0:
                    continue

                # Skip valence decomposition for centers with explicit disorder
                # groups (dg != 0).  These are already handled by conformer and
                # explicit conflict logic; adding valence conflicts would create
                # spurious exclusion edges.
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
                    _max_n = DISORDER_CONFIG["CLIQUE_DECOMP_MAX_NEIGHBORS"]
                    # SP-center pre-filtering: when n_low exceeds the clique-decomp
                    # cap AND the center sits on a special position (sso > 1), the
                    # large neighbor count is caused by symmetry expansion producing
                    # many copies of each unique H site.  Pre-filter to one
                    # representative per asym_id group (nearest to center), run
                    # tetrahedral/trigonal decomposition on the reduced set, then
                    # expand the picked representatives back to their full site
                    # clusters and mark the non-picked clusters as conflicting.
                    center_sso = 1
                    if has_site_sym and center_idx < len(self.info.site_symmetry_order):
                        center_sso = self.info.site_symmetry_order[center_idx]

                    if len(low_occ) > _max_n and has_asym_info and center_sso > 1:
                        low_occ, asym_id_to_cluster = self._prefilter_sp_neighbors(
                            low_occ, center_idx
                        )
                    else:
                        asym_id_to_cluster = None

                    self._decompose_cliques(low_occ, center_idx,
                                           asym_id_to_cluster=asym_id_to_cluster)

    def _prefilter_sp_neighbors(self, low_occ: List[int], center_idx: int):
        """
        Pre-filter low-occupancy neighbors for a special-position center.

        When a full-occupancy center on a special position (sso > 1) has far
        more partial-occ neighbors than CLIQUE_DECOMP_MAX_NEIGHBORS, the excess
        is caused by symmetry expansion: each unique asymmetric-unit H site
        generates many symmetry copies, all of which appear as neighbors.

        This method:
        1. Groups low_occ neighbors by asym_id.
        2. For each asym_id group, picks the single representative nearest to
           the center atom.
        3. Returns the reduced list of representatives plus a mapping
           {asym_id: [all_copies_in_group]} for later expansion.

        Parameters
        ----------
        low_occ : List[int]
            Indices of low-occupancy neighbor atoms.
        center_idx : int
            Index of the center atom.

        Returns
        -------
        representatives : List[int]
            One representative index per asym_id group.
        asym_id_to_cluster : dict
            Maps asym_id → list of all atom indices in that group.
        """
        from collections import defaultdict
        asym_id_to_cluster = defaultdict(list)
        for n in low_occ:
            ai = (
                self.info.asym_id[n]
                if n < len(self.info.asym_id)
                else n  # fallback: treat each atom as its own group
            )
            asym_id_to_cluster[ai].append(n)

        representatives = []
        for ai, cluster in asym_id_to_cluster.items():
            # Pick the atom nearest to the center
            nearest = min(cluster, key=lambda n: self.dist_matrix[center_idx, n])
            representatives.append(nearest)

        logger.debug(
            "SP pre-filter: center=%d sso>1, reduced %d neighbors → %d "
            "representatives (%d asym_id groups)",
            center_idx, len(low_occ), len(representatives), len(asym_id_to_cluster)
        )
        return representatives, dict(asym_id_to_cluster)

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

    def _decompose_cliques(self, atom_indices: List[int], center_idx: int,
                           asym_id_to_cluster=None):
        """
        Decompose a set of low-occupancy neighbors into mutually-exclusive groups.

        Parameters
        ----------
        atom_indices : List[int]
            Indices of low-occupancy neighbor atoms (may be pre-filtered
            representatives when asym_id_to_cluster is provided).
        center_idx : int
            Index of the center atom.
        asym_id_to_cluster : dict or None
            When provided (SP pre-filter case), maps asym_id → list of all
            atom indices in that site cluster.  After geometry matching on
            the representative level, the disjoint-group constraints are
            expanded back to full clusters.
        """
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

        _max_n = DISORDER_CONFIG["CLIQUE_DECOMP_MAX_NEIGHBORS"]

        # SP pre-filter case: asym_id_to_cluster is provided.
        # atom_indices are one representative per asym_id group (e.g., 4 reps
        # for NH4+: H1C, H1D, H1E, H1F).
        #
        # Strategy:
        #   1. Within each cluster, add mutual-exclusion edges (only 1 copy
        #      per H site can survive).
        #   2. Run geometry matching (tetrahedral/trigonal) on the
        #      representative positions to find valid coordination groups.
        #   3. Expand the geometry results back to full clusters so that
        #      atoms in incompatible clusters are properly excluded.
        if asym_id_to_cluster is not None:
            # Step 1: within-cluster mutual exclusion
            self._apply_sp_cluster_conflicts(asym_id_to_cluster)

            # Step 2+3: geometry matching on representatives, expanded to clusters
            if n_low == 4 and center_sym in ["N", "P", "S"]:
                # Exactly 4 groups — check if they form a single valid tetrahedron
                self._sp_tetrahedral_single(
                    atom_indices, cart_positions, asym_id_to_cluster,
                    center_idx
                )
            elif n_low >= 8 and center_sym in ["N", "P", "S"]:
                # Multiple tetrahedral alternatives — enumerate on representatives
                self._sp_geometry_match(
                    atom_indices, cart_positions, asym_id_to_cluster,
                    group_size=4, target_angle=109.5
                )
            elif n_low == 3 and center_sym in ["N", "C"]:
                # Exactly 3 groups — check if they form a single valid trigonal
                self._sp_trigonal_single(
                    atom_indices, cart_positions, asym_id_to_cluster,
                    center_idx
                )
            elif n_low >= 6 and center_sym in ["N", "C"]:
                # Multiple trigonal alternatives
                self._sp_geometry_match(
                    atom_indices, cart_positions, asym_id_to_cluster,
                    group_size=3, target_angle=109.5
                )
            else:
                # Fallback: mutual exclusion on representatives, expanded
                self._sp_fallback_exclusion(
                    atom_indices, asym_id_to_cluster
                )
            return

        # --- Normal (non-SP) path ---

        # Tetrahedral decomposition: NH4+, PO4, SO4, etc.
        if n_low >= 8 and n_low <= _max_n and center_sym in ["N", "P", "S"]:
            self._find_tetrahedral_groups(atom_indices, cart_positions)

        # Trigonal decomposition: NH3, CH3 (methyl rotation disorder)
        elif n_low >= 6 and n_low <= _max_n and center_sym in ["N", "C"]:
            self._find_trigonal_groups(atom_indices, cart_positions)

        else:
            # Fallback: mutually exclusive
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
                        if self._is_same_parent_pair(i_idx, j_idx):
                            continue
                        g_i = self.info.disorder_groups[i_idx]
                        g_j = self.info.disorder_groups[j_idx]
                        if g_i != 0 and g_j != 0 and g_i == g_j:
                            continue
                        add_or_promote_edge(self.graph, i_idx, j_idx, "valence")

    # ------------------------------------------------------------------
    # SP cluster helpers
    # ------------------------------------------------------------------

    def _apply_sp_cluster_conflicts(self, asym_id_to_cluster: dict):
        """
        Add within-cluster mutual-exclusion edges for SP pre-filtered sites.

        For each asym_id group (e.g., all copies of H1C bonded to a given N1),
        add conflict edges between every pair of copies so the MWIS solver
        picks at most one copy per H site.

        NOTE: This method intentionally does NOT touch cross-cluster edges.
        Cross-cluster relationships (compatible vs incompatible groups) are
        determined by the geometry-matching methods that run after this.
        """
        for ai, cluster in asym_id_to_cluster.items():
            for ii in range(len(cluster)):
                for jj in range(ii + 1, len(cluster)):
                    u, v = cluster[ii], cluster[jj]
                    if self._is_same_parent_pair(u, v):
                        continue
                    add_or_promote_edge(self.graph, u, v, "valence_geometry")

    def _sp_tetrahedral_single(self, reps, cart_positions,
                               asym_id_to_cluster, center_idx):
        """
        Handle exactly 4 representative groups around a tetrahedral center.

        Finds all valid tetrahedra (1-per-cluster combos with good geometry)
        and applies disjoint-group constraints: atoms in the same tetrahedron
        are compatible (no cross-cluster conflict); atoms in different
        tetrahedra conflict.  Any atom not in any valid tetrahedron conflicts
        with all atoms from other clusters.
        """
        valid_groups = self._sp_collect_valid_groups(
            asym_id_to_cluster, center_idx,
            group_size=4, target_angle=109.5, threshold=60.0,
            max_collect=200
        )
        self._sp_apply_group_constraints(
            valid_groups, asym_id_to_cluster, group_size=4
        )
        logger.debug(
            "SP tetrahedral single: found %d valid tetrahedra for center %d",
            len(valid_groups), center_idx
        )

    def _sp_trigonal_single(self, reps, cart_positions,
                            asym_id_to_cluster, center_idx):
        """
        Handle exactly 3 representative groups around a trigonal center.
        """
        valid_groups = self._sp_collect_valid_groups(
            asym_id_to_cluster, center_idx,
            group_size=3, target_angle=109.5, threshold=30.0,
            max_collect=200
        )
        self._sp_apply_group_constraints(
            valid_groups, asym_id_to_cluster, group_size=3
        )
        logger.debug(
            "SP trigonal single: found %d valid trigonal groups for center %d",
            len(valid_groups), center_idx
        )

    def _sp_collect_valid_groups(self, asym_id_to_cluster, center_idx,
                                 group_size, target_angle, threshold,
                                 max_collect=200):
        """
        Enumerate 1-per-cluster combinations and collect those with geometry
        scores below threshold.  Uses vectorized numpy angle matrices to avoid
        per-combo trig calls.

        Returns list of tuples, each tuple = (score, atom_indices) where
        atom_indices is a tuple of `group_size` atom indices (one per cluster).
        """
        center_frac = self.info.frac_coords[center_idx]
        cluster_keys = list(asym_id_to_cluster.keys())[:group_size]

        # Build atom lists and unit-vector arrays per cluster
        cluster_atoms = []   # cluster_atoms[k] = list of atom indices
        cluster_units = []   # cluster_units[k] = (n_k, 3) unit vectors
        for ai in cluster_keys:
            atoms = asym_id_to_cluster[ai]
            # Compute relative Cartesian positions via MIC
            delta_fracs = self.info.frac_coords[atoms] - center_frac
            delta_fracs = delta_fracs - np.round(delta_fracs)  # MIC
            rel_carts = delta_fracs @ self.lattice  # (n_k, 3)
            norms = np.linalg.norm(rel_carts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)  # avoid /0
            units = rel_carts / norms
            cluster_atoms.append(list(atoms))
            cluster_units.append(units)

        # Pre-compute pairwise angle matrices for all cluster pairs
        # angle_mats[(ki, kj)] is shape (n_ki, n_kj) in degrees
        pair_indices = [(ki, kj)
                        for ki in range(group_size)
                        for kj in range(ki + 1, group_size)]
        angle_mats = {}
        for ki, kj in pair_indices:
            # dot product matrix: (n_ki, n_kj)
            dots = cluster_units[ki] @ cluster_units[kj].T
            dots = np.clip(dots, -1.0, 1.0)
            angle_mats[(ki, kj)] = np.abs(
                np.degrees(np.arccos(dots)) - target_angle
            )  # pre-compute |angle - target|

        cluster_sizes = [len(c) for c in cluster_atoms]
        total_combos = 1
        for s in cluster_sizes:
            total_combos *= s

        MAX_ENUM = 200000
        MAX_SAMPLE = 50000

        valid_groups = []

        if group_size == 4 and total_combos <= MAX_ENUM:
            # Fully vectorized for tetrahedral case (most common)
            # Build all combos as index arrays and score in batches
            s0, s1, s2, s3 = cluster_sizes
            # Create index grids
            i0 = np.arange(s0)
            i1 = np.arange(s1)
            i2 = np.arange(s2)
            i3 = np.arange(s3)

            # Sum up 6 pairwise deviation matrices
            # For combo (a,b,c,d): score = AM01[a,b] + AM02[a,c] + AM03[a,d]
            #                              + AM12[b,c] + AM13[b,d] + AM23[c,d]
            AM01 = angle_mats[(0, 1)]  # (s0, s1)
            AM02 = angle_mats[(0, 2)]  # (s0, s2)
            AM03 = angle_mats[(0, 3)]  # (s0, s3)
            AM12 = angle_mats[(1, 2)]  # (s1, s2)
            AM13 = angle_mats[(1, 3)]  # (s1, s3)
            AM23 = angle_mats[(2, 3)]  # (s2, s3)

            # Build 4D score array via broadcasting:
            # score[a,b,c,d] = AM01[a,b] + AM02[a,c] + AM03[a,d]
            #                + AM12[b,c] + AM13[b,d] + AM23[c,d]
            scores_4d = (
                AM01[:, :, None, None]   # (s0,s1,1,1)
                + AM02[:, None, :, None] # (s0,1,s2,1)
                + AM03[:, None, None, :] # (s0,1,1,s3)
                + AM12[None, :, :, None] # (1,s1,s2,1)
                + AM13[None, :, None, :] # (1,s1,1,s3)
                + AM23[None, None, :, :] # (1,1,s2,s3)
            )

            # Find all combos below threshold
            valid_mask = scores_4d < threshold
            valid_indices = np.argwhere(valid_mask)  # (N_valid, 4)

            if len(valid_indices) > 0:
                valid_scores = scores_4d[valid_mask]
                # Sort by score, keep top max_collect
                order = np.argsort(valid_scores)[:max_collect]
                for idx in order:
                    combo = valid_indices[idx]
                    score = valid_scores[idx]
                    atoms = tuple(
                        cluster_atoms[k][combo[k]] for k in range(4)
                    )
                    valid_groups.append((float(score), atoms))

        elif group_size == 3 and total_combos <= MAX_ENUM:
            # Vectorized for trigonal case
            s0, s1, s2 = cluster_sizes
            AM01 = angle_mats[(0, 1)]
            AM02 = angle_mats[(0, 2)]
            AM12 = angle_mats[(1, 2)]

            scores_3d = (
                AM01[:, :, None]
                + AM02[:, None, :]
                + AM12[None, :, :]
            )

            valid_mask = scores_3d < threshold
            valid_indices = np.argwhere(valid_mask)

            if len(valid_indices) > 0:
                valid_scores = scores_3d[valid_mask]
                order = np.argsort(valid_scores)[:max_collect]
                for idx in order:
                    combo = valid_indices[idx]
                    score = valid_scores[idx]
                    atoms = tuple(
                        cluster_atoms[k][combo[k]] for k in range(3)
                    )
                    valid_groups.append((float(score), atoms))

        else:
            # Fallback: sampling for very large combo spaces
            import random
            from itertools import product as iproduct

            def _score_combo(combo):
                total = 0.0
                for ki, kj in pair_indices:
                    total += angle_mats[(ki, kj)][combo[ki], combo[kj]]
                return total

            if total_combos <= MAX_ENUM:
                index_lists = [list(range(s)) for s in cluster_sizes]
                for combo in iproduct(*index_lists):
                    s = _score_combo(combo)
                    if s < threshold:
                        atoms = tuple(cluster_atoms[k][combo[k]]
                                      for k in range(group_size))
                        valid_groups.append((float(s), atoms))
                        if len(valid_groups) >= max_collect:
                            break
            else:
                seen = set()
                for _ in range(MAX_SAMPLE):
                    combo = tuple(random.randint(0, cluster_sizes[k] - 1)
                                  for k in range(group_size))
                    if combo in seen:
                        continue
                    seen.add(combo)
                    s = _score_combo(combo)
                    if s < threshold:
                        atoms = tuple(cluster_atoms[k][combo[k]]
                                      for k in range(group_size))
                        valid_groups.append((float(s), atoms))
                        if len(valid_groups) >= max_collect:
                            break

        valid_groups.sort(key=lambda x: x[0])
        return valid_groups

    def _sp_apply_group_constraints(self, valid_groups, asym_id_to_cluster,
                                     group_size):
        """
        Apply graph constraints based on valid geometry groups.

        Uses numpy boolean arrays for fast compatibility checks instead of
        Python set intersections.
        """
        cluster_keys = list(asym_id_to_cluster.keys())[:group_size]

        if not valid_groups:
            # No valid geometry found — fallback: all cross-cluster pairs conflict
            for ki in range(len(cluster_keys)):
                for kj in range(ki + 1, len(cluster_keys)):
                    for u in asym_id_to_cluster[cluster_keys[ki]]:
                        for v in asym_id_to_cluster[cluster_keys[kj]]:
                            if self._is_same_parent_pair(u, v):
                                continue
                            add_or_promote_edge(self.graph, u, v, "valence_geometry")
            return

        n_groups = len(valid_groups)

        # For each cluster, build atom→local_index map and
        # a boolean matrix: membership[local_idx, group_idx]
        cluster_local_maps = []  # list of {atom_idx: local_idx}
        cluster_memberships = []  # list of (n_atoms_in_cluster, n_groups) bool arrays
        for ki, ai in enumerate(cluster_keys):
            atoms = asym_id_to_cluster[ai]
            local_map = {a: li for li, a in enumerate(atoms)}
            membership = np.zeros((len(atoms), n_groups), dtype=bool)
            cluster_local_maps.append(local_map)
            cluster_memberships.append(membership)

        # Fill membership arrays from valid_groups
        for gi, (score, group_atoms) in enumerate(valid_groups):
            for ki in range(group_size):
                a = group_atoms[ki]
                local_map = cluster_local_maps[ki]
                if a in local_map:
                    cluster_memberships[ki][local_map[a], gi] = True

        # For each pair of clusters, compute compatibility matrix
        # compatible[u_local, v_local] = any(membership_ki[u] & membership_kj[v])
        for ki in range(group_size):
            for kj in range(ki + 1, group_size):
                atoms_ki = asym_id_to_cluster[cluster_keys[ki]]
                atoms_kj = asym_id_to_cluster[cluster_keys[kj]]
                mem_ki = cluster_memberships[ki]  # (n_ki, n_groups)
                mem_kj = cluster_memberships[kj]  # (n_kj, n_groups)

                # compatible_mat[u, v] = True if they share any group
                # = (mem_ki @ mem_kj.T) > 0
                compatible_mat = (mem_ki.astype(np.int8) @ mem_kj.astype(np.int8).T) > 0

                for ui, u in enumerate(atoms_ki):
                    for vi, v in enumerate(atoms_kj):
                        if self._is_same_parent_pair(u, v):
                            continue
                        if compatible_mat[ui, vi]:
                            # Compatible — remove geometric conflict if present
                            if self.graph.has_edge(u, v):
                                if self.graph[u][v].get("conflict_type") == "geometric":
                                    self.graph.remove_edge(u, v)
                        else:
                            add_or_promote_edge(self.graph, u, v, "valence_geometry")

    def _sp_geometry_match(self, reps, cart_positions, asym_id_to_cluster,
                           group_size, target_angle):
        """
        Geometry matching on representatives with >= 2*group_size candidates.

        Enumerate combinations of `group_size` representatives, score each by
        angular deviation from `target_angle`, then apply disjoint-group
        constraints expanded to full clusters.
        """
        n = len(reps)
        candidates = []
        for combo in combinations(range(n), group_size):
            combo_pos = [cart_positions[i] for i in combo]
            angles = []
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    v1 = combo_pos[i]
                    v2 = combo_pos[j]
                    if np.allclose(v1, 0) or np.allclose(v2, 0):
                        continue
                    angles.append(np.degrees(angle_between_vectors(v1, v2)))
            if angles:
                score = sum(abs(a - target_angle) for a in angles)
                candidates.append((score, [reps[i] for i in combo]))

        candidates.sort(key=lambda x: x[0])

        # Apply disjoint groups expanded to clusters
        self._expand_sp_disjoint_groups(candidates, reps, asym_id_to_cluster)

    def _sp_fallback_exclusion(self, reps, asym_id_to_cluster):
        """
        Fallback: make all representative groups mutually exclusive.

        Adds conflict edges between ALL atoms in different clusters.
        """
        cluster_keys = list(asym_id_to_cluster.keys())
        for ki in range(len(cluster_keys)):
            for kj in range(ki + 1, len(cluster_keys)):
                for u in asym_id_to_cluster[cluster_keys[ki]]:
                    for v in asym_id_to_cluster[cluster_keys[kj]]:
                        if self._is_same_parent_pair(u, v):
                            continue
                        add_or_promote_edge(self.graph, u, v, "valence_geometry")

    def _remove_cross_cluster_geometric_edges(self, asym_id_to_cluster):
        """
        Remove geometric / valence_geometry conflict edges between atoms in
        different asym_id clusters.  Called only when geometry matching has
        confirmed that the clusters are compatible (e.g., valid tetrahedron).
        """
        cluster_keys = list(asym_id_to_cluster.keys())
        removed = 0
        for ki in range(len(cluster_keys)):
            for kj in range(ki + 1, len(cluster_keys)):
                for u in asym_id_to_cluster[cluster_keys[ki]]:
                    for v in asym_id_to_cluster[cluster_keys[kj]]:
                        if self.graph.has_edge(u, v):
                            ct = self.graph[u][v]["conflict_type"]
                            if ct in ("geometric", "valence_geometry"):
                                self.graph.remove_edge(u, v)
                                removed += 1
        if removed > 0:
            logger.debug(
                "Removed %d cross-cluster geometric edges between %d groups",
                removed, len(cluster_keys)
            )

    def _expand_sp_disjoint_groups(self, candidates, reps, asym_id_to_cluster):
        """
        Apply disjoint-group constraints from geometry matching on
        representatives, expanded to full clusters.

        Works like ``_apply_disjoint_groups()`` but each representative atom
        is expanded to its full asym_id cluster for edge generation.
        """
        # Build rep → asym_id mapping
        rep_to_ai = {}
        for ai, cluster in asym_id_to_cluster.items():
            for idx in cluster:
                if idx in set(reps):
                    rep_to_ai[idx] = ai
                    break

        all_reps_set = set(reps)

        for i in range(len(candidates)):
            part_a_reps = candidates[i][1]
            for j in range(i + 1, len(candidates)):
                part_b_reps = candidates[j][1]
                if set(part_a_reps).isdisjoint(set(part_b_reps)):
                    rogue_reps = list(
                        all_reps_set - set(part_a_reps) - set(part_b_reps)
                    )

                    # Expand to clusters
                    part_a_atoms = []
                    for r in part_a_reps:
                        ai = rep_to_ai.get(r)
                        if ai is not None:
                            part_a_atoms.extend(asym_id_to_cluster[ai])
                        else:
                            part_a_atoms.append(r)

                    part_b_atoms = []
                    for r in part_b_reps:
                        ai = rep_to_ai.get(r)
                        if ai is not None:
                            part_b_atoms.extend(asym_id_to_cluster[ai])
                        else:
                            part_b_atoms.append(r)

                    rogue_atoms = []
                    for r in rogue_reps:
                        ai = rep_to_ai.get(r)
                        if ai is not None:
                            rogue_atoms.extend(asym_id_to_cluster[ai])
                        else:
                            rogue_atoms.append(r)

                    # Part A vs Part B: mutual exclusion
                    for u in part_a_atoms:
                        for v in part_b_atoms:
                            if self._is_same_parent_pair(u, v):
                                continue
                            add_or_promote_edge(self.graph, u, v, "valence_geometry")

                    # Rogues vs Part A + Part B
                    for r in rogue_atoms:
                        for t in part_a_atoms + part_b_atoms:
                            if self._is_same_parent_pair(r, t):
                                continue
                            add_or_promote_edge(self.graph, r, t, "valence_geometry")
                    return

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
                    rogue_atoms = list(all_atoms_set - set(part_a) - set(part_b))

                    # Mutually exclude Part A and Part B
                    for u in part_a:
                        for v in part_b:
                            if self._is_same_parent_pair(u, v):
                                continue
                            add_or_promote_edge(self.graph, u, v, "valence_geometry")

                    # Exclude rogue atoms from both part_a and part_b
                    for r in rogue_atoms:
                        for target in list(part_a) + list(part_b):
                            if self._is_same_parent_pair(r, target):
                                continue
                            add_or_promote_edge(
                                self.graph, r, target, "valence_geometry"
                            )
                    return
