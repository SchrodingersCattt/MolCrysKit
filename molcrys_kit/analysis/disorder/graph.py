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
from ...constants.config import DISORDER_CONFIG, BONDING_THRESHOLDS, MAX_COORDINATION_NUMBERS, DEFAULT_MAX_COORDINATION, TRANSITION_METALS
from ...utils.geometry import (
    angle_between_vectors,
    frac_to_cart,
)


class DisorderGraphBuilder:
    """
    Builds an exclusion graph from DisorderInfo data.

    The exclusion graph represents mutual exclusivity between atoms -
    if there's an edge between two atoms, they cannot coexist in the same structure.
    """

    def __init__(self, info: DisorderInfo, lattice: np.ndarray):
        """
        Initialize the graph builder.

        Parameters:
        -----------
        info : DisorderInfo
            Raw disorder data from Phase 1
        lattice : np.ndarray
            3x3 matrix representing the lattice vectors
        """
        self.info = info
        self.lattice = lattice
        self.graph = nx.Graph()

        # Add nodes to the graph (each node is an atom index)
        for i in range(len(info.labels)):
            self.graph.add_node(
                i,
                label=info.labels[i],
                symbol=info.symbols[i],
                frac_coord=info.frac_coords[i],
                occupancy=info.occupancies[i],
                disorder_group=info.disorder_groups[i],
                assembly=info.assemblies[i] if i < len(info.assemblies) else ""
            )

        # Precompute distance matrix and root labels
        self._precompute_metrics()

    def _precompute_metrics(self):
        """
        Precompute distance matrix and root labels to avoid repeated calculations.
        """
        # Precompute distance matrix using vectorization
        coords = self.info.frac_coords  # shape (n, 3)

        # Calculate coordinate differences with broadcasting: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
        coord_diffs = coords[:, None, :] - coords[None, :, :]  # Shape: (n, n, 3)

        # Apply minimum image convention
        coord_diffs = coord_diffs - np.round(coord_diffs)

        # Convert to Cartesian coordinates using lattice
        cart_diffs = np.einsum("nij,jk->nik", coord_diffs, self.lattice)

        # Calculate distances
        distances = np.linalg.norm(cart_diffs, axis=2)

        self.dist_matrix = distances

        # Precompute root labels (e.g., "C1" from "C1A")
        self.root_labels = []
        for label in self.info.labels:
            match = re.match(r"([A-Za-z]+[0-9]*)", label)
            root_label = match.group(1) if match else label
            self.root_labels.append(root_label)

    def build(self) -> nx.Graph:
        """
        Build the exclusion graph by applying the multi-layer conflict detection logic.
        """
        # Layer 1: Explicit conflicts based on disorder groups and assemblies
        self._add_explicit_conflicts()

        # Layer 1.5: Global Symmetry Clashes (NEW)
        # Must run BEFORE geometric checks to catch "Same Group" overlapping ghosts
        self._add_symmetry_conflicts(default_threshold=1.35)

        # Layer 2: Geometric conflicts based on atomic distances with context-aware thresholds
        self._add_geometric_conflicts()

        # Layer 3: Valence/inferred conflicts based on coordination geometry
        self._resolve_valence_conflicts()

        return self.graph

    def _add_explicit_conflicts(self):
        """
        Add edges between atoms that have different non-zero disorder groups.

        Rule: If two atoms have non-zero disorder groups and group_A != group_B,
        they are mutually exclusive IF AND ONLY IF they belong to the same assembly
        or they are close enough (< ASSEMBLY_CONFLICT_THRESHOLD Å) if assemblies are not specified.
        """
        n_atoms = len(self.info.labels)

        for i in range(n_atoms):
            group_i = self.info.disorder_groups[i]
            assembly_i = self.graph.nodes[i]["assembly"]

            # Only consider non-zero groups
            if group_i == 0:
                continue

            for j in range(i + 1, n_atoms):
                group_j = self.info.disorder_groups[j]
                assembly_j = self.graph.nodes[j]["assembly"]

                # If both have non-zero groups and they are different, check additional conditions
                if group_j != 0 and group_i != group_j:
                    # Determine if there should be a conflict based on assembly or distance
                    has_conflict = False

                    if (
                        assembly_i != ""
                        and assembly_j != ""
                        and assembly_i == assembly_j
                    ):
                        # Same non-empty assembly: they have a conflict
                        has_conflict = True
                    elif assembly_i == "" and assembly_j == "":
                        # Both have empty assemblies: use distance heuristic (ASSEMBLY_CONFLICT_THRESHOLD)
                        distance = self.dist_matrix[i, j]
                        if distance < DISORDER_CONFIG["ASSEMBLY_CONFLICT_THRESHOLD"]:
                            has_conflict = True
                    # If one has assembly and the other doesn't, no conflict by this rule

                    # Only add the conflict if conditions are met
                    if has_conflict:
                        # Only add if not already added with a different conflict type
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(i, j, conflict_type="explicit")
                        else:
                            # If already exists, update to explicit if it's not already explicit
                            if self.graph[i][j]["conflict_type"] != "explicit":
                                self.graph[i][j]["conflict_type"] = "explicit"

    def _add_symmetry_conflicts(self, default_threshold: float = 1.35):
        """
        Layer 1.5: Detect conflicts between atoms from different symmetry images in negative PART groups.

        This prevents "Frankenstein Molecules" by ensuring that atoms from competing symmetry images
        (e.g., Image A vs Image B) cannot coexist in the same structure, regardless of their element types.
        """
        n_atoms = len(self.info.labels)

        # Define strict threshold for low occupancy atoms
        STRICT_THRESHOLD = 2.0

        # Get the symmetry site radius from config
        radius_threshold = DISORDER_CONFIG["SYMMETRY_SITE_RADIUS"]

        for i in range(n_atoms):
            group_i = self.info.disorder_groups[i]
            if group_i == 0:
                continue

            occ_i = self.info.occupancies[i]

            for j in range(i + 1, n_atoms):
                group_j = self.info.disorder_groups[j]

                # Check if both atoms are in the same disorder group and the group is negative (SHELX-style PART -n)
                if group_i == group_j and group_i < 0:
                    # Check spatial overlap
                    dist = self.dist_matrix[i, j]

                    if dist < radius_threshold:
                        # Check if atoms have different symmetry operation origins
                        if (
                            hasattr(self.info, "sym_op_indices")
                            and len(self.info.sym_op_indices) > i
                            and len(self.info.sym_op_indices) > j
                            and self.info.sym_op_indices[i]
                            != self.info.sym_op_indices[j]
                        ):
                            # These atoms belong to competing symmetry images (e.g., Image A vs Image B)
                            # CRITICAL: Do NOT check if labels are the same. Even if one is Nitrogen
                            # and the other is Hydrogen, they CANNOT coexist if from different images.
                            # Action: Add Exclusion Edge (conflict_type="symmetry_provenance")
                            if not self.graph.has_edge(i, j):
                                self.graph.add_edge(
                                    i,
                                    j,
                                    conflict_type="symmetry_provenance",
                                    distance=dist,
                                )
                            else:
                                self.graph[i][j][
                                    "conflict_type"
                                ] = "symmetry_provenance"
                        # If same SymOp Index, atoms are part of the same molecular image, do nothing

                # Legacy fallback: ensure existing occupancy/distance logic runs for non-negative groups
                elif group_i == group_j:
                    # Use precomputed root labels for general disorder cases
                    root_i = self.root_labels[i]
                    root_j = self.root_labels[j]

                    # Check Identity (Clones)
                    if root_i == root_j:
                        # Determine Dynamic Threshold
                        # If either atom has low occupancy (< 0.5), we assume they cannot
                        # bond to their own clone. Be strict.
                        occ_j = self.info.occupancies[j]
                        if occ_i < 0.5 or occ_j < 0.5:
                            threshold = STRICT_THRESHOLD
                        else:
                            threshold = default_threshold

                        # Use precomputed distance
                        dist = self.dist_matrix[i, j]

                        if dist < threshold:
                            if not self.graph.has_edge(i, j):
                                self.graph.add_edge(
                                    i, j, conflict_type="symmetry_clash", distance=dist
                                )
                            else:
                                self.graph[i][j]["conflict_type"] = "symmetry_clash"

    def _add_geometric_conflicts(self):
        """
        Add edges between atoms that are too close to coexist (hard sphere collision).
        Uses context-aware thresholds based on disorder groups.
        """
        n_atoms = len(self.info.labels)
        
        # Vectorized computation of geometric conflicts with context-aware thresholds
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get distance
                dist = self.dist_matrix[i, j]
                
                # Get disorder groups
                group_i = self.info.disorder_groups[i]
                group_j = self.info.disorder_groups[j]
                
                # Get symbols
                symbol_i = self.info.symbols[i]
                symbol_j = self.info.symbols[j]
                
                # Exclusion Rule: If atoms are in different non-zero groups, they represent competing realities
                # and should not be considered for bonding but rather for exclusion
                if group_i != 0 and group_j != 0 and group_i != group_j:
                    threshold = DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]  # 2.2
                else:
                    threshold = DISORDER_CONFIG["HARD_SPHERE_THRESHOLD"]  # 0.85
                
                # Check if atoms are bonded using context-aware bonding logic
                if not self._are_bonded(symbol_i, symbol_j, dist, group_i, group_j):
                    # If distance is below threshold, add geometric conflict
                    if dist < threshold:
                        # Only add if not already added with explicit conflict
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(
                                i, j, conflict_type="geometric", distance=dist
                            )
                        else:
                            # If there's already a connection, check if we should update the type
                            # Explicit conflicts take priority over geometric
                            if self.graph[i][j]["conflict_type"] != "explicit":
                                self.graph[i][j]["conflict_type"] = "geometric"
                                self.graph[i][j]["distance"] = dist

    def _are_bonded(self, s1, s2, dist, group1=0, group2=0):
        """
        Determine if two atoms are bonded based on element types, distance and disorder groups.
        
        Parameters:
        -----------
        s1, s2 : str
            Element symbols
        dist : float
            Distance between atoms
        group1, group2 : int
            Disorder groups of the atoms (default to 0 for backward compatibility)
            
        Returns:
        --------
        bool
            True if atoms are considered bonded
        """
        # Exclusion Rule: If atoms are in different non-zero groups, they represent alternative realities
        # and never bond
        if group1 != 0 and group2 != 0 and group1 != group2:
            return False
        
        # Ionic Rule: If one is Metal and other is Non-Metal and dist > METAL_NONMETAL_COVALENT_MAX
        is_metal1 = s1 in TRANSITION_METALS
        is_metal2 = s2 in TRANSITION_METALS
        is_nonmetal1 = not is_metal1
        is_nonmetal2 = not is_metal2
        
        if (is_metal1 and is_nonmetal2) or (is_metal2 and is_nonmetal1):
            if dist > BONDING_THRESHOLDS["METAL_NONMETAL_COVALENT_MAX"]:
                return False
        
        # Original bonding logic
        if bool({"H", "D"}.intersection({s1, s2})):
            # H/D with any other element
            if bool({"C", "N", "O", "S", "P"}.intersection({s1, s2})):
                return BONDING_THRESHOLDS["H_CNO_THRESHOLD_MIN"] < dist < BONDING_THRESHOLDS["H_CNO_THRESHOLD_MAX"]
            elif bool({"H", "D"}.intersection({s1, s2})):
                return BONDING_THRESHOLDS["HH_BOND_POSSIBLE"]  # H-H unlikely to bond
            else:
                return BONDING_THRESHOLDS["H_OTHER_THRESHOLD_MIN"] < dist < BONDING_THRESHOLDS["H_OTHER_THRESHOLD_MAX"]
        elif bool({"C", "N", "O"}.intersection({s1, s2})):
            # C, N, O with each other
            return BONDING_THRESHOLDS["CNO_THRESHOLD_MIN"] < dist < BONDING_THRESHOLDS["CNO_THRESHOLD_MAX"]
        elif bool(
            {"C", "N", "O"}.intersection({s1})
            and {"C", "N", "O"}.intersection({s2})
        ):
            # C-N, C-O, N-O
            return BONDING_THRESHOLDS["CNO_PAIR_THRESHOLD_MIN"] < dist < BONDING_THRESHOLDS["CNO_PAIR_THRESHOLD_MAX"]
        else:
            # General threshold for other element pairs
            return BONDING_THRESHOLDS["GENERAL_THRESHOLD_MIN"] < dist < BONDING_THRESHOLDS["GENERAL_THRESHOLD_MAX"]

    def _resolve_valence_conflicts(self):
        """
        Detect and resolve valence conflicts (implicit disorder like in 'DAP-4').

        This handles cases where atoms are not explicitly grouped but would
        overcrowd a coordination environment (like 8 H atoms around a N atom).
        """
        n_atoms = len(self.info.labels)

        # Build connectivity graph for neighbors
        connectivity_graph = nx.Graph()
        for i in range(n_atoms):
            connectivity_graph.add_node(i)

        # Vectorized computation of bonded atoms (using a generous threshold)
        # Create a mask for distances that indicate bonding
        dist_mask = self.dist_matrix < DISORDER_CONFIG["VALENCE_PRESCREEN_THRESHOLD"]  # Use new config value

        # Only consider upper triangle to avoid duplicate edges
        triu_mask = np.triu(dist_mask, k=1)

        # Get indices where bonds exist
        bond_indices = np.where(triu_mask)

        for i, j in zip(bond_indices[0], bond_indices[1]):
            # Calculate distance (already available in dist_matrix)
            distance = self.dist_matrix[i, j]

            # Determine if bonded based on element types
            symbol_i = self.info.symbols[i].strip()
            symbol_j = self.info.symbols[j].strip()

            # Determine if bonded based on element types and disorder groups
            group_i = self.info.disorder_groups[i]
            group_j = self.info.disorder_groups[j]
            
            is_bonded = self._are_bonded(symbol_i, symbol_j, distance, group_i, group_j)

            if is_bonded:
                connectivity_graph.add_edge(i, j, distance=distance)

        # Identify overcrowded centers
        for center_idx in range(n_atoms):
            # Ammonium Protection: If center is 'N' and neighbors <= 4, skip decomposition
            center_symbol = self.info.symbols[center_idx]
            if center_symbol == 'N':
                neighbors = list(connectivity_graph.neighbors(center_idx))
                if len(neighbors) <= 4:
                    continue  # Skip decomposition for N with <= 4 neighbors

            neighbors = list(connectivity_graph.neighbors(center_idx))

            # Get max coordination number for this element
            max_coordination = self._get_max_coordination(center_symbol)

            # Check if overcrowded and has low occupancy neighbors
            if len(neighbors) > max_coordination:
                low_occ_neighbors = [
                    n for n in neighbors if self.info.occupancies[n] < 1.0
                ]

                if len(low_occ_neighbors) > 1:
                    # Try to decompose the overcrowded situation using geometric analysis
                    self._decompose_cliques(low_occ_neighbors, center_idx)

    def _get_max_coordination(self, element_symbol: str) -> int:
        """
        Get the maximum coordination number for an element.

        Parameters:
        -----------
        element_symbol : str
            Element symbol

        Returns:
        --------
        int
            Maximum expected coordination number
        """
        element = element_symbol.strip().title()  # Capitalize first letter

        return MAX_COORDINATION_NUMBERS.get(element, DEFAULT_MAX_COORDINATION)  # Default to 6 if unknown

    def _decompose_cliques(self, atom_indices: List[int], center_idx: int):
        """
        Decompose overcrowded atom sets into mutually exclusive subsets.

        Parameters:
        -----------
        atom_indices : List[int]
            Indices of atoms that are overcrowding around a center
        center_idx : int
            Index of the center atom
        """
        if len(atom_indices) < 2:
            return  # Need at least 2 atoms to create exclusions

        # Calculate positions relative to the center
        center_frac = self.info.frac_coords[center_idx]
        relative_positions = []

        for idx in atom_indices:
            atom_frac = self.info.frac_coords[idx]
            # Apply minimum image convention relative to center
            delta = atom_frac - center_frac
            delta = delta - np.round(delta)  # Minimum image
            relative_positions.append(delta)

        # Convert to Cartesian coordinates
        cart_positions = []
        for rel_pos in relative_positions:
            cart_pos = frac_to_cart(rel_pos, self.lattice)
            cart_positions.append(cart_pos)

        # For now, use a simple approach: group atoms that form valid geometries
        # In the case of the "ammonium" example (8 H around N), we expect 2 sets of 4
        if len(atom_indices) == 8 and self.info.symbols[center_idx] in ["N", "P"]:
            # Try to find two tetrahedral arrangements among 8 atoms
            self._find_tetrahedral_groups(atom_indices, cart_positions)
        else:
            # For other cases, if all have low occupancy, make them all mutually exclusive
            # CRITICAL FIX: Only make atoms mutually exclusive if they belong to different disorder groups
            for i_idx in atom_indices:
                for j_idx in atom_indices:
                    if i_idx < j_idx:
                        group_i = self.info.disorder_groups[i_idx]
                        group_j = self.info.disorder_groups[j_idx]

                        # CRITICAL FIX: Allow atoms from the same disorder group to coexist in a clique
                        if group_i != 0 and group_j != 0 and group_i == group_j:
                            continue

                        if not self.graph.has_edge(i_idx, j_idx):
                            self.graph.add_edge(i_idx, j_idx, conflict_type="valence")
                        else:
                            # Upgrade existing conflict type if necessary
                            current_type = self.graph[i_idx][j_idx].get("conflict_type")
                            if current_type not in ["explicit", "geometric"]:
                                self.graph[i_idx][j_idx]["conflict_type"] = "valence"

    def _find_tetrahedral_groups(
        self, atom_indices: List[int], cart_positions: List[np.ndarray]
    ):
        """
        Find two disjoint tetrahedral groups among 8 atoms around a center (like in DAP-4).

        Parameters:
        -----------
        atom_indices : List[int]
            Indices of the 8 surrounding atoms
        cart_positions : List[np.ndarray]
            Cartesian positions of the 8 atoms relative to center
        """
        if len(atom_indices) != 8:
            return  # Need exactly 8 atoms for this logic

        # Calculate angles between all atom pairs as seen from the center (which is at origin now)
        n_atoms = len(atom_indices)

        # For each combination of 4 atoms, check if they form a valid tetrahedron
        candidates = []  # Store (score, group_indices) tuples

        for combo in combinations(range(n_atoms), 4):
            combo_indices = [atom_indices[i] for i in combo]
            combo_positions = [cart_positions[i] for i in combo]

            # Calculate angles between all pairs of positions
            angles = []
            for i in range(4):
                for j in range(i + 1, 4):
                    v1 = combo_positions[i]
                    v2 = combo_positions[j]

                    # Calculate angle between vectors from center to these atoms
                    if np.allclose(v1, 0) or np.allclose(v2, 0):
                        # If position is at center, skip (would cause issues with angle calculation)
                        continue

                    angle_rad = angle_between_vectors(v1, v2)
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)

            # Calculate score based on deviation from ideal tetrahedral angle (109.5°)
            # Lower score means better tetrahedral geometry
            if len(angles) > 0:
                score = sum(abs(angle - 109.5) for angle in angles)
                candidates.append((score, combo_indices))

        # Sort candidates by score (ascending - best geometry first)
        candidates.sort(key=lambda x: x[0])

        # Look for two disjoint groups among the best candidates
        for i in range(len(candidates)):
            part_a = candidates[i][1]  # Best group A
            for j in range(i + 1, len(candidates)):
                part_b = candidates[j][1]  # Next best group B

                # Check if they are disjoint
                if set(part_a).isdisjoint(set(part_b)):
                    # Found two disjoint tetrahedral groups - add exclusions between them
                    # Rule 1: Full exclusion edges between all atoms in Part A and all atoms in Part B
                    for atom_a in part_a:
                        for atom_b in part_b:
                            # Only add if not already added with explicit or geometric conflict
                            if not self.graph.has_edge(atom_a, atom_b):
                                self.graph.add_edge(
                                    atom_a, atom_b, conflict_type="valence_geometry"
                                )
                            else:
                                # Update to valence_geometry if it's not already explicit or geometric
                                current_type = self.graph[atom_a][atom_b][
                                    "conflict_type"
                                ]
                                if current_type not in ["explicit", "geometric"]:
                                    self.graph[atom_a][atom_b][
                                        "conflict_type"
                                    ] = "valence_geometry"

                    # Rule 2: Identify "Rogue Atoms" and enforce strict exclusions
                    # These are atoms that didn't make the cut for the best geometries
                    all_neighbors_set = set(atom_indices)
                    valid_geometry_atoms = set(part_a + part_b)
                    rogue_atoms = all_neighbors_set - valid_geometry_atoms

                    # Add exclusion edges between Rogue Atoms and EVERYONE in A and B
                    for rogue in rogue_atoms:
                        for atom_a in part_a:
                            # Add exclusion between rogue and atom in A
                            if not self.graph.has_edge(rogue, atom_a):
                                self.graph.add_edge(
                                    rogue, atom_a, conflict_type="valence_geometry"
                                )
                            else:
                                current_type = self.graph[rogue][atom_a][
                                    "conflict_type"
                                ]
                                if current_type not in ["explicit", "geometric"]:
                                    self.graph[rogue][atom_a][
                                        "conflict_type"
                                    ] = "valence_geometry"

                        for atom_b in part_b:
                            # Add exclusion between rogue and atom in B
                            if not self.graph.has_edge(rogue, atom_b):
                                self.graph.add_edge(
                                    rogue, atom_b, conflict_type="valence_geometry"
                                )
                            else:
                                current_type = self.graph[rogue][atom_b][
                                    "conflict_type"
                                ]
                                if current_type not in ["explicit", "geometric"]:
                                    self.graph[rogue][atom_b][
                                        "conflict_type"
                                    ] = "valence_geometry"

                    # Return after finding the best disjoint pair
                    return