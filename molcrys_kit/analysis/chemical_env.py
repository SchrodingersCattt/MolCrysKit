"""
Chemical environment analyzer for molecular crystals.

This module provides classes and functions to analyze the local environment
of atoms in molecular crystals, including coordination numbers, bond angles,
and ring detection.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx
from abc import ABC, abstractmethod
from ..structures.crystal import MolecularCrystal
from ..utils.geometry import angle_between_vectors


class ChemicalEnvironment:
    """
    Analyzes the local chemical environment of atoms in a molecular crystal.
    
    Provides methods to calculate coordination numbers, bond angles, 
    planarity, and ring membership for atoms in a crystal structure.
    """
    
    def __init__(self, crystal_molecule):
        """
        Initialize with a CrystalMolecule or graph + positions.
        
        Parameters
        ----------
        crystal_molecule : CrystalMolecule or tuple
            Either a CrystalMolecule object or a tuple containing
            (graph, positions) where graph is a NetworkX graph and 
            positions is a list of 3D coordinates.
        """
        if hasattr(crystal_molecule, 'graph') and hasattr(crystal_molecule, 'get_positions'):
            # It's a CrystalMolecule object
            self.graph = crystal_molecule.graph
            self.positions = crystal_molecule.get_positions()
        else:
            # It's a (graph, positions) tuple
            self.graph, self.positions = crystal_molecule
        
        # Precompute cycle basis ONCE per molecule to avoid repeated heavy calculations
        self._precompute_ring_info()
    
    def _precompute_ring_info(self):
        """Precompute ring membership for all atoms to avoid repeated heavy calculations.

        Note: nx.minimum_cycle_basis can fail when graph node IDs are numpy integer
        types (np.int64) because networkx 3.5's internal Dijkstra compares node IDs
        with ``==`` which may raise a ValueError for numpy arrays.  We work around
        this by relabelling the graph with plain Python ints before calling the
        function, then mapping the results back.
        """
        try:
            # Convert node IDs to plain Python ints to avoid numpy comparison issues
            node_ids = list(self.graph.nodes())
            int_to_orig = {i: n for i, n in enumerate(node_ids)}
            orig_to_int = {n: i for i, n in enumerate(node_ids)}
            relabelled = nx.relabel_nodes(self.graph, orig_to_int)

            raw_cycles = nx.minimum_cycle_basis(relabelled, weight=None)

            self._atom_rings = {}
            for idx in self.graph.nodes():
                int_idx = orig_to_int[idx]
                atom_cycles = [
                    [int_to_orig[n] for n in cycle]
                    for cycle in raw_cycles
                    if int_idx in cycle
                ]
                self._atom_rings[idx] = atom_cycles
        except Exception:
            self._atom_rings = {}

        self._precompute_aromatic_rings()

    def _precompute_aromatic_rings(self):
        """
        Identify aromatic rings and cache membership per atom.

        Aromaticity is detected geometrically (no Hückel) using three criteria:
        1. Ring size ∈ {5, 6, 7}
        2. All ring atoms are C, N, O (S excluded: C-S ~1.71 Å exceeds length window)
        3. All consecutive ring-bond lengths ∈ [1.20, 1.45] Å
           The lower bound is 1.20 (not 1.30) to accommodate short N=N double bonds
           (~1.25 Å) in pseudo-aromatic rings such as sydnones, oxadiazines, and
           triazine-like heterocycles.  Bonds below ~1.20 Å do not occur between
           ring heavy atoms in organic molecules.
        4. Ring is planar by the max-out-of-plane criterion

        Result: self._atom_aromatic_ring_sizes[atom_idx] -> list of ring sizes
        (empty list means not in any aromatic ring).
        """
        _AROM_ATOMS = {'C', 'N', 'O'}
        _BOND_MIN = 1.20  # allows N=N (~1.25 Å) in pseudo-aromatic rings
        _BOND_MAX = 1.45

        self._atom_aromatic_ring_sizes: Dict[int, List[int]] = {
            idx: [] for idx in self.graph.nodes()
        }

        all_cycles = []
        for cycles in self._atom_rings.values():
            for c in cycles:
                # Use frozenset to deduplicate cycles across atoms
                key = frozenset(c)
                if key not in {frozenset(x) for x in all_cycles}:
                    all_cycles.append(c)

        for cycle in all_cycles:
            n = len(cycle)
            if n not in (5, 6, 7):
                continue

            # Criterion 2: all atoms must be C/N/O
            symbols = [self.graph.nodes[i].get('symbol', '') for i in cycle]
            if not all(s in _AROM_ATOMS for s in symbols):
                continue

            # Criterion 3: consecutive ring-bond lengths in aromatic window
            ok = True
            for k in range(n):
                a, b = cycle[k], cycle[(k + 1) % n]
                if not self.graph.has_edge(a, b):
                    ok = False
                    break
                d = np.linalg.norm(self.positions[b] - self.positions[a])
                if not (_BOND_MIN <= d <= _BOND_MAX):
                    ok = False
                    break
            if not ok:
                continue

            # Criterion 4: planarity
            if not self._is_ring_planar(cycle):
                continue

            # All criteria met -> mark atoms
            for idx in cycle:
                self._atom_aromatic_ring_sizes[idx].append(n)

    def atom_aromatic_ring_sizes(self, atom_index: int) -> List[int]:
        """
        Return list of aromatic ring sizes the atom belongs to (empty if none).

        Parameters
        ----------
        atom_index : int
            Index of the atom to query.

        Returns
        -------
        List[int]
            Sizes of aromatic rings containing this atom.  Empty list means the
            atom is not in any detected aromatic ring.
        """
        return self._atom_aromatic_ring_sizes.get(atom_index, [])

    def has_conjugated_ring_bond(self, atom_index: int, threshold: float = 1.43) -> bool:
        """
        Return True if any ring bond *not* directly involving *atom_index* has length
        ≤ *threshold* Å.

        This is a robust indicator of π-delocalization in the ring around the atom.
        Unlike checking the atom's own bond lengths (which can be contracted by ring
        strain), the OTHER ring bonds sharply distinguish aromatic and sp3 rings:

        * Aromatic C–C / C–N:  ~1.33–1.42 Å  →  comfortably ≤ 1.43 Å
        * sp3 C–C:             ~1.50–1.54 Å  →  clearly  >  1.43 Å  (Δ ≈ 0.10 Å)
        * sp3 C–N:             ~1.46–1.48 Å  →  clearly  >  1.43 Å  (Δ ≈ 0.07 Å)

        Checking only bonds that do NOT touch the query atom avoids the ambiguity of
        the atom's own bonds, which may be shortened by strain (sp3 cage O–C ~1.38 Å)
        yet still fail the conjugated-ring criterion because the adjacent C–C bonds
        (~1.52 Å) exceed the threshold.

        Parameters
        ----------
        atom_index : int
            The ring atom whose neighbouring ring bonds are checked.
        threshold : float
            Upper bound for "aromatic-like" bond length (Å).  Default 1.43 Å.

        Returns
        -------
        bool
            True if the ring contains at least one bond (not touching atom_index)
            shorter than *threshold*.
        """
        rings = self._atom_rings.get(atom_index, [])
        for ring in rings:
            n = len(ring)
            for k in range(n):
                a, b = ring[k], ring[(k + 1) % n]
                if atom_index in (a, b):
                    continue  # skip bonds that touch the query atom
                d = float(np.linalg.norm(self.positions[a] - self.positions[b]))
                if d <= threshold:
                    return True
        return False

    def get_local_geometry_stats(self, atom_index: int) -> Dict:
        """
        Calculate raw geometry statistics. Does NOT make decisions (e.g. is_planar).
        Returns raw angles and lengths for heuristics to decide.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom to analyze
            
        Returns
        -------
        dict
            Dictionary containing:
            - coordination_number: Number of neighbors
            - bond_angle_sum: Sum of angles between all neighbor pairs
            - bond_angle_single: Specifically for coordination=2, the angle between the two neighbors
            - average_bond_length: Average distance to neighbors
        """
        neighbors = list(self.graph.neighbors(atom_index))
        n_neighbors = len(neighbors)
        
        center_pos = self.positions[atom_index]
        
        distances = []
        neighbor_positions = []
        for neighbor_idx in neighbors:
            neighbor_pos = self.positions[neighbor_idx]
            dist = np.linalg.norm(neighbor_pos - center_pos)
            distances.append(dist)
            neighbor_positions.append(neighbor_pos)
        avg_bond_length = sum(distances) / len(distances) if distances else 0.0
        
        # Calculate angle statistics
        bond_angle_sum = 0.0
        bond_angle_avg = 0.0  # Only meaningful for coord=2
        
        if n_neighbors >= 2:
            angles = []
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    vec1 = neighbor_positions[i] - center_pos
                    vec2 = neighbor_positions[j] - center_pos
                    angle = np.degrees(angle_between_vectors(vec1, vec2))
                    angles.append(angle)
            
            bond_angle_sum = sum(angles)
            if n_neighbors == 2 and angles:
                bond_angle_avg = angles[0]  # For coord=2, there is only 1 angle
        
        return {
            'coordination_number': n_neighbors,
            'bond_angle_sum': bond_angle_sum,
            'bond_angle_single': bond_angle_avg,  # Specifically for coord=2
            'average_bond_length': avg_bond_length
        }
    
    def detect_ring_info(self, atom_index: int) -> Dict:
        """
        Detect ring information for an atom using precomputed ring info.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom to analyze
            
        Returns
        -------
        dict
            Dictionary containing:
            - in_ring: Boolean indicating if atom is in a ring
            - ring_sizes: List of sizes of rings this atom belongs to
            - is_ring_planar: Boolean for the smallest ring, whether it's planar
        """
        try:
            # Use cached ring info
            atom_cycles = self._atom_rings.get(atom_index, [])
            
            if not atom_cycles:
                return {
                    'in_ring': False,
                    'ring_sizes': [],
                    'is_ring_planar': False
                }
            
            # Get ring sizes
            ring_sizes = [len(cycle) for cycle in atom_cycles]
            
            # Determine if the smallest ring is planar
            is_ring_planar = False
            if ring_sizes:
                # Find the smallest ring
                min_size = min(ring_sizes)
                smallest_ring_cycles = [cycle for cycle in atom_cycles if len(cycle) == min_size]
                
                if smallest_ring_cycles:
                    # Take the first occurrence of the smallest ring size
                    smallest_ring = smallest_ring_cycles[0]
                    
                    # Check if the atoms in the ring lie on a plane
                    is_ring_planar = self._is_ring_planar(smallest_ring)
            
            return {
                'in_ring': True,
                'ring_sizes': sorted(ring_sizes),
                'is_ring_planar': is_ring_planar
            }
        except Exception:
            # If there's an issue with cycle detection, fall back to basic check
            neighbors = list(self.graph.neighbors(atom_index))
            if len(neighbors) < 2:
                return {
                    'in_ring': False,
                    'ring_sizes': [],
                    'is_ring_planar': False
                }
            
            # Simple check: if neighbors are interconnected, it might be in a ring
            # This is a simplified fallback
            return {
                'in_ring': False,
                'ring_sizes': [],
                'is_ring_planar': False
            }
    
    def _is_ring_planar(self, ring_atom_indices: List[int], max_dev_tolerance: float = 0.25) -> bool:
        """
        Check if the atoms in a ring lie on a plane using max-out-of-plane deviation.

        Uses SVD to find the best-fit plane normal, then reports the maximum
        perpendicular distance of any ring atom from that plane.  This is
        more robust than the previous smallest-singular-value criterion, which
        grows with ring size and was sensitive to CSD refinement noise (~0.1 Å).

        Parameters
        ----------
        ring_atom_indices : List[int]
            Indices of atoms forming the ring.
        max_dev_tolerance : float
            Maximum allowed out-of-plane deviation in Angstroms.  Default 0.25 Å
            is chosen to accept aromatic rings in structures refined to R~0.07
            (typical CSD noise ±0.04-0.06 Å) while still rejecting sp3 chair-
            like distortions (~0.5 Å).

        Returns
        -------
        bool
            True if the maximum out-of-plane deviation is below the tolerance.
        """
        if len(ring_atom_indices) < 3:
            return False

        pts = np.array([self.positions[idx] for idx in ring_atom_indices], dtype=float)
        centroid = pts.mean(axis=0)
        centered = pts - centroid

        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # Last row of Vt is the direction of minimum variance (ring normal)
        normal = Vt[-1]

        deviations = np.abs(centered @ normal)
        return float(deviations.max()) < max_dev_tolerance

    def get_site(self, atom_index: int) -> 'HybridizedSite':
        """
        Factory method to create the appropriate HybridizedSite subclass based on the atom's symbol.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom in the crystal
            
        Returns
        -------
        HybridizedSite
            An instance of the appropriate subclass based on the atom's symbol
        """
        atom_symbol = self.graph.nodes[atom_index]['symbol'] if 'symbol' in self.graph.nodes[atom_index] else self.graph.nodes[atom_index]
        if isinstance(atom_symbol, dict):
            atom_symbol = atom_symbol.get('symbol', 'X')  # Fallback to 'X' if no symbol
        elif hasattr(atom_symbol, 'symbol'):
            atom_symbol = atom_symbol.symbol
        
        if atom_symbol == 'C':
            return CarbonSite(atom_index, atom_symbol, self)
        elif atom_symbol == 'N':
            return NitrogenSite(atom_index, atom_symbol, self)
        else:
            return GenericSite(atom_index, atom_symbol, self)


class HybridizedSite(ABC):
    """
    Abstract base class for representing hybridized sites in a chemical environment.
    """
    
    def __init__(self, atom_index: int, element: str, env: 'ChemicalEnvironment'):
        """
        Initialize with atom index, element, and chemical environment.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom in the crystal
        element : str
            Element symbol
        env : ChemicalEnvironment
            The chemical environment of the crystal
        """
        self.atom_index = atom_index
        self.element = element
        self.env = env
    
    @property
    def geometry_stats(self) -> Dict:
        """
        Lazy access to local geometry statistics.
        """
        return self.env.get_local_geometry_stats(self.atom_index)
    
    @property
    def ring_info(self) -> Dict:
        """
        Lazy access to ring information.
        """
        return self.env.detect_ring_info(self.atom_index)
    
    @abstractmethod
    def get_hydrogen_completion_strategy(self) -> Dict:
        """
        Abstract method to determine hydrogen_completion strategy for this site.
        
        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        pass


class CarbonSite(HybridizedSite):
    """
    Carbon-specific hybridized site implementation.
    """
    
    def get_hydrogen_completion_strategy(self) -> Dict:
        """
        Determine hydrogen_completion strategy for carbon based on its environment.

        Aromatic-ring membership (detected by geometric pre-pass) short-circuits
        the local-geometry heuristic to avoid misclassifying 5-membered aromatic
        ring carbons as sp3 (their internal angle ~108° is nearly identical to the
        tetrahedral ideal of 109.5°).

        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        from ..constants.config import BOND_LENGTHS

        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)

        # Aromatic short-circuit: if the atom is in a detected aromatic ring,
        # decide purely from coordination number (no geometry scoring needed).
        arom_sizes = self.env.atom_aromatic_ring_sizes(self.atom_index)
        if arom_sizes:
            coord = self.geometry_stats['coordination_number']
            if coord == 2:
                # Aromatic CH (e.g. indole C2, pyrrole C3/C4, benzene CH)
                return {'num_h': 1, 'geometry': 'trigonal_planar', 'bond_length': bond_length}
            elif coord == 3:
                # Aromatic C with three ring/substituent bonds — no H needed
                return {'num_h': 0, 'geometry': 'trigonal_planar', 'bond_length': bond_length}
            # coord==1 or coord==4 in an aromatic ring: fall through to general logic
            # (unusual but possible in distorted/disordered structures)

        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']

        # Defaults
        num_h = 0
        geometry = 'tetrahedral'

        if coord == 3:
            # Case: sp2 (Planar, ~360 sum) vs sp3 (Pyramidal, ~328.5 sum)
            # Previous logic was flawed: it prioritized ring planarity over local geometry
            # New logic: Local geometry takes precedence over global ring properties
            
            angle_sum = self.geometry_stats['bond_angle_sum']
            # --- NEW LOGIC: Local Geometry First ---
            
            # 1. Definitely sp3 region (Pyramidal)
            # Ideal tetrahedral is 328.5 degrees. With tolerance to 345 degrees.
            # If less than this value, regardless of ring environment, it must be pyramidal.
            if angle_sum < 345.0:
                num_h = 1
                geometry = 'tetrahedral'
                
            # 2. Definitely sp2 region (Planar)
            # Close to 360 degrees, definitely planar.
            elif angle_sum > 355.0:
                num_h = 0
                geometry = 'trigonal_planar'
                
            # 3. Ambiguous region (Distorted/Intermediate)
            # E.g. 348 degrees. Could be a strained sp3 or a distorted sp2.
            # Only in this case do we consider the ring environment for arbitration.
            else:
                if self.ring_info['in_ring'] and self.ring_info['is_ring_planar']:
                    # Ring is planar -> likely aromatic/conjugated system -> sp2
                    num_h = 0
                    geometry = 'trigonal_planar'
                else:
                    # Ring not planar, or not in ring -> default to sp3
                    num_h = 1
                    geometry = 'tetrahedral'

        elif coord == 2:
            # Case: -CH2- (sp3) vs =CH- (sp2 aromatic) vs =C= (sp linear)
            angle = self.geometry_stats['bond_angle_single']
            
            # --- SCORING SYSTEM ---
            # Ideal models
            # sp:   Angle 180,   Len ~1.2-1.3 (cumulene)
            # sp2:  Angle 120,   Len ~1.34-1.42 (aromatic)
            # sp3:  Angle 109.5, Len ~1.50-1.54 (aliphatic)
            
            # 1. Angle Penalty (Weighted heavily)
            score_sp   = abs(angle - 180.0)
            score_sp2  = abs(angle - 120.0)
            score_sp3  = abs(angle - 109.5)
            
            # 2. Bond Length Bias (Adjust scores based on length)
            # If length > 1.46 (typical single bond), heavily penalize sp/sp2
            if avg_len > 1.46:
                score_sp3 -= 15.0  # Strong bonus for sp3
                score_sp  += 20.0  # Penalty for sp
            # If length < 1.42 (aromatic / double bond range), penalize sp3.
            # Threshold raised from 1.38 to 1.42: covers aromatic C-C/C-N up to
            # ~1.42 Å (e.g. indole C2-C3 1.385 Å), acting as a safety net when
            # the aromatic pre-pass misses a distorted ring.
            elif avg_len < 1.42:
                score_sp2 -= 10.0  # Bonus for sp2
                score_sp3 += 20.0  # Penalty for sp3
            
            # 3. Decision
            best_match = min(score_sp, score_sp2, score_sp3)
            
            if best_match == score_sp and score_sp < 15.0: # Must be reasonably close
                num_h = 0
                geometry = 'linear'
            elif best_match == score_sp2:
                num_h = 1
                geometry = 'trigonal_planar'
            else:
                # Default to sp3 if ambiguous or matches sp3 best
                num_h = 2
                geometry = 'tetrahedral'
                
        elif coord == 1:
            # Case: Terminal Carbon
            # Possibilities:
            # 1. Alkyne (-C#C-H) or Isonitrile (-NC): sp, Triple bond (~1.20 A)
            # 2. Terminal Alkene (=CH2) or Imine (=CH2): sp2, Double bond (~1.34 A)
            # 3. Methyl (-CH3): sp3, Single bond (~1.54 A, C-N ~1.47 A)
            
            # Thresholds:
            # Triple < 1.28 (Safe cutoff for 1.20)
            # 1.28 <= Double < 1.42 (Safe cutoff between 1.34 and 1.47)
            # Single >= 1.42
            
            if avg_len < 1.28 and avg_len > 0.1: 
                # Case 1: Triple Bond (sp) -> Add 1 H
                num_h = 1
                geometry = 'linear'
                
            elif avg_len <= 1.42:
                # Case 2: Double Bond (sp2) -> Add 2 H (Terminal Alkene)
                # Note: "trigonal_planar" for a terminal atom means adding 2 H 
                # in the plane defined by the double bond vector (if possible)
                num_h = 2
                geometry = 'trigonal_planar'
                
            else:
                # Case 3: Single Bond (sp3) -> Add 3 H (Methyl)
                num_h = 3
                geometry = 'tetrahedral'
        
        return {
            'num_h': num_h,
            'geometry': geometry,
            'bond_length': bond_length
        }


class NitrogenSite(HybridizedSite):
    """
    Nitrogen-specific hybridized site implementation.
    
    Hybridization is inferred primarily from the shortest heavy-atom bond length,
    which is the most reliable single indicator of bond order and hybridization.
    
    Reference bond lengths (literature values, Å):
      N≡C (nitrile):   ~1.16   → sp
      N=C (imine):     ~1.28   → sp2
      N-C(amide):      ~1.34   → sp2 (partial double bond character)
      N-C(aniline):    ~1.40   → sp2 (resonance)
      N-C(amine sp3):  ~1.47   → sp3
      N=N (azo):       ~1.25   → sp2
      N-N (hydrazine): ~1.45   → sp3
    """
    
    def _min_heavy_bond_len(self) -> float:
        """Return the shortest bond length to a non-H neighbor."""
        import numpy as np
        graph = self.env.graph
        positions = self.env.positions
        center = positions[self.atom_index]
        dists = [
            np.linalg.norm(positions[nb] - center)
            for nb in graph.neighbors(self.atom_index)
            if graph.nodes[nb].get('symbol', '') != 'H'
        ]
        return min(dists) if dists else self.geometry_stats['average_bond_length']
    
    def get_hydrogen_completion_strategy(self) -> Dict:
        """
        Determine hydrogen_completion strategy for nitrogen based on its environment.

        Aromatic-ring membership (detected by geometric pre-pass) short-circuits
        the heuristic to correctly distinguish pyridine-like N (coord=2, 0H) from
        pyrrole-like N (coord=2, 1H) and N-substituted aromatic N (coord=3, 0H).

        Primary decision criterion (non-aromatic): shortest heavy-atom bond length.
        Secondary criterion: ring planarity (for aromatic systems).
        Tertiary criterion: bond angle (for linear sp detection).
        """
        from ..constants.config import BOND_LENGTHS

        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)

        # Aromatic short-circuit
        arom_sizes = self.env.atom_aromatic_ring_sizes(self.atom_index)
        if arom_sizes:
            coord = self.geometry_stats['coordination_number']
            if coord == 2:
                min_heavy = self._min_heavy_bond_len()
                # Pyridine-like (lone pair in plane, 0H): 6-membered ring, OR
                # bond to a neighbouring aromatic C is short (imidazole C2-N ~1.32 Å)
                # indicating genuine imine character.
                if 6 in arom_sizes or min_heavy < 1.34:
                    return {'num_h': 0, 'geometry': 'planar_aromatic', 'bond_length': bond_length}
                else:
                    # Pyrrole-like (lone pair in π system, 1H): typically 5-ring,
                    # N–C bonds ~1.37–1.40 Å (longer than the imine threshold).
                    return {'num_h': 1, 'geometry': 'planar_bisector', 'bond_length': bond_length}
            elif coord == 3:
                # N-substituted aromatic N (e.g. indole N, N-methyl pyrrole) — no H
                return {'num_h': 0, 'geometry': 'trigonal_planar', 'bond_length': bond_length}
            # coord==1 or coord==4: unusual in aromatic context, fall through

        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']

        # Defaults
        num_h = 0
        geometry = 'tetrahedral'

        # Shortest heavy-atom bond: most reliable hybridization indicator
        min_heavy = self._min_heavy_bond_len()
        
        in_ring = self.ring_info['in_ring']
        is_planar_ring = self.ring_info['is_ring_planar']
        ring_sizes = self.ring_info['ring_sizes']
        
        # Strained 3-membered rings (aziridine): usually sp3, BUT if there is an
        # exocyclic bond with double-bond character (min_heavy < 1.38, e.g. C=N
        # amidine, C=O amide attached to ring N), the N is sp2 (planar).
        # The lone pair participates in conjugation with the exocyclic π system.
        if in_ring and ring_sizes and min(ring_sizes) <= 3:
            # Check for exocyclic double-bond character
            min_heavy = self._min_heavy_bond_len()
            if min_heavy < 1.38:
                # Exocyclic conjugation (amidine, amide, imine) → sp2
                geometry = 'trigonal_planar'
                num_h = 0
            else:
                geometry = 'tetrahedral'
                if coord == 1:
                    num_h = 2
            return {'num_h': num_h, 'geometry': geometry, 'bond_length': bond_length}
        
        if coord == 2:
            bond_angle = self.geometry_stats['bond_angle_single']
            # sp: linear geometry (nitrile C≡N, isocyanate N=C=O, carbodiimide)
            # Threshold: bond_angle > 160° OR min_heavy < 1.22 (triple/cumulated bond)
            if bond_angle > 160.0 or min_heavy < 1.22:
                num_h = 0
                geometry = 'linear'
            # sp2: aromatic ring N (pyridine, pyrrole, imidazole, etc.)
            # Guard: min_heavy < 1.42 Å checks the bonds TO THIS NITROGEN specifically.
            # sp2 N bonds (pyridine, pyrrole, amide, aniline): 1.33–1.40 Å — all < 1.42.
            # sp3 N bonds (tertiary amine, cage N): ≥ 1.44 Å — excluded by guard.
            # We deliberately check bonds to N (not max_ring_bond_len) because some
            # non-fully-aromatic rings contain a long C-C bond between two sp2 carbons
            # (e.g. isatin, where C2-C3 ~1.52 Å) yet the N itself IS sp2 with short
            # N-C bonds (~1.37-1.40 Å).  max_ring_bond_len would incorrectly reject
            # such rings, while min_heavy correctly accepts them.
            elif in_ring and is_planar_ring and min_heavy < 1.42:
                if 6 in ring_sizes:  # Pyridine-like: lone pair in plane, no H
                    num_h = 0
                    geometry = 'planar_aromatic'
                else:  # Pyrrole-like (5-membered): lone pair in pi system, 1H
                    num_h = 1
                    geometry = 'planar_bisector'
            # sp2: imine/amide N with short bond (C=N ~1.28, amide N-C ~1.34)
            elif min_heavy < 1.38:
                num_h = 0
                geometry = 'trigonal_planar'
            else:
                # sp3: secondary amine (N-C ~1.47, N-N ~1.45)
                num_h = 1
                geometry = 'tetrahedral'
                
        elif coord == 1:
            # Terminal N: distinguish by bond length
            # sp:  C≡N nitrile (~1.16), N≡N (~1.10)
            # sp2: C=N imine (~1.28), amide N-C(=O) (~1.34)
            # sp3: C-N amine (~1.47)
            if min_heavy < 1.22:
                num_h = 0
                geometry = 'linear'
            elif min_heavy < 1.38:
                # sp2: amide or imine terminus
                num_h = 1
                geometry = 'trigonal_planar'
            else:
                # sp3: primary amine -NH2
                num_h = 2
                geometry = 'tetrahedral'
        
        elif coord == 3:
            # Tertiary N: sp3 amine vs sp2 (aniline, amide, enamine)
            # Key: sp2 N has at least one short bond due to resonance/conjugation
            # sp2 N-C(aromatic/carbonyl): min_heavy ~1.34-1.40
            # sp3 N-C(alkyl):             min_heavy ~1.46-1.47
            # sp3 N-N(hydrazine):         min_heavy ~1.45
            # Threshold 1.42 Å: covers aniline (1.40), amide (1.34), enamine (1.37)
            # while excluding sp3 amine (1.46+) and hydrazine N-N (1.45)
            # Same min_heavy guard as coord==2 for the same reason: the N-atom's own
            # bond lengths reliably reflect its hybridization even when the ring
            # contains long C-C bonds between sp2 carbons (as in isatin or maleimide).
            if in_ring and is_planar_ring and min_heavy < 1.42:
                # Aromatic ring N (e.g. N-substituted pyrrole, indole)
                num_h = 0
                geometry = 'trigonal_planar'
            elif min_heavy < 1.42:
                # sp2: aniline, amide, enamine — shortest bond shows conjugation
                num_h = 0
                geometry = 'trigonal_planar'
            else:
                # sp3: tertiary amine, hydrazine N
                num_h = 0
                geometry = 'tetrahedral'
        
        return {
            'num_h': num_h,
            'geometry': geometry,
            'bond_length': bond_length
        }


class GenericSite(HybridizedSite):
    """
    Generic hybridized site implementation for elements other than C and N.
    """
    
    def get_hydrogen_completion_strategy(self) -> Dict:
        """
        Determine hydrogen_completion strategy for generic elements based on their environment.

        For oxygen specifically, aromatic-ring membership (furan-like O) is detected
        via the geometric pre-pass and short-circuits the heuristic to avoid adding
        a spurious H to a furan-type O.

        Returns
        -------
        dict
            Contains 'num_h', 'geometry', and 'bond_length' keys
        """
        import numpy as np
        from ..constants.config import BOND_LENGTHS

        coord = self.geometry_stats['coordination_number']
        avg_len = self.geometry_stats['average_bond_length']

        # Defaults
        num_h = 0
        geometry = 'tetrahedral'
        bond_length = BOND_LENGTHS.get(f"{self.element}-H", 1.0)

        atom_symbol = self.element

        # Aromatic short-circuit for oxygen (furan-like: lone pair in π system, 0H)
        if atom_symbol == 'O' and self.env.atom_aromatic_ring_sizes(self.atom_index):
            return {'num_h': 0, 'geometry': 'trigonal_planar', 'bond_length': bond_length}

        # Shortest heavy-atom bond length: primary hybridization indicator
        # (avoids distortion from short X-H bonds, e.g. O-H ~0.97 Å)
        graph = self.env.graph
        positions = self.env.positions
        center = positions[self.atom_index]
        heavy_dists = [
            np.linalg.norm(positions[nb] - center)
            for nb in graph.neighbors(self.atom_index)
            if graph.nodes[nb].get('symbol', '') != 'H'
        ]
        min_heavy = min(heavy_dists) if heavy_dists else avg_len
        
        in_ring = self.ring_info['in_ring']
        is_planar_ring = self.ring_info['is_ring_planar']
        ring_sizes = self.ring_info['ring_sizes']
        
        # Oxygen rules
        # Reference bond lengths (Å):
        #   C=O carbonyl:    ~1.23   sp2
        #   C-O ester/enol:  ~1.34   sp2 (partial double bond)
        #   C-O aromatic:    ~1.37   sp2 (furan)
        #   C-O ether/alc:   ~1.43   sp3
        #   3-membered ring: strained → sp3 regardless of length
        if atom_symbol == 'O':
            if coord == 1:
                # Terminal O: only one heavy bond, no H confusion
                # Decision hierarchy based on bond length:
                #   C=O carbonyl:   ~1.20-1.25  → sp2 (no H)
                #   N=O oxime/nitroso: ~1.21-1.40 → sp2 (no H)
                #   C-OH alcohol:   ~1.41-1.43  → sp3 (add 1 H)
                # Threshold 1.42: covers all sp2 terminal O (carbonyl + oxime),
                # while the alcohol O-C bond is typically ≥1.41 in 3D optimised geom.
                # In practice the gap is small but bond length is the best available
                # single descriptor without explicit bond-order information.
                if min_heavy < 1.42:
                    num_h = 0
                    geometry = 'trigonal_planar'  # sp2: carbonyl or oxime O
                else:
                    num_h = 1
                    geometry = 'bent'  # sp3 hydroxyl
            elif coord == 2:
                if in_ring and ring_sizes and min(ring_sizes) <= 3:
                    # Oxirane: highly strained 3-membered ring → always sp3
                    num_h = 0
                    geometry = 'bent'
                elif in_ring and ring_sizes and min(ring_sizes) == 4:
                    # Oxetane: 4-membered ring.
                    # Usually sp3 (C-O-C ~1.44-1.46), but if one bond shows
                    # genuine double-bond character (min_heavy < 1.37, e.g.
                    # iminolactone O-C=N ~1.34, beta-lactone O-C=O ~1.35)
                    # the O is sp2 despite being in a small ring.
                    if min_heavy < 1.37:
                        num_h = 0
                        geometry = 'trigonal_planar'  # sp2: iminolactone / beta-lactone
                    else:
                        num_h = 0
                        geometry = 'bent'  # sp3 oxetane ether
                elif in_ring and is_planar_ring and ring_sizes and min(ring_sizes) >= 5:
                    # Fallback for O in a planar 5/6/7-membered ring that was NOT
                    # caught by the aromatic pre-pass short-circuit above.
                    # The pre-pass uses bond-length window [1.20, 1.45] Å, covering
                    # standard furan-like O AND pseudo-aromatic rings with N=N.
                    # This branch fires only for rare edge cases (e.g. one ring bond
                    # marginally > 1.45 Å due to coordinate noise in experimental data).
                    #
                    # In this residual population we use O's own bond length (min_heavy)
                    # as the discriminant.  In real crystal structures:
                    #   Aromatic ring O (furan, benzofuran, chromene, …):  C–O ~1.36 Å
                    #   sp3 ether O (THF, THP, dioxane, cage compounds):   C–O ~1.43 Å
                    # The ~0.07 Å gap gives robust separation for the experimental data
                    # that MolCrysKit is designed for (CSD / CIF files).
                    # Threshold 1.40 Å places the boundary in the middle of this gap.
                    if min_heavy < 1.40:
                        num_h = 0
                        geometry = 'trigonal_planar'  # furan-like sp2 (missed by pre-pass)
                    else:
                        num_h = 0
                        geometry = 'bent'  # sp3 ring ether
                elif min_heavy < 1.38:
                    # sp2 O with genuine conjugation (not in a small ring):
                    #   vinyl ether O-C(sp2): ~1.36-1.37
                    #   ester bridging O-C=O: ~1.34-1.37
                    num_h = 0
                    geometry = 'trigonal_planar'
                else:
                    # sp3: aliphatic ether (~1.43), alcohol O (~1.43)
                    num_h = 0
                    geometry = 'bent'
                
        elif atom_symbol == 'S':
            if coord == 1:
                num_h = 1
                geometry = 'bent'
            elif coord == 2:
                in_ring = self.ring_info['in_ring']
                is_planar_ring = self.ring_info['is_ring_planar']
                if in_ring and is_planar_ring:
                    # Thiophene-like aromatic S (sp2)
                    num_h = 0
                    geometry = 'trigonal_planar'
                else:
                    num_h = 0
                    geometry = 'bent'
        
        return {
            'num_h': num_h,
            'geometry': geometry,
            'bond_length': bond_length
        }