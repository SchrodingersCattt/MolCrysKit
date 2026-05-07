"""
hydrogen_completion operations for molecular crystals.

This module provides functionality to add hydrogen atoms to molecular crystals.
It builds a heuristic per-atom H plan, optionally corrects fragment H counts
from CIF `_chemical_formula_moiety`, and then places hydrogens geometrically.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional
from ase import Atoms
from ..structures.crystal import MolecularCrystal
from ..analysis.chemical_env import ChemicalEnvironment
from ..analysis.formula_moiety import match_molecule_to_fragment, parse_moiety_string
from ..utils.geometry import (
    get_missing_vectors,
    calculate_dihedral_and_adjustment,
    normalize_vector,
    cart_to_frac,
    frac_to_cart,
    rotate_vector,
    calculate_center_of_mass,
)


def add_hydrogens(
    crystal: MolecularCrystal,
    target_elements: Optional[List[str]] = None,
    optimize_torsion: bool = False,
    rules=None,
    bond_lengths=None,
    use_formula_moiety: bool = True,
):
    """
    Add hydrogen atoms to a molecular crystal.

    Geometric heuristics are always used to choose candidate atoms and
    placement geometry.  When ``use_formula_moiety=True`` and the crystal was
    read from a CIF containing ``_chemical_formula_moiety``, the moiety formula
    is used as a per-fragment hydrogen-count correction before placement.  If
    the moiety value is absent, unknown (``?``), unparseable, or ambiguous, the
    function falls back to the heuristic result for the affected molecule.

    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to hydrogenate.
    target_elements : Optional[List[str]]
        Whitelist of atomic symbols to hydrogenate (e.g. ``["C", "N", "O"]``).
        When provided, only atoms whose symbol is in this list are processed.
        When ``None`` (the default), all heavy atoms are processed.
    optimize_torsion : bool
        Whether to perform conformational optimization to adjust torsion angles.
        Default is False to preserve ring structures and crystal packing.
    rules : Optional[List[Dict]]
        Per-atom override rules for coordination geometry. Each rule is a dict:

        .. code-block:: python

            {
                "symbol": str,              # Required. E.g. "O", "N"
                "neighbors": List[str],     # Optional. Context filter (e.g. ["Cl", "S"])
                "target_coordination": int, # Optional. Desired total coordination.
                "geometry": str,            # Optional. Override placement geometry.
            }

        Priority order: specific rules (with ``neighbors``) > general rules
        (without ``neighbors``) > built-in defaults.
    bond_lengths : Optional[Dict]
        Override bond lengths for specific atom pairs, keyed as ``"X-H"``
        (e.g. ``{"C-H": 1.09, "N-H": 1.01}``).
    use_formula_moiety : bool
        When ``True`` (default), use a CIF ``_chemical_formula_moiety`` value
        attached to the crystal to correct per-fragment hydrogen counts before
        placing atoms. Geometry and placement still come from the heuristic path.
        Use ``False`` to reproduce the heuristic-only behaviour.

    Returns
    -------
    MolecularCrystal
        New crystal with hydrogen atoms added.
    """
    h_completer = HydrogenCompleter(crystal)
    return h_completer.add_hydrogens(
        target_elements=target_elements,
        optimize_torsion=optimize_torsion,
        rules=rules,
        bond_lengths=bond_lengths,
        use_formula_moiety=use_formula_moiety,
    )


class HydrogenCompleter:
    """
    Class for adding hydrogen atoms to molecular crystals.

    The completer keeps geometric placement heuristics as the base path and can
    use CIF `_chemical_formula_moiety` metadata as a per-fragment H-count
    constraint before placement.
    """

    def __init__(self, crystal: MolecularCrystal):
        """
        Initialize the h_completer with a molecular crystal.

        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to hydrogenate.
        """
        self.crystal = crystal
        # Use unwrapped molecules to handle periodic boundary conditions properly
        self.unwrapped_molecules = crystal.get_unwrapped_molecules()

        # Import the constants here to avoid circular imports
        from ..constants.config import COORDINATION_RULES, BOND_LENGTHS

        # Use imported constants instead of hardcoding them
        self.default_rules = COORDINATION_RULES
        self.default_bond_lengths = BOND_LENGTHS

    def _find_matching_rule(self, chem_env, atom_idx, symbol, specific_rules, general_rules):
        """
        Find a matching rule for the given atom based on its chemical environment.
        
        Parameters
        ----------
        chem_env : ChemicalEnvironment
            The chemical environment of the molecule
        atom_idx : int
            Index of the atom to find rules for
        symbol : str
            Chemical symbol of the atom
        specific_rules : list
            List of rules with neighbor conditions
        general_rules : dict
            Dictionary of general rules indexed by symbol
            
        Returns
        -------
        dict or None
            Matching rule or None if no rule matches
        """
        # Retrieve neighbors and their symbols
        neighbors = list(chem_env.graph.neighbors(atom_idx))
        neighbor_symbols = [chem_env.graph.nodes[n]['symbol'] for n in neighbors]
        
        # Check specific rules first
        for rule in specific_rules:
            if rule['symbol'] == symbol:
                # Check if any neighbor symbols match the rule's neighbor conditions
                if any(neighbor_symbol in rule['neighbors'] for neighbor_symbol in neighbor_symbols):
                    return rule
        
        # Check general rules
        if symbol in general_rules:
            return general_rules[symbol]
        
        # No rule matched
        return None

    def _expand_periodic_images(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """Expand crystal-context positions over the nearest periodic images."""
        if not positions:
            return []

        lattice = np.asarray(self.crystal.lattice)
        image_positions = []
        for pos in positions:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        image_positions.append(pos + np.dot([i, j, k], lattice))
        return image_positions

    def _collect_anion_positions(self) -> List[np.ndarray]:
        """
        Collect Cartesian positions of likely anion atoms across the crystal.

        These positions are used only by charge-aware protonation heuristics.
        Detection is deliberately conservative: terminal O atoms on sulfonate,
        phosphonate, carboxylate, or inorganic oxo-anion centers, plus
        standalone halide ions.
        """
        anion_positions = []
        halides = {"F", "Cl", "Br", "I"}

        for unwrapped_mol in self.unwrapped_molecules:
            symbols = unwrapped_mol.get_chemical_symbols()
            positions = unwrapped_mol.get_positions()

            if len(symbols) == 1 and symbols[0] in halides:
                anion_positions.append(positions[0])
                continue

            chem_env = ChemicalEnvironment(unwrapped_mol)
            for atom_idx, symbol in enumerate(symbols):
                if symbol != "O":
                    continue

                heavy_neighbors = chem_env._heavy_neighbors(atom_idx)
                if len(heavy_neighbors) != 1:
                    continue

                neighbor_idx = heavy_neighbors[0]
                neighbor_symbol = chem_env.graph.nodes[neighbor_idx].get('symbol', '')
                if (
                    chem_env._is_hypercoordinate_oxo_center(neighbor_idx)
                    or
                    (neighbor_symbol == "S" and chem_env._is_sulfonate_like_S(neighbor_idx))
                    or (neighbor_symbol == "P" and chem_env._is_phosphonate_like_P(neighbor_idx))
                    or (neighbor_symbol == "C" and chem_env._is_carboxylate_like_C(neighbor_idx))
                ):
                    anion_positions.append(positions[atom_idx])

        return self._expand_periodic_images(anion_positions)

    def _collect_cation_positions(self) -> List[np.ndarray]:
        """
        Collect Cartesian positions of simple inorganic cations across the crystal.

        This intentionally covers only single-atom alkali/alkaline-earth metals,
        used for context-aware isolated O defaults (water vs hydroxide).
        """
        cation_positions = []
        simple_cations = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}

        for unwrapped_mol in self.unwrapped_molecules:
            symbols = unwrapped_mol.get_chemical_symbols()
            if len(symbols) == 1 and symbols[0] in simple_cations:
                cation_positions.append(unwrapped_mol.get_positions()[0])

        return self._expand_periodic_images(cation_positions)

    def add_hydrogens(
        self,
        target_elements: Optional[List[str]] = None,
        optimize_torsion: bool = False,
        rules: Optional[List[Dict]] = None,
        bond_lengths: Optional[Dict] = None,
        use_formula_moiety: bool = True,
    ) -> MolecularCrystal:
        """
        Add hydrogen atoms to the crystal.

        The heuristic path computes per-atom H counts and geometries first.
        If enabled and available, CIF _chemical_formula_moiety then corrects
        the per-fragment H count while preserving the heuristic placement
        geometries.

        Parameters
        ----------
        target_elements : Optional[List[str]]
            Whitelist of atomic symbols to hydrogenate (e.g. ``["C", "N", "O"]``).
            When provided, only atoms whose symbol is in this list are processed.
            When ``None`` (the default), all heavy atoms are processed.
        optimize_torsion : bool
            Whether to perform conformational optimization to adjust torsion angles.
            Default is False to preserve ring structures and crystal packing.
        rules : Optional[List[Dict]]
            Override rules for coordination geometry. Format:
            [
                {
                    "symbol": str,              # Required. E.g., "O", "N"
                    "neighbors": List[str],     # Optional. E.g., ["Cl", "S"]. Context condition.
                    "target_coordination": int, # Optional. Override coordination.
                    "geometry": str             # Optional. Override geometry.
                },
                ...
            ]

            Processing Logic:
            1. Specific rules (with neighbors) take priority
            2. General rules (without neighbors) take second priority
            3. Default rules are used if no user rules match
        bond_lengths : Optional[Dict]
            Override bond lengths for specific atom pairs (e.g., "C-H": 1.09).
        use_formula_moiety : bool
            Use CIF _chemical_formula_moiety, when available, to correct
            per-fragment hydrogen counts before placing atoms. Missing,
            unknown, unparseable, or ambiguous moiety values fall back to the
            heuristic-only result for the affected molecule.

        Returns
        -------
        MolecularCrystal
            New crystal with hydrogen atoms added.
        """
        # Process the rules into specific and general categories
        specific_rules = []
        general_rules = {}

        if rules:
            for rule in rules:
                if "neighbors" in rule and rule["neighbors"]:
                    # This is a specific rule with neighbor conditions
                    specific_rules.append(rule)
                else:
                    # This is a general rule without neighbor conditions
                    symbol = rule["symbol"]
                    general_rules[symbol] = {
                        k: v for k, v in rule.items() if k != "symbol"
                    }

        # Merge default and user-provided bond lengths
        effective_bond_lengths = self.default_bond_lengths.copy()
        if bond_lengths:
            effective_bond_lengths.update(bond_lengths)

        # Create new molecules with hydrogen atoms added
        new_molecules = []
        anion_positions = self._collect_anion_positions()
        cation_positions = self._collect_cation_positions()
        moiety_fragments = None
        formula_moiety = getattr(self.crystal, "formula_moiety", None)
        if use_formula_moiety and formula_moiety:
            moiety_fragments = parse_moiety_string(formula_moiety)

        for mol_idx, unwrapped_mol in enumerate(self.unwrapped_molecules):
            # Get atomic symbols and positions
            symbols = unwrapped_mol.get_chemical_symbols()
            positions = unwrapped_mol.get_positions()

            # Create a new ASE Atoms object to add hydrogens to
            # Start with the same atoms and positions
            new_atoms = unwrapped_mol.copy()

            # Create chemical environment analyzer for this molecule
            chem_env = ChemicalEnvironment(
                unwrapped_mol,
                anion_positions=anion_positions,
                cation_positions=cation_positions,
            )

            h_plan = []

            # For each atom in the molecule, compute the hydrogen plan first.
            for atom_idx in range(len(symbols)):
                symbol = symbols[atom_idx]
                if target_elements and symbol not in target_elements:
                    h_plan.append({
                        'num_h': 0,
                        'geometry': 'tetrahedral',
                        'bond_length': effective_bond_lengths.get(f"{symbol}-H", 1.0),
                        'process': False,
                    })
                    continue

                env_stats = chem_env.get_local_geometry_stats(atom_idx)
                site = chem_env.get_site(atom_idx)
                h_strategy = site.get_hydrogen_completion_strategy()
                
                # Check for user rule override
                user_rule = self._find_matching_rule(chem_env, atom_idx, symbol, specific_rules, general_rules)
                if user_rule:
                    if "geometry" in user_rule:
                        h_strategy['geometry'] = user_rule["geometry"]
                    if "target_coordination" in user_rule:
                        current_coord = env_stats['coordination_number']
                        h_strategy['num_h'] = max(0, user_rule["target_coordination"] - current_coord)

                # Check for bond length override
                bond_key = f"{symbol}-H"
                if bond_key in effective_bond_lengths:
                    h_strategy['bond_length'] = effective_bond_lengths[bond_key]

                h_strategy['process'] = True
                h_plan.append(h_strategy)

            if moiety_fragments is not None:
                fragment = match_molecule_to_fragment(symbols, moiety_fragments)
                if fragment is not None:
                    existing_h = symbols.count('H')
                    expected_total_h = fragment.composition.get('H', 0)
                    expected_added_h = expected_total_h - existing_h
                    predicted_added_h = self._plan_h_count(h_plan)

                    if expected_added_h < 0:
                        warnings.warn(
                            "_chemical_formula_moiety expects fewer H atoms "
                            f"({expected_total_h}) than are already present "
                            f"({existing_h}) in fragment {fragment.raw}; "
                            "leaving existing atoms unchanged.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    elif predicted_added_h != expected_added_h:
                        self._redistribute_h(
                            symbols,
                            chem_env,
                            h_plan,
                            expected_added_h,
                            fragment,
                        )
                        adjusted_added_h = self._plan_h_count(h_plan)
                        warnings.warn(
                            "_chemical_formula_moiety drove added-H count from "
                            f"{predicted_added_h} to {adjusted_added_h} "
                            f"for fragment {fragment.raw}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            # Place hydrogens from the final plan.
            for atom_idx, h_strategy in enumerate(h_plan):
                # Extract final values for vector calculation
                num_h = h_strategy['num_h']
                geometry_type = h_strategy['geometry']
                bond_len = h_strategy['bond_length']

                if num_h <= 0:
                    continue  # No hydrogens to add

                # Get the position of the center atom
                center_pos = positions[atom_idx]

                # Get positions of existing neighbors
                neighbors = list(unwrapped_mol.graph.neighbors(atom_idx))
                neighbor_positions = [positions[n] for n in neighbors]

                # Calculate positions for new hydrogen atoms
                missing_vectors = get_missing_vectors(
                    center_pos, neighbor_positions, geometry_type, bond_length=bond_len
                )

                # Add only as many hydrogens as needed
                for i in range(min(num_h, len(missing_vectors))):
                    h_pos = center_pos + missing_vectors[i]

                    # Append the new hydrogen atom
                    new_atoms = new_atoms + Atoms(symbols=["H"], positions=[h_pos])

            # Add the modified molecule to the new molecules list
            new_molecules.append(new_atoms)

        # Create a new crystal with the modified molecules
        new_crystal = MolecularCrystal(
            lattice=self.crystal.lattice,
            molecules=new_molecules,
            pbc=self.crystal.pbc,
            formula_moiety=getattr(self.crystal, "formula_moiety", None),
        )

        # Perform optimization step for staggered conformations only if enabled
        if optimize_torsion:
            new_crystal = self._optimize_conformations(new_crystal, effective_bond_lengths)

        # Wrap the molecules back into the unit cell to handle PBC correctly
        wrapped_molecules = self._wrap_molecules_to_cell(new_crystal.molecules)

        # Create the final crystal with wrapped molecules
        final_crystal = MolecularCrystal(
            lattice=self.crystal.lattice,
            molecules=wrapped_molecules,
            pbc=self.crystal.pbc,
            formula_moiety=getattr(self.crystal, "formula_moiety", None),
        )

        return final_crystal

    @staticmethod
    def _plan_h_count(h_plan: List[Dict]) -> int:
        """Return total H atoms scheduled for addition."""
        return int(sum(max(0, int(item.get('num_h', 0))) for item in h_plan))

    def _redistribute_h(
        self,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
        expected_total: int,
        fragment,
    ) -> None:
        """Adjust a molecule-level H plan to match a moiety fragment H count."""
        expected_total = max(0, int(expected_total))
        current_total = self._plan_h_count(h_plan)

        if current_total < expected_total:
            self._add_h_to_match(symbols, chem_env, h_plan, expected_total)
        elif current_total > expected_total:
            self._remove_h_to_match(symbols, chem_env, h_plan, expected_total)

        final_total = self._plan_h_count(h_plan)
        if final_total != expected_total:
            warnings.warn(
                "Could not fully satisfy _chemical_formula_moiety H count for "
                f"fragment {fragment.raw}: expected {expected_total}, got {final_total}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _add_h_to_match(
        self,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
        expected_total: int,
    ) -> None:
        """Add H to the most chemically plausible underfilled atoms first."""
        while self._plan_h_count(h_plan) < expected_total:
            added = False

            for atom_idx in self._addition_candidates(symbols, chem_env, h_plan):
                if self._can_add_h(atom_idx, symbols, chem_env, h_plan):
                    h_plan[atom_idx]['num_h'] += 1
                    added = True
                    break

            if not added:
                break

    def _remove_h_to_match(
        self,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
        expected_total: int,
    ) -> None:
        """Remove H from likely over-protonated atoms first."""
        while self._plan_h_count(h_plan) > expected_total:
            removed = False

            for atom_idx in self._removal_candidates(symbols, chem_env, h_plan):
                if h_plan[atom_idx].get('num_h', 0) <= 0:
                    continue
                h_plan[atom_idx]['num_h'] -= 1
                removed = True
                break

            if not removed:
                break

    def _addition_candidates(
        self,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> List[int]:
        """Return atom indices ordered by moiety-driven H-addition priority."""
        processable = [
            idx for idx, symbol in enumerate(symbols)
            if symbol != 'H' and h_plan[idx].get('process', True)
        ]

        tertiary_n = [
            idx for idx in processable
            if self._is_underfilled_tertiary_amine(idx, symbols, chem_env, h_plan)
        ]
        amine_n = [
            idx for idx in processable
            if self._is_underfilled_primary_or_secondary_amine(idx, symbols, chem_env, h_plan)
        ]
        alcohol_o = [
            idx for idx in processable
            if self._is_underfilled_alcohol_oxygen(idx, symbols, chem_env, h_plan)
        ]

        used = set(tertiary_n + amine_n + alcohol_o)
        fallback = [idx for idx in processable if idx not in used]
        fallback.sort(
            key=lambda idx: (
                4 - len(chem_env._heavy_neighbors(idx)) - h_plan[idx].get('num_h', 0),
                -self._min_heavy_distance(idx, chem_env),
            ),
            reverse=True,
        )

        return tertiary_n + amine_n + alcohol_o + fallback

    def _removal_candidates(
        self,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> List[int]:
        """Return atom indices ordered by moiety-driven H-removal priority."""
        processable = [
            idx for idx, symbol in enumerate(symbols)
            if (
                symbol != 'H'
                and h_plan[idx].get('process', True)
                and h_plan[idx].get('num_h', 0) > 0
            )
        ]

        hyper_oxo_o = [
            idx for idx in processable
            if self._is_terminal_hyper_oxo_oxygen(idx, symbols, chem_env)
        ]
        used = set(hyper_oxo_o)
        fallback = [idx for idx in processable if idx not in used]
        fallback.sort(key=lambda idx: self._min_heavy_distance(idx, chem_env))

        return hyper_oxo_o + fallback

    def _can_add_h(
        self,
        atom_idx: int,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> bool:
        symbol = symbols[atom_idx]
        if symbol == 'H' or not h_plan[atom_idx].get('process', True):
            return False
        if symbol not in {'C', 'N', 'O', 'S'}:
            return False

        heavy_coord = len(chem_env._heavy_neighbors(atom_idx))
        current_h = h_plan[atom_idx].get('num_h', 0)
        max_total_coord = {'C': 4, 'N': 4, 'O': 3, 'S': 2}.get(symbol, 4)
        return heavy_coord + current_h < max_total_coord

    def _is_underfilled_tertiary_amine(
        self,
        atom_idx: int,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> bool:
        if symbols[atom_idx] != 'N':
            return False

        neighbors = chem_env._heavy_neighbors(atom_idx)
        if len(neighbors) != 3 or h_plan[atom_idx].get('num_h', 0) != 0:
            return False
        if any(chem_env.graph.nodes[nb].get('symbol', '') != 'C' for nb in neighbors):
            return False
        if self._min_heavy_distance(atom_idx, chem_env) < 1.42:
            return False
        return True

    def _is_underfilled_primary_or_secondary_amine(
        self,
        atom_idx: int,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> bool:
        if symbols[atom_idx] != 'N':
            return False

        heavy_coord = len(chem_env._heavy_neighbors(atom_idx))
        current_h = h_plan[atom_idx].get('num_h', 0)
        if heavy_coord not in (1, 2):
            return False
        if heavy_coord + current_h >= 4:
            return False

        # Avoid assigning formula-driven protons to clear imine/nitrile N first.
        return self._min_heavy_distance(atom_idx, chem_env) >= 1.38

    def _is_underfilled_alcohol_oxygen(
        self,
        atom_idx: int,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
        h_plan: List[Dict],
    ) -> bool:
        if symbols[atom_idx] != 'O' or h_plan[atom_idx].get('num_h', 0) != 0:
            return False

        neighbors = chem_env._heavy_neighbors(atom_idx)
        return (
            len(neighbors) == 1
            and chem_env.graph.nodes[neighbors[0]].get('symbol', '') == 'C'
            and not chem_env._is_carboxylate_like_C(neighbors[0])
            and not chem_env._is_carbonyl_like_C(neighbors[0])
        )

    def _is_terminal_hyper_oxo_oxygen(
        self,
        atom_idx: int,
        symbols: List[str],
        chem_env: ChemicalEnvironment,
    ) -> bool:
        if symbols[atom_idx] != 'O':
            return False

        neighbors = chem_env._heavy_neighbors(atom_idx)
        return (
            len(neighbors) == 1
            and chem_env._is_hypercoordinate_oxo_center(neighbors[0])
        )

    def _min_heavy_distance(self, atom_idx: int, chem_env: ChemicalEnvironment) -> float:
        center = chem_env.positions[atom_idx]
        distances = [
            float(np.linalg.norm(chem_env.positions[nb] - center))
            for nb in chem_env._heavy_neighbors(atom_idx)
        ]
        return min(distances) if distances else float('inf')

    def _wrap_molecules_to_cell(self, molecules: List[Atoms]) -> List[Atoms]:
        """
        Wrap molecules back into the unit cell to handle PBC correctly.

        Parameters
        ----------
        molecules : List[Atoms]
            List of molecules to wrap.

        Returns
        -------
        List[Atoms]
            List of wrapped molecules.
        """
        wrapped_molecules = []

        for mol in molecules:
            # Calculate the center of mass of the molecule
            symbols = mol.get_chemical_symbols()
            positions = mol.get_positions()

            com = calculate_center_of_mass(positions, symbols)

            # Convert COM to fractional coordinates
            frac_com = cart_to_frac(com, self.crystal.lattice)

            # Determine the shift integer
            shift = np.floor(frac_com)

            # Convert shift back to Cartesian coordinates
            shift_cart = frac_to_cart(shift, self.crystal.lattice)

            # Shift all atoms in the molecule by the same amount
            new_positions = positions - shift_cart

            # Create a new molecule with shifted positions
            new_mol = mol.copy()
            new_mol.set_positions(new_positions)

            wrapped_molecules.append(new_mol)

        return wrapped_molecules

    def _optimize_conformations(
        self, crystal: MolecularCrystal, bond_lengths: Dict
    ) -> MolecularCrystal:
        """
        Adjust dihedral angles around sp3–sp3 single bonds to favour
        staggered conformations.

        Only bonds between atoms in ``["C", "N", "O", "S", "P"]`` are
        considered.  Rotations smaller than 5° are skipped.  Ring bonds
        are not explicitly excluded, so this helper should only be called
        when ``optimize_torsion=True`` and the user accepts that caveat.
        """
        # Get unwrapped molecules for this crystal
        unwrapped_mols = crystal.get_unwrapped_molecules()
        new_molecules = []

        for mol_idx, unwrapped_mol in enumerate(unwrapped_mols):
            original_mol = crystal.molecules[mol_idx]
            symbols = original_mol.get_chemical_symbols()
            positions = original_mol.get_positions()

            # Create a copy of the original molecule
            new_mol = original_mol.copy()

            # Get the connectivity graph
            graph = unwrapped_mol.graph

            # Look for pairs of sp3 atoms connected by a single bond
            # For now, we'll look for C-N, C-C, N-N bonds where both atoms have H
            sp3_elements = ["C", "N", "O", "S", "P"]  # Common sp3 hybridized atoms

            for atom1, atom2 in graph.edges():
                elem1 = symbols[atom1]
                elem2 = symbols[atom2]

                # Check if both atoms are sp3 hybridized
                if elem1 in sp3_elements and elem2 in sp3_elements:
                    # Get neighbors of both atoms
                    neighbors1 = [n for n in graph.neighbors(atom1)]
                    neighbors2 = [n for n in graph.neighbors(atom2)]

                    # Get positions of neighbors for dihedral calculation
                    pos1 = positions[atom1]
                    pos2 = positions[atom2]

                    # Calculate the adjustment needed for optimal dihedral angle
                    adjustment = calculate_dihedral_and_adjustment(
                        pos1,
                        pos2,
                        [positions[n] for n in neighbors1 if n != atom2],
                        [positions[n] for n in neighbors2 if n != atom1],
                    )

                    # If adjustment is significant, rotate the attached H atoms
                    if abs(adjustment) > 5:  # Only if significant adjustment needed
                        # Rotate H atoms connected to atom1 around the bond axis
                        self._rotate_hydrogens_around_bond(
                            new_mol, atom1, atom2, adjustment
                        )

            new_molecules.append(new_mol)

        # Create new crystal with optimized conformations
        optimized_crystal = MolecularCrystal(
            lattice=crystal.lattice,
            molecules=new_molecules,
            pbc=crystal.pbc,
            formula_moiety=getattr(crystal, "formula_moiety", None),
        )

        return optimized_crystal

    def _rotate_hydrogens_around_bond(
        self, mol: Atoms, atom1_idx: int, atom2_idx: int, angle_deg: float
    ):
        """
        Rotate hydrogen atoms connected to atom1 around the atom1-atom2 bond.
        """
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()

        # Find hydrogen atoms connected to atom1
        h_indices = []
        for atom_idx, symbol in enumerate(symbols):
            if symbol == "H":
                # Check if this hydrogen is connected to atom1 by checking distance
                dist = mol.get_distance(atom1_idx, atom_idx, mic=False)
                if dist < 1.5:  # Typical H-X distance is less than 1.5 Å
                    h_indices.append(atom_idx)

        if not h_indices:
            return  # No hydrogens to rotate

        # Calculate the rotation axis (bond vector from atom1 to atom2)
        bond_vector = positions[atom2_idx] - positions[atom1_idx]
        rotation_axis = normalize_vector(bond_vector)

        # Rotate each hydrogen atom around the bond axis
        new_positions = positions.copy()
        for h_idx in h_indices:
            # Vector from atom1 to hydrogen
            h_vector = positions[h_idx] - positions[atom1_idx]

            # Rotate the vector around the bond axis using the utility function
            rotated_h_vector = rotate_vector(h_vector, rotation_axis, angle_deg)

            # Update the hydrogen position
            new_positions[h_idx] = positions[atom1_idx] + rotated_h_vector

        mol.set_positions(new_positions)