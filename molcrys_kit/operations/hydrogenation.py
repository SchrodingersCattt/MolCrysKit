"""
Hydrogenation operations for molecular crystals.

This module provides functionality to add hydrogen atoms to molecular crystals
based on geometric rules and chemical constraints.
"""

import numpy as np
from typing import Dict, List, Optional
from ase import Atoms
from ..structures.crystal import MolecularCrystal
from ..analysis.chemical_env import ChemicalEnvironment
from ..utils.geometry import (
    get_missing_vectors,
    calculate_dihedral_and_adjustment,
    normalize_vector,
    cart_to_frac,
    frac_to_cart,
    rotate_vector,
    calculate_center_of_mass,
)


def add_hydrogens(crystal, target_elements: Optional[List[str]] = None, optimize_torsion: bool = False, rules=None, bond_lengths=None):
    """
    Add hydrogen atoms to a molecular crystal based on geometric rules.

    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to hydrogenate.
    target_elements : Optional[List[str]]
        List of atomic symbols to specifically hydrogenate. Only these elements
        will have hydrogens added. If None or empty, no elements will be targeted.
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
        Override bond lengths for specific atom pairs.

    Returns
    -------
    MolecularCrystal
        New crystal with hydrogen atoms added.
    """
    h_completer = HydrogenCompleter(crystal)
    return h_completer.add_hydrogens(target_elements=target_elements, optimize_torsion=optimize_torsion, rules=rules, bond_lengths=bond_lengths)


class HydrogenCompleter:
    """
    Class for adding hydrogen atoms to molecular crystals based on geometric rules.
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

    def add_hydrogens(
        self, target_elements: Optional[List[str]] = None, optimize_torsion: bool = False, rules: Optional[List[Dict]] = None, bond_lengths: Optional[Dict] = None
    ) -> MolecularCrystal:
        """
        Add hydrogen atoms to the crystal based on geometric rules.

        Parameters
        ----------
        target_elements : Optional[List[str]]
            List of atomic symbols to specifically hydrogenate. Only these elements
            will have hydrogens added. If None or empty, no elements will be targeted.
            This implements a whitelist-only approach for explicit targeting.
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

        for mol_idx, unwrapped_mol in enumerate(self.unwrapped_molecules):
            # Get atomic symbols and positions
            symbols = unwrapped_mol.get_chemical_symbols()
            positions = unwrapped_mol.get_positions()

            # Create a new ASE Atoms object to add hydrogens to
            # Start with the same atoms and positions
            new_atoms = unwrapped_mol.copy()

            # Create chemical environment analyzer for this molecule
            chem_env = ChemicalEnvironment(unwrapped_mol)

            # For each atom in the molecule, check if it needs hydrogens
            for atom_idx in range(len(symbols)):
                symbol = symbols[atom_idx]
                if target_elements and symbol not in target_elements:
                    continue

                # Get local geometry stats and ring info
                env_stats = chem_env.get_local_geometry_stats(atom_idx)
                ring_info = chem_env.detect_ring_info(atom_idx)
                # --- DEBUG START ---
                if symbol == "C":
                    neighbors = list(unwrapped_mol.graph.neighbors(atom_idx))
                    dists = [np.linalg.norm(positions[n] - positions[atom_idx]) for n in neighbors]
                    print(f"\n--- Checking Carbon {atom_idx} ---")
                    print(f"Position: {positions[atom_idx]}")
                    print(f"Coordination Number: {len(neighbors)}")
                    print(f"Neighbor Distances: {dists}")
                    print(f"Angle Sum: {env_stats['bond_angle_sum']:.2f}")
                    
                    # 模拟运行 heuristics
                    if len(neighbors) == 3 and env_stats['bond_angle_sum'] < 345.0:
                        print("Prediction: Should add 1 H (sp3)")
                    elif len(neighbors) == 4:
                        print("Prediction: Saturated (4 neighbors), adds 0 H")
                    else:
                        print(f"Prediction: Unsure state (Coord={len(neighbors)})")
                # --- DEBUG END ---
                # Determine hydrogenation strategy using the new API
                site = chem_env.get_site(atom_idx)
                h_strategy = site.get_hydrogenation_strategy()
                
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
            lattice=self.crystal.lattice, molecules=new_molecules, pbc=self.crystal.pbc
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
        )

        return final_crystal

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
        Optimize conformations to achieve staggered arrangements where possible.
        """
        # For now, we'll implement a basic version that adjusts dihedral angles
        # between sp3 centers connected by single bonds

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
            lattice=crystal.lattice, molecules=new_molecules, pbc=crystal.pbc
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