"""
Desolvation (solvent removal) module for molecular crystals.

This module provides functionality for removing common solvents from molecular crystals.
Only explicit removal mode is supported, where the user specifies what to remove by name or formula.
"""

from typing import List
from ..structures.crystal import MolecularCrystal
from ..analysis.stoichiometry import StoichiometryAnalyzer
from ..constants.config import COMMON_SOLVENTS


class Desolvator:
    """
    A class to handle desolvation (removal of solvents) from molecular crystals.
    
    Only supports explicit mode where the user specifies what to remove by Name or Formula.
    """

    @staticmethod
    def remove_solvents(crystal: MolecularCrystal, targets: List[str]) -> MolecularCrystal:
        """
        Remove specified solvents from the crystal.

        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to desolvate.
        targets : List[str]
            A list of strings. Each string can be a Name (e.g., "Water", "DMF") 
            OR a Formula (e.g., "H2O", "C3H7NO").

        Returns
        -------
        MolecularCrystal
            A new crystal object with specified solvents removed.

        Raises
        ------
        ValueError
            If the resulting crystal is empty after desolvation.
        """
        # Parse targets to build a set of "target formulas" and "target heavy formulas"
        target_formulas = set()
        target_heavy_formulas = set()

        for target in targets:
            # Check if target matches a key in COMMON_SOLVENTS
            if target in COMMON_SOLVENTS:
                solvent_info = COMMON_SOLVENTS[target]
                target_formulas.add(solvent_info["formula"])
                target_heavy_formulas.add(solvent_info["heavy_formula"])
            else:
                # Treat the target string directly as a formula to remove
                target_formulas.add(target)

        # Identify species to remove based on stoichiometry analysis
        analyzer = StoichiometryAnalyzer(crystal)
        
        # Find molecules to remove
        molecules_to_remove = set()
        
        for species_id, molecule_indices in analyzer.species_map.items():
            # Extract formula from species ID (before the underscore and number)
            formula_parts = species_id.split("_")
            formula = "_".join(formula_parts[:-1])
            
            # Check if this formula matches any target
            should_remove = (
                formula in target_formulas or 
                (formula in target_heavy_formulas and 'H' not in formula)  # Heavy formula match requires no hydrogen
            )
            
            if should_remove:
                molecules_to_remove.update(molecule_indices)

        # Create a new crystal excluding the marked molecules
        remaining_molecules = [
            mol for idx, mol in enumerate(crystal.molecules) 
            if idx not in molecules_to_remove
        ]

        if not remaining_molecules:
            raise ValueError("Resulting crystal is empty after removing specified solvents.")

        # Create new crystal with remaining molecules
        new_crystal = MolecularCrystal(
            lattice=crystal.lattice,
            molecules=remaining_molecules,
            pbc=crystal.pbc
        )

        return new_crystal


def remove_solvents(crystal: MolecularCrystal, targets: List[str]) -> MolecularCrystal:
    """
    Public API function to remove specified solvents from a crystal.

    Parameters
    ----------
    crystal : MolecularCrystal
        The molecular crystal to desolvate.
    targets : List[str]
        A list of strings. Each string can be a Name (e.g., "Water", "DMF") 
        OR a Formula (e.g., "H2O", "C3H7NO").

    Returns
    -------
    MolecularCrystal
        A new crystal object with specified solvents removed.
    """
    return Desolvator.remove_solvents(crystal, targets)