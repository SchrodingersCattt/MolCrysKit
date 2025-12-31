"""
Species recognition for molecular crystals.

This module identifies discrete molecular units in periodic crystals.
"""

from typing import List

from ase import Atoms

from ..structures.crystal import MolecularCrystal


def identify_molecules(crystal: MolecularCrystal) -> List[Atoms]:
    """
    Identify discrete molecular units in a crystal.

    NOTE: In the current design, molecules are already ASE Atoms objects after parsing with
    parse_cif_advanced(). This function simply returns the molecules list. If you parsed
    your CIF file with parse_cif(), you should first use a molecular identification method
    or re-parse with parse_cif_advanced().

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.

    Returns
    -------
    List[Atoms]
        List of identified molecular units as ASE Atoms objects.
    """

    # For the new design, molecules are already ASE Atoms objects
    # NOTE: If you used parse_cif() instead of parse_cif_advanced(), all atoms might be in one molecule
    return crystal.molecules


def assign_atoms_to_molecules(crystal: MolecularCrystal) -> MolecularCrystal:
    """
    Reorganize a crystal by assigning atoms to discrete molecular units.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to reorganize.

    Returns
    -------
    MolecularCrystal
        New crystal with atoms organized into molecular units.
    """
    # For the new design, this function is not needed as molecules are already properly organized
    return crystal
