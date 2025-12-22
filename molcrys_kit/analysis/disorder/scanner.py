"""
Disorder site scanning.

This module scans molecular crystals for disordered atomic sites.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict

try:
    from ase import Atoms

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder

from ..structures.crystal import MolecularCrystal


def scan_disordered_atoms(crystal: MolecularCrystal) -> List[Atoms]:
    """
    Scan the crystal for atoms with partial occupancy.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to scan.

    Returns
    -------
    List[Atoms]
        List of atoms with occupancy < 1.0.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is required for disorder scanning. Please install it with 'pip install ase'"
        )

    disordered_atoms = []

    for molecule in crystal.molecules:
        # Filter atoms with partial occupancy
        partial_occupancy_atoms = []
        for atom in molecule:
            if hasattr(atom, "occupancy") and atom.occupancy < 1.0:
                partial_occupancy_atoms.append(atom)
            elif not hasattr(atom, "occupancy"):
                # Default to full occupancy
                atom.occupancy = 1.0
                partial_occupancy_atoms.append(atom)

        if partial_occupancy_atoms:
            # Create a new Atoms object with only the partially occupied atoms
            symbols = [atom.symbol for atom in partial_occupancy_atoms]
            positions = [atom.position for atom in partial_occupancy_atoms]
            occupancies = [atom.occupancy for atom in partial_occupancy_atoms]

            disordered_molecule = Atoms(symbols=symbols, positions=positions)
            # Store occupancies as an array attribute
            disordered_molecule.set_array("occupancies", np.array(occupancies))
            disordered_atoms.append(disordered_molecule)

    return disordered_atoms


def group_disordered_atoms(crystal: MolecularCrystal) -> Dict[str, List[Atoms]]:
    """
    Group disordered atoms by site.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.

    Returns
    -------
    Dict[str, List[Atoms]]
        Dictionary mapping site identifiers to lists of atoms.
    """
    disordered = scan_disordered_atoms(crystal)

    # Group by rounded fractional coordinates
    site_groups = defaultdict(list)

    for molecule in disordered:
        for atom in molecule:
            # Create a site identifier based on rounded coordinates
            coords = atom.position
            site_id = f"{coords[0]:.3f}_{coords[1]:.3f}_{coords[2]:.3f}"
            site_groups[site_id].append(atom)

    return dict(site_groups)


def enumerate_disorder_configurations(
    crystal: MolecularCrystal, max_configurations: int = 100
) -> List[MolecularCrystal]:
    """
    Enumerate possible disorder configurations.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to enumerate configurations for.
    max_configurations : int, default=100
        Maximum number of configurations to generate.

    Returns
    -------
    List[MolecularCrystal]
        List of possible crystal configurations.
    """
    # This is a simplified implementation
    # A full implementation would systematically enumerate all valid combinations

    configurations = []

    # For demonstration, we'll just return the original crystal
    # A real implementation would generate multiple configurations
    configurations.append(crystal)

    return configurations[:max_configurations]
