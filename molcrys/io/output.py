"""
Output functionality for molecular crystals.

This module provides functionality to export MolecularCrystal objects to various formats.
"""

import numpy as np
from typing import List
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


def write_xyz(molecule: Molecule, filename: str = None) -> str:
    """
    Export a molecule to XYZ format.
    
    Parameters
    ----------
    molecule : Molecule
        The molecule to export.
    filename : str, optional
        If provided, write to this file. Otherwise, return the XYZ string.
        
    Returns
    -------
    str
        XYZ format string if filename is not provided, otherwise None.
    """
    lines = [f"{len(molecule.atoms)}", ""]  # Number of atoms and comment line
    
    for atom in molecule.atoms:
        # Convert fractional to Cartesian coordinates (assuming a default unit cell for visualization)
        # In a real implementation, we would use the actual lattice
        cart_coords = atom.frac_coords  # Simplified for now
        lines.append(f"{atom.symbol} {cart_coords[0]:.6f} {cart_coords[1]:.6f} {cart_coords[2]:.6f}")
    
    xyz_content = "\n".join(lines) + "\n"
    
    if filename:
        with open(filename, 'w') as f:
            f.write(xyz_content)
        return None
    else:
        return xyz_content


def write_vesta(crystal: MolecularCrystal, filename: str = None) -> str:
    """
    Generate VESTA format representation of a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to export.
    filename : str, optional
        If provided, write to this file. Otherwise, return the VESTA string.
        
    Returns
    -------
    str
        VESTA format string if filename is not provided, otherwise None.
    """
    lines = ["#VESTA_FORMAT_VERSION 3.5.0", ""]  # Header
    
    # Add crystal structure information
    lines.append("CRYSTAL")
    lines.append("INFO")
    lines.append("LEN a=1.0 b=1.0 c=1.0")
    lines.append("ANG alpha=90.0 beta=90.0 gamma=90.0")
    lines.append("SPGP I1")
    lines.append("SPGN 1")
    lines.append("")
    
    # Add lattice information
    lines.append("CELLP")
    a, b, c = crystal.lattice[0], crystal.lattice[1], crystal.lattice[2]
    lines.append(f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}")
    lines.append(f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f}")
    lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("  0.000000   0.000000   0.000000")
    lines.append("")
    
    # Count total atoms
    total_atoms = sum(len(mol.atoms) for mol in crystal.molecules)
    
    # Add atom positions
    lines.append("STRUC")
    atom_index = 1
    for mol in crystal.molecules:
        for atom in mol.atoms:
            lines.append(f"  {atom_index:4d} {atom.symbol:4s} {atom_index:4d}  1.0000  {atom.frac_coords[0]:8.5f}  {atom.frac_coords[1]:8.5f}  {atom.frac_coords[2]:8.5f}  1a  1  0  0  0  0")
            atom_index += 1
    lines.append("  0 0 0 0 0 0 0")
    lines.append("")
    
    vesta_content = "\n".join(lines) + "\n"
    
    if filename:
        with open(filename, 'w') as f:
            f.write(vesta_content)
        return None
    else:
        return vesta_content


def export_for_vesta(crystal: MolecularCrystal, filename: str) -> None:
    """
    Export a crystal structure to a VESTA-compatible file.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to export.
    filename : str
        Output filename.
    """
    write_vesta(crystal, filename)