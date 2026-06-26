"""
Input/Output module for MolCrysKit.

This module handles reading and writing of molecular crystal data.
"""

from .cif import (
    DisorderInfo,
    identify_molecule_indices,
    parse_cif_advanced,
    read_mol_crystal,
    scan_cif_disorder,
)
from .extxyz import read_extxyz, write_extxyz
from .output import (
    write_cif,
    write_cif_sequence,
    write_poscar,
    write_poscar_sequence,
    write_trajectory,
    write_xyz,
    write_xyz_with_freeze,
)
from .poscar import read_poscar
from .xyz import read_xyz

__all__ = [
    "DisorderInfo",
    "read_mol_crystal",
    "parse_cif_advanced",
    "identify_molecule_indices",
    "scan_cif_disorder",
    "write_cif",
    "write_cif_sequence",
    "write_poscar",
    "write_poscar_sequence",
    "write_trajectory",
    "write_xyz",
    "write_xyz_with_freeze",
    "read_xyz",
    "read_poscar",
    "read_extxyz",
    "write_extxyz",
]
