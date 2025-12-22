"""
Output functionality for molecular crystal structures.

This module provides functions for writing molecular crystal data to various formats.
"""

from ..structures.molecule import CrystalMolecule
from ..structures.crystal import MolecularCrystal


def write_xyz(molecule: CrystalMolecule, filename: str = None) -> str:
    """
    Write a molecule to XYZ format.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to write.
    filename : str, optional
        The filename to write to. If None, returns the XYZ string.

    Returns
    -------
    str
        XYZ format string if filename is None, otherwise None.
    """
    # Get atomic symbols and positions
    symbols = molecule.get_chemical_symbols()
    positions = molecule.get_positions()

    # Create XYZ format string
    xyz_lines = [str(len(symbols)), ""]  # Number of atoms and empty comment line
    for symbol, pos in zip(symbols, positions):
        xyz_lines.append(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

    xyz_string = "\n".join(xyz_lines) + "\n"

    if filename is None:
        return xyz_string
    else:
        with open(filename, "w") as f:
            f.write(xyz_string)


def write_molecule_summary(molecule: CrystalMolecule) -> str:
    """
    Generate a summary string for a molecule.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to summarize.

    Returns
    -------
    str
        Summary string with molecular information.
    """
    summary = []
    summary.append(f"Molecule: {molecule.get_chemical_formula()}")
    summary.append(f"Number of atoms: {len(molecule)}")

    if hasattr(molecule, "crystal") and molecule.crystal is not None:
        centroid = molecule.get_centroid()
        centroid_frac = molecule.get_centroid_frac()
        summary.append(
            f"Centroid (Cartesian): ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})"
        )
        summary.append(
            f"Centroid (Fractional): ({centroid_frac[0]:.4f}, {centroid_frac[1]:.4f}, {centroid_frac[2]:.4f})"
        )

    com = molecule.get_center_of_mass()
    summary.append(f"Center of mass: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")

    # Get ellipsoid radii
    radii = molecule.get_ellipsoid_radii()
    summary.append(f"Ellipsoid radii: {radii[0]:.3f} × {radii[1]:.3f} × {radii[2]:.3f}")

    return "\n".join(summary)


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

    # Add atom positions
    lines.append("STRUC")
    atom_index = 1
    for mol in crystal.molecules:
        for atom in mol.atoms:
            lines.append(
                f"  {atom_index:4d} {atom.symbol:4s} {atom_index:4d}  1.0000  {atom.frac_coords[0]:8.5f}  {atom.frac_coords[1]:8.5f}  {atom.frac_coords[2]:8.5f}  1a  1  0  0  0  0"
            )
            atom_index += 1
    lines.append("  0 0 0 0 0 0 0")
    lines.append("")

    vesta_content = "\n".join(lines) + "\n"

    if filename:
        with open(filename, "w") as f:
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
