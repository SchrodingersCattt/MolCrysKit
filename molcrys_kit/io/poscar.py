"""
VASP POSCAR/CONTCAR file parsing for molecular crystals.

POSCAR stores periodic structure data but does not carry MolCrysKit-specific
metadata such as occupancies, disorder groups, assemblies, or CIF atom labels.
Those fields are therefore populated with defaults when the ASE structure is
wrapped in a MolecularCrystal.
"""

from typing import Dict, Optional, Tuple

from ase.io import read as ase_read

from ..structures.crystal import MolecularCrystal


def read_poscar(
    filepath: str,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    max_atoms: Optional[int] = None,
    bond_scale: float = 1.0,
) -> MolecularCrystal:
    """
    Read a VASP POSCAR/CONTCAR file and return a MolecularCrystal object.

    Uses ASE's VASP reader for robust parsing of scale factors, Direct versus
    Cartesian coordinates, VASP 4/5 element headers, and selective dynamics
    syntax. POSCAR does not encode bonds, so molecular units are identified
    using MolCrysKit's graph-based bonding pipeline. The resulting molecules
    follow the normal MolCrysKit convention of unwrapped, contiguous molecular
    coordinates.

    Parameters
    ----------
    filepath : str
        Path to the POSCAR/CONTCAR file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as
        values. Keys should be tuples of element symbols, e.g. ``("H", "O")``.
    max_atoms : int, optional
        Optional maximum molecule size passed to molecule identification.

    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units. Occupancy,
        disorder, assembly, and atom-label metadata are defaulted because VASP
        POSCAR cannot represent them.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed as a valid VASP POSCAR/CONTCAR file.
    """
    try:
        atoms = ase_read(filepath, format="vasp")
    except FileNotFoundError:
        raise FileNotFoundError(f"POSCAR file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to parse POSCAR file '{filepath}': {e}")

    atoms.set_pbc(True)
    return MolecularCrystal.from_ase(
        atoms,
        bond_thresholds=bond_thresholds,
        max_atoms=max_atoms,
        bond_scale=bond_scale,
    )
