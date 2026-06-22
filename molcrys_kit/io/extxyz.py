"""
Extended XYZ (extxyz) I/O for molecular crystals.

Provides first-class read / write support for the ASE Extended XYZ format,
which carries per-frame lattice vectors, periodic-boundary flags, and
arbitrary key–value metadata in the comment line.

All functions round-trip through :meth:`MolecularCrystal.to_ase` and
:meth:`MolecularCrystal.from_ase_atoms`, so molecule partitioning is
preserved exactly.
"""

from __future__ import annotations

from typing import List, Union

import ase.io
from ase import Atoms

from ..structures.crystal import MolecularCrystal


__all__ = [
    "read_extxyz",
    "write_extxyz",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def write_extxyz(
    crystals: Union[MolecularCrystal, List[MolecularCrystal]],
    filepath: str,
    *,
    append: bool = False,
    **write_kwargs,
) -> None:
    """Write one or more molecular crystals to an Extended XYZ file.

    Each crystal is flattened to a single ASE :class:`~ase.Atoms` frame via
    :meth:`MolecularCrystal.to_ase`.  Lattice vectors, PBC flags, molecule
    indices, and disorder metadata are stored as per-frame arrays and
    ``atoms.info`` entries so the crystal can be reconstructed losslessly
    with :func:`read_extxyz`.

    Parameters
    ----------
    crystals:
        A single :class:`MolecularCrystal` or a list of them.
    filepath:
        Path to the output file (``.xyz`` extension recommended).
    append:
        If ``True``, append frames to *filepath* instead of overwriting.
    **write_kwargs:
        Forwarded to :func:`ase.io.write` (``format="extxyz"``).
        Useful keys include ``write_info=True`` (default), ``write_results=True``,
        ``columns``, and ``plain``.
    """
    if isinstance(crystals, MolecularCrystal):
        crystals = [crystals]

    images = [c.to_ase() for c in crystals]

    ase.io.write(
        filepath,
        images,
        format="extxyz",
        append=append,
        write_info=write_kwargs.pop("write_info", True),
        write_results=write_kwargs.pop("write_results", True),
        **write_kwargs,
    )


def read_extxyz(
    filepath: str,
    index: Union[int, slice, str, None] = None,
) -> Union[MolecularCrystal, List[MolecularCrystal]]:
    """Read one or more molecular crystals from an Extended XYZ file.

    This is a thin wrapper around :func:`ase.io.read` (``format="extxyz"``)
    that converts every returned :class:`~ase.Atoms` frame into a
    :class:`MolecularCrystal`.

    Parameters
    ----------
    filepath:
        Path to the input file.
    index:
        Frame selector (ASE conventions):

        * ``int``  – return a single :class:`MolecularCrystal`.
        * ``slice`` – return a ``list[MolecularCrystal]``.
        * ``":"``   – return all frames as a list.
        * ``None``  – return the **last** frame (single crystal).

    Returns
    -------
    MolecularCrystal or list[MolecularCrystal]
    """
    # Normalise None → -1 (ASE convention: last frame, single Atoms).
    _index = -1 if index is None else index
    raw = ase.io.read(filepath, index=_index, format="extxyz")

    if raw is None:
        raise ValueError(f"No frames found in {filepath}")

    if isinstance(raw, Atoms):
        return MolecularCrystal.from_ase_atoms(raw)

    # list of Atoms
    return [MolecularCrystal.from_ase_atoms(atoms) for atoms in raw]
