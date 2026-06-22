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

from collections.abc import Mapping, Sequence
from typing import Any, List, Union

import ase.io
from ase import Atoms
import numpy as np

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
    info: Union[Mapping[str, Any], Sequence[Mapping[str, Any]], None] = None,
    arrays: Union[Mapping[str, Any], Sequence[Mapping[str, Any]], None] = None,
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
    info:
        Custom per-frame ExtXYZ header fields.  A single mapping is applied to
        every frame; a sequence of mappings is matched frame-by-frame.
    arrays:
        Custom per-atom arrays to write as ExtXYZ ``Properties`` columns.  A
        single mapping is applied to every frame; a sequence of mappings is
        matched frame-by-frame.  Each array length must equal the atom count of
        the corresponding frame.
    **write_kwargs:
        Forwarded to :func:`ase.io.write` (``format="extxyz"``).
        Useful keys include ``write_info=True`` (default), ``write_results=True``,
        ``columns``, and ``plain``.

    Notes
    -----
    For dataset bundles, store per-frame provenance (for example ``refcode``,
    ``motif``, ``dataset_id``, and ``frame_index``) in ``info``.  Read bundles
    with ``read_extxyz(path, index=":")``; the default ``index=None`` follows
    ASE convention and returns only the last frame.
    """
    if isinstance(crystals, MolecularCrystal):
        crystals = [crystals]

    images = [c.to_ase() for c in crystals]
    frame_infos = _normalise_frame_payload(info, len(images), "info")
    frame_arrays = _normalise_frame_payload(arrays, len(images), "arrays")

    for atoms, info_payload, arrays_payload in zip(images, frame_infos, frame_arrays):
        if info_payload:
            atoms.info.update(dict(info_payload))
        if arrays_payload:
            for name, values in arrays_payload.items():
                array = np.asarray(values)
                if len(array) != len(atoms):
                    raise ValueError(
                        f"Custom array {name!r} has length {len(array)}; "
                        f"expected {len(atoms)} for this frame."
                    )
                atoms.set_array(name, array)

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


def _normalise_frame_payload(payload, n_frames: int, name: str) -> list[dict]:
    """Normalise a custom payload into one mapping per frame."""
    if payload is None:
        return [{} for _ in range(n_frames)]

    if isinstance(payload, Mapping):
        return [dict(payload) for _ in range(n_frames)]

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if len(payload) != n_frames:
            raise ValueError(
                f"{name} sequence has length {len(payload)}; expected {n_frames}."
            )
        return [dict(item) for item in payload]

    raise TypeError(
        f"{name} must be a mapping, a sequence of mappings, or None "
        f"(got {type(payload).__name__})."
    )
