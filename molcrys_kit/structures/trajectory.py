"""
CrystalTrajectory — multi-frame container for molecular crystal sequences.

Provides a lightweight factory-style interface (mirroring ASE's ``Trajectory``)
for reading and writing sequences of :class:`~molcrys_kit.structures.crystal.MolecularCrystal`
frames stored in Extended XYZ format.

Typical usage:

    >>> from molcrys_kit.structures.trajectory import CrystalTrajectory

    # Write
    >>> with CrystalTrajectory("sweep.xyz", "w") as traj:
    ...     for angle in range(0, 360, 30):
    ...         rotated = rotate_crystal(crystal, axis=[0,0,1], angle=angle)
    ...         traj.write(rotated, rotation_angle=angle)

    # Read
    >>> traj = CrystalTrajectory("sweep.xyz", "r")
    >>> len(traj)
    12
    >>> frame_0 = traj[0]
    >>> for frame in traj:
    ...     ...
"""

from __future__ import annotations

from typing import Any, Iterator, List, Mapping, Union

import ase.io
from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

from .crystal import MolecularCrystal


__all__ = ["CrystalTrajectory"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def CrystalTrajectory(
    filename: str,
    mode: str = "r",
):
    """Factory for reading / writing crystal trajectories in ExtXYZ format.

    Parameters
    ----------
    filename:
        Path to the ``.xyz`` file.
    mode:
        ``"r"`` → :class:`CrystalTrajectoryReader`
        ``"w"`` → :class:`CrystalTrajectoryWriter`
        ``"a"`` → :class:`CrystalTrajectoryWriter` (append mode)
    """
    if mode == "r":
        return CrystalTrajectoryReader(filename)
    if mode in ("w", "a"):
        append = mode == "a"
        return CrystalTrajectoryWriter(filename, append=append)
    raise ValueError(f"Unknown mode {mode!r}; expected 'r', 'w', or 'a'")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class CrystalTrajectoryReader:
    """Read-only access to a sequence of molecular-crystal frames.

    Frames are loaded eagerly on construction for stable ``len()`` and
    inexpensive repeated indexing.

    Parameters
    ----------
    filename:
        Path to the ExtXYZ file.
    """

    def __init__(self, filename: str):
        self._filename = filename
        # Load all frames eagerly for consistent __len__.
        # For very large files, consider a lazy index in a future version.
        raw = ase.io.read(filename, index=":", format="extxyz")
        if raw is None:
            self._frames: List[MolecularCrystal] = []
        elif isinstance(raw, Atoms):
            self._frames = [MolecularCrystal.from_ase_atoms(raw)]
        else:
            self._frames = [MolecularCrystal.from_ase_atoms(a) for a in raw]

    # -- context manager --------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # nothing to close; file is read once and closed by ASE

    # -- sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[MolecularCrystal, List[MolecularCrystal]]:
        return self._frames[index]

    def __iter__(self) -> Iterator[MolecularCrystal]:
        return iter(self._frames)

    def __repr__(self) -> str:
        return (
            f"<CrystalTrajectoryReader '{self._filename}': "
            f"{len(self._frames)} frames>"
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class CrystalTrajectoryWriter:
    """Write a sequence of molecular-crystal frames to an ExtXYZ file.

    Each :meth:`write` call appends one :class:`MolecularCrystal` frame.
    Extra keyword arguments (``energy``, ``forces``, ``stress``, etc.) are
    attached to the frame via :class:`~ase.calculators.singlepoint.SinglePointCalculator`.

    Parameters
    ----------
    filename:
        Path to the output ``.xyz`` file.
    append:
        If ``True``, append to an existing file instead of overwriting.
    """

    def __init__(self, filename: str, append: bool = False):
        self._filename = filename
        self._append = append
        self._n_written = 0
        self._first = True

    # -- context manager --------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # ASE write flushes immediately; no file handle to close

    # -- write ------------------------------------------------------------

    def write(
        self,
        crystal: MolecularCrystal,
        *,
        info: Mapping[str, Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Append a single crystal frame.

        Parameters
        ----------
        crystal:
            The :class:`MolecularCrystal` frame to write.
        info:
            Custom per-frame ExtXYZ header fields.
        arrays:
            Custom per-atom ExtXYZ ``Properties`` arrays.
        **kwargs:
            ASE calculator results (``energy``, ``forces``, ``stress``, ...)
            are stored as a :class:`~ase.calculators.singlepoint.SinglePointCalculator`.
            Unknown keyword arguments are treated as additional per-frame
            metadata fields for convenience, e.g. ``rotation_angle=30``.
        """
        atoms = crystal.to_ase()

        calc_kwargs = {k: v for k, v in kwargs.items() if k in all_properties}
        extra_info = {k: v for k, v in kwargs.items() if k not in all_properties}
        if info:
            atoms.info.update(dict(info))
        if extra_info:
            atoms.info.update(extra_info)
        if arrays:
            for name, values in arrays.items():
                array = np.asarray(values)
                if len(array) != len(atoms):
                    raise ValueError(
                        f"Custom array {name!r} has length {len(array)}; "
                        f"expected {len(atoms)} for this frame."
                    )
                atoms.set_array(name, array)

        if calc_kwargs:
            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)

        # First frame overwrites; subsequent frames append (per extxyz convention).
        _append = not self._first or self._append
        ase.io.write(
            self._filename,
            atoms,
            format="extxyz",
            append=_append,
            write_info=True,
            write_results=True,
        )
        self._first = False
        self._n_written += 1

    def __len__(self) -> int:
        return self._n_written

    def __repr__(self) -> str:
        return (
            f"<CrystalTrajectoryWriter '{self._filename}': "
            f"{self._n_written} frames written>"
        )
