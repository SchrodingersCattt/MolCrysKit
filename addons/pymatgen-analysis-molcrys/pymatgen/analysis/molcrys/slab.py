"""Topology-preserving slab generation with a pymatgen Structure API."""

from __future__ import annotations

from typing import Optional, Tuple

from pymatgen.core import Structure

from molcrys_kit.operations.surface import (
    TopologicalSlabGenerator as _MckTopologicalSlabGenerator,
)
from molcrys_kit.operations.surface import generate_topological_slab as _mck_generate_topological_slab

from ._convert import as_molecular_crystal, as_structure

MillerIndex = Tuple[int, int, int]


class TopologicalSlabGenerator:
    """Pymatgen-facing wrapper around MolCrysKit topological slab generation.

    Accepts a :class:`~pymatgen.core.structure.Structure` (or MolCrysKit
    ``MolecularCrystal``) and returns slabs as ``Structure`` objects without
    cutting through molecules.
    """

    def __init__(self, structure: Structure, miller_index: Optional[MillerIndex] = None):
        self._crystal = as_molecular_crystal(structure)
        self.miller_index = miller_index
        self._generator = _MckTopologicalSlabGenerator(self._crystal)

    def get_slab(
        self,
        miller_index: Optional[MillerIndex] = None,
        *,
        layers: Optional[int] = None,
        min_thickness: Optional[float] = None,
        vacuum: float = 10.0,
    ) -> Structure:
        """Build one topology-preserving slab and return a ``Structure``."""
        miller = miller_index or self.miller_index
        if miller is None:
            raise ValueError("miller_index must be provided to get_slab() or the constructor")
        slab = self._generator.build(
            miller,
            layers=layers,
            min_thickness=min_thickness,
            vacuum=vacuum,
        )
        return as_structure(slab)


def generate_topological_slab(
    structure: Structure,
    miller_index: MillerIndex,
    *,
    layers: Optional[int] = None,
    min_thickness: Optional[float] = None,
    vacuum: float = 10.0,
) -> Structure:
    """Functional entry point mirroring MolCrysKit ``generate_topological_slab``.

    Parameters
    ----------
    structure:
        Bulk molecular crystal as a pymatgen ``Structure``.
    miller_index:
        Surface Miller indices ``(h, k, l)``.
    layers, min_thickness, vacuum:
        Forwarded to MolCrysKit.

    Returns
    -------
    Structure
        Slab cell with vacuum along the surface normal.
    """
    crystal = as_molecular_crystal(structure)
    slab = _mck_generate_topological_slab(
        crystal,
        miller_index,
        layers=layers,
        min_thickness=min_thickness,
        vacuum=vacuum,
    )
    return as_structure(slab)
