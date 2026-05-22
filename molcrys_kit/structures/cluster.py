"""Finite molecular cluster carved from a periodic crystal.

A ``CrystalCluster`` is a :class:`CrystalMolecule` that always carries
``pbc=False`` and that stores the bookkeeping required to feed a QM code
(Gaussian, ORCA, Psi4, ...) downstream:

* ``provenance``      -- :class:`ClusterProvenance` record of how it was carved.
* ``frozen_local_indices`` -- atoms to be held fixed during QM optimisation.
* ``cap_local_indices``    -- the H atoms that cap the dangling bonds.

The class deliberately stays at the atom + bond level; it does not
introduce any domain-specific ontology (SBU / linker / pore for MOFs,
ring / channel for zeolites, vertex / strut for COFs, ...).  Callers
wanting such typing should layer it on top of the existing
``ChemicalEnvironment`` analyser.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .molecule import CrystalMolecule
from ..analysis.cluster_provenance import ClusterProvenance


class CrystalCluster(CrystalMolecule):
    """A non-periodic molecular cluster derived from a :class:`MolecularCrystal`.

    Parameters
    ----------
    atoms : ase.Atoms
        The cluster atoms with explicit Cartesian positions.  ``pbc`` is
        forced to ``False`` on construction.
    provenance : ClusterProvenance
        Audit trail describing the carve (see
        :mod:`molcrys_kit.analysis.cluster_provenance`).
    frozen_local_indices : list[int]
        Local indices (into the emitted atom list) that downstream QM
        runs should hold fixed.  Stored as ``int`` tuple for immutability.
    cap_local_indices : list[int]
        Local indices of the H atoms that cap dangling bonds.
    parent_crystal : object, optional
        Reference to the parent crystal (matches ``CrystalMolecule.crystal``).
    """

    def __init__(
        self,
        atoms,
        provenance: ClusterProvenance,
        frozen_local_indices: Optional[List[int]] = None,
        cap_local_indices: Optional[List[int]] = None,
        parent_crystal=None,
    ):
        # Force pbc=False on the underlying Atoms before calling the parent
        # constructor; check_pbc=False prevents the unwrap pass since the
        # cluster is already contiguous in Cartesian space.
        atoms = atoms.copy()
        atoms.set_pbc(False)
        atoms.set_cell(np.zeros((3, 3)))
        super().__init__(atoms, crystal=parent_crystal, check_pbc=False)

        self._provenance = provenance
        self._frozen_local_indices = tuple(int(i) for i in (frozen_local_indices or []))
        self._cap_local_indices = tuple(int(i) for i in (cap_local_indices or []))

    @property
    def provenance(self) -> ClusterProvenance:
        return self._provenance

    @property
    def frozen_local_indices(self) -> List[int]:
        return list(self._frozen_local_indices)

    @property
    def cap_local_indices(self) -> List[int]:
        return list(self._cap_local_indices)

    def copy(self) -> "CrystalCluster":  # type: ignore[override]
        atoms_copy = CrystalMolecule.copy(self).to_ase()
        return CrystalCluster(
            atoms_copy,
            provenance=self._provenance,
            frozen_local_indices=list(self._frozen_local_indices),
            cap_local_indices=list(self._cap_local_indices),
            parent_crystal=self.crystal,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CrystalCluster(formula='{self.get_chemical_formula()}', "
            f"atoms={len(self)}, frozen={len(self._frozen_local_indices)}, "
            f"caps={len(self._cap_local_indices)}, mode={self._provenance.mode})"
        )


__all__ = ["CrystalCluster"]
