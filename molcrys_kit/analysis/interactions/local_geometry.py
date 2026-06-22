"""Cached molecule-local topology and ring geometry for interaction detectors.

``LocalGeometry`` wraps ``ChemicalEnvironment`` to expose repeated
neighbourhood, bonded-hydrogen, and ring queries through a small cache.  The
records here keep molecule-local geometry separate from periodic image
handling, which is managed by the interaction detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from ..chemical_env import ChemicalEnvironment
from .base import RingRef
from .geometry import best_fit_plane


@dataclass(frozen=True)
class AtomLocalGeometry:
    """Cached topology and simple geometry around one molecule-local atom.

    The record stores neighbour lists, heavy-atom neighbours, coordination
    number, bond-length/angle summaries, and ring/aromatic-ring membership.  It
    is intended as contextual metadata for interaction records rather than a
    full chemical environment model.
    """

    atom_index: int
    symbol: str
    neighbor_indices: tuple[int, ...]
    heavy_neighbor_indices: tuple[int, ...]
    coordination_number: int
    average_bond_length_A: float | None = None
    bond_angle_sum_deg: float | None = None
    bond_angle_single_deg: float | None = None
    in_ring: bool = False
    ring_sizes: tuple[int, ...] = ()
    aromatic_ring_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class RingGeometry:
    """Molecule-local ring geometry derived from a topology cycle.

    The record contains sorted ring atom indices, element symbols, centroid in
    Å, fitted normal, plane RMSD in Å, planarity/aromaticity flags, ring size,
    and an optional ``RingRef`` when the parent molecule index is known.
    """

    atom_indices: tuple[int, ...]
    symbols: tuple[str, ...]
    centroid_A: tuple[float, float, float]
    normal: tuple[float, float, float]
    plane_rmsd_A: float | None = None
    is_planar: bool = False
    is_aromatic: bool = False
    size: int | None = None
    ring_ref: RingRef | None = None


class LocalGeometry:
    """Cached geometry view for one molecule.

    The wrapper owns a ``ChemicalEnvironment`` and lazily computes atom-local
    records and ring geometries.  It provides the common local queries needed
    by weak-interaction detectors while hiding graph and ring-detection details.
    """

    def __init__(self, molecule, molecule_index: int | None = None):
        """Create a cached local-geometry view for a molecule.

        ``molecule_index`` is optional; when supplied, generated ring
        geometries include ``RingRef`` objects that can be embedded directly in
        interaction records.  The wrapper reads from the molecule but does not
        intentionally mutate it.
        """
        self.molecule = molecule
        self.molecule_index = molecule_index
        self.env = ChemicalEnvironment(molecule)
        self._atom_cache: dict[int, AtomLocalGeometry] = {}
        self._rings_cache: list[RingGeometry] | None = None

    def atom(self, atom_index: int) -> AtomLocalGeometry:
        """Return cached local geometry for one molecule-local atom.

        The first request reads neighbours and ring information from
        ``ChemicalEnvironment``, computes heavy-neighbour and aromatic-ring
        summaries, and stores the resulting ``AtomLocalGeometry`` for reuse.
        """
        atom_index = int(atom_index)
        if atom_index not in self._atom_cache:
            graph = self.env.graph
            symbols = self.molecule.get_chemical_symbols()
            neighbors = tuple(int(i) for i in graph.neighbors(atom_index))
            heavy_neighbors = tuple(
                int(i)
                for i in neighbors
                if graph.nodes[i].get("symbol", symbols[i]) != "H"
            )
            stats = self.env.get_local_geometry_stats(atom_index)
            ring_info = self.env.detect_ring_info(atom_index)
            self._atom_cache[atom_index] = AtomLocalGeometry(
                atom_index=atom_index,
                symbol=str(symbols[atom_index]),
                neighbor_indices=neighbors,
                heavy_neighbor_indices=heavy_neighbors,
                coordination_number=int(stats.get("coordination_number", len(neighbors))),
                average_bond_length_A=float(stats.get("average_bond_length", 0.0)),
                bond_angle_sum_deg=float(stats.get("bond_angle_sum", 0.0)),
                bond_angle_single_deg=float(stats.get("bond_angle_single", 0.0)),
                in_ring=bool(ring_info.get("in_ring", False)),
                ring_sizes=tuple(int(v) for v in ring_info.get("ring_sizes", ())),
                aromatic_ring_sizes=tuple(
                    int(v) for v in self.env.atom_aromatic_ring_sizes(atom_index)
                ),
            )
        return self._atom_cache[atom_index]

    def neighbors(self, atom_index: int, heavy_only: bool = False) -> tuple[int, ...]:
        """Return molecule-local neighbour indices for an atom.

        Set ``heavy_only=True`` to exclude hydrogens from the returned
        neighbour tuple.
        """
        local = self.atom(atom_index)
        return local.heavy_neighbor_indices if heavy_only else local.neighbor_indices

    def bonded_hydrogens(self, atom_index: int) -> tuple[int, ...]:
        """Return molecule-local hydrogen indices bonded to the given atom."""
        symbols = self.molecule.get_chemical_symbols()
        return tuple(i for i in self.neighbors(atom_index) if symbols[i] == "H")

    def rings(self, aromatic_only: bool = False) -> list[RingGeometry]:
        """Return cached ring geometries for the molecule.

        When ``aromatic_only`` is true, only rings marked aromatic by
        ``ChemicalEnvironment`` are returned.  A shallow list copy is returned
        so callers cannot mutate the internal cache list.
        """
        if self._rings_cache is None:
            self._rings_cache = self._build_rings()
        if aromatic_only:
            return [ring for ring in self._rings_cache if ring.is_aromatic]
        return list(self._rings_cache)

    def _build_rings(self) -> list[RingGeometry]:
        """Construct ring geometry records from topology cycles.

        Each cycle is converted to sorted atom indices, aromaticity is inferred
        from per-atom aromatic ring sizes, and a best-fit plane supplies
        centroid, normal, and planarity RMSD.  Plane-fitting failures produce a
        warning and a fallback centroid with a zero normal so ring detection can
        continue.
        """
        rings: list[RingGeometry] = []
        positions = self.molecule.get_positions()
        symbols = self.molecule.get_chemical_symbols()

        for cycle in self.env.rings():
            atom_indices = tuple(sorted(int(i) for i in cycle))
            size = len(atom_indices)
            is_aromatic = all(
                size in self.env.atom_aromatic_ring_sizes(i) for i in atom_indices
            )
            pts = positions[list(atom_indices)]
            try:
                centroid, normal, rmsd = best_fit_plane(pts)
                is_planar = bool(rmsd <= 0.25)
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"Could not fit plane for ring {atom_indices}: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                centroid = tuple(float(v) for v in np.asarray(pts).mean(axis=0))
                normal = (0.0, 0.0, 0.0)
                rmsd = None
                is_planar = False

            ring_ref = None
            if self.molecule_index is not None:
                ring_ref = RingRef.from_molecule(
                    self.molecule,
                    self.molecule_index,
                    atom_indices,
                    is_aromatic=is_aromatic,
                )

            rings.append(
                RingGeometry(
                    atom_indices=atom_indices,
                    symbols=tuple(symbols[i] for i in atom_indices),
                    centroid_A=centroid,
                    normal=normal,
                    plane_rmsd_A=rmsd,
                    is_planar=is_planar,
                    is_aromatic=is_aromatic,
                    size=size,
                    ring_ref=ring_ref,
                )
            )
        return rings


class LocalGeometryCache:
    """Lazy cache of ``LocalGeometry`` objects indexed by molecule number.

    The constructor accepts either a ``MolecularCrystal``-like object with a
    ``molecules`` attribute or an explicit molecule sequence.
    """

    def __init__(self, crystal_or_molecules):
        if hasattr(crystal_or_molecules, "molecules"):
            self.molecules = list(crystal_or_molecules.molecules)
        else:
            self.molecules = list(crystal_or_molecules)
        self._cache: dict[int, LocalGeometry] = {}

    def get(self, molecule_index: int) -> LocalGeometry:
        """Return cached ``LocalGeometry`` for one molecule index."""
        molecule_index = int(molecule_index)
        if molecule_index not in self._cache:
            self._cache[molecule_index] = LocalGeometry(
                self.molecules[molecule_index], molecule_index=molecule_index
            )
        return self._cache[molecule_index]

    def __getitem__(self, molecule_index: int) -> LocalGeometry:
        """Return ``get(molecule_index)`` for dictionary-like cache access."""
        return self.get(molecule_index)
