"""Molecule-local geometry views for interaction detectors."""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from ..chemical_env import ChemicalEnvironment
from .base import RingRef
from .geometry import best_fit_plane


@dataclass(frozen=True)
class AtomLocalGeometry:
    """Cached local topology/geometry around a molecule-local atom."""

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
    """Geometry of a molecule-local ring."""

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
    """Cached molecule-local geometry wrapper around ``ChemicalEnvironment``."""

    def __init__(self, molecule, molecule_index: int | None = None):
        self.molecule = molecule
        self.molecule_index = molecule_index
        self.env = ChemicalEnvironment(molecule)
        self._atom_cache: dict[int, AtomLocalGeometry] = {}
        self._rings_cache: list[RingGeometry] | None = None

    def atom(self, atom_index: int) -> AtomLocalGeometry:
        """Return local geometry for one atom."""
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
        """Return molecule-local neighbour indices."""
        local = self.atom(atom_index)
        return local.heavy_neighbor_indices if heavy_only else local.neighbor_indices

    def bonded_hydrogens(self, atom_index: int) -> tuple[int, ...]:
        """Return hydrogens bonded to a molecule-local atom."""
        symbols = self.molecule.get_chemical_symbols()
        return tuple(i for i in self.neighbors(atom_index) if symbols[i] == "H")

    def rings(self, aromatic_only: bool = False) -> list[RingGeometry]:
        """Return deduplicated ring geometries for the molecule."""
        if self._rings_cache is None:
            self._rings_cache = self._build_rings()
        if aromatic_only:
            return [ring for ring in self._rings_cache if ring.is_aromatic]
        return list(self._rings_cache)

    def _build_rings(self) -> list[RingGeometry]:
        seen: set[frozenset[int]] = set()
        rings: list[RingGeometry] = []
        atom_rings = getattr(self.env, "_atom_rings", {})
        aromatic_sizes_by_atom = getattr(self.env, "_atom_aromatic_ring_sizes", {})
        positions = self.molecule.get_positions()
        symbols = self.molecule.get_chemical_symbols()

        for cycles in atom_rings.values():
            for cycle in cycles:
                atom_indices = tuple(int(i) for i in cycle)
                key = frozenset(atom_indices)
                if key in seen:
                    continue
                seen.add(key)
                size = len(atom_indices)
                is_aromatic = all(
                    size in aromatic_sizes_by_atom.get(i, []) for i in atom_indices
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
    """Lazy cache of ``LocalGeometry`` objects for crystal molecules."""

    def __init__(self, crystal_or_molecules):
        if hasattr(crystal_or_molecules, "molecules"):
            self.molecules = list(crystal_or_molecules.molecules)
        else:
            self.molecules = list(crystal_or_molecules)
        self._cache: dict[int, LocalGeometry] = {}

    def get(self, molecule_index: int) -> LocalGeometry:
        molecule_index = int(molecule_index)
        if molecule_index not in self._cache:
            self._cache[molecule_index] = LocalGeometry(
                self.molecules[molecule_index], molecule_index=molecule_index
            )
        return self._cache[molecule_index]

    def __getitem__(self, molecule_index: int) -> LocalGeometry:
        return self.get(molecule_index)