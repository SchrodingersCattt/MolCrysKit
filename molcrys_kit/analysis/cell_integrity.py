"""Periodic-cell integrity checks for transformed molecular crystals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx
from ase.neighborlist import neighbor_list

from ..structures.crystal import _structure_hash
from ..constants.config import KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z

_DERIVED_FRACTIONAL_KEYS = {KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z}

__all__ = ["CellIntegrityReport", "check_cell_integrity"]


@dataclass
class CellIntegrityReport:
    """Machine-readable result of a periodic-cell audit."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "checks": {key: bool(value) for key, value in self.checks.items()},
            "details": self.details,
        }


def _inventory(crystal) -> list[tuple[str, int]]:
    return sorted(
        (str(molecule.get_chemical_formula()), len(molecule))
        for molecule in crystal.molecules
    )


def _topology_matches(before, after) -> bool:
    """Compare molecular graphs exactly, including element labels.

    ``graph_invariant`` is only a pre-filter: non-isomorphic graphs can share
    the same degree and element inventory.  Integrity validation is a
    correctness gate, so use a full isomorphism check and consume each target
    molecule at most once when identical molecules are present.
    """
    if len(before.molecules) != len(after.molecules):
        return False

    def node_match(left, right):
        return left.get("symbol") == right.get("symbol")

    remaining = list(after.molecules)
    for source in before.molecules:
        match_index = None
        for index, target in enumerate(remaining):
            if len(source) != len(target):
                continue
            if nx.is_isomorphic(
                source.graph,
                target.graph,
                node_match=node_match,
            ):
                match_index = index
                break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _metadata_arrays_match(before, after) -> bool:
    if len(before.molecules) != len(after.molecules):
        return False
    for old, new in zip(before.molecules, after.molecules):
        ignored = {"positions", "numbers", "image_shift"} | _DERIVED_FRACTIONAL_KEYS
        old_keys = set(old.arrays) - ignored
        new_keys = set(new.arrays) - ignored
        if old_keys != new_keys:
            return False
        for key in old_keys:
            if key in {"positions", "numbers", "image_shift"}:
                continue
            if not np.array_equal(old.arrays[key], new.arrays[key]):
                return False
    if set(before.extra_arrays) != set(after.extra_arrays):
        return False
    return all(
        np.array_equal(before.extra_arrays[key], after.extra_arrays[key])
        for key in before.extra_arrays
    )


def check_cell_integrity(crystal, reference=None, *, centroid_tolerance: float = 1e-8) -> CellIntegrityReport:
    """Audit a crystal's lattice, molecular images, topology, and provenance.

    If ``reference`` is supplied, inventory, topology, and arbitrary per-atom
    arrays are compared against it.  This is the recommended postcondition
    check for :meth:`MolecularCrystal.transform_cell`.
    """
    lattice = np.asarray(crystal.lattice, dtype=float)
    shape_ok = lattice.shape == (3, 3) and np.all(np.isfinite(lattice))
    determinant = float(np.linalg.det(lattice)) if shape_ok else 0.0
    nonsingular = abs(determinant) > 1e-10
    right_handed = determinant > 0
    pbc = tuple(bool(value) for value in crystal.pbc)
    pbc_ok = len(pbc) == 3

    centroid_rows: list[list[float]] = []
    centroids_inside = True
    if shape_ok and nonsingular:
        inverse = np.linalg.inv(lattice)
        periodic = np.asarray(pbc, dtype=bool)
        for molecule in crystal.molecules:
            centroid = molecule.get_positions().mean(axis=0) if len(molecule) else np.zeros(3)
            fractional = centroid @ inverse
            centroid_rows.append([float(value) for value in fractional])
            if np.any(fractional[periodic] < -centroid_tolerance) or np.any(
                fractional[periodic] >= 1.0 + centroid_tolerance
            ):
                centroids_inside = False
    else:
        centroids_inside = False

    seam_contact_count = 0
    try:
        atoms = crystal.to_ase()
        if len(atoms) and shape_ok and nonsingular and any(pbc):
            _, _, _, shifts = neighbor_list("ijdS", atoms, cutoff=3.0)
            seam_contact_count = int(np.count_nonzero(np.any(np.asarray(shifts) != 0, axis=1)))
    except Exception:
        # Contact reporting is diagnostic; lattice and inventory checks remain
        # authoritative when a third-party Atoms implementation is incomplete.
        seam_contact_count = 0

    # ``wrap_mode='none'`` deliberately preserves an unwrapped molecular
    # embedding.  A centroid outside the primary cell is then valid output,
    # so the in-cell check must be informational rather than a failure gate.
    transform = getattr(crystal, "metadata", {}).get("cell_transform")
    allows_unwrapped = (
        isinstance(transform, dict) and transform.get("wrap_mode") == "none"
    )
    checks = {
        "lattice_shape": shape_ok,
        "lattice_nonsingular": nonsingular,
        "lattice_right_handed": right_handed,
        "pbc_flags": pbc_ok,
        "molecule_centroids_in_cell": (
            True if allows_unwrapped else centroids_inside
        ),
    }
    details: dict[str, Any] = {
        "determinant_A3": determinant,
        "pbc": list(pbc),
        "centroids_fractional": centroid_rows,
        "centroids_wrapping_required": not allows_unwrapped,
        "molecule_inventory": _inventory(crystal),
        "seam_contact_count": seam_contact_count,
        "output_hash": _structure_hash(crystal),
    }

    if isinstance(transform, dict):
        expected_hash = transform.get("output_hash")
        checks["transform_provenance"] = expected_hash in {None, details["output_hash"]}
        if reference is not None and transform.get("source_hash") is not None:
            checks["input_provenance"] = transform["source_hash"] == _structure_hash(reference)
        details["cell_transform"] = transform

    if reference is not None:
        checks["molecular_inventory"] = _inventory(reference) == details["molecule_inventory"]
        checks["topology"] = _topology_matches(reference, crystal)
        checks["per_atom_metadata"] = _metadata_arrays_match(reference, crystal)
        details["input_hash"] = _structure_hash(reference)

    return CellIntegrityReport(passed=all(checks.values()), checks=checks, details=details)
