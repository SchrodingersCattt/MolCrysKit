"""Compose explicitly mapped atom subsets into an existing geometric path.

Composition changes coordinates only; it does not infer molecular identities,
reaction events, charge, spin, or connectivity. Inputs are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from ..constants.path_constraints import PATH_COMPOSITION_ENDPOINT_TOLERANCE_A
from ._periodic_path_geometry import nearest_image


@dataclass(frozen=True)
class PathLayer:
    """Coordinates for one explicitly mapped atom subset across a path.

    ``atom_indices`` maps local atom order in every layer image to global atom
    indices in the base images. If ``anchor_index`` is given, each layer image
    is translated so that this global atom coincides with the base image; no
    rotation is inferred. ``image_policy='nearest'`` explicitly reimages each
    layer atom to the closest image of its base position and can change the
    represented winding. The default ``'preserve'`` keeps supplied images.
    """

    atom_indices: tuple[int, ...]
    images: Sequence[Atoms]
    anchor_index: int | None = None
    image_policy: str = "preserve"
    name: str | None = None

    def __post_init__(self):
        indices = tuple(self.atom_indices)
        if not indices or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or i < 0
            for i in indices
        ):
            raise ValueError(
                "layer atom indices must be nonempty non-negative integers"
            )
        indices = tuple(int(i) for i in indices)
        if len(set(indices)) != len(indices):
            raise ValueError("layer atom indices must be unique")
        anchor = self.anchor_index
        if anchor is not None:
            if isinstance(anchor, (bool, np.bool_)) or not isinstance(
                anchor, (int, np.integer)
            ):
                raise ValueError("layer anchor must be an integer atom index")
            anchor = int(anchor)
            if anchor not in indices:
                raise ValueError("layer anchor must belong to the layer")
        if self.image_policy not in {"preserve", "nearest"}:
            raise ValueError("layer image_policy must be 'preserve' or 'nearest'")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("layer name must be a string or None")
        object.__setattr__(self, "atom_indices", indices)
        object.__setattr__(self, "anchor_index", anchor)
        object.__setattr__(self, "images", tuple(self.images))


@dataclass
class ComposedPathResult:
    """Composed images and provenance of the explicit layer mappings."""

    images: list[Atoms]
    diagnostics: dict


def _stable_atom_ids(frames, *, label):
    ids = frames[0].arrays.get("atom_id")
    if ids is not None and np.asarray(ids).shape != (len(frames[0]),):
        raise ValueError(f"{label} atom_id must contain one scalar per atom")
    if ids is not None and len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label} atom_id values must be unique")
    for frame in frames:
        current = frame.arrays.get("atom_id")
        if (ids is None) != (current is None):
            raise ValueError(
                f"{label} atom_id arrays must be present in all images or none"
            )
        if ids is not None and not np.array_equal(ids, current):
            raise ValueError(f"{label} atom_id order must be identical in all images")
    return ids


def compose_path_images(
    base_images: Sequence[Atoms],
    layers: Sequence[PathLayer],
    *,
    preserve_endpoints: bool = True,
    endpoint_tolerance_A: float = PATH_COMPOSITION_ENDPOINT_TOLERANCE_A,
) -> ComposedPathResult:
    """Replace disjoint atom subsets of a mapped path with caller-supplied layers.

    All base images retain their atoms, cell, PBC and existing FixAtoms
    constraints. Other ASE constraints are rejected because composition cannot
    safely interpret them. A layer may not move a FixAtoms atom. When
    ``preserve_endpoints`` is true, layer endpoints must agree with the base
    modulo PBC, and the original endpoint coordinates are retained exactly.
    Setting it to false explicitly permits endpoint changes. Cells of the
    base images can vary, but each supplied layer must already use the same
    Cartesian orientation and units as its corresponding base frame.

    Atom order must be stable; when ``atom_id`` arrays exist, both base and
    layers must have them and mapped IDs must match. Without IDs, the explicit
    row mapping is a caller assumption. Geometric or bond metadata is not
    recomputed. The known coordinate-derived ``_molcrys_bond_records`` cache
    is removed from changed images; callers must review other derived metadata.
    No potential, clash checks, energy optimization or bond inference is run.
    """
    frames = tuple(base_images)
    if len(frames) < 2 or any(not isinstance(a, Atoms) for a in frames):
        raise ValueError("base_images must contain at least two ASE Atoms images")
    if not isinstance(preserve_endpoints, (bool, np.bool_)):
        raise TypeError("preserve_endpoints must be boolean")
    if not np.isfinite(endpoint_tolerance_A) or endpoint_tolerance_A < 0:
        raise ValueError("endpoint tolerance must be finite and non-negative")
    count, numbers, pbc = len(frames[0]), frames[0].numbers, frames[0].pbc
    if not count:
        raise ValueError("base images must not be empty")
    for frame in frames:
        if len(frame) != count or not np.array_equal(frame.numbers, numbers):
            raise ValueError("base images must have identical atom order and elements")
        if not np.array_equal(frame.pbc, pbc):
            raise ValueError("base images must have identical periodic directions")
        if not np.all(np.isfinite(frame.positions)) or not np.all(
            np.isfinite(frame.cell)
        ):
            raise ValueError("base positions and cells must be finite")
        if any(not isinstance(c, FixAtoms) for c in frame.constraints):
            raise ValueError("only ASE FixAtoms constraints are supported")
    base_ids = _stable_atom_ids(frames, label="base")

    prepared = []
    used: set[int] = set()
    for layer in layers:
        if not isinstance(layer, PathLayer):
            raise TypeError("layers must contain PathLayer objects")
        indices = np.asarray(layer.atom_indices, dtype=int)
        if np.any(indices >= count):
            raise ValueError("layer atom index is outside the base images")
        overlap = used.intersection(layer.atom_indices)
        if overlap:
            raise ValueError(f"path layers overlap at atom indices {sorted(overlap)}")
        used.update(layer.atom_indices)
        images = tuple(layer.images)
        if len(images) != len(frames) or any(not isinstance(a, Atoms) for a in images):
            raise ValueError(
                "each layer must contain one ASE Atoms object per base image"
            )
        for image in images:
            if len(image) != len(indices) or not np.array_equal(
                image.numbers, numbers[indices]
            ):
                raise ValueError("layer elements must match the mapped base atoms")
            if not np.all(np.isfinite(image.positions)):
                raise ValueError("layer positions must be finite")
        local_ids = _stable_atom_ids(images, label="layer")
        if (base_ids is None) != (local_ids is None):
            raise ValueError(
                "base and layers must either both supply atom_id arrays or neither"
            )
        if base_ids is not None and not np.array_equal(base_ids[indices], local_ids):
            raise ValueError("layer atom_id values do not match the explicit mapping")
        prepared.append((layer, indices, images))

    output = [frame.copy() for frame in frames]
    records = [
        {
            "name": layer.name,
            "atom_indices": list(layer.atom_indices),
            "anchor_index": layer.anchor_index,
            "image_policy": layer.image_policy,
            "max_displacement_A": 0.0,
            "endpoint_error_A": [0.0, 0.0],
        }
        for layer, _, _ in prepared
    ]
    invalidated_bond_record_images = []
    for k, frame in enumerate(frames):
        fixed = {
            int(i) for constraint in frame.constraints for i in constraint.get_indices()
        }
        for record, (layer, indices, images) in zip(records, prepared):
            positions = np.asarray(images[k].positions, dtype=float).copy()
            if layer.anchor_index is not None:
                local = layer.atom_indices.index(layer.anchor_index)
                positions += frame.positions[layer.anchor_index] - positions[local]
            deltas = positions - frame.positions[indices]
            mic = np.array([nearest_image(v, frame.cell, frame.pbc)[0] for v in deltas])
            distances = np.linalg.norm(mic, axis=1)
            record["max_displacement_A"] = max(
                record["max_displacement_A"], float(distances.max())
            )
            if k in (0, len(frames) - 1):
                end = 0 if k == 0 else 1
                record["endpoint_error_A"][end] = float(distances.max())
                if preserve_endpoints:
                    if record["endpoint_error_A"][end] > endpoint_tolerance_A:
                        raise ValueError(
                            "layer endpoint does not match the base endpoint modulo PBC"
                        )
                    continue
            if layer.image_policy == "nearest":
                positions = frame.positions[indices] + mic
            for local, global_index in enumerate(indices):
                if int(global_index) in fixed:
                    if not np.allclose(
                        positions[local],
                        frame.positions[global_index],
                        rtol=0.0,
                        atol=endpoint_tolerance_A,
                    ):
                        raise ValueError(
                            "layer would move an atom constrained by FixAtoms"
                        )
                    positions[local] = frame.positions[global_index]
            output[k].positions[indices] = positions
        if not np.array_equal(output[k].positions, frame.positions):
            if "_molcrys_bond_records" in output[k].info:
                output[k].info.pop("_molcrys_bond_records")
                invalidated_bond_record_images.append(k)
        output[k].calc = None
    return ComposedPathResult(
        output,
        {
            "layers": records,
            "invalidated_bond_record_images": invalidated_bond_record_images,
            "preserve_endpoints": bool(preserve_endpoints),
            "atom_identity": "atom_id"
            if base_ids is not None
            else "explicit row mapping",
        },
    )
