"""Shared reference and result schemas for interaction analysis.

The classes here provide lightweight, serializable references to atoms and
rings participating in detected interactions.  They store indices, labels,
periodic images, and CIF/disorder metadata rather than live ASE atom objects,
so interaction records can be exported and compared independently of the
original molecule containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_LABEL,
    KEY_OCCUPANCY,
    KEY_SYM_OP_INDEX,
)


Image = tuple[int, int, int]


def build_crystal_atom_offsets(molecules: Any) -> tuple[int, ...]:
    """Return flattened atom-index offsets for a molecule sequence.

    The returned tuple maps each molecule index to the starting atom index it
    would have in a concatenated current-crystal atom list.  Interaction
    detectors use these offsets to populate ``AtomRef.crystal_atom_index``
    while preserving molecule-local atom indices.
    """
    offsets: list[int] = []
    current = 0
    for molecule in molecules:
        offsets.append(current)
        current += len(molecule)
    return tuple(offsets)


def _array_value(molecule: Any, key: str, atom_index: int, default: Any = None) -> Any:
    """Safely read one per-atom array value from a molecule.

    Missing arrays, invalid indices, and non-indexable array containers return
    ``default``.  NumPy scalar values are converted with ``.item()`` so the
    result can be stored in dataclasses and JSON-friendly dictionaries.
    """
    arrays = getattr(molecule, "arrays", {})
    if key not in arrays:
        return default
    try:
        value = arrays[key][atom_index]
    except (IndexError, KeyError, TypeError):
        return default
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class AtomRef:
    """Serializable reference to an atom participating in an interaction.

    ``AtomRef`` identifies an atom by molecule-local index plus optional
    current-crystal and asymmetric-unit/source indices.  ``crystal_atom_index``
    is the flattened atom index in the current molecule sequence; together with
    ``image`` it identifies a periodic image atom.  ``asu_atom_index`` stores
    source provenance when available from ``molecule.info["atom_indices"]``.

    Coordinates are not stored; they can be reconstructed from the parent
    molecule and ``image``.  The remaining fields carry common CIF/disorder
    metadata such as label, occupancy, disorder group, assembly, and symmetry
    operation index.
    """

    molecule_index: int
    atom_index: int
    symbol: str
    label: str | None = None
    crystal_atom_index: int | None = None
    asu_atom_index: int | None = None
    image: Image = (0, 0, 0)
    occupancy: float | None = None
    disorder_group: int | None = None
    assembly: str | None = None
    sym_op_index: int | None = None

    @classmethod
    def from_molecule(
        cls,
        molecule: Any,
        molecule_index: int,
        atom_index: int,
        image: Image = (0, 0, 0),
        crystal_atom_offset: int | None = None,
    ) -> "AtomRef":
        """Build an atom reference from a molecule-local atom index.

        The method reads the chemical symbol, optional CIF-style per-atom
        arrays, and optional source asymmetric-unit index from
        ``molecule.info["atom_indices"]``.  When ``crystal_atom_offset`` is
        supplied, it also fills the flattened ``crystal_atom_index`` used by
        interaction records.
        """
        symbols = molecule.get_chemical_symbols()
        asu_indices = getattr(molecule, "info", {}).get("atom_indices")
        asu_atom_index = None
        if asu_indices is not None and atom_index < len(asu_indices):
            asu_atom_index = int(asu_indices[atom_index])
        crystal_atom_index = None
        if crystal_atom_offset is not None:
            crystal_atom_index = int(crystal_atom_offset) + int(atom_index)

        return cls(
            molecule_index=int(molecule_index),
            atom_index=int(atom_index),
            symbol=str(symbols[atom_index]),
            label=_coerce_optional_str(
                _array_value(molecule, KEY_LABEL, atom_index, symbols[atom_index])
            ),
            crystal_atom_index=crystal_atom_index,
            asu_atom_index=asu_atom_index,
            image=tuple(int(v) for v in image),
            occupancy=_coerce_optional_float(
                _array_value(molecule, KEY_OCCUPANCY, atom_index, None)
            ),
            disorder_group=_coerce_optional_int(
                _array_value(molecule, KEY_DISORDER_GROUP, atom_index, None)
            ),
            assembly=_coerce_optional_str(
                _array_value(molecule, KEY_ASSEMBLY, atom_index, None)
            ),
            sym_op_index=_coerce_optional_int(
                _array_value(molecule, KEY_SYM_OP_INDEX, atom_index, None)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation.

        The periodic image tuple is converted to a list and all optional
        metadata fields are included explicitly, allowing consumers to
        distinguish unavailable metadata from omitted keys.
        """
        return {
            "molecule_index": self.molecule_index,
            "atom_index": self.atom_index,
            "symbol": self.symbol,
            "label": self.label,
            "crystal_atom_index": self.crystal_atom_index,
            "asu_atom_index": self.asu_atom_index,
            "image": list(self.image),
            "occupancy": self.occupancy,
            "disorder_group": self.disorder_group,
            "assembly": self.assembly,
            "sym_op_index": self.sym_op_index,
        }


@dataclass(frozen=True)
class RingRef:
    """Serializable reference to a molecular ring in an interaction.

    A ring is identified by sorted molecule-local atom indices, optional
    per-atom ``AtomRef`` records, ring size, aromaticity, and periodic image.
    The object is a reference/metadata container; geometric quantities such as
    centroid and normal live in ``RingGeometry``.
    """

    molecule_index: int
    atom_indices: tuple[int, ...]
    atom_refs: tuple[AtomRef, ...] = ()
    size: int | None = None
    is_aromatic: bool = False
    image: Image = (0, 0, 0)

    @classmethod
    def from_molecule(
        cls,
        molecule: Any,
        molecule_index: int,
        atom_indices: tuple[int, ...],
        is_aromatic: bool = False,
        image: Image = (0, 0, 0),
        crystal_atom_offset: int | None = None,
    ) -> "RingRef":
        """Build a ring reference from molecule-local atom indices.

        Atom indices are sorted for stable identity, and an ``AtomRef`` is
        created for each ring atom using the same image and optional
        crystal-atom offset.  Ring-based detectors use this to keep ring
        identity and per-atom provenance together.
        """
        atom_indices = tuple(sorted(int(i) for i in atom_indices))
        return cls(
            molecule_index=int(molecule_index),
            atom_indices=atom_indices,
            atom_refs=tuple(
                AtomRef.from_molecule(
                    molecule,
                    molecule_index,
                    i,
                    image=image,
                    crystal_atom_offset=crystal_atom_offset,
                )
                for i in atom_indices
            ),
            size=len(atom_indices),
            is_aromatic=bool(is_aromatic),
            image=tuple(int(v) for v in image),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation of the ring."""
        return {
            "molecule_index": self.molecule_index,
            "atom_indices": list(self.atom_indices),
            "atom_refs": [ref.to_dict() for ref in self.atom_refs],
            "size": self.size,
            "is_aromatic": self.is_aromatic,
            "image": list(self.image),
        }


ParticipantRef = AtomRef | RingRef


@dataclass
class BaseInteraction:
    """Common base record for detected weak interactions.

    Concrete interaction records initialize this class with a ``kind``,
    role-named participants, primary distance and angle metrics, optional score,
    periodic image translation, and detector metadata.  Distances are in Å and
    angles are in degrees.  The base fields provide a uniform export surface
    while subclasses expose interaction-specific attributes.
    """

    kind: str
    participants: Mapping[str, ParticipantRef] = field(default_factory=dict)
    distance_A: float | None = None
    angle_deg: float | None = None
    score: float | None = None
    image: Image | None = None
    translation_A: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_optional_str(value: Any) -> str | None:
    """Convert metadata to ``str``, preserving ``None`` and empty as missing."""
    if value is None:
        return None
    value = str(value)
    return value if value != "" else None


def _coerce_optional_float(value: Any) -> float | None:
    """Convert metadata to ``float``, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    """Convert metadata to ``int``, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
