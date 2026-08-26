"""Immutable chemistry-domain records independent of crystal geometry.

The records in this module intentionally do not import ASE, NetworkX, or any
MolCrysKit crystal class. Chemistry may therefore be reasoned about and tested
without conflating a molecular graph with one particular periodic embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


Vector3 = tuple[float, float, float]
ImageShift = tuple[int, int, int]


class EvidenceSource(str, Enum):
    """Origin of a chemical assertion."""

    EXPLICIT_CIF = "explicit_cif"
    LINE_NOTATION = "line_notation"
    INFERRED = "inferred"
    USER_CONFIRMED = "user_confirmed"


class InferenceStatus(str, Enum):
    """Resolution state of an entity or analysis result."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    INDETERMINATE = "indeterminate"


class BondKind(str, Enum):
    """Chemical semantics of an edge, independent of its numeric order."""

    UNKNOWN = "unknown"
    COVALENT = "covalent"
    COORDINATION = "coordination"
    IONIC = "ionic"
    METALLIC = "metallic"


@dataclass(frozen=True)
class Evidence:
    """Auditable support for one or more chemical assertions."""

    source: EvidenceSource
    method: str
    detail: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True)
class ChemicalAtom:
    """One atom in a chemistry graph."""

    atom_id: str
    element: str
    label: str | None = None
    isotope: int | None = None
    formal_charge: int | None = None
    radical_electrons: int = 0
    explicit_hydrogens: int | None = None
    implicit_hydrogens: int | None = None
    oxidation_state: int | None = None
    stereochemistry: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ValueError("atom_id must not be empty")
        if not self.element:
            raise ValueError("element must not be empty")
        if self.isotope is not None and self.isotope <= 0:
            raise ValueError("isotope must be positive")
        if self.radical_electrons < 0:
            raise ValueError("radical_electrons must not be negative")


@dataclass(frozen=True)
class ChemicalBond:
    """One canonical chemistry edge, optionally crossing a unit cell."""

    atom1_id: str
    atom2_id: str
    order: float | None = None
    kind: BondKind = BondKind.UNKNOWN
    aromatic: bool = False
    atom2_image_shift: ImageShift = (0, 0, 0)
    stereochemistry: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.atom1_id or not self.atom2_id:
            raise ValueError("bond endpoints must not be empty")
        if self.order is not None and self.order <= 0:
            raise ValueError("bond order must be positive")
        if len(self.atom2_image_shift) != 3:
            raise ValueError("atom2_image_shift must contain three integers")


@dataclass(frozen=True)
class Embedding:
    """Coordinates for atoms in one entity, separate from graph identity."""

    coordinates_A: tuple[tuple[str, Vector3], ...]
    coordinate_system: str = "cartesian"
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        ids = [atom_id for atom_id, _ in self.coordinates_A]
        if len(ids) != len(set(ids)):
            raise ValueError("embedding atom identities must be unique")
        if any(len(vector) != 3 for _, vector in self.coordinates_A):
            raise ValueError("embedding coordinates must be three-dimensional")

    def position(self, atom_id: str) -> Vector3:
        """Return coordinates for one stable atom identity."""
        for candidate, vector in self.coordinates_A:
            if candidate == atom_id:
                return vector
        raise KeyError(atom_id)


def _validate_graph(atoms: tuple[ChemicalAtom, ...], bonds: tuple[ChemicalBond, ...]) -> None:
    atom_ids = [atom.atom_id for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("chemical entity atom identities must be unique")
    known = set(atom_ids)
    for bond in bonds:
        if bond.atom1_id not in known or bond.atom2_id not in known:
            raise ValueError("chemical bond endpoint is absent from entity atoms")


@dataclass(frozen=True)
class FiniteChemicalEntity:
    """A finite molecule, ion, or discrete coordination entity."""

    entity_id: str
    atoms: tuple[ChemicalAtom, ...]
    bonds: tuple[ChemicalBond, ...]
    embedding: Embedding | None = None
    net_charge: int | None = None
    status: InferenceStatus = InferenceStatus.INDETERMINATE
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()
    dimension: int = 0

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if self.dimension != 0:
            raise ValueError("finite entities must have dimension zero")
        _validate_graph(self.atoms, self.bonds)


@dataclass(frozen=True)
class PeriodicChemicalEntity:
    """A connected 1D, 2D, or 3D periodic chemical graph."""

    entity_id: str
    atoms: tuple[ChemicalAtom, ...]
    bonds: tuple[ChemicalBond, ...]
    periodic_rank: int
    translation_generators: tuple[ImageShift, ...]
    embedding: Embedding | None = None
    net_charge_per_repeat: int | None = None
    status: InferenceStatus = InferenceStatus.INDETERMINATE
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.periodic_rank not in {1, 2, 3}:
            raise ValueError("periodic_rank must be one, two, or three")
        _validate_graph(self.atoms, self.bonds)

    @property
    def dimension(self) -> int:
        return self.periodic_rank


@dataclass(frozen=True)
class PolymerChemicalEntity:
    """A polymer represented by repeat entities and connection descriptors."""

    entity_id: str
    repeat_units: tuple[FiniteChemicalEntity, ...]
    connections: tuple[str, ...] = ()
    status: InferenceStatus = InferenceStatus.INDETERMINATE
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MulticomponentEntity:
    """A salt, solvate, adduct, or other stoichiometric entity collection."""

    entity_id: str
    components: tuple[tuple["ChemicalEntity", int], ...]
    status: InferenceStatus = InferenceStatus.INDETERMINATE
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(count <= 0 for _, count in self.components):
            raise ValueError("component counts must be positive")


ChemicalEntity = Union[
    FiniteChemicalEntity,
    PeriodicChemicalEntity,
    PolymerChemicalEntity,
    MulticomponentEntity,
]


@dataclass(frozen=True)
class CrystalChemistry:
    """Chemistry annotations attached to one crystal snapshot."""

    components: tuple[ChemicalEntity, ...]
    atom_ids_by_global_index: tuple[str, ...]
    status: InferenceStatus
    evidence: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()
    alternatives: tuple[tuple[ChemicalEntity, ...], ...] = ()

    @property
    def component_dimensions(self) -> tuple[int | None, ...]:
        """Chemical dimensionality of each component in stable order."""
        return tuple(getattr(component, "dimension", None) for component in self.components)

    @property
    def is_molecular_crystal(self) -> bool:
        """Whether every attached component is a finite 0D entity."""
        return bool(self.components) and all(
            isinstance(component, FiniteChemicalEntity) for component in self.components
        )


__all__ = [
    "BondKind",
    "ChemicalAtom",
    "ChemicalBond",
    "ChemicalEntity",
    "CrystalChemistry",
    "Embedding",
    "Evidence",
    "EvidenceSource",
    "FiniteChemicalEntity",
    "InferenceStatus",
    "MulticomponentEntity",
    "PeriodicChemicalEntity",
    "PolymerChemicalEntity",
]
