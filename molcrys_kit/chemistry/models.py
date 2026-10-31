"""Immutable chemistry-domain records independent of crystal geometry.

The records in this module intentionally do not import ASE, NetworkX, or any
MolCrysKit crystal class. Chemistry may therefore be reasoned about and tested
without conflating a molecular graph with one particular periodic embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np


Vector3 = tuple[float, float, float]
ImageShift = tuple[int, int, int]

# Shared *default* valences for the bounded OpenSMILES/naming subset.  Keeping
# this table in the chemistry model avoids separate completion and validation
# rules drifting apart.  ``DEFAULT_VALENCE`` is deliberately the ordinary
# organic-subset target used when an atom has no higher-valence environment.
# In particular, sulfur remains divalent here: changing this value to six
# would make an unbracketed ``S``/``CS`` acquire the wrong implicit hydrogen
# count.
DEFAULT_VALENCE = {
    "H": 1.0,
    "B": 3.0,
    "C": 4.0,
    "N": 3.0,
    "O": 2.0,
    "P": 3.0,
    "S": 2.0,
    "F": 1.0,
    "Cl": 1.0,
    "Br": 1.0,
    "I": 1.0,
}

# Some elements have more than one neutral valence state.  The bounded
# OpenSMILES subset currently needs this for sulfur (for example,
# ``C-S(=O)-C`` has bond-order sum four).  Keep the alternatives separate from
# ``DEFAULT_VALENCE`` so callers that infer implicit hydrogens can select the
# lowest compatible state for the local bond environment.
ALLOWED_VALENCES = {
    "H": (1.0,),
    "B": (3.0,),
    "C": (4.0,),
    "N": (3.0,),
    "O": (2.0,),
    "P": (3.0,),
    "S": (2.0, 4.0, 6.0),
    "F": (1.0,),
    "Cl": (1.0,),
    "Br": (1.0,),
    "I": (1.0,),
}


class EvidenceSource(str, Enum):
    """Origin of a chemical assertion."""

    EXPLICIT_CIF = "explicit_cif"
    LINE_NOTATION = "line_notation"
    IUPAC_NAME = "iupac_name"
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
    # Optional row-vector lattice.  Keeping this on the embedding makes the
    # pure chemistry API usable for periodic graphs without importing ASE.
    cell_A: tuple[Vector3, Vector3, Vector3] | None = None
    _index: dict[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        ids = [atom_id for atom_id, _ in self.coordinates_A]
        if len(ids) != len(set(ids)):
            raise ValueError("embedding atom identities must be unique")
        if any(len(vector) != 3 for _, vector in self.coordinates_A):
            raise ValueError("embedding coordinates must be three-dimensional")
        if self.cell_A is not None:
            if len(self.cell_A) != 3 or any(len(vector) != 3 for vector in self.cell_A):
                raise ValueError("cell_A must be a 3x3 row-vector lattice")
            if abs(float(np.linalg.det(np.asarray(self.cell_A, dtype=float)))) < 1e-12:
                raise ValueError("cell_A must be nonsingular")
        object.__setattr__(
            self,
            "_index",
            {atom_id: i for i, (atom_id, _) in enumerate(self.coordinates_A)},
        )

    def position(self, atom_id: str) -> Vector3:
        """Return coordinates for one stable atom identity."""
        index = self._index.get(atom_id) if self._index is not None else None
        if index is None:
            raise KeyError(atom_id)
        return self.coordinates_A[index][1]

    def vector(
        self, atom1_id: str, atom2_id: str, image_shift: ImageShift = (0, 0, 0)
    ) -> Vector3:
        """Return ``r2 + shift @ cell - r1`` using the row-vector convention."""
        left = np.asarray(self.position(atom1_id), dtype=float)
        right = np.asarray(self.position(atom2_id), dtype=float)
        if image_shift != (0, 0, 0):
            if self.cell_A is None:
                raise ValueError("image_shift requires a periodic cell")
            right = right + np.asarray(image_shift, dtype=float) @ np.asarray(
                self.cell_A, dtype=float
            )
        return tuple(float(value) for value in right - left)


def _validate_graph(
    atoms: tuple[ChemicalAtom, ...], bonds: tuple[ChemicalBond, ...]
) -> None:
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
        return tuple(
            getattr(component, "dimension", None) for component in self.components
        )

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
