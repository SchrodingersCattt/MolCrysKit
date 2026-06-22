"""
Intermolecular interaction analysis for molecular crystals.

The package separates interaction analysis into four layers:

* references: lightweight ``AtomRef`` and ``RingRef`` objects that identify
  participating atoms/rings, including their periodic image;
* identity: ``ChemicalIdentity`` objects that annotate molecule/fragment/species
  identity;
* local geometry: cached molecule-local neighbour/ring/coordination information;
* detectors: concrete interaction records and finder functions.

The legacy import path ``molcrys_kit.analysis.interactions`` is kept stable.
"""

from .base import AtomRef, BaseInteraction, RingRef
from .bonding import get_bonding_threshold
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .hydrogen_bond import HydrogenBond, HydrogenBondCriteria, find_hydrogen_bonds
from .identity import ChemicalIdentity, ChemicalIdentityCache
from .local_geometry import (
    AtomLocalGeometry,
    LocalGeometry,
    LocalGeometryCache,
    RingGeometry,
)

__all__ = [
    "AtomLocalGeometry",
    "AtomRef",
    "BaseInteraction",
    "ChemicalIdentity",
    "ChemicalIdentityCache",
    "HydrogenBond",
    "HydrogenBondCriteria",
    "LocalGeometry",
    "LocalGeometryCache",
    "RingGeometry",
    "RingRef",
    "enumerate_lattice_images",
    "find_hydrogen_bonds",
    "get_bonding_threshold",
    "image_translation",
    "vector_angle_deg",
]