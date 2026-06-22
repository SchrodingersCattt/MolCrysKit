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

from .base import AtomRef, BaseInteraction, RingRef, build_crystal_atom_offsets
from .bonding import get_bonding_threshold
from .ch_pi import CHPiInteraction, CHPiInteractionCriteria, find_ch_pi, find_ch_pi_interactions
from .h_h_contact import HHContact, HHContactCriteria, find_h_h_contacts
from .halogen_bond import HalogenBond, HalogenBondCriteria, find_halogen_bonds
from .hydrogen_bond import HydrogenBond, HydrogenBondCriteria, find_hydrogen_bonds
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .local_geometry import (
    AtomLocalGeometry,
    LocalGeometry,
    LocalGeometryCache,
    RingGeometry,
)
from .pi_stacking import PiStacking, PiStackingCriteria, PiStackingSubtype, find_pi_stacking, find_pi_stacks

__all__ = [
    "AtomLocalGeometry",
    "AtomRef",
    "BaseInteraction",
    "CHPiInteraction",
    "CHPiInteractionCriteria",
    "ChemicalIdentity",
    "ChemicalIdentityCache",
    "HHContact",
    "HHContactCriteria",
    "HalogenBond",
    "HalogenBondCriteria",
    "HydrogenBond",
    "HydrogenBondCriteria",
    "LocalGeometry",
    "LocalGeometryCache",
    "PiStacking",
    "PiStackingCriteria",
    "PiStackingSubtype",
    "RingGeometry",
    "RingRef",
    "build_crystal_atom_offsets",
    "find_ch_pi",
    "find_ch_pi_interactions",
    "find_h_h_contacts",
    "find_halogen_bonds",
    "find_hydrogen_bonds",
    "find_pi_stacking",
    "find_pi_stacks",
    "get_bonding_threshold",
]
