"""Material-agnostic periodic-net analysis on the shared quotient graph.

Isomorphism, linking, and duality are importable while crystal adapters and
the typed report are assembled.
"""

from .duality import (
    DualityAssessment,
    PeriodicTiling,
    TilingAdjacency,
    TilingFace,
    assess_duality,
)
from .isomorphism import (
    ComponentTopology,
    NetEquivalence,
    TopologyCharacterization,
    characterize_periodic_net,
    compare_periodic_nets,
    verify_topology_witness,
)
from .limits import NetAnalysisLimits
from .linking import LinkingAssessment, assess_linking

__all__ = [
    "ComponentTopology",
    "DualityAssessment",
    "LinkingAssessment",
    "NetAnalysisLimits",
    "NetEquivalence",
    "PeriodicTiling",
    "TilingAdjacency",
    "TilingFace",
    "TopologyCharacterization",
    "assess_duality",
    "assess_linking",
    "characterize_periodic_net",
    "compare_periodic_nets",
    "verify_topology_witness",
]
