"""Material-agnostic periodic-net analysis on the shared quotient graph.

Isomorphism is available while the crystal adapters and report are assembled.
"""

from .isomorphism import (
    ComponentTopology,
    NetEquivalence,
    TopologyCharacterization,
    characterize_periodic_net,
    compare_periodic_nets,
    verify_topology_witness,
)
from .limits import NetAnalysisLimits

__all__ = [
    "ComponentTopology",
    "NetAnalysisLimits",
    "NetEquivalence",
    "TopologyCharacterization",
    "characterize_periodic_net",
    "compare_periodic_nets",
    "verify_topology_witness",
]
