"""Material-agnostic periodic-net analysis on the shared quotient graph.

Build a site graph, decompose candidate subnets, and read a typed report.
Isomorphism, linking, and duality each return an explicit status. A capped
search stays undetermined or not assessed; it is not a classification.
"""

from .decompose import (
    NetCandidate,
    NetDecomposition,
    NetProposal,
    decompose_periodic_nets,
)
from .duality import (
    DualityAssessment,
    PeriodicTiling,
    TilingAdjacency,
    TilingFace,
    assess_duality,
)
from .invariants import NetSymbols, NodeSymbols, periodic_net_symbols
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
from .policies import (
    DISORDER_MODES,
    EDGE_KINDS,
    NODE_KINDS,
    PARTITION_MODES,
    DisorderPolicy,
    EdgePolicy,
    NodePolicy,
    PartitionPolicy,
    policies_from_mapping,
)
from .report import (
    SCHEMA,
    NetTopologyReport,
    analyze_periodic_net_system,
    limits_from_mapping,
)
from .site_graph import (
    EdgeProvenance,
    NodeProvenance,
    PeriodicSiteGraph,
    build_periodic_site_graph,
    site_graph_from_quotient,
)

__all__ = [
    "DISORDER_MODES",
    "EDGE_KINDS",
    "NODE_KINDS",
    "PARTITION_MODES",
    "SCHEMA",
    "ComponentTopology",
    "DisorderPolicy",
    "DualityAssessment",
    "EdgePolicy",
    "EdgeProvenance",
    "LinkingAssessment",
    "NetAnalysisLimits",
    "NetCandidate",
    "NetDecomposition",
    "NetEquivalence",
    "NetProposal",
    "NetSymbols",
    "NetTopologyReport",
    "NodeSymbols",
    "NodePolicy",
    "NodeProvenance",
    "PartitionPolicy",
    "PeriodicSiteGraph",
    "PeriodicTiling",
    "TilingAdjacency",
    "TilingFace",
    "TopologyCharacterization",
    "analyze_periodic_net_system",
    "assess_duality",
    "assess_linking",
    "build_periodic_site_graph",
    "characterize_periodic_net",
    "compare_periodic_nets",
    "decompose_periodic_nets",
    "limits_from_mapping",
    "periodic_net_symbols",
    "policies_from_mapping",
    "site_graph_from_quotient",
    "verify_topology_witness",
]
