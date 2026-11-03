"""Candidate nets inside one periodic site graph.

Explicit mode checks the caller's labels. Discovery mode returns the
connectivity components, the single whole-graph partition, and, when
asymmetric-unit indices exist, the orbit partition. It does not invent orbits
from missing indices, and it does not drop a second admissible partition.
Translated copies stay inside one candidate and are counted by cover
multiplicity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from molcrys_kit.periodic import (
    PeriodicEmbedding,
    PeriodicQuotientGraph,
    analyze_periodic_graph,
)

from .limits import NetAnalysisLimits
from .policies import PartitionPolicy
from .site_graph import PeriodicSiteGraph


@dataclass(frozen=True)
class NetCandidate:
    """One induced subnet and its cover rank."""

    candidate_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    cross_edge_ids: tuple[str, ...]
    graph: PeriodicQuotientGraph
    embedding: PeriodicEmbedding | None
    periodic_rank: int
    quotient_cycle_rank: int
    multiplicity: int | None
    component_ranks: tuple[int, ...]
    site_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "cross_edge_ids": list(self.cross_edge_ids),
            "graph": self.graph.to_dict(),
            "periodic_rank": self.periodic_rank,
            "quotient_cycle_rank": self.quotient_cycle_rank,
            "multiplicity": self.multiplicity,
            "component_ranks": list(self.component_ranks),
            "site_ids": list(self.site_ids),
            "translated_copies_are_one_topology": True,
        }


@dataclass(frozen=True)
class NetProposal:
    """One partition of the site graph."""

    proposal_id: str
    status: str
    reason: str
    candidates: tuple[NetCandidate, ...]
    uncovered_node_ids: tuple[str, ...]
    overlap_node_ids: tuple[str, ...]
    contradictory_edge_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "status": self.status,
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "uncovered_node_ids": list(self.uncovered_node_ids),
            "overlap_node_ids": list(self.overlap_node_ids),
            "contradictory_edge_ids": list(self.contradictory_edge_ids),
        }


@dataclass(frozen=True)
class NetDecomposition:
    """The proposals examined for one site graph."""

    site_graph: PeriodicSiteGraph
    mode: str
    proposals: tuple[NetProposal, ...]
    primary_index: int
    primary_is_not_unique: bool
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "primary_index": self.primary_index,
            "primary_proposal_id": self.proposals[self.primary_index].proposal_id,
            "primary_is_not_unique": self.primary_is_not_unique,
            "truncated": self.truncated,
        }

    @property
    def primary(self) -> NetProposal:
        return self.proposals[self.primary_index]


def decompose_periodic_nets(
    site_graph: PeriodicSiteGraph,
    partition_policy: PartitionPolicy | None = None,
    limits: NetAnalysisLimits | None = None,
) -> NetDecomposition:
    """Validate an explicit partition or list the discovery proposals."""
    policy = partition_policy or PartitionPolicy()
    limits = limits or NetAnalysisLimits.from_config()
    if policy.mode == "explicit":
        proposal = _explicit_proposal(site_graph, policy)
        return NetDecomposition(site_graph, "explicit", (proposal,), 0, False, False)
    proposals, truncated = _discovery(site_graph, limits)
    signatures = {
        _signature(proposal)
        for proposal in proposals
        if proposal.status == "admissible"
    }
    primary = _primary_index(proposals)
    return NetDecomposition(
        site_graph,
        "discovery",
        tuple(proposals),
        primary,
        len(signatures) > 1,
        truncated,
    )


def _explicit_proposal(
    site_graph: PeriodicSiteGraph, policy: PartitionPolicy
) -> NetProposal:
    assignment: dict[str, str] = {}
    overlaps: list[str] = []
    for node_id, net_id in policy.assignment:
        if node_id in assignment and assignment[node_id] != net_id:
            overlaps.append(node_id)
            continue
        assignment[node_id] = net_id
    known = set(site_graph.graph.node_ids)
    unknown = sorted(set(assignment) - known)
    uncovered = tuple(sorted(known - set(assignment)))
    contradictory: list[str] = []
    edge_nets = dict(policy.edge_nets)
    for edge in site_graph.graph.edges:
        if edge.edge_id not in edge_nets:
            continue
        net_id = edge_nets[edge.edge_id]
        ends = {assignment.get(edge.source), assignment.get(edge.target)}
        if ends != {net_id}:
            contradictory.append(edge.edge_id)
    invalid = bool(overlaps or uncovered or contradictory or unknown)
    grouped: dict[str, list[str]] = defaultdict(list)
    for node_id, net_id in assignment.items():
        if node_id in known:
            grouped[net_id].append(node_id)
    candidates = tuple(
        _candidate(site_graph, net_id, tuple(sorted(nodes)))
        for net_id, nodes in sorted(grouped.items())
    )
    reason = "admissible"
    if unknown:
        reason = "unknown_node"
    elif overlaps:
        reason = "overlapping_nodes"
    elif uncovered:
        reason = "uncovered_nodes"
    elif contradictory:
        reason = "contradictory_edges"
    return NetProposal(
        "explicit",
        "invalid" if invalid else "admissible",
        reason,
        candidates,
        uncovered,
        tuple(sorted(set(overlaps))),
        tuple(contradictory),
    )


def _discovery(site_graph: PeriodicSiteGraph, limits: NetAnalysisLimits):
    graph = site_graph.graph
    analysis = analyze_periodic_graph(graph)
    proposals = [
        _proposal_from_groups(
            site_graph,
            "components",
            [tuple(component.node_ids) for component in analysis.quotient_components]
            or [()],
        )
    ]
    if graph.node_ids:
        proposals.append(_proposal_from_groups(site_graph, "entire", [graph.node_ids]))
    truncated = False
    orbits = _asym_orbits(site_graph)
    if orbits is not None and len(orbits) > limits.discovery_partition_limit:
        truncated = True
    elif orbits is not None:
        proposals.append(_proposal_from_groups(site_graph, "orbits", orbits))
    # Drop a duplicate of another proposal's partition, keeping the earlier id.
    kept = []
    seen = set()
    for proposal in proposals:
        signature = _signature(proposal)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(proposal)
    return kept, truncated


def _asym_orbits(site_graph: PeriodicSiteGraph) -> list[tuple[str, ...]] | None:
    """Group nodes that carry asymmetric-unit indices.

    Nodes with no index stay in their own group so unequal sites are not
    merged. The whole proposal is omitted when no node has an index: there
    is then no orbit evidence, and a singleton split would not be a discovered
    net.
    """
    grouped: dict[tuple[str, object], list[str]] = defaultdict(list)
    any_index = False
    for node in site_graph.nodes:
        present = tuple(
            sorted({index for index in node.asym_indices if index is not None})
        )
        if not present:
            key: tuple[str, object] = ("node", node.node_id)
        else:
            any_index = True
            key = ("asym", present)
        grouped[key].append(node.node_id)
    if not any_index:
        return None
    return [
        tuple(sorted(nodes))
        for _key, nodes in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _proposal_from_groups(
    site_graph, proposal_id: str, groups: list[tuple[str, ...]]
) -> NetProposal:
    nonempty = [tuple(sorted(group)) for group in groups if group]
    assigned = {node_id for group in nonempty for node_id in group}
    uncovered = tuple(sorted(set(site_graph.graph.node_ids) - assigned))
    grouped = {f"{proposal_id}-{index}": group for index, group in enumerate(nonempty)}
    candidates = tuple(
        _candidate(site_graph, net_id, nodes) for net_id, nodes in grouped.items()
    )
    status = "admissible" if not uncovered else "invalid"
    return NetProposal(
        proposal_id,
        status,
        "admissible" if status == "admissible" else "uncovered_nodes",
        candidates,
        uncovered,
        (),
        (),
    )


def _candidate(site_graph, candidate_id, node_ids) -> NetCandidate:
    node_set = set(node_ids)
    inside = []
    cross = []
    for edge in site_graph.graph.edges:
        ends = {edge.source, edge.target}
        if ends <= node_set:
            inside.append(edge)
        elif ends & node_set:
            cross.append(edge.edge_id)
    graph = PeriodicQuotientGraph.from_edges(
        node_ids,
        tuple(
            (edge.edge_id, edge.source, edge.target, edge.image_shift)
            for edge in inside
        ),
        pbc=site_graph.graph.pbc,
    )
    embedding = _subset_embedding(site_graph.embedding, node_ids)
    analysis = analyze_periodic_graph(graph)
    multiplicities = [
        family.subgroup.multiplicity for family in analysis.cover_families
    ]
    if not multiplicities:
        multiplicity: int | None = 0
    elif any(value is None for value in multiplicities):
        multiplicity = None
    else:
        multiplicity = sum(value for value in multiplicities if value is not None)
    site_ids = []
    lookup = {node.node_id: node for node in site_graph.nodes}
    for node_id in graph.node_ids:
        record = lookup.get(node_id)
        if record is not None:
            site_ids.extend(record.site_ids)
    return NetCandidate(
        candidate_id,
        graph.node_ids,
        tuple(edge.edge_id for edge in graph.edges),
        tuple(cross),
        graph,
        embedding,
        analysis.periodic_rank,
        analysis.quotient_cycle_rank,
        multiplicity,
        tuple(family.subgroup.rank for family in analysis.cover_families),
        tuple(site_ids),
    )


def _subset_embedding(embedding: PeriodicEmbedding | None, node_ids: tuple[str, ...]):
    if embedding is None or not node_ids:
        return None
    present = []
    for node_id in node_ids:
        try:
            present.append((node_id, embedding.position(node_id)))
        except KeyError:
            return None
    return PeriodicEmbedding(tuple(present), embedding.lattice_A)


def _signature(proposal: NetProposal) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(candidate.node_ids) for candidate in proposal.candidates))


def _primary_index(proposals: list[NetProposal]) -> int:
    for preferred in ("components", "entire", "orbits", "explicit"):
        for index, proposal in enumerate(proposals):
            if proposal.proposal_id == preferred and proposal.status == "admissible":
                return index
    return 0


__all__ = [
    "NetCandidate",
    "NetDecomposition",
    "NetProposal",
    "decompose_periodic_nets",
]
