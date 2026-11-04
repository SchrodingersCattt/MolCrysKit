"""Candidate nets inside one periodic site graph.

Explicit mode checks the caller's labels. A non-empty ``edge_nets`` list is a
complete assignment: unknown ids, conflicting duplicates, and edges left out
are invalid, and each candidate keeps only the edges assigned to it.

Discovery mode returns the connectivity components, the whole graph, the
asymmetry-orbit partition, and, within ``discovery_partition_limit``, connected
merges of those orbit groups ranked by how many edges cross between blocks.
It does not invent orbits from missing indices, and it does not drop a second
admissible partition. Translated copies stay inside one candidate and are
counted by cover multiplicity.
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
    known_edges = {edge.edge_id for edge in site_graph.graph.edges}
    edge_nets: dict[str, str] = {}
    unknown_edges: list[str] = []
    conflicting: list[str] = []
    for edge_id, net_id in policy.edge_nets:
        if edge_id not in known_edges:
            unknown_edges.append(edge_id)
            continue
        previous = edge_nets.get(edge_id)
        if previous is not None and previous != net_id:
            conflicting.append(edge_id)
            continue
        edge_nets[edge_id] = net_id
    unassigned = sorted(known_edges - set(edge_nets)) if policy.edge_nets else []
    contradictory: list[str] = []
    for edge in site_graph.graph.edges:
        if edge.edge_id not in edge_nets:
            continue
        net_id = edge_nets[edge.edge_id]
        ends = {assignment.get(edge.source), assignment.get(edge.target)}
        if ends != {net_id}:
            contradictory.append(edge.edge_id)
    invalid = bool(
        overlaps
        or uncovered
        or contradictory
        or unknown
        or unknown_edges
        or conflicting
        or unassigned
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for node_id, net_id in assignment.items():
        if node_id in known:
            grouped[net_id].append(node_id)
    selected = edge_nets if policy.edge_nets else None
    candidates = tuple(
        _candidate(site_graph, net_id, tuple(sorted(nodes)), edge_assignment=selected)
        for net_id, nodes in sorted(grouped.items())
    )
    reason = "admissible"
    if unknown:
        reason = "unknown_node"
    elif unknown_edges:
        reason = "unknown_edge"
    elif conflicting:
        reason = "conflicting_edge_assignment"
    elif overlaps:
        reason = "overlapping_nodes"
    elif uncovered:
        reason = "uncovered_nodes"
    elif unassigned:
        reason = "unassigned_edges"
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
        merges, merge_truncated = _orbit_merges(site_graph, orbits, limits)
        proposals.extend(merges)
        truncated = truncated or merge_truncated
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


def _orbit_merges(site_graph: PeriodicSiteGraph, orbits, limits: NetAnalysisLimits):
    """Connected partitions of orbit groups, fewest crossing edges first.

    The single-block and all-singleton partitions are already returned as
    ``entire`` / ``components`` and ``orbits``. Each extra block must contain
    an induced edge, so a bare orbit is not proposed as its own net. At most
    ``discovery_partition_limit`` further partitions are kept. A larger orbit
    count than can be partitioned exactly sets ``truncated``.
    """
    if len(orbits) > 8:
        return [], True
    ranked = []
    for partition in _index_partitions(len(orbits)):
        if len(partition) <= 1 or all(len(block) == 1 for block in partition):
            continue
        blocks = [
            tuple(node_id for index in block for node_id in orbits[index])
            for block in partition
        ]
        if not all(_block_has_edge(site_graph.graph, block) for block in blocks):
            continue
        crosses = _cross_count(site_graph.graph, blocks)
        signature = tuple(sorted(tuple(sorted(block)) for block in blocks))
        ranked.append((crosses, len(blocks), signature, blocks))
    ranked.sort(key=lambda item: item[:3])
    kept = ranked[: limits.discovery_partition_limit]
    proposals = [
        _proposal_from_groups(
            site_graph,
            f"orbit-merge-{index}",
            [tuple(sorted(block)) for block in blocks],
            reason="ranked_orbit_partition",
        )
        for index, (_crosses, _count, _signature, blocks) in enumerate(kept)
    ]
    return proposals, len(ranked) > len(kept)


def _index_partitions(count: int):
    def generate(items: tuple[int, ...]):
        if not items:
            yield ()
            return
        head, tail = items[0], items[1:]
        for partition in generate(tail):
            yield ((head,),) + partition
            for index, block in enumerate(partition):
                yield tuple(
                    block + (head,) if cursor == index else block
                    for cursor, block in enumerate(partition)
                )

    yield from generate(tuple(range(count)))


def _block_has_edge(graph: PeriodicQuotientGraph, nodes: tuple[str, ...]) -> bool:
    """A block must be connected and contain an edge, not a bare orbit."""
    node_set = set(nodes)
    if not _induced_connected(graph, node_set):
        return False
    return any(
        edge.source in node_set and edge.target in node_set for edge in graph.edges
    )


def _induced_connected(graph: PeriodicQuotientGraph, nodes: set[str]) -> bool:
    if len(nodes) <= 1:
        return True
    parent = {node_id: node_id for node_id in nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in graph.edges:
        if edge.source in nodes and edge.target in nodes and edge.source != edge.target:
            union(edge.source, edge.target)
    return len({find(node_id) for node_id in nodes}) == 1


def _cross_count(graph: PeriodicQuotientGraph, blocks: list[tuple[str, ...]]) -> int:
    index = {
        node_id: block_index
        for block_index, block in enumerate(blocks)
        for node_id in block
    }
    return sum(index.get(edge.source) != index.get(edge.target) for edge in graph.edges)


def _proposal_from_groups(
    site_graph,
    proposal_id: str,
    groups: list[tuple[str, ...]],
    reason: str = "admissible",
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
        reason if status == "admissible" else "uncovered_nodes",
        candidates,
        uncovered,
        (),
        (),
    )


def _candidate(
    site_graph, candidate_id, node_ids, edge_assignment: dict[str, str] | None = None
) -> NetCandidate:
    node_set = set(node_ids)
    inside = []
    cross = []
    for edge in site_graph.graph.edges:
        ends = {edge.source, edge.target}
        if edge_assignment is None:
            chosen = ends <= node_set
        else:
            chosen = (
                edge_assignment.get(edge.edge_id) == candidate_id and ends <= node_set
            )
        if chosen:
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
