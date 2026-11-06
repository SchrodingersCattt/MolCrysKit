"""Typed net-topology report.

The report records ranks, proof status, embedding relations, and duality.
A matching coordination sequence is never upgraded to an isomorphism, and a
missing tiling is ``not_assessed`` rather than a dual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from molcrys_kit.periodic import PeriodicEmbedding, PeriodicQuotientGraph

from .decompose import NetCandidate, NetDecomposition, decompose_periodic_nets
from .duality import PeriodicTiling, assess_duality
from .isomorphism import compare_periodic_nets
from .limits import NetAnalysisLimits
from .linking import assess_linking
from .policies import PartitionPolicy
from .site_graph import PeriodicSiteGraph

SCHEMA = "mck.net_topology/v1"


@dataclass(frozen=True)
class NetTopologyReport:
    """JSON-stable description of one multi-net analysis."""

    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return self.payload


def analyze_periodic_net_system(
    candidates,
    *,
    topology: bool = True,
    linking: bool = True,
    duality: PeriodicTiling | None = None,
    limits: NetAnalysisLimits | None = None,
    partition_policy: PartitionPolicy | None = None,
) -> NetTopologyReport:
    """Analyze discovered or supplied periodic nets.

    ``candidates`` may be a :class:`PeriodicSiteGraph`, a
    :class:`NetDecomposition`, a quotient graph, or a sequence of graphs,
    ``(graph, embedding)`` pairs, or :class:`NetCandidate` objects.
    """
    limits = limits or NetAnalysisLimits.from_config()
    decomposition, supplied = _coerce(candidates, partition_policy, limits)
    disorder_status = decomposition.site_graph.disorder_status
    disorder_reason = decomposition.site_graph.disorder_reason
    proposals = []
    ambiguities = _ambiguities(decomposition, disorder_status, disorder_reason, duality)
    for proposal in decomposition.proposals:
        blocks = []
        for candidate in proposal.candidates:
            blocks.append(
                _candidate_block(
                    candidate,
                    proposal,
                    disorder_status,
                    topology=topology,
                    limits=limits,
                )
            )
        pairs = []
        if proposal.status == "admissible" and disorder_status != "indeterminate":
            pairs = _pairs(
                proposal.candidates,
                linking=linking,
                duality=duality,
                limits=limits,
            )
        elif proposal.status != "admissible":
            ambiguities.append(f"partition:{proposal.proposal_id}:{proposal.reason}")
        proposals.append(
            {
                "proposal_id": proposal.proposal_id,
                "status": proposal.status,
                "reason": proposal.reason,
                "candidates": blocks,
                "pairwise": pairs,
                "uncovered_node_ids": list(proposal.uncovered_node_ids),
                "overlap_node_ids": list(proposal.overlap_node_ids),
                "contradictory_edge_ids": list(proposal.contradictory_edge_ids),
            }
        )
    payload = {
        "schema": SCHEMA,
        "disorder_status": disorder_status,
        "disorder_reason": disorder_reason,
        "source": "supplied_graphs" if supplied else "site_graph",
        "primary_proposal_id": decomposition.primary.proposal_id,
        "primary_is_not_unique": decomposition.primary_is_not_unique,
        "truncated": decomposition.truncated,
        "settings": {
            "limits": limits.to_dict(),
            "topology": topology,
            "linking": linking,
            "duality": None if duality is None else duality.to_dict(),
            "site_graph": decomposition.site_graph.settings,
        },
        "proposals": proposals,
        "ambiguities": sorted(set(ambiguities)),
    }
    return NetTopologyReport(payload)


def _coerce(candidates, partition_policy, limits):
    if isinstance(candidates, NetDecomposition):
        return candidates, False
    if isinstance(candidates, PeriodicSiteGraph):
        return decompose_periodic_nets(candidates, partition_policy, limits), False
    if isinstance(candidates, PeriodicQuotientGraph):
        graph = site_graph_of([(candidates, None)])
        return decompose_periodic_nets(graph, partition_policy, limits), True
    if isinstance(candidates, NetCandidate):
        candidates = [candidates]
    sequence = list(candidates)
    if sequence and isinstance(sequence[0], NetCandidate):
        graph = _from_candidates(sequence)
        seen: dict[str, int] = {}
        assignment: list[tuple[str, str]] = []
        for candidate in sequence:
            seen[candidate.candidate_id] = seen.get(candidate.candidate_id, 0) + 1
            net_id = candidate.candidate_id
            if seen[candidate.candidate_id] > 1:
                net_id = f"{candidate.candidate_id}#{seen[candidate.candidate_id]}"
            assignment.extend((node_id, net_id) for node_id in candidate.node_ids)
        decomposition = decompose_periodic_nets(
            graph, PartitionPolicy("explicit", tuple(assignment)), limits
        )
        return decomposition, True
    pairs = []
    for item in sequence:
        if isinstance(item, PeriodicQuotientGraph):
            pairs.append((item, None))
        else:
            graph, embedding = item
            pairs.append((graph, embedding))
    return decompose_periodic_nets(site_graph_of(pairs), partition_policy, limits), True


def site_graph_of(
    pairs: list[tuple[PeriodicQuotientGraph, PeriodicEmbedding | None]],
) -> PeriodicSiteGraph:
    """Disjoint union of supplied graphs, with node ids prefixed by their index."""
    from .site_graph import site_graph_from_quotient

    nodes = []
    edges = []
    positions = []
    lattice = None
    pbc = pairs[0][0].pbc if pairs else (True, True, True)
    for index, (graph, embedding) in enumerate(pairs):
        if graph.pbc != pbc:
            raise ValueError("supplied graphs must share periodic boundary flags")
        prefix = f"g{index}:"
        nodes.extend(prefix + node_id for node_id in graph.node_ids)
        for edge in graph.edges:
            edges.append(
                (
                    f"{prefix}{edge.edge_id}",
                    prefix + edge.source,
                    prefix + edge.target,
                    edge.image_shift,
                )
            )
        if embedding is None or embedding.lattice_A is None:
            continue
        if lattice is None:
            lattice = embedding.lattice_A
        elif embedding.lattice_A != lattice:
            raise ValueError("supplied graphs must share one lattice")
        for node_id in graph.node_ids:
            positions.append((prefix + node_id, embedding.position(node_id)))
    graph = PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)
    embedding = None
    if positions and lattice is not None and len(positions) == len(nodes):
        embedding = PeriodicEmbedding(tuple(positions), lattice)
    return site_graph_from_quotient(graph, embedding)


def _from_candidates(candidates: list[NetCandidate]) -> PeriodicSiteGraph:
    from .site_graph import site_graph_from_quotient

    # Candidates already carry disjoint ids when they come from one graph.
    nodes: list[str] = []
    edges = []
    positions = []
    lattice = None
    complete = True
    pbc = candidates[0].graph.pbc if candidates else (True, True, True)
    for candidate in candidates:
        if candidate.graph.pbc != pbc:
            raise ValueError("supplied candidates must share periodic boundary flags")
        nodes.extend(candidate.node_ids)
        for edge in candidate.graph.edges:
            edges.append((edge.edge_id, edge.source, edge.target, edge.image_shift))
        embedding = candidate.embedding
        if embedding is None or embedding.lattice_A is None:
            complete = False
            continue
        if lattice is None:
            lattice = embedding.lattice_A
        elif embedding.lattice_A != lattice:
            raise ValueError("supplied candidates must share one lattice")
        for node_id in candidate.node_ids:
            try:
                positions.append((node_id, embedding.position(node_id)))
            except KeyError:
                complete = False
    graph = PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)
    merged = None
    if complete and lattice is not None and nodes and len(positions) == len(nodes):
        merged = PeriodicEmbedding(tuple(positions), lattice)
    return site_graph_from_quotient(graph, merged)


def _candidate_block(candidate, proposal, disorder_status, *, topology: bool, limits):
    block = {
        "candidate_id": candidate.candidate_id,
        "node_ids": list(candidate.node_ids),
        "edge_ids": list(candidate.edge_ids),
        "cross_edge_ids": list(candidate.cross_edge_ids),
        "site_ids": list(candidate.site_ids),
        "periodic_rank": candidate.periodic_rank,
        "quotient_cycle_rank": candidate.quotient_cycle_rank,
        "component_ranks": list(candidate.component_ranks),
        "multiplicity": candidate.multiplicity,
        "translated_copies_are_one_topology": True,
    }
    if disorder_status == "indeterminate" or proposal.status != "admissible":
        block["topology"] = {
            "status": "invalid_or_indeterminate",
            "proof": "none",
            "reason": (
                "disorder_indeterminate"
                if disorder_status == "indeterminate"
                else proposal.reason
            ),
            "canonical_key": None,
        }
        return block
    if not topology:
        block["topology"] = {
            "status": "not_assessed",
            "proof": "none",
            "reason": "topology_disabled",
            "canonical_key": None,
        }
        return block
    compared = compare_periodic_nets(candidate.graph, candidate.graph, limits)
    record = compared.left
    block["topology"] = {
        "status": record.status,
        "proof": compared.proof if record.status == "identified" else "none",
        "reason": record.reason,
        "canonical_key": record.canonical_key,
        "conditional_on_disorder": disorder_status == "conditional",
        "coordination_census": record.to_dict()["coordination_census"],
        "components": [component.to_dict() for component in record.components],
    }
    if disorder_status == "conditional":
        block["topology"]["reason"] = f"conditional:{record.reason}"
    return block


def _pairs(candidates, *, linking: bool, duality, limits):
    rows = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            equivalence = compare_periodic_nets(left.graph, right.graph, limits)
            row = {
                "left": left.candidate_id,
                "right": right.candidate_id,
                "equivalence": {
                    "status": equivalence.status,
                    "proof": equivalence.proof,
                    "reason": equivalence.reason,
                    "canonical_key": equivalence.left.canonical_key
                    if equivalence.status == "isomorphic"
                    else None,
                },
                "embedding": _embedding_row(left, right, linking, limits),
                "duality": _duality_row(left, right, duality, limits),
            }
            rows.append(row)
    if len(candidates) == 1 and duality is not None:
        only = candidates[0]
        equivalence = compare_periodic_nets(only.graph, only.graph, limits)
        rows.append(
            {
                "left": only.candidate_id,
                "right": only.candidate_id,
                "equivalence": {
                    "status": equivalence.status,
                    "proof": equivalence.proof,
                    "reason": equivalence.reason,
                    "canonical_key": (
                        equivalence.left.canonical_key
                        if equivalence.status == "isomorphic"
                        else None
                    ),
                },
                "embedding": {
                    "status": "invalid_or_indeterminate",
                    "relation": "shared_vertices",
                    "reason": "same_candidate",
                    "witness": None,
                },
                "duality": assess_duality(
                    only.graph, only.graph, duality, limits
                ).to_dict(),
            }
        )
    return rows


def _embedding_row(left, right, linking: bool, limits):
    if not linking:
        return {
            "status": "not_assessed",
            "relation": "unknown",
            "reason": "linking_disabled",
            "witness": None,
            "translational_offset": None,
        }
    return assess_linking(
        left.graph,
        left.embedding,
        right.graph,
        right.embedding,
        limits=limits,
        same_vertex_namespace=True,
    ).to_dict()


def _duality_row(left, right, duality, limits):
    if duality is None:
        return {"status": "not_assessed", "reason": "no_tiling", "witness": None}
    return assess_duality(left.graph, right.graph, duality, limits).to_dict()


def _ambiguities(decomposition, disorder_status, disorder_reason, duality) -> list[str]:
    notes = []
    if decomposition.primary_is_not_unique:
        notes.append("primary_partition_is_not_unique")
    if decomposition.truncated:
        notes.append("discovery_truncated")
    if disorder_status == "conditional":
        notes.append(f"disorder_conditional:{disorder_reason}")
    if disorder_status == "indeterminate":
        notes.append(f"disorder_indeterminate:{disorder_reason}")
    if duality is None:
        notes.append("duality_not_assessed")
    return notes


def limits_from_mapping(
    base: NetAnalysisLimits, overrides: dict[str, object]
) -> NetAnalysisLimits:
    """Replace limit fields named in a policy object."""
    if not overrides:
        return base
    current = base.to_dict()
    unknown = set(overrides) - set(current)
    if unknown:
        raise ValueError(f"unknown limit overrides: {sorted(unknown)}")
    updates = {}
    for key, value in overrides.items():
        sample = current[key]
        if isinstance(sample, bool):
            updates[key] = bool(value)
        elif isinstance(sample, int):
            updates[key] = int(value)
        elif isinstance(sample, float):
            updates[key] = float(value)
        else:
            updates[key] = value
    return replace(base, **updates)


__all__ = [
    "SCHEMA",
    "NetTopologyReport",
    "analyze_periodic_net_system",
    "limits_from_mapping",
]
