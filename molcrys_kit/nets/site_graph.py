"""Crystal sites and contacts as a periodic quotient graph.

Node and edge policies are explicit. Sites that share a formula stay distinct
when they are different nodes. Disorder alternatives are kept together only
when the policy says they are one realizable crystal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from molcrys_kit.constants import DEFAULT_NEIGHBOR_CUTOFF
from molcrys_kit.constants.config import PARTIAL_OCCUPANCY_TOLERANCE
from molcrys_kit.periodic import PeriodicEmbedding, PeriodicQuotientGraph
from molcrys_kit.periodic._lattice import ImageShift, canonical_shift

from .limits import NetAnalysisLimits
from .policies import DisorderPolicy, EdgePolicy, NodePolicy

_OCCUPANCY_TOLERANCE = PARTIAL_OCCUPANCY_TOLERANCE
_GEOMETRY_TOLERANCE_A = 1e-4


@dataclass(frozen=True)
class NodeProvenance:
    """Sites contracted into one quotient node, with a reverse map."""

    node_id: str
    kind: str
    site_ids: tuple[str, ...]
    molecule_indices: tuple[int, ...]
    asym_indices: tuple[int | None, ...]
    symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "site_ids": list(self.site_ids),
            "molecule_indices": list(self.molecule_indices),
            "asym_indices": list(self.asym_indices),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True)
class EdgeProvenance:
    """One accepted edge or a recorded alternative that was not used."""

    record_id: str
    kind: str
    status: str
    source_node: str | None
    target_node: str | None
    image_shift: ImageShift | None
    site_ids: tuple[str, ...]
    distance_A: float | None
    criteria: dict[str, object]
    geometry_mismatch: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "status": self.status,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "image_shift": None if self.image_shift is None else list(self.image_shift),
            "site_ids": list(self.site_ids),
            "distance_A": self.distance_A,
            "criteria": self.criteria,
            "geometry_mismatch": self.geometry_mismatch,
        }


@dataclass(frozen=True)
class PeriodicSiteGraph:
    """Quotient graph plus the crystal records that produced it."""

    graph: PeriodicQuotientGraph
    embedding: PeriodicEmbedding | None
    nodes: tuple[NodeProvenance, ...]
    records: tuple[EdgeProvenance, ...]
    disorder_status: str
    disorder_reason: str
    settings: dict[str, object]

    def node(self, node_id: str) -> NodeProvenance:
        for record in self.nodes:
            if record.node_id == node_id:
                return record
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "graph": self.graph.to_dict(),
            "embedding": None if self.embedding is None else self.embedding.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "records": [record.to_dict() for record in self.records],
            "disorder_status": self.disorder_status,
            "disorder_reason": self.disorder_reason,
            "settings": self.settings,
        }


def site_graph_from_quotient(
    graph: PeriodicQuotientGraph,
    embedding: PeriodicEmbedding | None = None,
    *,
    site_ids: dict[str, tuple[str, ...]] | None = None,
) -> PeriodicSiteGraph:
    """Wrap an existing quotient graph without inventing crystal chemistry."""
    mapping = site_ids or {node_id: (node_id,) for node_id in graph.node_ids}
    missing = set(graph.node_ids) - set(mapping)
    if missing:
        raise ValueError(f"site map is missing nodes: {sorted(missing)}")
    nodes = tuple(
        NodeProvenance(
            node_id=node_id,
            kind="quotient",
            site_ids=tuple(mapping[node_id]),
            molecule_indices=(),
            asym_indices=tuple(None for _ in mapping[node_id]),
            symbols=(),
        )
        for node_id in graph.node_ids
    )
    records = tuple(
        EdgeProvenance(
            record_id=edge.edge_id,
            kind="supplied",
            status="defining",
            source_node=edge.source,
            target_node=edge.target,
            image_shift=edge.image_shift,
            site_ids=(),
            distance_A=None,
            criteria={"source": "periodic_quotient_graph"},
        )
        for edge in graph.edges
    )
    return PeriodicSiteGraph(
        graph,
        embedding,
        nodes,
        records,
        "resolved",
        "explicit_graph",
        {"source": "periodic_quotient_graph"},
    )


def build_periodic_site_graph(
    crystal,
    node_policy: NodePolicy | None = None,
    edge_policy: EdgePolicy | None = None,
    disorder_policy: DisorderPolicy | None = None,
    limits: NetAnalysisLimits | None = None,
) -> PeriodicSiteGraph:
    """Copy one realizable crystal into the shared periodic quotient graph."""
    node_policy = node_policy or NodePolicy()
    edge_policy = edge_policy or EdgePolicy()
    disorder_policy = disorder_policy or DisorderPolicy()
    limits = limits or NetAnalysisLimits.from_config()
    sites = list(crystal.get_site_records())
    kept, disorder_status, disorder_reason, disorder_detail = _select_sites(
        sites, disorder_policy
    )
    lattice, pbc = _cell(crystal)
    node_records, site_to_node, unassigned = _nodes(kept, node_policy, crystal)
    points = _node_points(node_records, kept, site_to_node, lattice)
    cutoff = (
        float(DEFAULT_NEIGHBOR_CUTOFF)
        if edge_policy.cutoff_A is None
        else float(edge_policy.cutoff_A)
    )
    records: list[EdgeProvenance] = []
    defining: list[tuple[str, str, str, ImageShift]] = []
    _add_defining_edges(
        crystal,
        kept,
        site_to_node,
        node_records,
        node_policy,
        edge_policy,
        lattice,
        pbc,
        defining,
        records,
    )
    _add_contact_observations(
        node_records,
        points,
        lattice,
        pbc,
        cutoff,
        limits,
        defining if edge_policy.kind == "contacts" else [],
        records,
        define=edge_policy.kind == "contacts",
    )
    graph = PeriodicQuotientGraph.from_edges(
        tuple(node.node_id for node in node_records),
        defining,
        pbc=pbc,
    )
    embedding = None
    if node_records and lattice is not None:
        embedding = PeriodicEmbedding(
            tuple(
                (node.node_id, tuple(float(value) for value in points[node.node_id]))
                for node in node_records
            ),
            lattice,
        )
    settings = {
        "node_policy": node_policy.to_dict(),
        "edge_policy": edge_policy.to_dict(),
        "disorder_policy": disorder_policy.to_dict(),
        "cutoff_A": cutoff,
        "contact_borderline_margin_A": limits.contact_borderline_margin_A,
        "unassigned_site_ids": sorted(unassigned),
        "sites_not_in_nodes": sorted(
            site.site_id for site in kept if site.site_id not in site_to_node
        ),
        "disorder": disorder_detail,
        "limits": limits.to_dict(),
    }
    return PeriodicSiteGraph(
        graph,
        embedding,
        node_records,
        tuple(records),
        disorder_status,
        disorder_reason,
        settings,
    )


def _select_sites(sites, policy: DisorderPolicy):
    families: dict[str, set[int]] = defaultdict(set)
    for site in sites:
        if site.disorder_group > 0 and site.disorder_assembly:
            families[site.disorder_assembly].add(int(site.disorder_group))
    exclusive = {name: groups for name, groups in families.items() if len(groups) > 1}
    detail = {
        "exclusive_assemblies": {
            name: sorted(groups) for name, groups in exclusive.items()
        }
    }
    if policy.mode == "as_refined":
        status, reason = _refined_status(sites, exclusive)
        return list(sites), status, reason, detail
    if policy.mode == "explicit":
        return _explicit_sites(sites, exclusive, policy, detail)
    return _major_sites(sites, exclusive, policy, detail)


def _refined_status(sites, exclusive: dict[str, set[int]]) -> tuple[str, str]:
    if exclusive:
        return "indeterminate", "competitors_retained"
    if any(not _full_occupancy(site) for site in sites):
        return "conditional", "partial_occupancy_retained"
    return "resolved", "no_competitors"


def _explicit_sites(sites, exclusive, policy: DisorderPolicy, detail):
    chosen: dict[str, int] = {}
    for assembly, group in policy.selected_groups:
        if assembly in chosen:
            detail["duplicate_assembly"] = assembly
            return list(sites), "indeterminate", "explicit_selects_two_groups", detail
        chosen[assembly] = group
    for assembly, groups in exclusive.items():
        if assembly not in chosen or chosen[assembly] not in groups:
            detail["unspecified_assembly"] = assembly
            return list(sites), "indeterminate", "explicit_selection_incomplete", detail
    kept = [
        site
        for site in sites
        if site.disorder_group == 0
        or not site.disorder_assembly
        or site.disorder_assembly not in exclusive
        or chosen.get(site.disorder_assembly) == site.disorder_group
    ]
    status = (
        "resolved" if all(_full_occupancy(site) for site in kept) else "conditional"
    )
    reason = (
        "explicit_selection" if status == "conditional" else "explicit_full_occupancy"
    )
    return kept, status, reason, detail


def _major_sites(sites, exclusive, policy: DisorderPolicy, detail):
    if len(exclusive) > 1 and not policy.independent_assemblies:
        detail["assemblies"] = sorted(exclusive)
        return list(sites), "indeterminate", "joint_realization_not_unique", detail
    chosen: dict[str, int] = {}
    for assembly, groups in exclusive.items():
        totals: dict[int, float] = defaultdict(float)
        for site in sites:
            if site.disorder_assembly == assembly and site.disorder_group in groups:
                totals[site.disorder_group] += float(site.occupancy)
        ranking = sorted(totals, key=lambda group: (-totals[group], group))
        if len(ranking) > 1 and totals[ranking[0]] == totals[ranking[1]]:
            detail["tied_assembly"] = assembly
            return list(sites), "indeterminate", "major_occupancy_tie", detail
        chosen[assembly] = ranking[0]
    kept = [
        site
        for site in sites
        if site.disorder_group == 0
        or not site.disorder_assembly
        or site.disorder_assembly not in exclusive
        or chosen.get(site.disorder_assembly) == site.disorder_group
    ]
    dropped = len(kept) != len(sites)
    partial = any(not _full_occupancy(site) for site in kept)
    if dropped or partial:
        return kept, "conditional", "major_component", detail
    return kept, "resolved", "no_competitors", detail


def _full_occupancy(site) -> bool:
    return abs(float(site.occupancy) - 1.0) <= _OCCUPANCY_TOLERANCE


def _cell(crystal):
    lattice_raw = np.asarray(crystal.lattice, dtype=float)
    lattice = tuple(tuple(float(value) for value in row) for row in lattice_raw)
    pbc = tuple(bool(flag) for flag in crystal.pbc)
    return lattice, pbc


def _nodes(sites, policy: NodePolicy, crystal):
    unassigned: list[str] = []
    if policy.kind == "atom_sites":
        groups = [((site,), f"atom:{_site_token(site, sites)}") for site in sites]
    elif policy.kind == "molecular_units":
        by_molecule: dict[int, list] = defaultdict(list)
        for site in sites:
            by_molecule[site.molecule_index].append(site)
        groups = [
            (tuple(by_molecule[index]), f"molecule:{index}")
            for index in sorted(by_molecule)
        ]
    elif policy.kind == "coordination_centres":
        groups = _centre_groups(sites, policy, crystal.get_bond_records())
    elif policy.kind == "explicit_nodes":
        by_id = {site.site_id: site for site in sites}
        groups = []
        used: set[str] = set()
        for index, group in enumerate(policy.groups):
            members = []
            for site_id in group:
                site = by_id.get(site_id)
                if site is None:
                    raise ValueError(
                        f"explicit node group references unknown site {site_id}"
                    )
                if site_id in used:
                    raise ValueError(f"explicit node groups repeat site {site_id}")
                members.append(site)
                used.add(site_id)
            groups.append((tuple(members), f"group:{index}"))
        unassigned = [site.site_id for site in sites if site.site_id not in used]
    else:
        raise ValueError(policy.kind)
    records = []
    site_to_node: dict[str, str] = {}
    for members, node_id in groups:
        records.append(
            NodeProvenance(
                node_id=node_id,
                kind=policy.kind,
                site_ids=tuple(site.site_id for site in members),
                molecule_indices=tuple(
                    sorted({site.molecule_index for site in members})
                ),
                asym_indices=tuple(site.asym_index for site in members),
                symbols=tuple(site.symbol for site in members),
            )
        )
        for site in members:
            site_to_node[site.site_id] = node_id
    return tuple(records), site_to_node, unassigned


def _site_token(site, sites) -> str:
    duplicates = [other.site_id for other in sites].count(site.site_id) > 1
    if duplicates:
        return f"{site.site_id}#{site.global_index}"
    return site.site_id


def _centre_groups(sites, policy: NodePolicy, bonds):
    by_id = {site.site_id: site for site in sites}
    if policy.centre_ids:
        missing = [site_id for site_id in policy.centre_ids if site_id not in by_id]
        if missing:
            raise ValueError(
                f"coordination centres are not in the realized sites: {missing}"
            )
        if len(set(policy.centre_ids)) != len(policy.centre_ids):
            raise ValueError("coordination centres repeat a site id")
        return [
            ((by_id[site_id],), f"centre:{site_id}") for site_id in policy.centre_ids
        ]
    by_local = {(site.molecule_index, site.local_index): site for site in sites}
    counts: dict[str, int] = defaultdict(int)
    for bond in bonds:
        if bond.role != policy.bond_role:
            continue
        for key in (
            (bond.molecule_index, bond.left_local_index),
            (bond.molecule_index, bond.right_local_index),
        ):
            site = by_local.get(key)
            if site is not None:
                counts[site.site_id] += 1
    chosen = [site for site in sites if counts[site.site_id] >= policy.minimum_incident]
    return [((site,), f"centre:{site.site_id}") for site in chosen]


def _node_points(node_records, sites, site_to_node, lattice):
    grouped: dict[str, list] = defaultdict(list)
    for site in sites:
        node_id = site_to_node.get(site.site_id)
        if node_id is not None:
            grouped[node_id].append(np.asarray(site.cartesian_position_A, dtype=float))
    points = {}
    for node in node_records:
        stacked = np.vstack(grouped[node.node_id])
        points[node.node_id] = tuple(float(value) for value in stacked.mean(axis=0))
    return points


def _add_defining_edges(
    crystal,
    sites,
    site_to_node,
    node_records,
    node_policy: NodePolicy,
    policy: EdgePolicy,
    lattice,
    pbc,
    defining,
    records,
):
    by_local = {(site.molecule_index, site.local_index): site for site in sites}
    if policy.kind == "supplied":
        _add_supplied_bonds(crystal, by_local, site_to_node, lattice, defining, records)
    elif policy.kind == "coordination_shell":
        _add_shell_edges(
            crystal,
            by_local,
            site_to_node,
            node_policy.bond_role,
            lattice,
            defining,
            records,
        )
    elif policy.kind == "hydrogen_bonds":
        _add_hydrogen_bonds(crystal, by_local, site_to_node, policy, defining, records)
    elif policy.kind == "user_edges":
        known = {node.node_id for node in node_records}
        for index, (source, target, shift) in enumerate(policy.user_edges):
            if (
                source not in known
                or target not in known
                or not _shift_allowed(shift, pbc)
            ):
                records.append(
                    EdgeProvenance(
                        f"user-{index}",
                        "user_edges",
                        "rejected",
                        source,
                        target,
                        shift,
                        (),
                        None,
                        {"reason": "unknown_node_or_nonperiodic_shift"},
                    )
                )
                continue
            edge_id = f"user-{index}"
            defining.append((edge_id, source, target, shift))
            records.append(
                EdgeProvenance(
                    edge_id,
                    "user_edges",
                    "defining",
                    source,
                    target,
                    shift,
                    (),
                    None,
                    {"source": "user_edges"},
                )
            )
    elif policy.kind == "contacts":
        return
    else:
        raise ValueError(policy.kind)


def _add_supplied_bonds(crystal, by_local, site_to_node, lattice, defining, records):
    for bond in crystal.get_bond_records():
        left = by_local.get((bond.molecule_index, bond.left_local_index))
        right = by_local.get((bond.molecule_index, bond.right_local_index))
        if left is None or right is None:
            continue
        source = site_to_node.get(left.site_id)
        target = site_to_node.get(right.site_id)
        if source is None or target is None:
            continue
        stored = tuple(int(value) for value in bond.right_image_shift)
        shift = _vector_shift(left, right, bond.vector_A, lattice)
        criteria = {"role": bond.role, "recorded_image_shift": list(stored)}
        if shift is None:
            records.append(
                EdgeProvenance(
                    f"bond-{len(records)}",
                    "supplied",
                    "rejected",
                    source,
                    target,
                    stored,
                    (left.site_id, right.site_id),
                    float(bond.distance_A),
                    {**criteria, "reason": "unresolvable_geometry"},
                    True,
                )
            )
            continue
        mismatch = _geometry_mismatch(stored, shift)
        criteria["vector_image_shift"] = list(shift)
        if source == target and shift == (0, 0, 0):
            records.append(
                EdgeProvenance(
                    f"bond-{len(records)}",
                    "supplied",
                    "contracted",
                    source,
                    target,
                    shift,
                    (left.site_id, right.site_id),
                    float(bond.distance_A),
                    criteria,
                    mismatch,
                )
            )
            continue
        edge_id = f"bond-{len(records)}"
        defining.append((edge_id, source, target, shift))
        records.append(
            EdgeProvenance(
                edge_id,
                "supplied",
                "defining",
                source,
                target,
                shift,
                (left.site_id, right.site_id),
                float(bond.distance_A),
                criteria,
                mismatch,
            )
        )


def _vector_shift(left, right, vector, lattice):
    """Integer image that makes ``vector`` agree with the site coordinates.

    A stored site-image difference is not used by itself. A finite molecule
    whose fractional coordinates cross a cell face can carry a nonzero site
    image while its Cartesian bond stays inside one molecule. The shift
    returned here is the one implied by that Cartesian vector. ``None`` means
    the vector is not an integer lattice image of the two positions.
    """
    matrix = np.asarray(lattice, dtype=float)
    delta = np.asarray(right.cartesian_position_A, dtype=float) - np.asarray(
        left.cartesian_position_A, dtype=float
    )
    try:
        raw = (np.asarray(vector, dtype=float) - delta) @ np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    rounded = np.rint(raw)
    error = float(np.linalg.norm((raw - rounded) @ matrix))
    if error > _GEOMETRY_TOLERANCE_A:
        return None
    return tuple(int(value) for value in rounded)


def _geometry_mismatch(stored: ImageShift, resolved: ImageShift) -> bool:
    return stored != resolved


def _add_shell_edges(
    crystal, by_local, site_to_node, bond_role: str, lattice, defining, records
):
    """Connect centres that share a ligand of ``bond_role``.

    The shift from centre A to centre B is the difference of the shifts from
    each centre to the shared ligand. Ligands are realized sites that were
    not selected as centres.
    """
    centre_ids = {
        site_id
        for site_id, node_id in site_to_node.items()
        if node_id.startswith("centre:")
    }
    attachments: dict[str, list[tuple[str, ImageShift]]] = defaultdict(list)
    for bond in crystal.get_bond_records():
        if bond.role != bond_role:
            continue
        left = by_local.get((bond.molecule_index, bond.left_local_index))
        right = by_local.get((bond.molecule_index, bond.right_local_index))
        if left is None or right is None:
            continue
        stored = tuple(int(value) for value in bond.right_image_shift)
        shift = _vector_shift(left, right, bond.vector_A, lattice)
        if shift is None:
            records.append(
                EdgeProvenance(
                    f"shell-rejected-{len(records)}",
                    "coordination_shell",
                    "rejected",
                    site_to_node.get(left.site_id),
                    site_to_node.get(right.site_id),
                    stored,
                    (left.site_id, right.site_id),
                    float(bond.distance_A),
                    {"reason": "unresolvable_geometry", "bond_role": bond_role},
                    True,
                )
            )
            continue
        directed = (
            (left, right, shift),
            (right, left, (-shift[0], -shift[1], -shift[2])),
        )
        for centre, ligand, centre_to_ligand in directed:
            if centre.site_id not in centre_ids or ligand.site_id in centre_ids:
                continue
            attachments[ligand.site_id].append(
                (site_to_node[centre.site_id], centre_to_ligand)
            )
    seen: set[tuple[str, str, ImageShift, str]] = set()
    for ligand_id, linked in attachments.items():
        for index, (left_node, left_shift) in enumerate(linked):
            for right_node, right_shift in linked[index + 1 :]:
                if left_node == right_node:
                    continue
                bridge = (
                    left_shift[0] - right_shift[0],
                    left_shift[1] - right_shift[1],
                    left_shift[2] - right_shift[2],
                )
                key = (left_node, right_node, bridge, ligand_id)
                if key in seen:
                    continue
                seen.add(key)
                edge_id = f"shell-{len(defining)}"
                defining.append((edge_id, left_node, right_node, bridge))
                records.append(
                    EdgeProvenance(
                        edge_id,
                        "coordination_shell",
                        "defining",
                        left_node,
                        right_node,
                        bridge,
                        (ligand_id,),
                        None,
                        {"bond_role": bond_role, "ligand_site_id": ligand_id},
                    )
                )


def _add_hydrogen_bonds(
    crystal, by_local, site_to_node, policy: EdgePolicy, defining, records
):
    from molcrys_kit.analysis.interactions import (
        HydrogenBondCriteria,
        find_hydrogen_bonds,
    )

    criteria_kwargs = {}
    if policy.max_h_acceptor_distance_A is not None:
        criteria_kwargs["max_h_acceptor_distance_A"] = policy.max_h_acceptor_distance_A
    if policy.min_dha_angle_deg is not None:
        criteria_kwargs["min_dha_angle_deg"] = policy.min_dha_angle_deg
    criteria = HydrogenBondCriteria(**criteria_kwargs) if criteria_kwargs else None
    found = find_hydrogen_bonds(crystal, criteria=criteria)
    for bond in found:
        donor = by_local.get((bond.donor.molecule_index, bond.donor.atom_index))
        acceptor = by_local.get(
            (bond.acceptor.molecule_index, bond.acceptor.atom_index)
        )
        if donor is None or acceptor is None:
            continue
        source = site_to_node.get(donor.site_id)
        target = site_to_node.get(acceptor.site_id)
        if source is None or target is None:
            continue
        shift = tuple(int(value) for value in bond.acceptor.image)
        edge_id = f"hbond-{len(defining)}"
        defining.append((edge_id, source, target, shift))
        records.append(
            EdgeProvenance(
                edge_id,
                "hydrogen_bonds",
                "defining",
                source,
                target,
                shift,
                (donor.site_id, acceptor.site_id),
                float(bond.distance),
                {
                    "max_h_acceptor_distance_A": (
                        criteria.max_h_acceptor_distance_A
                        if criteria is not None
                        else None
                    ),
                    "min_dha_angle_deg": (
                        criteria.min_dha_angle_deg if criteria is not None else None
                    ),
                },
            )
        )


def _add_contact_observations(
    node_records,
    points,
    lattice,
    pbc,
    cutoff,
    limits: NetAnalysisLimits,
    existing,
    records,
    *,
    define: bool,
):
    if len(node_records) < 1 or lattice is None:
        return
    margin = float(limits.contact_borderline_margin_A)
    matrix = np.asarray(lattice, dtype=float)
    fractional = {}
    inverse = np.linalg.inv(matrix)
    for node in node_records:
        cartesian = np.asarray(points[node.node_id], dtype=float)
        fractional[node.node_id] = cartesian @ inverse
    images = _image_box(matrix, pbc, cutoff + margin)
    ids = [node.node_id for node in node_records]
    seen = {
        _pair_key(source, target, shift) for _edge_id, source, target, shift in existing
    }
    for left_index, source in enumerate(ids):
        for target in ids[left_index:]:
            for image in images:
                if source == target and image == (0, 0, 0):
                    continue
                if source == target and canonical_shift(image) != image:
                    continue
                if not _shift_allowed(image, pbc):
                    continue
                delta = (
                    fractional[target]
                    - fractional[source]
                    + np.asarray(image, dtype=float)
                )
                distance = float(np.linalg.norm(delta @ matrix))
                key = _pair_key(source, target, image)
                if distance <= cutoff:
                    if key in seen:
                        continue
                    seen.add(key)
                    edge_id = f"contact-{len(records)}"
                    status = "defining" if define else "observed"
                    if define:
                        existing.append((edge_id, source, target, image))
                    records.append(
                        EdgeProvenance(
                            edge_id,
                            "contacts",
                            status,
                            source,
                            target,
                            image,
                            (),
                            distance,
                            {"cutoff_A": cutoff, "margin_A": margin},
                        )
                    )
                elif distance <= cutoff + margin:
                    records.append(
                        EdgeProvenance(
                            f"border-{len(records)}",
                            "contacts",
                            "borderline",
                            source,
                            target,
                            image,
                            (),
                            distance,
                            {"cutoff_A": cutoff, "margin_A": margin},
                        )
                    )


def _image_box(lattice, pbc, reach: float):
    spans = []
    for row, periodic in zip(lattice, pbc):
        if not periodic:
            spans.append((0,))
            continue
        length = float(np.linalg.norm(row))
        if length <= 0:
            spans.append((0,))
            continue
        limit = int(np.ceil(reach / length))
        spans.append(tuple(range(-limit, limit + 1)))
    images = []
    for i in spans[0]:
        for j in spans[1]:
            for k in spans[2]:
                images.append((i, j, k))
    return tuple(images)


def _shift_allowed(shift: ImageShift, pbc) -> bool:
    return all(component == 0 or periodic for component, periodic in zip(shift, pbc))


def _pair_key(source: str, target: str, shift: ImageShift):
    if source < target or (source == target and canonical_shift(shift) == shift):
        return (source, target, shift)
    return (target, source, (-shift[0], -shift[1], -shift[2]))


__all__ = [
    "EdgeProvenance",
    "NodeProvenance",
    "PeriodicSiteGraph",
    "build_periodic_site_graph",
    "site_graph_from_quotient",
]
