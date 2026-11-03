"""Adapters from existing periodic graphs onto the quotient multigraph.

The three domain representations stay in place. Nothing here subclasses them
or changes their constructors, winding, chemistry, or coordination-shell JSON.

========================  ================================================
Representation            What stays outside the quotient graph
========================  ================================================
Fragment ``PeriodicGraph``  closure mode, rule id, ports, closure flag
Chemical entity             element, bond kind/order, evidence, declared rank
Polyhedron network          shells, shared vertices, relation, disorder policy
Caller-supplied edges       whatever scalar attributes the caller attaches
========================  ================================================

Shared mathematical fields (node id, edge id, exact image shift, PBC) are
copied into :class:`~molcrys_kit.periodic.graph.PeriodicQuotientGraph`.
Provenance keeps the domain fields, including the original edge orientation,
so a later reversal inside the quotient does not drop ports or bond kind.

``PeriodicGraph.winding_cycles`` is not used. Chain construction keeps that
optimized path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from molcrys_kit.analysis.polyhedron_network import (
    PolyhedronNetwork,
    SharedContact,
)
from molcrys_kit.chemistry.models import (
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    FiniteChemicalEntity,
    PeriodicChemicalEntity,
)
from molcrys_kit.structures.periodic_geometry import PeriodicEdge, PeriodicGraph

from ._lattice import ImageShift, exact_shift
from .graph import (
    PeriodicEmbedding,
    PeriodicQuotientGraph,
    _prepare_edges,
)

_ChemicalEntity = FiniteChemicalEntity | PeriodicChemicalEntity


@dataclass(frozen=True)
class FragmentEdgeProvenance:
    """One fragment-chain edge, in the orientation stored on ``PeriodicEdge``."""

    edge_id: str
    left_node: str
    right_node: str
    right_image_shift: ImageShift
    rule_id: str
    closure: bool
    left_port: str | None
    right_port: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "left_node": self.left_node,
            "right_node": self.right_node,
            "right_image_shift": list(self.right_image_shift),
            "rule_id": self.rule_id,
            "closure": bool(self.closure),
            "left_port": self.left_port,
            "right_port": self.right_port,
        }


@dataclass(frozen=True)
class FragmentGraphProvenance:
    """Lossless snapshot of a fragment ``PeriodicGraph``."""

    kind: str
    closure: str
    node_order: tuple[str, ...]
    edges: tuple[FragmentEdgeProvenance, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "closure": self.closure,
            "node_order": list(self.node_order),
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class ChemicalGraphProvenance:
    """Chemical atoms and bonds that produced a quotient graph.

    ``declared_rank`` and ``declared_generators`` are the values stored on a
    :class:`~molcrys_kit.chemistry.models.PeriodicChemicalEntity`. They are
    not recomputed. Finite entities leave them as ``None``. The mathematical
    rank is :func:`molcrys_kit.periodic.algorithms.periodic_rank` of the graph.
    """

    kind: str
    entity_id: str
    declared_dimension: int
    declared_rank: int | None
    declared_generators: tuple[ImageShift, ...] | None
    status: str
    net_charge: int | None
    warnings: tuple[str, ...]
    atoms: tuple[ChemicalAtom, ...]
    bonds: tuple[ChemicalBond, ...]
    edge_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.edge_ids) != len(self.bonds):
            raise ValueError("chemical edge ids must align with bonds")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "entity_id": self.entity_id,
            "declared_dimension": self.declared_dimension,
            "declared_rank": self.declared_rank,
            "declared_generators": None
            if self.declared_generators is None
            else [list(vector) for vector in self.declared_generators],
            "status": self.status,
            "net_charge": self.net_charge,
            "warnings": list(self.warnings),
            "atoms": [_atom_dict(atom) for atom in self.atoms],
            "bonds": [
                _bond_dict(bond, edge_id)
                for bond, edge_id in zip(self.bonds, self.edge_ids)
            ],
            "evidence": [_evidence_dict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class PolyhedronGraphProvenance:
    """Coordination-shell network kept beside its quotient graph.

    ``network`` is the original object. ``node_ids`` rows are
    ``(node_id, molecule_index, realization_index)``. ``edge_ids`` rows are
    ``(edge_id, contact_index)`` into ``network.contacts``.
    """

    kind: str
    node_ids: tuple[tuple[str, int, int], ...]
    edge_ids: tuple[tuple[str, int], ...]
    center_kind: str | None
    centre_disorder_policy: str | None
    ligand_disorder_policy: str | None
    network: PolyhedronNetwork

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "node_ids": [
                [node_id, molecule_index, realization_index]
                for node_id, molecule_index, realization_index in self.node_ids
            ],
            "edge_ids": [
                [edge_id, contact_index] for edge_id, contact_index in self.edge_ids
            ],
            "center_kind": self.center_kind,
            "centre_disorder_policy": self.centre_disorder_policy,
            "ligand_disorder_policy": self.ligand_disorder_policy,
            "network": self.network.to_dict(),
        }


@dataclass(frozen=True)
class GraphAttribute:
    """One JSON scalar attached to a caller-supplied node or edge."""

    key: str
    type_name: str
    value: str

    def to_dict(self) -> dict[str, object]:
        parsed: object
        if self.type_name == "bool":
            parsed = self.value == "true"
        elif self.type_name == "int":
            parsed = int(self.value)
        elif self.type_name == "float":
            parsed = float(self.value)
        elif self.type_name == "none":
            parsed = None
        else:
            parsed = self.value
        return {"key": self.key, "type": self.type_name, "value": parsed}


@dataclass(frozen=True)
class CallerGraphProvenance:
    """Scalar attributes for a graph supplied by the caller."""

    kind: str
    node_attributes: tuple[tuple[str, tuple[GraphAttribute, ...]], ...] = ()
    edge_attributes: tuple[tuple[str, tuple[GraphAttribute, ...]], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "node_attributes": [
                [node_id, [attribute.to_dict() for attribute in attributes]]
                for node_id, attributes in self.node_attributes
            ],
            "edge_attributes": [
                [edge_id, [attribute.to_dict() for attribute in attributes]]
                for edge_id, attributes in self.edge_attributes
            ],
        }


@dataclass(frozen=True)
class QuotientGraphProvenance:
    """Marker for a graph that is already a periodic quotient."""

    kind: str = "quotient_graph"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind}


PeriodicGraphProvenance = (
    FragmentGraphProvenance
    | ChemicalGraphProvenance
    | PolyhedronGraphProvenance
    | CallerGraphProvenance
    | QuotientGraphProvenance
)


@dataclass(frozen=True)
class PeriodicGraphAdaptation:
    """Quotient graph, optional embedding, and domain provenance."""

    graph: PeriodicQuotientGraph
    embedding: PeriodicEmbedding | None
    provenance: PeriodicGraphProvenance

    def __post_init__(self) -> None:
        if self.embedding is None:
            return
        embedded = {node_id for node_id, _vector in self.embedding.positions_A}
        if embedded != set(self.graph.node_ids):
            raise ValueError("embedding nodes must match the quotient graph")

    def to_dict(self) -> dict[str, object]:
        return {
            "graph": self.graph.to_dict(),
            "embedding": None if self.embedding is None else self.embedding.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


def _evidence_dict(evidence: Evidence) -> dict[str, object]:
    return {
        "source": evidence.source.value,
        "method": evidence.method,
        "detail": evidence.detail,
        "confidence": evidence.confidence,
    }


def _atom_dict(atom: ChemicalAtom) -> dict[str, object]:
    return {
        "atom_id": atom.atom_id,
        "element": atom.element,
        "label": atom.label,
        "isotope": atom.isotope,
        "formal_charge": atom.formal_charge,
        "radical_electrons": atom.radical_electrons,
        "explicit_hydrogens": atom.explicit_hydrogens,
        "implicit_hydrogens": atom.implicit_hydrogens,
        "oxidation_state": atom.oxidation_state,
        "stereochemistry": atom.stereochemistry,
        "evidence": [_evidence_dict(item) for item in atom.evidence],
    }


def _bond_dict(bond: ChemicalBond, edge_id: str) -> dict[str, object]:
    return {
        "edge_id": edge_id,
        "atom1_id": bond.atom1_id,
        "atom2_id": bond.atom2_id,
        "order": bond.order,
        "kind": bond.kind.value,
        "aromatic": bool(bond.aromatic),
        "atom2_image_shift": list(exact_shift(bond.atom2_image_shift)),
        "stereochemistry": bond.stereochemistry,
        "evidence": [_evidence_dict(item) for item in bond.evidence],
    }


def _numbered(
    items: Sequence[object], keys: Sequence[object]
) -> tuple[tuple[object, int], ...]:
    buckets: dict[object, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        buckets[key].append(index)
    occurrence_of = [0] * len(items)
    for indices in buckets.values():
        for occurrence, index in enumerate(indices):
            occurrence_of[index] = occurrence
    return tuple((item, occurrence_of[index]) for index, item in enumerate(items))


def _fragment_edge_id(edge: PeriodicEdge, occurrence: int) -> str:
    shift = edge.right_image_shift
    left_port = "" if edge.left_port is None else edge.left_port
    right_port = "" if edge.right_port is None else edge.right_port
    return (
        f"fragment:{edge.left_node}->{edge.right_node}"
        f"@{shift[0]}_{shift[1]}_{shift[2]}"
        f"/rule={edge.rule_id}/closure={int(bool(edge.closure))}"
        f"/ports={left_port},{right_port}#{occurrence}"
    )


def adapt_fragment_graph(
    graph: PeriodicGraph,
    *,
    pbc: Sequence[bool] = (True, True, True),
) -> PeriodicGraphAdaptation:
    """Adapt a fragment-chain graph without using its winding fast path.

    ``PeriodicGraph`` does not store PBC flags. The default is full
    periodicity, matching graphs whose shifts were produced under
    ``pbc=(True, True, True)``. Pass the builder flags when an axis is not
    periodic; multiplicity depends on that choice, the periodic rank does not
    when every illegal shift component is already zero.
    """
    if not isinstance(graph, PeriodicGraph):
        raise TypeError("adapt_fragment_graph expects a PeriodicGraph")
    keys = [
        (
            edge.left_node,
            edge.right_node,
            exact_shift(edge.right_image_shift),
            edge.rule_id,
            bool(edge.closure),
            edge.left_port,
            edge.right_port,
        )
        for edge in graph.edges
    ]
    edge_ids = tuple(
        _fragment_edge_id(edge, occurrence)
        for edge, occurrence in _numbered(graph.edges, keys)
    )
    provenance_edges = tuple(
        FragmentEdgeProvenance(
            edge_id=edge_id,
            left_node=edge.left_node,
            right_node=edge.right_node,
            right_image_shift=exact_shift(edge.right_image_shift),
            rule_id=edge.rule_id,
            closure=bool(edge.closure),
            left_port=edge.left_port,
            right_port=edge.right_port,
        )
        for edge, edge_id in zip(graph.edges, edge_ids)
    )
    quotient_edges = tuple(
        (edge_id, edge.left_node, edge.right_node, edge.right_image_shift)
        for edge, edge_id in zip(graph.edges, edge_ids)
    )
    quotient = PeriodicQuotientGraph.from_edges(graph.nodes, quotient_edges, pbc=pbc)
    provenance = FragmentGraphProvenance(
        kind="fragment_graph",
        closure=graph.closure,
        node_order=tuple(graph.nodes),
        edges=provenance_edges,
    )
    return PeriodicGraphAdaptation(quotient, None, provenance)


def _chemical_edge_id(bond: ChemicalBond, occurrence: int) -> str:
    shift = exact_shift(bond.atom2_image_shift)
    order = "" if bond.order is None else format(float(bond.order), ".16g")
    stereo = "" if bond.stereochemistry is None else bond.stereochemistry
    return (
        f"bond:{bond.atom1_id}->{bond.atom2_id}"
        f"@{shift[0]}_{shift[1]}_{shift[2]}"
        f"/kind={bond.kind.value}/order={order}/aromatic={int(bool(bond.aromatic))}"
        f"/stereo={stereo}#{occurrence}"
    )


def _embedding_from_chemical(entity: _ChemicalEntity) -> PeriodicEmbedding | None:
    embedding = entity.embedding
    if embedding is None:
        return None
    atom_ids = {atom.atom_id for atom in entity.atoms}
    coordinates = tuple(
        (atom_id, tuple(float(component) for component in vector))
        for atom_id, vector in embedding.coordinates_A
    )
    if {atom_id for atom_id, _vector in coordinates} != atom_ids:
        raise ValueError("chemical embedding must include every atom in the entity")
    lattice = None
    if embedding.cell_A is not None:
        lattice = tuple(
            tuple(float(component) for component in row) for row in embedding.cell_A
        )
    if embedding.coordinate_system == "cartesian":
        return PeriodicEmbedding(coordinates, lattice)
    if embedding.coordinate_system == "fractional":
        if lattice is None:
            raise ValueError("fractional chemical embedding requires cell_A")
        matrix = np.asarray(lattice, dtype=float)
        positions = tuple(
            (
                atom_id,
                tuple(float(component) for component in np.asarray(vector) @ matrix),
            )
            for atom_id, vector in coordinates
        )
        return PeriodicEmbedding(positions, lattice, coordinates)
    raise ValueError(
        "unsupported chemical coordinate system "
        f"{embedding.coordinate_system!r}; expected 'cartesian' or 'fractional'"
    )


def adapt_chemical_entity(
    entity: _ChemicalEntity,
    *,
    pbc: Sequence[bool] = (True, True, True),
) -> PeriodicGraphAdaptation:
    """Adapt a finite or periodic chemical entity.

    Polymer and multicomponent records are not bond graphs; pass one
    component entity instead. Chemical entities do not store PBC flags. The
    default matches :func:`molcrys_kit.chemistry.topology.analyze_periodic_topology`.
    Pass the crystal flags when a non-periodic axis should not contribute
    extra cover copies.
    """
    if not isinstance(entity, (FiniteChemicalEntity, PeriodicChemicalEntity)):
        raise TypeError(
            "adapt_chemical_entity expects FiniteChemicalEntity or "
            "PeriodicChemicalEntity"
        )
    keys = [
        (
            bond.atom1_id,
            bond.atom2_id,
            exact_shift(bond.atom2_image_shift),
            bond.kind.value,
            None if bond.order is None else float(bond.order),
            bool(bond.aromatic),
            bond.stereochemistry,
        )
        for bond in entity.bonds
    ]
    numbered = _numbered(entity.bonds, keys)
    edge_ids = tuple(
        _chemical_edge_id(bond, occurrence) for bond, occurrence in numbered
    )
    quotient = PeriodicQuotientGraph.from_edges(
        tuple(atom.atom_id for atom in entity.atoms),
        tuple(
            (edge_id, bond.atom1_id, bond.atom2_id, exact_shift(bond.atom2_image_shift))
            for bond, edge_id in zip(entity.bonds, edge_ids)
        ),
        pbc=pbc,
    )
    if isinstance(entity, PeriodicChemicalEntity):
        declared_rank: int | None = int(entity.periodic_rank)
        declared_generators: tuple[ImageShift, ...] | None = tuple(
            exact_shift(vector) for vector in entity.translation_generators
        )
        net_charge = entity.net_charge_per_repeat
    else:
        declared_rank = None
        declared_generators = None
        net_charge = entity.net_charge
    provenance = ChemicalGraphProvenance(
        kind="chemical_entity",
        entity_id=entity.entity_id,
        declared_dimension=int(entity.dimension),
        declared_rank=declared_rank,
        declared_generators=declared_generators,
        status=entity.status.value,
        net_charge=net_charge,
        warnings=tuple(entity.warnings),
        atoms=entity.atoms,
        bonds=entity.bonds,
        edge_ids=edge_ids,
        evidence=entity.evidence,
    )
    return PeriodicGraphAdaptation(
        quotient, _embedding_from_chemical(entity), provenance
    )


def _centre_node_id(molecule_index: int, realization_index: int) -> str:
    return f"centre:{int(molecule_index)}:{int(realization_index)}"


def _contact_edge_id(contact: SharedContact, occurrence: int) -> str:
    shift = exact_shift(contact.right_image_shift)
    return (
        f"contact:{contact.left.molecule_index}:{contact.left.realization_index}"
        f"->{contact.right.molecule_index}:{contact.right.realization_index}"
        f"@{shift[0]}_{shift[1]}_{shift[2]}#{occurrence}"
    )


def adapt_polyhedron_network(network: PolyhedronNetwork) -> PeriodicGraphAdaptation:
    """Adapt one coordination-shell network, preserving contacts and shells."""
    if not isinstance(network, PolyhedronNetwork):
        raise TypeError("adapt_polyhedron_network expects a PolyhedronNetwork")
    node_rows = []
    seen: set[str] = set()
    for shell in network.shells:
        node_id = _centre_node_id(shell.molecule_index, shell.realization_index)
        if node_id in seen:
            raise ValueError(f"duplicate coordination centre {node_id}")
        seen.add(node_id)
        node_rows.append(
            (node_id, int(shell.molecule_index), int(shell.realization_index))
        )
    keys = [
        (
            contact.left.molecule_index,
            contact.left.realization_index,
            contact.right.molecule_index,
            contact.right.realization_index,
            exact_shift(contact.right_image_shift),
        )
        for contact in network.contacts
    ]
    numbered = _numbered(network.contacts, keys)
    edge_ids = tuple(
        _contact_edge_id(contact, occurrence) for contact, occurrence in numbered
    )
    quotient_edges = []
    for contact, edge_id in zip(network.contacts, edge_ids):
        source = _centre_node_id(
            contact.left.molecule_index, contact.left.realization_index
        )
        target = _centre_node_id(
            contact.right.molecule_index, contact.right.realization_index
        )
        if source not in seen or target not in seen:
            raise ValueError(f"contact {edge_id} references a centre that has no shell")
        quotient_edges.append(
            (edge_id, source, target, exact_shift(contact.right_image_shift))
        )
    edge_rows = [(edge_id, index) for index, edge_id in enumerate(edge_ids)]
    quotient = PeriodicQuotientGraph.from_edges(
        tuple(node_id for node_id, _molecule, _realization in node_rows),
        quotient_edges,
        pbc=tuple(bool(flag) for flag in network.pbc),
    )
    lattice = tuple(
        tuple(float(component) for component in row) for row in network.lattice_A
    )
    positions = tuple(
        (
            _centre_node_id(shell.molecule_index, shell.realization_index),
            tuple(float(component) for component in shell.center_position_A),
        )
        for shell in network.shells
    )
    embedding = PeriodicEmbedding(positions, lattice)
    provenance = PolyhedronGraphProvenance(
        kind="polyhedron_network",
        node_ids=tuple(node_rows),
        edge_ids=tuple(edge_rows),
        center_kind=network.center_kind,
        centre_disorder_policy=network.centre_disorder_policy,
        ligand_disorder_policy=network.ligand_disorder_policy,
        network=network,
    )
    return PeriodicGraphAdaptation(quotient, embedding, provenance)


def _one_attribute(key: str, value: object) -> GraphAttribute:
    if isinstance(value, bool):
        return GraphAttribute(key, "bool", "true" if value else "false")
    if isinstance(value, int):
        return GraphAttribute(key, "int", str(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"attribute {key!r} must be finite")
        return GraphAttribute(key, "float", format(value, ".16g"))
    if isinstance(value, str):
        return GraphAttribute(key, "str", value)
    if value is None:
        return GraphAttribute(key, "none", "")
    raise TypeError(
        f"attribute {key!r} must be a string, finite number, boolean, or null"
    )


def _freeze_attributes(
    mapping: Mapping[str, object] | None,
) -> tuple[GraphAttribute, ...]:
    if not mapping:
        return ()
    attributes = []
    for key in sorted(mapping):
        if not isinstance(key, str) or not key:
            raise ValueError("attribute keys must be non-empty strings")
        attributes.append(_one_attribute(key, mapping[key]))
    return tuple(attributes)


def adapt_caller_graph(
    nodes: Sequence[str],
    edges: Sequence[object],
    *,
    pbc: Sequence[object] = (True, True, True),
    embedding: PeriodicEmbedding | None = None,
    node_attributes: Mapping[str, Mapping[str, object]] | None = None,
    edge_attributes: Sequence[Mapping[str, object] | None] | None = None,
) -> PeriodicGraphAdaptation:
    """Adapt an explicit periodic graph.

    Each edge is a :class:`~molcrys_kit.periodic.graph.PeriodicQuotientEdge`,
    ``(source, target, shift)``, or ``(edge_id, source, target, shift)``.
    ``edge_attributes``, when given, aligns with ``edges`` and may contain
    ``None``. Attributes are JSON scalars.
    """
    prepared, input_ids = _prepare_edges(edges)  # type: ignore[arg-type]
    graph = PeriodicQuotientGraph(tuple(nodes), prepared, pbc=pbc)  # type: ignore[arg-type]
    if edge_attributes is None:
        stored_edges: tuple[tuple[str, tuple[GraphAttribute, ...]], ...] = ()
    else:
        if len(edge_attributes) != len(input_ids):
            raise ValueError("edge_attributes must align with edges")
        stored_edges = tuple(
            (edge_id, _freeze_attributes(attributes))
            for edge_id, attributes in zip(input_ids, edge_attributes)
            if attributes
        )
    if node_attributes is None:
        stored_nodes: tuple[tuple[str, tuple[GraphAttribute, ...]], ...] = ()
    else:
        unknown = set(node_attributes) - set(graph.node_ids)
        if unknown:
            raise ValueError(
                "node attributes reference unknown nodes: " + ", ".join(sorted(unknown))
            )
        stored_nodes = tuple(
            (node_id, _freeze_attributes(node_attributes[node_id]))
            for node_id in graph.node_ids
            if node_id in node_attributes and node_attributes[node_id]
        )
    return PeriodicGraphAdaptation(
        graph,
        embedding,
        CallerGraphProvenance("caller", stored_nodes, stored_edges),
    )


def adapt_periodic_graph(source: object) -> PeriodicGraphAdaptation:
    """Adapt a supported periodic graph, or return an existing adaptation."""
    if isinstance(source, PeriodicGraphAdaptation):
        return source
    if isinstance(source, PeriodicQuotientGraph):
        return PeriodicGraphAdaptation(source, None, QuotientGraphProvenance())
    if isinstance(source, PeriodicGraph):
        return adapt_fragment_graph(source)
    if isinstance(source, (FiniteChemicalEntity, PeriodicChemicalEntity)):
        return adapt_chemical_entity(source)
    if isinstance(source, PolyhedronNetwork):
        return adapt_polyhedron_network(source)
    raise TypeError(
        "unsupported periodic graph source "
        f"{type(source).__name__}; expected PeriodicQuotientGraph, PeriodicGraph, "
        "FiniteChemicalEntity, PeriodicChemicalEntity, or PolyhedronNetwork. "
        "Build a caller-supplied graph with adapt_caller_graph."
    )


def to_periodic_quotient_graph(source: object) -> PeriodicQuotientGraph:
    """Return the canonical quotient graph for a supported domain graph.

    Provenance is available from :func:`adapt_periodic_graph`. Rank is
    :func:`molcrys_kit.periodic.algorithms.periodic_rank` of the result.
    This does not classify nets.
    """
    return adapt_periodic_graph(source).graph


def fragment_graph_from_provenance(
    provenance: FragmentGraphProvenance,
) -> PeriodicGraph:
    """Rebuild the fragment graph stored in an adaptation."""
    if provenance.kind != "fragment_graph":
        raise TypeError("fragment provenance is required")
    return PeriodicGraph(
        provenance.node_order,
        tuple(
            PeriodicEdge(
                edge.left_node,
                edge.right_node,
                edge.right_image_shift,
                rule_id=edge.rule_id,
                closure=edge.closure,
                left_port=edge.left_port,
                right_port=edge.right_port,
            )
            for edge in provenance.edges
        ),
        closure=provenance.closure,
    )


__all__ = [
    "CallerGraphProvenance",
    "ChemicalGraphProvenance",
    "FragmentEdgeProvenance",
    "FragmentGraphProvenance",
    "GraphAttribute",
    "PeriodicGraphAdaptation",
    "PeriodicGraphProvenance",
    "PolyhedronGraphProvenance",
    "QuotientGraphProvenance",
    "adapt_caller_graph",
    "adapt_chemical_entity",
    "adapt_fragment_graph",
    "adapt_periodic_graph",
    "adapt_polyhedron_network",
    "fragment_graph_from_provenance",
    "to_periodic_quotient_graph",
]
