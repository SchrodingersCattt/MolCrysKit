"""Serializable node, edge, disorder, and partition policies.

Policies are predicates and explicit selections. They do not match element
symbols, formulas, Wyckoff letters, or a fixed number of nets.
"""

from __future__ import annotations

from dataclasses import dataclass

NODE_KINDS = ("atom_sites", "molecular_units", "coordination_centres", "explicit_nodes")
EDGE_KINDS = (
    "supplied",
    "coordination_shell",
    "contacts",
    "hydrogen_bonds",
    "user_edges",
)
DISORDER_MODES = ("major", "as_refined", "explicit")
PARTITION_MODES = ("discovery", "explicit")


@dataclass(frozen=True)
class NodePolicy:
    """How crystal sites become quotient nodes.

    ``explicit_nodes`` uses ``groups`` of site ids. ``coordination_centres``
    uses ``centre_ids`` when that tuple is non-empty; otherwise a site is a
    centre when it has at least ``minimum_incident`` bonds of ``bond_role``.
    """

    kind: str = "atom_sites"
    groups: tuple[tuple[str, ...], ...] = ()
    bond_role: str = "coordination"
    minimum_incident: int = 1
    centre_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(
                f"node policy must be one of {NODE_KINDS}, got {self.kind!r}"
            )
        if self.minimum_incident < 1:
            raise ValueError("minimum_incident must be positive")
        object.__setattr__(self, "groups", tuple(tuple(group) for group in self.groups))
        object.__setattr__(self, "centre_ids", tuple(self.centre_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "groups": [list(group) for group in self.groups],
            "bond_role": self.bond_role,
            "minimum_incident": self.minimum_incident,
            "centre_ids": list(self.centre_ids),
        }


@dataclass(frozen=True)
class EdgePolicy:
    """How quotient edges are chosen.

    ``cutoff_A`` is required for the meaning of a contact scan. ``None`` asks
    the builder to use ``DEFAULT_NEIGHBOR_CUTOFF`` and to record that value.
    ``user_edges`` rows are ``(source_node, target_node, image_shift)``.
    """

    kind: str = "supplied"
    cutoff_A: float | None = None
    user_edges: tuple[tuple[str, str, tuple[int, int, int]], ...] = ()
    max_h_acceptor_distance_A: float | None = None
    min_dha_angle_deg: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(
                f"edge policy must be one of {EDGE_KINDS}, got {self.kind!r}"
            )
        object.__setattr__(
            self,
            "user_edges",
            tuple(
                (source, target, (int(shift[0]), int(shift[1]), int(shift[2])))
                for source, target, shift in self.user_edges
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "cutoff_A": self.cutoff_A,
            "user_edges": [
                [source, target, list(shift)]
                for source, target, shift in self.user_edges
            ],
            "max_h_acceptor_distance_A": self.max_h_acceptor_distance_A,
            "min_dha_angle_deg": self.min_dha_angle_deg,
        }


@dataclass(frozen=True)
class DisorderPolicy:
    """Which refined sites may be occupied together.

    ``selected_groups`` rows are ``(assembly, group)`` for ``explicit`` mode.
    ``independent_assemblies`` is opt-in. Leaving it false does not combine
    alternatives from two assemblies into one crystal.
    """

    mode: str = "major"
    selected_groups: tuple[tuple[str, int], ...] = ()
    independent_assemblies: bool = False

    def __post_init__(self) -> None:
        if self.mode not in DISORDER_MODES:
            raise ValueError(
                f"disorder mode must be one of {DISORDER_MODES}, got {self.mode!r}"
            )
        object.__setattr__(
            self,
            "selected_groups",
            tuple(
                (str(assembly), int(group)) for assembly, group in self.selected_groups
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "selected_groups": [
                [assembly, group] for assembly, group in self.selected_groups
            ],
            "independent_assemblies": self.independent_assemblies,
        }


@dataclass(frozen=True)
class PartitionPolicy:
    """Explicit net labels or a request to search for partitions.

    ``assignment`` rows are ``(node_id, net_id)``. ``edge_nets`` rows are
    ``(edge_id, net_id)`` and are checked against the node assignment.
    """

    mode: str = "discovery"
    assignment: tuple[tuple[str, str], ...] = ()
    edge_nets: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in PARTITION_MODES:
            raise ValueError(
                f"partition mode must be one of {PARTITION_MODES}, got {self.mode!r}"
            )
        object.__setattr__(
            self,
            "assignment",
            tuple((str(node), str(net)) for node, net in self.assignment),
        )
        object.__setattr__(
            self,
            "edge_nets",
            tuple((str(edge), str(net)) for edge, net in self.edge_nets),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "assignment": [[node, net] for node, net in self.assignment],
            "edge_nets": [[edge, net] for edge, net in self.edge_nets],
        }


def policies_from_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Build policies from a JSON object. Unknown keys are rejected."""
    if not isinstance(payload, dict):
        raise TypeError("a net-topology policy must be a JSON object")
    allowed = {"nodes", "edges", "disorder", "partition", "limits"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    nodes = _object(
        payload.get("nodes", {}),
        "nodes",
        {"kind", "groups", "bond_role", "minimum_incident", "centre_ids"},
    )
    edges = _object(
        payload.get("edges", {}),
        "edges",
        {
            "kind",
            "cutoff_A",
            "user_edges",
            "max_h_acceptor_distance_A",
            "min_dha_angle_deg",
        },
    )
    disorder = _object(
        payload.get("disorder", {}),
        "disorder",
        {"mode", "selected_groups", "independent_assemblies"},
    )
    partition = _object(
        payload.get("partition", {}),
        "partition",
        {"mode", "assignment", "edge_nets"},
    )
    limits = payload.get("limits", {})
    if limits is not None and not isinstance(limits, dict):
        raise TypeError("limits must be an object")
    return {
        "nodes": NodePolicy(
            kind=str(nodes.get("kind", "atom_sites")),
            groups=tuple(
                tuple(str(site) for site in group) for group in nodes.get("groups", ())
            ),
            bond_role=str(nodes.get("bond_role", "coordination")),
            minimum_incident=int(nodes.get("minimum_incident", 1)),
            centre_ids=tuple(str(site) for site in nodes.get("centre_ids", ())),
        ),
        "edges": EdgePolicy(
            kind=str(edges.get("kind", "supplied")),
            cutoff_A=None
            if edges.get("cutoff_A") is None
            else float(edges["cutoff_A"]),
            user_edges=tuple(
                (
                    str(row[0]),
                    str(row[1]),
                    (int(row[2][0]), int(row[2][1]), int(row[2][2])),
                )
                for row in edges.get("user_edges", ())
            ),
            max_h_acceptor_distance_A=(
                None
                if edges.get("max_h_acceptor_distance_A") is None
                else float(edges["max_h_acceptor_distance_A"])
            ),
            min_dha_angle_deg=(
                None
                if edges.get("min_dha_angle_deg") is None
                else float(edges["min_dha_angle_deg"])
            ),
        ),
        "disorder": DisorderPolicy(
            mode=str(disorder.get("mode", "major")),
            selected_groups=tuple(
                (str(row[0]), int(row[1]))
                for row in disorder.get("selected_groups", ())
            ),
            independent_assemblies=bool(disorder.get("independent_assemblies", False)),
        ),
        "partition": PartitionPolicy(
            mode=str(partition.get("mode", "discovery")),
            assignment=tuple(
                (str(row[0]), str(row[1])) for row in partition.get("assignment", ())
            ),
            edge_nets=tuple(
                (str(row[0]), str(row[1])) for row in partition.get("edge_nets", ())
            ),
        ),
        "limits": dict(limits or {}),
    }


def _object(value: object, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    return value


__all__ = [
    "DISORDER_MODES",
    "EDGE_KINDS",
    "NODE_KINDS",
    "PARTITION_MODES",
    "DisorderPolicy",
    "EdgePolicy",
    "NodePolicy",
    "PartitionPolicy",
    "policies_from_mapping",
]
