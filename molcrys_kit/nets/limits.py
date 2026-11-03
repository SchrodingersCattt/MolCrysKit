"""Search limits for periodic-net analysis.

Every limit is reported with the result that used it. A search that stops at
a limit is not a negative proof.

``basis_entry_limit`` and ``lll_step_limit`` are part of that report. The
isomorphism proof is a Hermite form, so those two fields do not decide it.
"""

from __future__ import annotations

from dataclasses import dataclass

from molcrys_kit.constants.config import NET_TOPOLOGY_CONFIG


@dataclass(frozen=True)
class NetAnalysisLimits:
    """Caps copied from ``NET_TOPOLOGY_CONFIG`` unless a caller overrides them."""

    coordination_depth: int
    cover_state_limit: int
    primitive_node_limit: int
    canonical_node_limit: int
    backtrack_limit: int
    basis_entry_limit: int
    lll_step_limit: int
    ring_length_limit: int
    ring_count_limit: int
    ring_state_limit: int
    link_pair_limit: int
    link_image_limit: int
    discovery_partition_limit: int
    contact_borderline_margin_A: float
    fraction_residual_A: float
    fraction_denominator_limit: int

    @classmethod
    def from_config(cls) -> NetAnalysisLimits:
        """Return the packaged limits."""
        config = NET_TOPOLOGY_CONFIG
        return cls(
            coordination_depth=int(config["COORDINATION_SEQUENCE_DEPTH"]),
            cover_state_limit=int(config["COVER_BFS_STATE_LIMIT"]),
            primitive_node_limit=int(config["PRIMITIVE_NODE_LIMIT"]),
            canonical_node_limit=int(config["CANONICAL_NODE_LIMIT"]),
            backtrack_limit=int(config["CANONICAL_BACKTRACK_LIMIT"]),
            basis_entry_limit=int(config["BASIS_ENTRY_LIMIT"]),
            lll_step_limit=int(config["LLL_STEP_LIMIT"]),
            ring_length_limit=int(config["RING_LENGTH_LIMIT"]),
            ring_count_limit=int(config["RING_COUNT_LIMIT"]),
            ring_state_limit=int(config["RING_STATE_LIMIT"]),
            link_pair_limit=int(config["LINK_PAIR_LIMIT"]),
            link_image_limit=int(config["LINK_IMAGE_LIMIT"]),
            discovery_partition_limit=int(config["DISCOVERY_PARTITION_LIMIT"]),
            contact_borderline_margin_A=float(config["CONTACT_BORDERLINE_MARGIN_A"]),
            fraction_residual_A=float(config["FRACTION_RESIDUAL_A"]),
            fraction_denominator_limit=int(config["FRACTION_DENOMINATOR_LIMIT"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "coordination_depth": self.coordination_depth,
            "cover_state_limit": self.cover_state_limit,
            "primitive_node_limit": self.primitive_node_limit,
            "canonical_node_limit": self.canonical_node_limit,
            "backtrack_limit": self.backtrack_limit,
            "basis_entry_limit": self.basis_entry_limit,
            "lll_step_limit": self.lll_step_limit,
            "ring_length_limit": self.ring_length_limit,
            "ring_count_limit": self.ring_count_limit,
            "ring_state_limit": self.ring_state_limit,
            "link_pair_limit": self.link_pair_limit,
            "link_image_limit": self.link_image_limit,
            "discovery_partition_limit": self.discovery_partition_limit,
            "contact_borderline_margin_A": self.contact_borderline_margin_A,
            "fraction_residual_A": self.fraction_residual_A,
            "fraction_denominator_limit": self.fraction_denominator_limit,
        }


__all__ = ["NetAnalysisLimits"]
