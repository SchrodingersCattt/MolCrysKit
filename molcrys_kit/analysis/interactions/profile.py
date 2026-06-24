"""Aggregate weak-interaction scoring profiles for molecular crystals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ...structures.crystal import MolecularCrystal
from .base import BaseInteraction
from .ch_pi import CHPiInteractionCriteria, find_ch_pi
from .h_h_contact import HHContactCriteria, find_h_h_contacts
from .halogen_bond import HalogenBondCriteria, find_halogen_bonds
from .hydrogen_bond import HydrogenBondCriteria, find_hydrogen_bonds
from .pi_stacking import PiStackingCriteria, find_pi_stacking


@dataclass(frozen=True)
class InteractionScoreSummary:
    """Count and score statistics for one interaction family."""

    count: int = 0
    max: float = 0.0
    mean: float = 0.0
    sum: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "count": self.count,
            "max": self.max,
            "mean": self.mean,
            "sum": self.sum,
        }


@dataclass(frozen=True)
class InteractionProfile:
    """Per-type interaction score statistics plus raw interaction records."""

    summaries: dict[str, InteractionScoreSummary]
    interactions: tuple[BaseInteraction, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_interactions: bool = False) -> dict[str, Any]:
        """Return profile summaries and, optionally, raw interaction metadata."""
        payload: dict[str, Any] = {
            key: value.to_dict() for key, value in self.summaries.items()
        }
        if include_interactions:
            payload["interactions"] = [
                {
                    "kind": interaction.kind,
                    "distance_A": interaction.distance_A,
                    "angle_deg": interaction.angle_deg,
                    "score": interaction.score,
                    "image": list(interaction.image) if interaction.image is not None else None,
                    "metadata": interaction.metadata,
                }
                for interaction in self.interactions
            ]
        return payload


DEFAULT_PROFILE_KEYS = (
    "hydrogen_bond",
    "halogen_bond",
    "pi_stacking",
    "ch_pi",
    "close_contact",
)

PROFILE_KIND_ALIASES = {
    "h_h_contact": "close_contact",
}


def interaction_profile(
    target: MolecularCrystal | Sequence,
    *,
    hydrogen_bond_criteria: HydrogenBondCriteria | None = None,
    halogen_bond_criteria: HalogenBondCriteria | None = None,
    pi_stacking_criteria: PiStackingCriteria | None = None,
    ch_pi_criteria: CHPiInteractionCriteria | None = None,
    h_h_contact_criteria: HHContactCriteria | None = None,
) -> InteractionProfile:
    """Return count/max/mean/sum scoring profile for all interaction detectors.

    The returned profile keeps the full raw interaction list while summarizing
    scores by profile kind.  Legacy H···H contact records keep their
    ``kind="h_h_contact"`` value for backward compatibility and are summarized
    under the more descriptive ``close_contact`` profile key.
    """
    interactions: list[BaseInteraction] = []
    interactions.extend(find_hydrogen_bonds(target, criteria=hydrogen_bond_criteria))
    interactions.extend(find_halogen_bonds(target, criteria=halogen_bond_criteria))
    interactions.extend(find_pi_stacking(target, criteria=pi_stacking_criteria))
    interactions.extend(find_ch_pi(target, criteria=ch_pi_criteria))
    interactions.extend(find_h_h_contacts(target, criteria=h_h_contact_criteria))

    summaries = {
        key: _score_summary(
            [item for item in interactions if _profile_kind(item.kind) == key]
        )
        for key in DEFAULT_PROFILE_KEYS
    }
    return InteractionProfile(summaries=summaries, interactions=tuple(interactions))


def _profile_kind(kind: str) -> str:
    """Return aggregation key for a raw interaction kind."""
    return PROFILE_KIND_ALIASES.get(kind, kind)


def _score_summary(interactions: Sequence[BaseInteraction]) -> InteractionScoreSummary:
    scores = [float(item.score) for item in interactions if item.score is not None]
    if not scores:
        return InteractionScoreSummary(count=len(interactions))
    total = float(sum(scores))
    return InteractionScoreSummary(
        count=len(interactions),
        max=float(max(scores)),
        mean=total / len(scores),
        sum=total,
    )
