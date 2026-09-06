"""Compare chemical entities parsed from linear notation.

This is a convenience wrapper over :func:`classify_entity_relationship` and
:func:`from_line_notation` for callers that start from notation strings rather
than pre-built entity objects.
"""

from __future__ import annotations

from .crystal_stereo import EntityRelationship, classify_entity_relationship
from .line_notation import LineNotationError, from_line_notation
from .models import FiniteChemicalEntity


def notations_equivalent(left: str, right: str) -> bool | None:
    """Return whether two line notations describe chemically equivalent entities.

    Parses each notation with :func:`from_line_notation` (OpenSMILES or MCK-LN,
    auto-detected), then compares constitution and stereochemistry via
    :func:`classify_entity_relationship`.

    Returns
    -------
    bool or None
        ``True`` if the entities are the same stereoisomer, ``False`` if they
        differ in constitution or stereochemistry, or ``None`` if the
        comparison is indeterminate (e.g. unresolved stereocenters) or either
        notation is not a finite entity.
    """
    try:
        left_entity = from_line_notation(left)
        right_entity = from_line_notation(right)
    except (LineNotationError, ValueError, TypeError):
        return None
    if not isinstance(left_entity, FiniteChemicalEntity) or not isinstance(
        right_entity, FiniteChemicalEntity
    ):
        return None
    result = classify_entity_relationship(left_entity, right_entity)
    if result is EntityRelationship.SAME_STEREOISOMER:
        return True
    if result is EntityRelationship.INDETERMINATE:
        return None
    # MIRROR, STEREOISOMER, DIFFERENT_CONSTITUTION → not equivalent.
    return False


__all__ = ["notations_equivalent"]
