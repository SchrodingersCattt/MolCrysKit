"""Compare chemical entities parsed from linear notation.

This is a convenience wrapper over :func:`classify_entity_relationship` and
:func:`from_line_notation` for callers that start from notation strings rather
than pre-built entity objects.
"""

from __future__ import annotations

from .crystal_stereo import EntityRelationship, classify_entity_relationship
from .line_notation import LineNotationError, from_line_notation
from .models import FiniteChemicalEntity, InferenceStatus
from .stereo import StereoReport


def notations_equivalent(left: str, right: str) -> bool | None:
    """Return whether two line notations describe chemically equivalent entities.

    Parses each notation with :func:`from_line_notation` (OpenSMILES or MCK-LN,
    auto-detected), applies OpenSMILES default-valence hydrogens to
    unbracketed organic-subset atoms, then compares constitution and
    stereochemistry via :func:`classify_entity_relationship`.

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
    # OpenSMILES organic-subset atoms carry default valence hydrogens even
    # when they are written without brackets (for example ``CCO``).  The
    # low-level parser intentionally leaves those defaults unresolved, so use
    # the same bounded completion as the reversible naming API before graph
    # comparison.  MCK-LN remains an exact field-preserving comparison.
    from .name_conversion import _complete_open_smiles_hydrogens

    if not left.strip().startswith("MCK-LN1|"):
        left_entity = _complete_open_smiles_hydrogens(left_entity)
    if not right.strip().startswith("MCK-LN1|"):
        right_entity = _complete_open_smiles_hydrogens(right_entity)
    result = classify_entity_relationship(left_entity, right_entity)
    if result is EntityRelationship.SAME_STEREOISOMER:
        return True
    if result is EntityRelationship.INDETERMINATE:
        # OpenSMILES bracket hydrogens can make the coordinate-free stereo
        # helper report indistinguishable implicit-H centers.  If neither
        # notation carries stereo tokens, compare constitution with empty
        # stereo reports instead of returning an avoidable indeterminate.
        if not any(
            atom.stereochemistry is not None for atom in (*left_entity.atoms, *right_entity.atoms)
        ) and not any(
            bond.stereochemistry is not None for bond in (*left_entity.bonds, *right_entity.bonds)
        ):
            left_report = StereoReport(
                entity_id=left_entity.entity_id,
                descriptors=(),
                status=InferenceStatus.INFERRED,
                evidence=(),
            )
            right_report = StereoReport(
                entity_id=right_entity.entity_id,
                descriptors=(),
                status=InferenceStatus.INFERRED,
                evidence=(),
            )
            fallback = classify_entity_relationship(
                left_entity,
                right_entity,
                left_report,
                right_report,
            )
            if fallback is EntityRelationship.SAME_STEREOISOMER:
                return True
        return None
    # MIRROR, STEREOISOMER, DIFFERENT_CONSTITUTION → not equivalent.
    return False


__all__ = ["notations_equivalent"]
