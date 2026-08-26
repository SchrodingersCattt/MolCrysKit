"""Crystal-level stereochemical relationships without external engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .models import (
    CrystalChemistry,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)
from .stereo import StereoReport, assign_stereochemistry


class EntityRelationship(str, Enum):
    """Relationship between two finite chemical entities."""

    SAME_STEREOISOMER = "same_stereoisomer"
    MIRROR = "mirror"
    STEREOISOMER = "stereoisomer"
    DIFFERENT_CONSTITUTION = "different_constitution"
    INDETERMINATE = "indeterminate"


class CrystalStereoClass(str, Enum):
    """Coordinate-model stereochemical composition of one crystal snapshot."""

    ENANTIOPURE = "enantiopure"
    RACEMIC_CRYSTAL = "racemic_crystal"
    MESO_ACHIRAL = "meso_or_achiral"
    STEREO_HETEROGENEOUS = "stereo_heterogeneous"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class AbsoluteStructureParameter:
    """One experimental absolute-structure value preserved without a verdict."""

    method: str
    raw: str
    value: float
    standard_uncertainty: float | None


@dataclass(frozen=True)
class EntityStereoSummary:
    """Stereo state of one entity in the coordinate model."""

    entity_id: str
    descriptor_count: int
    assigned_descriptors: tuple[tuple[str, str, str], ...]
    is_internal_mirror: bool
    status: InferenceStatus
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnantiomerCount:
    """Counts for one constitution/stereoisomer cluster and its mirror."""

    representative_entity_id: str
    count: int
    mirror_entity_id: str | None
    mirror_count: int


@dataclass(frozen=True)
class CrystalStereoReport:
    """Separate coordinate-model composition from experimental evidence."""

    classification: CrystalStereoClass
    status: InferenceStatus
    entities: tuple[EntityStereoSummary, ...]
    relationships: tuple[tuple[str, str, EntityRelationship], ...]
    enantiomer_counts: tuple[EnantiomerCount, ...]
    symmetry_category: str
    absolute_structure: tuple[AbsoluteStructureParameter, ...]
    absolute_structure_details: str | None
    reason: str
    warnings: tuple[str, ...]
    evidence: tuple[Evidence, ...]


class CrystalStereoIndeterminateError(ValueError):
    """Raised when strict crystal-stereo analysis is not conclusive."""


def classify_entity_relationship(
    left: FiniteChemicalEntity,
    right: FiniteChemicalEntity,
    left_stereo: StereoReport | None = None,
    right_stereo: StereoReport | None = None,
) -> EntityRelationship:
    """Compare constitution and assigned stereochemistry by graph isomorphism."""
    if not _isomorphic(left, right):
        return EntityRelationship.DIFFERENT_CONSTITUTION
    left_report = left_stereo or assign_stereochemistry(left, left.embedding)
    right_report = right_stereo or assign_stereochemistry(right, right.embedding)
    if _has_indeterminate_descriptor(left_report) or _has_indeterminate_descriptor(
        right_report
    ):
        return EntityRelationship.INDETERMINATE
    if _isomorphic(left, right, left_report, right_report, relation="same"):
        return EntityRelationship.SAME_STEREOISOMER
    if _isomorphic(left, right, left_report, right_report, relation="mirror"):
        return EntityRelationship.MIRROR
    return EntityRelationship.STEREOISOMER


def analyze_crystal_stereochemistry(
    structure_or_chemistry,
    *,
    stereo_reports: Mapping[str, StereoReport] | None = None,
    strict: bool = False,
) -> CrystalStereoReport:
    """Aggregate molecular stereochemistry for a crystal coordinate model.

    The result never uses Flack, Hooft, Rogers, or Parsons values as a binary
    threshold. Those experimental values are preserved in a separate field.
    A single structure is classified as a racemic *crystal* when appropriate;
    it is never called a conglomerate because that requires sample-level data.
    """
    chemistry, metadata, symmetry = _analysis_inputs(structure_or_chemistry)
    finite = tuple(
        component
        for component in chemistry.components
        if isinstance(component, FiniteChemicalEntity)
    )
    warnings = []
    if len(finite) != len(chemistry.components):
        warnings.append(
            "periodic or non-finite entities are outside molecular enantiomer counting"
        )

    reports = {
        entity.entity_id: (
            stereo_reports[entity.entity_id]
            if stereo_reports is not None and entity.entity_id in stereo_reports
            else assign_stereochemistry(entity, entity.embedding)
        )
        for entity in finite
    }
    summaries = tuple(
        _entity_summary(entity, reports[entity.entity_id]) for entity in finite
    )
    relationships = _pair_relationships(finite, reports)
    classification, counts, reason = _classify_crystal(finite, reports, summaries)

    # The current descriptor engine advertises its supported scope rather than
    # silently treating absence of a tetrahedral center as proof of achirality.
    warnings.append(
        "coordinate classification currently covers tetrahedral centers and double bonds; other stereogenic-unit families remain unevaluated"
    )
    warnings.extend(
        warning for report in reports.values() for warning in report.warnings
    )
    warnings = list(dict.fromkeys(warnings))
    status = (
        InferenceStatus.INDETERMINATE
        if classification is CrystalStereoClass.INDETERMINATE
        else InferenceStatus.PROVISIONAL
    )
    if strict and status in {
        InferenceStatus.PROVISIONAL,
        InferenceStatus.INDETERMINATE,
    }:
        raise CrystalStereoIndeterminateError(reason)

    evidence = Evidence(
        source=EvidenceSource.INFERRED,
        method="self_contained_graph_isomorphism_and_CIP_aggregation",
        detail="Coordinate-model classification; experimental absolute-structure evidence is separate.",
    )
    absolute, details = _absolute_structure(metadata)
    return CrystalStereoReport(
        classification=classification,
        status=status,
        entities=summaries,
        relationships=relationships,
        enantiomer_counts=counts,
        symmetry_category=_symmetry_category(symmetry),
        absolute_structure=absolute,
        absolute_structure_details=details,
        reason=reason,
        warnings=tuple(warnings),
        evidence=(evidence,),
    )


def _analysis_inputs(value):
    if isinstance(value, CrystalChemistry):
        return value, {}, None
    chemistry = getattr(value, "chemistry", None)
    if chemistry is None:
        from .perception import infer_chemistry

        chemistry = infer_chemistry(value)
    metadata = dict(getattr(value, "metadata", {}) or {})
    return chemistry, metadata, metadata.get("crystal_symmetry")


def _entity_summary(entity, report: StereoReport) -> EntityStereoSummary:
    assigned = tuple(
        sorted(
            (
                descriptor.center_atom_id,
                descriptor.kind.value,
                descriptor.descriptor,
            )
            for descriptor in report.descriptors
            if descriptor.descriptor is not None
        )
    )
    internal_mirror = bool(assigned) and _isomorphic(
        entity,
        entity,
        report,
        report,
        relation="mirror",
    )
    return EntityStereoSummary(
        entity_id=entity.entity_id,
        descriptor_count=len(report.descriptors),
        assigned_descriptors=assigned,
        is_internal_mirror=internal_mirror,
        status=report.status,
        warnings=report.warnings,
    )


def _pair_relationships(entities, reports):
    values = []
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            relationship = classify_entity_relationship(
                left,
                right,
                reports[left.entity_id],
                reports[right.entity_id],
            )
            if relationship is not EntityRelationship.DIFFERENT_CONSTITUTION:
                values.append((left.entity_id, right.entity_id, relationship))
    return tuple(values)


def _classify_crystal(entities, reports, summaries):
    if not entities:
        return (
            CrystalStereoClass.INDETERMINATE,
            (),
            "crystal contains no finite chemical entities",
        )
    if any(_has_indeterminate_descriptor(reports[item.entity_id]) for item in entities):
        return (
            CrystalStereoClass.INDETERMINATE,
            (),
            "one or more stereogenic units are indeterminate",
        )

    active = [
        entity
        for entity, summary in zip(entities, summaries)
        if summary.assigned_descriptors and not summary.is_internal_mirror
    ]
    if not active:
        if any(summary.is_internal_mirror for summary in summaries):
            return (
                CrystalStereoClass.MESO_ACHIRAL,
                (),
                "all assigned stereogenic entities are superposable on their descriptor-inverted graph",
            )
        return (
            CrystalStereoClass.INDETERMINATE,
            (),
            "no supported stereogenic units were assigned; achirality is not assumed",
        )

    constitution_groups = _partition(
        active,
        lambda left, right: (
            classify_entity_relationship(
                left,
                right,
                reports[left.entity_id],
                reports[right.entity_id],
            )
            is not EntityRelationship.DIFFERENT_CONSTITUTION
        ),
    )
    group_classes = []
    all_counts = []
    for group in constitution_groups:
        stereo_groups = _partition(
            group,
            lambda left, right: (
                classify_entity_relationship(
                    left,
                    right,
                    reports[left.entity_id],
                    reports[right.entity_id],
                )
                is EntityRelationship.SAME_STEREOISOMER
            ),
        )
        if len(stereo_groups) == 1:
            representative = stereo_groups[0][0]
            all_counts.append(
                EnantiomerCount(
                    representative.entity_id, len(stereo_groups[0]), None, 0
                )
            )
            group_classes.append(CrystalStereoClass.ENANTIOPURE)
            continue
        if len(stereo_groups) == 2:
            left, right = stereo_groups[0][0], stereo_groups[1][0]
            relationship = classify_entity_relationship(
                left,
                right,
                reports[left.entity_id],
                reports[right.entity_id],
            )
            all_counts.append(
                EnantiomerCount(
                    left.entity_id,
                    len(stereo_groups[0]),
                    right.entity_id
                    if relationship is EntityRelationship.MIRROR
                    else None,
                    len(stereo_groups[1]),
                )
            )
            if relationship is EntityRelationship.MIRROR and len(
                stereo_groups[0]
            ) == len(stereo_groups[1]):
                group_classes.append(CrystalStereoClass.RACEMIC_CRYSTAL)
                continue
        group_classes.append(CrystalStereoClass.STEREO_HETEROGENEOUS)

    if all(value is CrystalStereoClass.ENANTIOPURE for value in group_classes):
        return (
            CrystalStereoClass.ENANTIOPURE,
            tuple(all_counts),
            "one handed stereoisomer occurs for each chiral constitution",
        )
    if all(value is CrystalStereoClass.RACEMIC_CRYSTAL for value in group_classes):
        return (
            CrystalStereoClass.RACEMIC_CRYSTAL,
            tuple(all_counts),
            "each chiral constitution occurs as equal mirror-related counts in this crystal model",
        )
    return (
        CrystalStereoClass.STEREO_HETEROGENEOUS,
        tuple(all_counts),
        "stereoisomer counts are neither uniformly single-handed nor equal mirror pairs",
    )


def _partition(values, equivalent):
    groups = []
    for value in values:
        for group in groups:
            if equivalent(value, group[0]):
                group.append(value)
                break
        else:
            groups.append([value])
    return groups


def _isomorphic(left, right, left_stereo=None, right_stereo=None, *, relation=None):
    if len(left.atoms) != len(right.atoms) or len(left.bonds) != len(right.bonds):
        return False
    left_atoms = {atom.atom_id: atom for atom in left.atoms}
    right_atoms = {atom.atom_id: atom for atom in right.atoms}
    left_adj = _adjacency(left)
    right_adj = _adjacency(right)
    candidates = {
        atom_id: tuple(
            candidate_id
            for candidate_id, candidate in right_atoms.items()
            if _atom_invariant(atom, left_adj[atom_id])
            == _atom_invariant(candidate, right_adj[candidate_id])
        )
        for atom_id, atom in left_atoms.items()
    }
    if any(not values for values in candidates.values()):
        return False
    left_descriptors = _descriptor_map(left_stereo)
    right_descriptors = _descriptor_map(right_stereo)
    order = sorted(
        left_atoms,
        key=lambda atom_id: (
            len(candidates[atom_id]),
            -len(left_adj[atom_id]),
            atom_id,
        ),
    )
    mapping = {}
    used = set()

    def visit(position):
        if position == len(order):
            return _stereo_mapping_matches(
                mapping, left_descriptors, right_descriptors, relation
            )
        atom_id = order[position]
        for candidate_id in candidates[atom_id]:
            if candidate_id in used or not _mapping_consistent(
                atom_id, candidate_id, mapping, left_adj, right_adj
            ):
                continue
            mapping[atom_id] = candidate_id
            used.add(candidate_id)
            if visit(position + 1):
                return True
            used.remove(candidate_id)
            del mapping[atom_id]
        return False

    return visit(0)


def _adjacency(entity):
    result = {atom.atom_id: {} for atom in entity.atoms}
    for bond in entity.bonds:
        label = (
            bond.order,
            bond.kind.value,
            bond.aromatic,
            tuple(abs(value) for value in bond.atom2_image_shift),
        )
        result[bond.atom1_id][bond.atom2_id] = label
        result[bond.atom2_id][bond.atom1_id] = label
    return result


def _atom_invariant(atom, neighbors):
    return (
        atom.element,
        atom.isotope,
        atom.formal_charge,
        atom.radical_electrons,
        atom.implicit_hydrogens,
        len(neighbors),
        tuple(sorted(neighbors.values(), key=repr)),
    )


def _mapping_consistent(left_id, right_id, mapping, left_adj, right_adj):
    for mapped_left, mapped_right in mapping.items():
        left_edge = left_adj[left_id].get(mapped_left)
        right_edge = right_adj[right_id].get(mapped_right)
        if left_edge != right_edge:
            return False
    return True


def _descriptor_map(report):
    if report is None:
        return {}
    return {
        descriptor.center_atom_id: (descriptor.kind.value, descriptor.descriptor)
        for descriptor in report.descriptors
        if descriptor.descriptor is not None
    }


def _stereo_mapping_matches(mapping, left, right, relation):
    if relation is None:
        return True
    if len(left) != len(right):
        return False
    if relation == "mirror" and not left:
        return False
    for atom_id, descriptor in left.items():
        mapped = right.get(mapping[atom_id])
        if mapped is None or mapped[0] != descriptor[0]:
            return False
        expected = (
            descriptor[1] if relation == "same" else _invert_descriptor(descriptor[1])
        )
        if mapped[1] != expected:
            return False
    return True


def _invert_descriptor(value):
    return {
        "R": "S",
        "S": "R",
        "r": "s",
        "s": "r",
        "P": "M",
        "M": "P",
        "Ra": "Sa",
        "Sa": "Ra",
        "Rp": "Sp",
        "Sp": "Rp",
        "E": "E",
        "Z": "Z",
    }.get(value, f"mirror({value})")


def _has_indeterminate_descriptor(report):
    return any(
        descriptor.status is InferenceStatus.INDETERMINATE
        for descriptor in report.descriptors
    )


def _absolute_structure(metadata):
    cif = dict(metadata.get("cif_chemistry", {}) or {})
    absolute = dict(cif.get("absolute_structure", {}) or {})
    records = []
    for method in ("flack", "hooft", "rogers"):
        value = absolute.get(method)
        if not isinstance(value, dict) or value.get("value") is None:
            continue
        records.append(
            AbsoluteStructureParameter(
                method=method,
                raw=str(value.get("raw", value["value"])),
                value=float(value["value"]),
                standard_uncertainty=(
                    None
                    if value.get("standard_uncertainty") is None
                    else float(value["standard_uncertainty"])
                ),
            )
        )
    details = absolute.get("details")
    return tuple(records), None if details is None else str(details)


def _symmetry_category(symmetry) -> str:
    if symmetry is None:
        return "indeterminate (symmetry operations unavailable)"
    operations = tuple(getattr(symmetry, "operations", ()))
    if not operations:
        return "indeterminate (symmetry operations unavailable)"
    return (
        "Sohncke (proper operations only)"
        if all(item.is_proper for item in operations)
        else "non-Sohncke (contains improper operation)"
    )


__all__ = [
    "AbsoluteStructureParameter",
    "CrystalStereoClass",
    "CrystalStereoIndeterminateError",
    "CrystalStereoReport",
    "EnantiomerCount",
    "EntityRelationship",
    "EntityStereoSummary",
    "analyze_crystal_stereochemistry",
    "classify_entity_relationship",
]
