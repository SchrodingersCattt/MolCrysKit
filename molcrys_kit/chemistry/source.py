"""Source precedence for bond-order and aromaticity constraints.

User-confirmed evidence outranks an explicit file value, which outranks
inference. A contradiction at the winning rank clears a pre-populated order and does
not lock one. A lower-ranked value that disagrees is flagged and left unused.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import ChemicalAtom, ChemicalBond, Evidence, EvidenceSource
from .valence import integral_bond_order

# Higher rank wins. Explicit file-like sources share one rank so two CIF
# types that disagree are a real contradiction rather than a silent override.
_RANK = {
    EvidenceSource.USER_CONFIRMED: 3,
    EvidenceSource.EXPLICIT_CIF: 2,
    EvidenceSource.LINE_NOTATION: 2,
    EvidenceSource.IUPAC_NAME: 2,
    EvidenceSource.INFERRED: 1,
}

_ORDER_TOKENS = {
    "S": 1,
    "SINGLE": 1,
    "SING": 1,
    "D": 2,
    "DOUBLE": 2,
    "DOUB": 2,
    "T": 3,
    "TRIPLE": 3,
    "TRIP": 3,
}
_AROMATIC_TOKENS = {"A", "AROM", "AROMATIC", "DELOCALIZED", "DELOCALISED", "DEL"}
_EMPTY_TOKENS = {"", ".", "?", "NONE", "N/A", "NA"}


@dataclass(frozen=True, slots=True)
class BondConstraint:
    """One parsed order or aromaticity assertion."""

    source: EvidenceSource
    order: int | None = None
    aromatic: bool | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SourceResolution:
    """Bonds after precedence, plus diagnostics that must stay visible."""

    bonds: tuple[ChemicalBond, ...]
    warnings: tuple[str, ...]
    locked_orders: int
    contradictions: int


def parse_bond_type(text: str | None) -> BondConstraint | None:
    """Map a CIF ``_geom_bond_type`` token onto an order or aromatic flag.

    ``A`` / ``aromatic`` / ``delocalised`` mark the edge aromatic and do not
    invent a numeric order. Unrecognized text returns None.
    """
    if text is None:
        return None
    token = "".join(ch for ch in str(text).strip().upper() if ch.isalnum())
    if token in _EMPTY_TOKENS or token == "":
        return None
    if token in _ORDER_TOKENS:
        return BondConstraint(
            EvidenceSource.EXPLICIT_CIF,
            order=_ORDER_TOKENS[token],
            detail=str(text).strip(),
        )
    if token in _AROMATIC_TOKENS:
        return BondConstraint(
            EvidenceSource.EXPLICIT_CIF, aromatic=True, detail=str(text).strip()
        )
    return None


def apply_source_precedence(
    bonds: tuple[ChemicalBond, ...] | list[ChemicalBond],
) -> SourceResolution:
    """Lock each bond to its highest-precedence consistent constraint."""
    updated = []
    warnings: list[str] = []
    locked = 0
    contradictions = 0
    for bond in bonds:
        resolved, warning, did_lock, contradicted = _resolve_bond(bond)
        updated.append(resolved)
        if warning:
            warnings.append(warning)
        locked += int(did_lock)
        contradictions += int(contradicted)
    return SourceResolution(tuple(updated), tuple(warnings), locked, contradictions)


def apply_published_bond_types(
    atoms: tuple[ChemicalAtom, ...] | list[ChemicalAtom],
    bonds: tuple[ChemicalBond, ...] | list[ChemicalBond],
    published: tuple[dict, ...] | list[dict] | None,
) -> tuple[tuple[ChemicalBond, ...], tuple[str, ...]]:
    """Attach CIF bond-type evidence by atom label.

    Distance-only rows and unrecognized types do not set an order. Two
    published types for one unordered label pair are both retained and do not
    lock the edge. Ambiguous labels (the same label on two atoms) are skipped.
    """
    if not published:
        return tuple(bonds), ()
    ids_for_label: dict[str, list[str]] = {}
    for atom in atoms:
        if atom.label:
            ids_for_label.setdefault(str(atom.label), []).append(atom.atom_id)
    ambiguous = {label for label, ids in ids_for_label.items() if len(ids) != 1}
    by_pair: dict[frozenset[str], list[dict]] = {}
    warnings: list[str] = []
    for row in published:
        left = str(row.get("atom1_label", "")).strip()
        right = str(row.get("atom2_label", "")).strip()
        if not left or not right:
            continue
        if left in ambiguous or right in ambiguous:
            warnings.append(
                f"CIF bond type for {left}-{right} skipped because a label is not unique"
            )
            continue
        by_pair.setdefault(frozenset((left, right)), []).append(row)
    label_of = {atom.atom_id: atom.label for atom in atoms}
    updated = []
    for bond in bonds:
        left = label_of.get(bond.atom1_id)
        right = label_of.get(bond.atom2_id)
        key = frozenset((str(left), str(right))) if left and right else None
        rows = by_pair.get(key) if key is not None else None
        if not rows:
            updated.append(bond)
            continue
        evidence = list(bond.evidence)
        orders: set[int] = set()
        aromatic = False
        for row in rows:
            parsed = parse_bond_type(row.get("bond_type"))
            if parsed is None:
                if row.get("bond_type") not in (None, "", ".", "?"):
                    warnings.append(
                        f"unrecognized CIF bond type {row.get('bond_type')!r} "
                        f"for {left}-{right}"
                    )
                continue
            evidence.append(
                Evidence(
                    source=EvidenceSource.EXPLICIT_CIF,
                    method="cif_geom_bond_type",
                    detail=f"bond_type:{parsed.detail}",
                )
            )
            if parsed.order is not None:
                orders.add(parsed.order)
            if parsed.aromatic:
                aromatic = True
        if len(orders) > 1:
            warnings.append(
                f"contradictory CIF bond types for {left}-{right}; order left unlocked"
            )
            updated.append(replace(bond, order=None, evidence=tuple(evidence)))
            continue
        order = float(next(iter(orders))) if len(orders) == 1 else bond.order
        updated.append(
            replace(
                bond,
                order=order,
                aromatic=bond.aromatic or aromatic,
                evidence=tuple(evidence),
            )
        )
    return tuple(updated), tuple(dict.fromkeys(warnings))


def _resolve_bond(bond: ChemicalBond) -> tuple[ChemicalBond, str | None, bool, bool]:
    constraints = _constraints(bond)
    if not constraints:
        return bond, None, False, False
    top = max(_RANK[item.source] for item in constraints)
    winners = [item for item in constraints if _RANK[item.source] == top]
    lower = [item for item in constraints if _RANK[item.source] < top]
    orders = {item.order for item in winners if item.order is not None}
    aromatics = {item.aromatic for item in winners if item.aromatic is not None}
    pair = f"{bond.atom1_id}--{bond.atom2_id}"
    if len(orders) > 1 or len(aromatics) > 1:
        # A pre-populated numeric order is not a higher-priority source. Leaving
        # it in place would keep the bond out of the variable graph.
        cleared = replace(bond, order=None) if len(orders) > 1 else bond
        return (
            cleared,
            f"contradictory {winners[0].source.value} constraints on {pair}; bond left unlocked",
            False,
            True,
        )
    notes = []
    if orders and any(
        item.order is not None and item.order not in orders for item in lower
    ):
        notes.append(
            f"lower-precedence bond order on {pair} disagrees and was not applied"
        )
    if aromatics and any(
        item.aromatic is not None and item.aromatic not in aromatics for item in lower
    ):
        notes.append(
            f"lower-precedence aromatic flag on {pair} disagrees and was not applied"
        )
    order = float(next(iter(orders))) if orders else bond.order
    aromatic = next(iter(aromatics)) if aromatics else bond.aromatic
    changed = order != bond.order or aromatic != bond.aromatic
    resolved = bond if not changed else replace(bond, order=order, aromatic=aromatic)
    locked = bool(orders) and top > _RANK[EvidenceSource.INFERRED]
    return resolved, ("; ".join(notes) or None), locked, bool(notes)


def _constraints(bond: ChemicalBond) -> list[BondConstraint]:
    found: list[BondConstraint] = []
    user_order_recorded = False
    for evidence in bond.evidence:
        parsed = _parse_detail(evidence)
        found.extend(parsed)
        if evidence.source is EvidenceSource.USER_CONFIRMED and any(
            item.order is not None for item in parsed
        ):
            user_order_recorded = True
    recorded_order = integral_bond_order(bond.order)
    if (
        recorded_order is not None
        and not user_order_recorded
        and any(
            evidence.source is EvidenceSource.USER_CONFIRMED
            for evidence in bond.evidence
        )
    ):
        found.append(
            BondConstraint(
                EvidenceSource.USER_CONFIRMED,
                order=recorded_order,
                detail="bond.order",
            )
        )
    return found


def _parse_detail(evidence: Evidence) -> list[BondConstraint]:
    detail = evidence.detail or ""
    constraints: list[BondConstraint] = []
    for part in detail.split(";"):
        text = part.strip()
        if not text:
            continue
        if text.startswith("bond_type:"):
            parsed = parse_bond_type(text.split(":", 1)[1])
            if parsed is not None:
                constraints.append(
                    BondConstraint(evidence.source, parsed.order, parsed.aromatic, text)
                )
            continue
        if text.startswith("order:"):
            try:
                constraints.append(
                    BondConstraint(
                        evidence.source, order=int(text.split(":", 1)[1]), detail=text
                    )
                )
            except ValueError:
                continue
        elif text.startswith("aromatic:"):
            flag = text.split(":", 1)[1].strip().lower()
            if flag in {"1", "true", "yes"}:
                constraints.append(
                    BondConstraint(evidence.source, aromatic=True, detail=text)
                )
            elif flag in {"0", "false", "no"}:
                constraints.append(
                    BondConstraint(evidence.source, aromatic=False, detail=text)
                )
    return constraints


__all__ = [
    "BondConstraint",
    "SourceResolution",
    "apply_published_bond_types",
    "apply_source_precedence",
    "parse_bond_type",
]
