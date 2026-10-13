"""Bounded, propagating bond-order search.

This module contains no crystal or mutable-structure dependencies.  It uses
incremental residual-valence bounds before branching, so long saturated chains
are solved in linear-ish time while genuinely ambiguous rings remain bounded
and are reported as truncated when the caller's budget is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Sequence

from .models import BondKind, ChemicalAtom, ChemicalBond
from .valence import (
    allowed_valences,
    atom_valence_target,
    inferred_charge,
    integral_bond_order,
    valence_offset,
)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Search candidates and auditable budget counters."""

    candidates: tuple[tuple[float, tuple[ChemicalBond, ...]], ...]
    states: int
    complete: bool
    rejected: int
    # Assignments dropped because a requested total charge was missing or wrong.
    charge_rejected: int = 0


def _possible(
    states: tuple[int, ...] | None,
    current: int,
    remaining_min: int,
    remaining_max: int,
    atom: ChemicalAtom,
) -> bool:
    if states is None:
        return True
    # Stored H/radicals are part of the valence target, not graph edges.
    offset = valence_offset(atom)
    lower = current + remaining_min + offset
    upper = current + remaining_max + offset
    return any(lower <= state <= upper for state in states)


def search_bond_orders(
    atoms: Sequence[ChemicalAtom],
    bonds: Sequence[ChemicalBond],
    variable_indices: Sequence[int],
    choices: Mapping[int, Sequence[int]],
    *,
    score: Callable[[Mapping[int, int]], float] | None = None,
    total_charge: int | None = None,
    max_search_states: int = 100_000,
    max_candidates: int = 64,
    valence_atom_ids: Collection[str] | None = None,
) -> SearchOutcome:
    """Enumerate valid order assignments with residual-valence propagation.

    ``max_search_states`` bounds visited nodes only.  ``max_candidates`` caps
    retained output and therefore never changes search completeness.
    ``valence_atom_ids`` limits the valence and charge checks to one
    component; the default still checks every atom.
    """
    if max_search_states < 1:
        raise ValueError("max_search_states must be positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    atoms = tuple(atoms)
    bonds = tuple(bonds)
    active = None if valence_atom_ids is None else set(valence_atom_ids)

    def relevant(atom: ChemicalAtom) -> bool:
        return active is None or atom.atom_id in active

    degrees = {atom.atom_id: 0 for atom in atoms}
    for idx in variable_indices:
        bond = bonds[idx]
        degrees[bond.atom1_id] += 1
        degrees[bond.atom2_id] += 1

    # Fixed source orders seed the residual sums and are never rewritten.
    sums = {atom.atom_id: 0 for atom in atoms}
    fixed = {}
    for idx, bond in enumerate(bonds):
        if bond.kind is not BondKind.COVALENT:
            continue
        # 1.5 is an aromatic Kekulé placeholder, not a fixed double bond.
        order = integral_bond_order(bond.order)
        if order is None:
            continue
        fixed[idx] = order
    for idx, order in fixed.items():
        bond = bonds[idx]
        sums[bond.atom1_id] += order
        sums[bond.atom2_id] += order

    variable = [idx for idx in variable_indices if idx not in fixed]
    domains = {
        idx: tuple(sorted(set(int(v) for v in choices[idx]))) for idx in variable
    }
    # Fail early on genuinely empty source domains.
    if any(not domain for domain in domains.values()):
        return SearchOutcome((), 1, True, 1)

    def bounds(unassigned: set[int]) -> tuple[dict[str, int], dict[str, int]]:
        low = {atom.atom_id: 0 for atom in atoms}
        high = {atom.atom_id: 0 for atom in atoms}
        for idx in unassigned:
            bond = bonds[idx]
            domain = domains[idx]
            minimum, maximum = min(domain), max(domain)
            low[bond.atom1_id] += minimum
            low[bond.atom2_id] += minimum
            high[bond.atom1_id] += maximum
            high[bond.atom2_id] += maximum
        return low, high

    def partial_valid(unassigned: set[int]) -> bool:
        low, high = bounds(unassigned)
        for atom in atoms:
            if not relevant(atom):
                continue
            if not _possible(
                allowed_valences(atom),
                sums[atom.atom_id],
                low[atom.atom_id],
                high[atom.atom_id],
                atom,
            ):
                return False
        return True

    results: list[tuple[float, tuple[ChemicalBond, ...]]] = []
    states = 0
    rejected = 0
    charge_rejected = 0
    exhausted = False

    def retain(candidate: tuple[float, tuple[ChemicalBond, ...]]) -> None:
        results.append(candidate)
        # Keep only the requested number while retaining deterministic ties.
        results.sort(key=lambda item: (item[0], order_signature(item[1])))
        if len(results) > max_candidates:
            results.pop()

    def visit(remaining: set[int], assignment: dict[int, int]) -> None:
        nonlocal states, rejected, charge_rejected, exhausted
        if exhausted:
            return
        states += 1
        if states > max_search_states:
            exhausted = True
            return
        if not partial_valid(remaining):
            rejected += 1
            return
        if not remaining:
            # Every known valence must be exact. Unknown elements are allowed
            # through but are marked provisional by the façade.
            for atom in atoms:
                if not relevant(atom):
                    continue
                states_for_atom = allowed_valences(atom)
                if states_for_atom is None:
                    continue
                target = atom_valence_target(atom, sums[atom.atom_id])
                if target not in states_for_atom:
                    rejected += 1
                    return
            # A requested total charge is a hard constraint.  It is checked
            # even when one or more atoms have no inferred formal charge:
            # an incomplete charge vector cannot be treated as a match.
            if total_charge is not None:
                charges: list[int] = []
                charge_known = True
                for atom in atoms:
                    if not relevant(atom):
                        continue
                    value = inferred_charge(
                        atom, atom_valence_target(atom, sums[atom.atom_id])
                    )
                    if value is None:
                        charge_known = False
                        break
                    charges.append(value)
                if not charge_known or sum(charges) != int(total_charge):
                    rejected += 1
                    charge_rejected += 1
                    return
            assigned = list(bonds)
            for idx, order in assignment.items():
                assigned[idx] = assigned[idx].__class__(
                    atom1_id=assigned[idx].atom1_id,
                    atom2_id=assigned[idx].atom2_id,
                    order=float(order),
                    kind=assigned[idx].kind,
                    aromatic=assigned[idx].aromatic,
                    atom2_image_shift=assigned[idx].atom2_image_shift,
                    stereochemistry=assigned[idx].stereochemistry,
                    evidence=assigned[idx].evidence,
                )
            value = float(score(assignment) if score is not None else 0.0)
            retain((value, tuple(assigned)))
            return
        # Most constrained first: small domains, then highest residual degree,
        # then stable source index for reproducibility.
        idx = min(
            remaining,
            key=lambda i: (
                len(domains[i]),
                -degrees[bonds[i].atom1_id] - degrees[bonds[i].atom2_id],
                i,
            ),
        )
        next_remaining = set(remaining)
        next_remaining.remove(idx)
        bond = bonds[idx]
        for order in domains[idx]:
            assignment[idx] = order
            sums[bond.atom1_id] += order
            sums[bond.atom2_id] += order
            visit(next_remaining, assignment)
            sums[bond.atom1_id] -= order
            sums[bond.atom2_id] -= order
            assignment.pop(idx, None)
            if exhausted:
                return

    all_indices = set(variable)
    visit(all_indices, {})
    return SearchOutcome(
        tuple(results), states, not exhausted, rejected, charge_rejected
    )


def order_signature(bonds: Sequence[ChemicalBond]) -> tuple[float, ...]:
    return tuple(0.0 if bond.order is None else float(bond.order) for bond in bonds)


__all__ = ["SearchOutcome", "search_bond_orders", "order_signature"]
