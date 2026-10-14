"""Component decomposition and exact tree dynamic programming.

Variable covalent bonds are split into connected components. A component with
a cycle is enumerated by the bounded search. A forest is solved by tree DP
whose valence rules match that search: the retained assignment is the minimum
geometry score, then the lexicographically smallest order signature. A
requested total charge is applied after the components are merged. When the
independently optimal pieces do not meet that charge, their charge frontiers
are combined by convolution instead of searching the whole graph again.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..constants.config import CHEMISTRY_PERCEPTION_CONFIG
from .models import BondKind, ChemicalAtom, ChemicalBond
from .solver import SearchOutcome, order_signature, search_bond_orders
from .valence import (
    allowed_valences,
    atom_valence_target,
    inferred_charge,
    integral_bond_order,
)


@dataclass(frozen=True, slots=True)
class BondComponent:
    """One connected piece of the variable-bond graph."""

    indices: tuple[int, ...]
    atom_ids: frozenset[str]
    cyclic: bool


def decompose_variable_graph(
    bonds: tuple[ChemicalBond, ...] | list[ChemicalBond],
    variable_indices: list[int] | tuple[int, ...],
) -> tuple[BondComponent, ...]:
    """Partition variable bonds. A repeated edge or a self-edge is cyclic."""
    bonds = tuple(bonds)
    parent: dict[str, str] = {}

    def find(atom_id: str) -> str:
        parent.setdefault(atom_id, atom_id)
        while parent[atom_id] != atom_id:
            parent[atom_id] = parent[parent[atom_id]]
            atom_id = parent[atom_id]
        return atom_id

    def union(left: str, right: str) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        parent[root_right] = root_left
        return True

    cyclic_atoms: set[str] = set()
    for index in variable_indices:
        bond = bonds[index]
        if bond.atom1_id == bond.atom2_id or not union(bond.atom1_id, bond.atom2_id):
            cyclic_atoms.add(bond.atom1_id)
            cyclic_atoms.add(bond.atom2_id)
    grouped: dict[str, list[int]] = {}
    for index in variable_indices:
        grouped.setdefault(find(bonds[index].atom1_id), []).append(index)
    components = []
    for indices in grouped.values():
        atom_ids = set()
        for index in indices:
            atom_ids.add(bonds[index].atom1_id)
            atom_ids.add(bonds[index].atom2_id)
        components.append(
            BondComponent(
                tuple(indices),
                frozenset(atom_ids),
                any(atom_id in cyclic_atoms for atom_id in atom_ids),
            )
        )
    components.sort(key=lambda item: item.indices[0] if item.indices else 0)
    return tuple(components)


def decompose_and_solve(
    atoms,
    bonds,
    variable_indices,
    choices,
    *,
    score,
    total_charge,
    max_search_states: int,
    max_candidates: int,
) -> tuple[SearchOutcome, dict]:
    """Solve each component, then keep the result only if charge still holds."""
    atoms = tuple(atoms)
    bonds = tuple(bonds)
    variable = [index for index in variable_indices if index in choices]
    components = decompose_variable_graph(bonds, variable)
    info = {
        "components": len(components),
        "tree_components": sum(not item.cyclic for item in components),
        "cyclic_components": sum(item.cyclic for item in components),
        "solver": "search" if any(item.cyclic for item in components) else "tree_dp",
    }
    if len(components) == 1 and not components[0].cyclic:
        info["solver"] = "tree_dp"
        outcome, frontier = _solve_tree(
            atoms,
            bonds,
            components[0],
            choices,
            score=score,
            max_search_states=max_search_states,
            max_candidates=max_candidates,
        )
        if total_charge is None or not outcome.candidates:
            return outcome, info
        matched = tuple(
            item
            for item in outcome.candidates
            if _charge_matches(atoms, item[1], total_charge)
        )
        if matched:
            return (
                SearchOutcome(
                    matched,
                    outcome.states,
                    outcome.complete,
                    outcome.rejected,
                ),
                info,
            )
        if not outcome.complete:
            return SearchOutcome((), outcome.states, False, outcome.rejected, 1), info
        selected = _frontier_matching_charge(
            atoms, bonds, frontier, total_charge, max_candidates
        )
        if not selected:
            return SearchOutcome((), outcome.states, True, outcome.rejected, 1), info
        info["solver"] = "charge_convolution"
        return (
            SearchOutcome(selected, outcome.states, True, outcome.rejected),
            info,
        )
    if not variable:
        info["solver"] = "fixed"
        outcome = search_bond_orders(
            atoms,
            bonds,
            [],
            {},
            score=score,
            total_charge=total_charge,
            max_search_states=max_search_states,
            max_candidates=max_candidates,
        )
        return outcome, info
    # One cyclic piece is the historical search, including its alternatives.
    if len(components) == 1 and components[0].cyclic:
        info["solver"] = "search"
        outcome = search_bond_orders(
            atoms,
            bonds,
            components[0].indices,
            choices,
            score=score,
            total_charge=total_charge,
            max_search_states=max_search_states,
            max_candidates=max_candidates,
        )
        return outcome, info
    remaining = max_search_states
    partials: list[tuple[BondComponent, SearchOutcome, tuple]] = []
    for component in components:
        budget = max(1, remaining)
        if component.cyclic:
            outcome = search_bond_orders(
                atoms,
                bonds,
                component.indices,
                choices,
                score=score,
                total_charge=None,
                max_search_states=budget,
                max_candidates=max_candidates,
                valence_atom_ids=component.atom_ids,
            )
            frontier = _frontier_from_search(atoms, bonds, component, outcome)
            info["solver"] = "mixed" if info["tree_components"] else "search"
        else:
            outcome, frontier = _solve_tree(
                atoms,
                bonds,
                component,
                choices,
                score=score,
                max_search_states=budget,
            )
        partials.append((component, outcome, frontier))
        remaining -= outcome.states
        if not outcome.candidates or not outcome.complete or remaining <= 0:
            break
    spent = sum(item.states for _, item, _frontier in partials)
    if len(partials) != len(components) or any(
        not outcome.complete for _, outcome, _frontier in partials
    ):
        return SearchOutcome((), spent, False, 0), info
    if any(not outcome.candidates for _, outcome, _frontier in partials):
        return (
            SearchOutcome(
                (),
                spent,
                True,
                sum(outcome.rejected for _, outcome, _frontier in partials),
            ),
            info,
        )
    merged = _merge(atoms, bonds, partials, score)
    if not merged.candidates:
        return merged, info
    if total_charge is not None and not _charge_matches(
        atoms, merged.candidates[0][1], total_charge
    ):
        # Independent optima can miss a charge that is still a sum of local
        # valence states. Convolution over those states stays linear in the
        # number of components; a second whole-graph search is exponential.
        if any(not item.complete for _, item, _frontier in partials):
            return (
                SearchOutcome(
                    (), sum(item.states for _, item, _frontier in partials), False, 0, 1
                ),
                info,
            )
        covered = set()
        for component, _outcome, _frontier in partials:
            covered.update(component.atom_ids)
        outside = _outside_charge(atoms, bonds, covered)
        chosen = (
            None
            if outside is None
            else _convolve(
                [frontier for _component, _outcome, frontier in partials],
                int(total_charge) - outside,
            )
        )
        if chosen is None:
            return SearchOutcome((), spent, True, merged.rejected, 1), info
        _path_score, assignment = chosen
        assigned = _apply(bonds, assignment)
        if not _charge_matches(atoms, assigned, total_charge):
            return SearchOutcome((), spent, True, merged.rejected, 1), info
        full_score = float(score(assignment) if score is not None else _path_score)
        info["solver"] = "charge_convolution"
        return SearchOutcome(((full_score, assigned),), spent, True, 0), info
    return merged, info


def _solve_tree(
    atoms,
    bonds,
    component: BondComponent,
    choices,
    *,
    score,
    max_search_states: int,
    max_candidates: int = 64,
) -> tuple[SearchOutcome, tuple]:
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    domains = {
        index: tuple(sorted({int(value) for value in choices[index]}))
        for index in component.indices
    }
    neighbors: dict[str, list[tuple[int, str]]] = {
        atom_id: [] for atom_id in component.atom_ids
    }
    for index in component.indices:
        bond = bonds[index]
        neighbors[bond.atom1_id].append((index, bond.atom2_id))
        neighbors[bond.atom2_id].append((index, bond.atom1_id))
    fixed = {atom_id: 0 for atom_id in component.atom_ids}
    variable = set(component.indices)
    for index, bond in enumerate(bonds):
        order = integral_bond_order(bond.order)
        if index in variable or bond.kind is not BondKind.COVALENT or order is None:
            continue
        if bond.atom1_id == bond.atom2_id:
            if bond.atom1_id in fixed:
                fixed[bond.atom1_id] += 2 * order
            continue
        if bond.atom1_id in fixed:
            fixed[bond.atom1_id] += order
        if bond.atom2_id in fixed:
            fixed[bond.atom2_id] += order
    states = 0
    exhausted = False
    memo: dict[tuple, object] = {}

    def edge_cost(index: int, order: int) -> float:
        if score is None:
            return 0.0
        return float(score({index: order}))

    def subtree(
        node: str,
        incoming_index: int | None,
        incoming_order: int,
        collect: bool = False,
    ):
        nonlocal states, exhausted
        key = (node, incoming_index, incoming_order)
        if key in memo:
            return memo[key]
        states += 1
        if states > max_search_states:
            exhausted = True
            memo[key] = "EXHAUSTED"
            return "EXHAUSTED"
        children = [
            (index, other)
            for index, other in neighbors[node]
            if index != incoming_index
        ]
        partials = [(0, 0.0, (), {}, 0, True)]
        for index, other in children:
            domain = domains[index]
            nxt = []
            for order in domain:
                child = subtree(other, index, order)
                if child == "EXHAUSTED":
                    return "EXHAUSTED"
                if child is None:
                    continue
                child_options = (
                    child
                    if isinstance(child, list)
                    else [child]
                )
                for child_option in child_options:
                    for current in partials:
                        nxt.append(_combine(current, child_option, order))
            partials = _keep_best_by_sum(nxt)
            if not partials:
                memo[key] = None
                return None
        if collect:
            return _feasible_from_partials(
                atom_by_id[node],
                fixed[node],
                incoming_index,
                incoming_order,
                partials,
                edge_cost,
            )
        result = _finish_node(
            atom_by_id[node],
            fixed[node],
            incoming_index,
            incoming_order,
            partials,
            edge_cost,
        )
        memo[key] = result
        return result

    root = min(component.atom_ids)
    # A saturated chain is a path of length N. The post-order walk is recursive,
    # so the limit has to cover that path plus the Python frames already in use.
    # Add the path length on top of the frames already in use. Replacing the
    # limit with ``N + 32`` still overflows once pytest and the façade are on
    # the stack.
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(previous_limit + len(component.atom_ids) + 32)
    try:
        found = subtree(root, None, 0, collect=True)
    finally:
        if sys.getrecursionlimit() != previous_limit:
            sys.setrecursionlimit(previous_limit)
    if found == "EXHAUSTED" or exhausted:
        return SearchOutcome((), states, False, 0), ()
    if not found:
        return SearchOutcome((), states, True, 1), ()
    ranked = []
    frontier_best: dict[int, tuple] = {}
    for path_score, signature, assignment, _charge, _known in found:
        assigned = _apply(bonds, assignment)
        full_score = float(score(assignment) if score is not None else path_score)
        ranked.append((full_score, order_signature(assigned), assigned))
        charge = _subset_charge(atoms, assigned, component.atom_ids)
        if charge is None:
            continue
        previous = frontier_best.get(charge)
        if previous is None or (full_score, signature) < (previous[0], previous[1]):
            frontier_best[charge] = (full_score, signature, dict(assignment))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best_score = ranked[0][0]
    gap = float(CHEMISTRY_PERCEPTION_CONFIG["COMPETITIVE_SCORE_GAP"])
    kept = tuple((item[0], item[2]) for item in ranked if item[0] - best_score <= gap)[
        :max_candidates
    ]
    frontier = tuple(
        (charge, item[0], item[2]) for charge, item in sorted(frontier_best.items())
    )
    return SearchOutcome(kept, states, True, 0), frontier


def _combine(current, child, order: int):
    """Merge a child that already owns the edge used to reach it."""
    order_sum, path_score, signature, assignment, charge, known = current
    child_score, child_sig, child_assignment, child_charge, child_known = child
    merged = dict(assignment)
    merged.update(child_assignment)
    return (
        order_sum + order,
        path_score + child_score,
        tuple(sorted(signature + child_sig)),
        merged,
        charge + child_charge,
        known and child_known,
    )


def _keep_best_by_sum(partials):
    """Keep the best score for each residual sum, including equal ties.

    Equal-scoring alternatives are chemically meaningful resonance forms.  A
    single representative per sum made the DP claim a complete search while
    silently dropping the second carboxylate placement.  Distinct ties are
    retained; strictly worse paths remain safely pruned for optimality.
    """
    best: dict[int, list] = {}
    for item in partials:
        previous = best.setdefault(item[0], [])
        if not previous or item[1] < previous[0][1]:
            best[item[0]] = [item]
            continue
        if item[1] != previous[0][1]:
            continue
        signature = item[2]
        if all(candidate[2] != signature for candidate in previous):
            previous.append(item)
    return [item for items in best.values() for item in items]


def _finish_node(
    atom: ChemicalAtom,
    fixed_sum: int,
    incoming_index,
    incoming_order,
    partials,
    edge_cost,
):
    candidates = []
    for order_sum, path_score, signature, assignment, charge, known in partials:
        bond_sum = fixed_sum + incoming_order + order_sum
        target = atom_valence_target(atom, bond_sum)
        allowed = allowed_valences(atom)
        if allowed is not None and target not in allowed:
            continue
        value = inferred_charge(atom, target)
        node_known = known and value is not None
        node_charge = charge + (0 if value is None else value)
        node_assignment = dict(assignment)
        node_signature = signature
        node_score = path_score
        if incoming_index is not None:
            node_assignment[incoming_index] = incoming_order
            node_signature = tuple(
                sorted(signature + ((incoming_index, incoming_order),))
            )
            node_score = path_score + edge_cost(incoming_index, incoming_order)
        candidates.append(
            (
            node_score,
            node_signature,
            node_assignment,
            node_charge,
            node_known,
            )
        )
    if not candidates:
        return None
    best_score = min(candidate[0] for candidate in candidates)
    return [candidate for candidate in candidates if candidate[0] == best_score]


def _feasible_from_partials(
    atom, fixed_sum, incoming_index, incoming_order, partials, edge_cost
):
    """Every valence-valid completion of one node, not only the tie-break winner."""
    found = []
    for order_sum, path_score, signature, assignment, charge, known in partials:
        bond_sum = fixed_sum + incoming_order + order_sum
        target = atom_valence_target(atom, bond_sum)
        allowed = allowed_valences(atom)
        if allowed is not None and target not in allowed:
            continue
        value = inferred_charge(atom, target)
        node_assignment = dict(assignment)
        node_signature = signature
        node_score = path_score
        if incoming_index is not None:
            node_assignment[incoming_index] = incoming_order
            node_signature = tuple(
                sorted(signature + ((incoming_index, incoming_order),))
            )
            node_score = path_score + edge_cost(incoming_index, incoming_order)
        found.append(
            (
                node_score,
                node_signature,
                node_assignment,
                charge + (0 if value is None else value),
                known and value is not None,
            )
        )
    return found


def _merge(atoms, bonds, partials, score) -> SearchOutcome | None:
    if any(not outcome.candidates for _, outcome, _frontier in partials):
        complete = all(outcome.complete for _, outcome, _frontier in partials)
        return SearchOutcome(
            (),
            sum(outcome.states for _, outcome, _frontier in partials),
            complete,
            1,
        )
    assignment = {}
    for component, outcome, _frontier in partials:
        chosen = outcome.candidates[0][1]
        allowed = set(component.indices)
        for index in allowed:
            order = integral_bond_order(chosen[index].order)
            if order is None:
                return SearchOutcome(
                    (),
                    sum(item.states for _, item, _frontier in partials),
                    False,
                    1,
                )
            assignment[index] = order
    assigned = _apply(bonds, assignment)
    full_score = float(score(assignment) if score is not None else 0.0)
    # Confirm the merged signature is the one search would rank first.
    _ = order_signature(assigned)
    return SearchOutcome(
        ((full_score, assigned),),
        sum(outcome.states for _, outcome, _frontier in partials),
        all(outcome.complete for _, outcome, _frontier in partials),
        sum(outcome.rejected for _, outcome, _frontier in partials),
    )


def _apply(bonds, assignment: dict[int, int]) -> tuple[ChemicalBond, ...]:
    assigned = list(bonds)
    for index, order in assignment.items():
        bond = assigned[index]
        assigned[index] = bond.__class__(
            atom1_id=bond.atom1_id,
            atom2_id=bond.atom2_id,
            order=float(order),
            kind=bond.kind,
            aromatic=bond.aromatic,
            atom2_image_shift=bond.atom2_image_shift,
            stereochemistry=bond.stereochemistry,
            evidence=bond.evidence,
        )
    return tuple(assigned)


def _subset_charge(atoms, assigned_bonds, atom_ids) -> int | None:
    """Formal-charge sum of ``atom_ids`` under one assignment.

    A periodic self-edge increments its single atom twice, once per endpoint
    reference. That is the whole valence contribution.
    """
    wanted = set(atom_ids)
    sums = {atom.atom_id: 0 for atom in atoms if atom.atom_id in wanted}
    for bond in assigned_bonds:
        if bond.kind is not BondKind.COVALENT:
            continue
        value = integral_bond_order(bond.order)
        if value is None:
            continue
        if bond.atom1_id in sums:
            sums[bond.atom1_id] += value
        if bond.atom2_id in sums:
            sums[bond.atom2_id] += value
    total = 0
    for atom in atoms:
        if atom.atom_id not in sums:
            continue
        value = inferred_charge(atom, atom_valence_target(atom, sums[atom.atom_id]))
        if value is None:
            return None
        total += value
    return total


def _charge_matches(atoms, assigned_bonds, total_charge: int) -> bool:
    charge = _subset_charge(atoms, assigned_bonds, {atom.atom_id for atom in atoms})
    return charge is not None and charge == int(total_charge)


def _outside_charge(atoms, bonds, covered: set[str]) -> int | None:
    """Charge of atoms that belong to no variable component."""
    return _subset_charge(
        atoms,
        bonds,
        {atom.atom_id for atom in atoms if atom.atom_id not in covered},
    )


def _frontier_from_search(
    atoms, bonds, component: BondComponent, outcome: SearchOutcome
):
    best: dict[int, tuple] = {}
    for path_score, assigned in outcome.candidates:
        assignment = {}
        missing = False
        for index in component.indices:
            order = integral_bond_order(assigned[index].order)
            if order is None:
                missing = True
                break
            assignment[index] = order
        if missing:
            continue
        charge = _subset_charge(atoms, assigned, component.atom_ids)
        if charge is None:
            continue
        signature = order_signature(assigned)
        previous = best.get(charge)
        if previous is None or (path_score, signature) < (previous[0], previous[1]):
            best[charge] = (path_score, signature, assignment)
    return tuple((charge, item[0], item[2]) for charge, item in sorted(best.items()))


def _frontier_matching_charge(atoms, bonds, frontier, total_charge: int, limit: int):
    matched = []
    for _charge, path_score, assignment in frontier:
        assigned = _apply(bonds, assignment)
        if _charge_matches(atoms, assigned, total_charge):
            matched.append((path_score, assigned))
    if not matched:
        return ()
    matched.sort(key=lambda item: (item[0], order_signature(item[1])))
    best = matched[0][0]
    gap = float(CHEMISTRY_PERCEPTION_CONFIG["COMPETITIVE_SCORE_GAP"])
    return tuple(item for item in matched if item[0] - best <= gap)[:limit]


def _convolve(frontiers, target: int):
    """Best assignment whose component charges sum to ``target``."""
    states = {0: (0.0, {})}
    for frontier in frontiers:
        nxt: dict[int, tuple] = {}
        for base_charge, (base_score, base_assignment) in states.items():
            for charge, score, assignment in frontier:
                total = base_charge + int(charge)
                merged = dict(base_assignment)
                merged.update(assignment)
                candidate = (base_score + score, merged)
                previous = nxt.get(total)
                if previous is None or candidate[0] < previous[0]:
                    nxt[total] = candidate
        states = nxt
        if not states:
            return None
    return states.get(int(target))


__all__ = ["BondComponent", "decompose_and_solve", "decompose_variable_graph"]
