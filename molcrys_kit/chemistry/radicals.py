"""Bracket-valence radical inference for OpenSMILES atoms.

OpenSMILES has no radical token. A bracket atom states an explicit hydrogen
count (omitted hydrogen means zero) and an optional formal charge. This module
is the single policy that turns the gap between that observed valence and a
closed-shell target into ``ChemicalAtom.radical_electrons``.

Policy
------
* Only bracket atoms are inferred. Unbracketed organic atoms keep the
  OpenSMILES default-hydrogen completion and stay closed-shell.
* If the observed valence equals an allowed closed-shell state for the
  element and charge, the atom is closed-shell (``radical_electrons`` 0).
  Ions are therefore not rewritten as radicals.
* If the observed valence is below every allowed state, the integer deficit
  is the radical-electron count. That is the methyl, hydroxyl, and peroxyl
  reading of ``[CH3]``, ``[OH]``, and ``[O]``.
* If the observed valence sits strictly between allowed states (sulfur is the
  common case), the count is left unset. The atom stays an explicit
  indeterminate valence, not a guessed radical and not a hydrogenated
  closed shell.
* An atom that already carries ``radical_electrons`` is preserved.
"""

from __future__ import annotations

from dataclasses import replace

from .models import ChemicalAtom, Evidence, EvidenceSource, FiniteChemicalEntity


# Closed-shell valence targets keyed by element and formal charge.
# Several states mean "any of these is closed-shell"; radicals are inferred
# only from a deficit below the lowest state.
_CLOSED_SHELL: dict[str, dict[int, tuple[int, ...]]] = {
    "H": {0: (1,), 1: (0,), -1: (0,)},
    "B": {0: (3,), -1: (4,)},
    "C": {0: (4,), 1: (3,), -1: (3,)},
    # N+ is closed-shell both as a three-coordinate azonia centre and as a
    # four-coordinate ammonium centre. A deficit below 3 is a radical.
    "N": {0: (3,), 1: (3, 4), -1: (2,)},
    "O": {0: (2,), 1: (3,), -1: (1,)},
    "F": {0: (1,), -1: (0,)},
    "P": {0: (3, 5), 1: (4,), -1: (2, 4)},
    "S": {0: (2, 4, 6), 1: (3, 5), -1: (1, 3, 5)},
    "Cl": {0: (1,), -1: (0,)},
    "Br": {0: (1,), -1: (0,)},
    "I": {0: (1,), -1: (0,)},
}


def is_bracket_atom(atom: ChemicalAtom) -> bool:
    """Return whether OpenSMILES parsed this atom inside brackets."""
    return any(
        evidence.method == "OpenSMILES bracket atom parser"
        for evidence in atom.evidence
    )


def observed_bracket_valence(atom: ChemicalAtom, bond_sum: float) -> float:
    """Bond-order sum plus the bracket hydrogen count.

    A bracket without an ``H`` token requested zero hydrogens. That is
    different from an unbracketed atom whose implicit hydrogens are still
    unresolved.
    """
    return float(bond_sum) + (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)


def bracket_radical_electrons(element: str, formal_charge: int | None, observed: float) -> int | None:
    """Return the inferred radical count, or ``None`` when the state is indeterminate."""
    states = _CLOSED_SHELL.get(element, {}).get(int(formal_charge or 0))
    if not states:
        return None
    if any(abs(observed - state) <= 1e-8 for state in states):
        return 0
    minimum = min(states)
    if observed < minimum - 1e-8:
        deficit = minimum - observed
        if abs(deficit - round(deficit)) > 1e-8:
            return None
        return int(round(deficit))
    return None


def assign_bracket_radicals(entity: FiniteChemicalEntity) -> FiniteChemicalEntity:
    """Populate radical-electron counts implied by bracket valence."""
    if not isinstance(entity, FiniteChemicalEntity):
        return entity
    adjacency: dict[str, float] = {atom.atom_id: 0.0 for atom in entity.atoms}
    for bond in entity.bonds:
        adjacency[bond.atom1_id] = adjacency.get(bond.atom1_id, 0.0) + float(bond.order or 0.0)
        adjacency[bond.atom2_id] = adjacency.get(bond.atom2_id, 0.0) + float(bond.order or 0.0)
    changed = False
    atoms = []
    for atom in entity.atoms:
        if atom.radical_electrons or not is_bracket_atom(atom):
            atoms.append(atom)
            continue
        inferred = bracket_radical_electrons(
            atom.element,
            atom.formal_charge,
            observed_bracket_valence(atom, adjacency.get(atom.atom_id, 0.0)),
        )
        if not inferred:
            atoms.append(atom)
            continue
        changed = True
        atoms.append(
            replace(
                atom,
                radical_electrons=inferred,
                evidence=(
                    *atom.evidence,
                    Evidence(
                        EvidenceSource.INFERRED,
                        "OpenSMILES bracket valence radical inference",
                        detail=(
                            f"{atom.element} valence deficit {inferred} "
                            "stored as radical_electrons"
                        ),
                    ),
                ),
            )
        )
    if not changed:
        return entity
    return replace(entity, atoms=tuple(atoms))


def radicals_match_bracket_valence(entity: FiniteChemicalEntity) -> bool:
    """Return whether every radical count is the bracket-valence deficit.

    OpenSMILES can round-trip those counts by emitting the bracket hydrogen
    count and letting the reader infer the deficit again. Any other radical
    count needs MCK-LN.
    """
    adjacency: dict[str, float] = {atom.atom_id: 0.0 for atom in entity.atoms}
    for bond in entity.bonds:
        adjacency[bond.atom1_id] = adjacency.get(bond.atom1_id, 0.0) + float(bond.order or 0.0)
        adjacency[bond.atom2_id] = adjacency.get(bond.atom2_id, 0.0) + float(bond.order or 0.0)
    for atom in entity.atoms:
        if not atom.radical_electrons:
            continue
        hydrogen = (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        inferred = bracket_radical_electrons(
            atom.element,
            atom.formal_charge,
            float(adjacency.get(atom.atom_id, 0.0)) + hydrogen,
        )
        if inferred != atom.radical_electrons:
            return False
    return True


__all__ = [
    "assign_bracket_radicals",
    "bracket_radical_electrons",
    "is_bracket_atom",
    "observed_bracket_valence",
    "radicals_match_bracket_valence",
]
