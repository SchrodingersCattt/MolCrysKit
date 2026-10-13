"""Small, conservative valence tables used by chemical perception.

The tables intentionally describe *candidate* Lewis valences.  They are not
oxidation-state or coordination-number tables; transition-metal edges are
handled as coordination/metallic evidence by the perception layer.
"""

from __future__ import annotations

from typing import Iterable

from .models import ChemicalAtom

# Ordinary main-group valence states.  Keeping this table in one module makes
# the solver and downstream hydrogen completion use the same constraints.
VALENCES: dict[str, tuple[int, ...]] = {
    "H": (1,),
    "D": (1,),
    "B": (3, 4),
    "C": (4,),
    "N": (2, 3, 4),
    "O": (1, 2, 3),
    "F": (1,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Se": (2, 4, 6),
    "Cl": (0, 1),
    "Br": (0, 1),
    "I": (0, 1),
}

VALENCES_FOR_CHARGE: dict[tuple[str, int], tuple[int, ...]] = {
    ("B", -1): (4,),
    ("C", -1): (3,),
    ("C", 1): (3,),
    ("N", -1): (2,),
    ("N", 1): (4,),
    ("O", -1): (1,),
    ("O", 1): (3,),
    ("F", -1): (0,),
    ("Cl", -1): (0,),
    ("Br", -1): (0,),
    ("I", -1): (0,),
}

CHARGE_BY_VALENCE: dict[str, dict[int, int]] = {
    "B": {3: 0, 4: -1},
    "C": {3: 0, 4: 0},
    "N": {2: -1, 3: 0, 4: 1},
    "O": {1: -1, 2: 0, 3: 1},
    "F": {0: -1, 1: 0},
    "Cl": {0: -1, 1: 0},
    "Br": {0: -1, 1: 0},
    "I": {0: -1, 1: 0},
    "Si": {4: 0},
    "P": {3: 0, 5: 0},
    "S": {2: 0, 4: 0, 6: 0},
    "Se": {2: 0, 4: 0, 6: 0},
    "H": {1: 0},
    "D": {1: 0},
}


def allowed_valences(atom: ChemicalAtom) -> tuple[int, ...] | None:
    """Return feasible valence sums for one atom.

    ``None`` means the element is outside the conservative main-group model;
    the caller should preserve its source state and report indeterminate
    chemistry rather than silently applying an organic rule.
    """
    if atom.formal_charge is not None:
        explicit = VALENCES_FOR_CHARGE.get((atom.element, int(atom.formal_charge)))
        if explicit is not None:
            return explicit
    return VALENCES.get(atom.element)


def inferred_charge(atom: ChemicalAtom, valence_sum: int) -> int | None:
    """Infer a formal charge only where the table has an explicit convention."""
    if atom.formal_charge is not None:
        return int(atom.formal_charge)
    return CHARGE_BY_VALENCE.get(atom.element, {}).get(int(valence_sum))


def atom_valence_target(atom: ChemicalAtom, value: int) -> int:
    """Convert a bond-order sum into the atom's valence accounting baseline."""
    # Explicit hydrogens are represented in the atom record when no H edge is
    # present.  Radical electrons consume one valence slot conservatively.
    return (
        int(value) + int(atom.explicit_hydrogens or 0) + int(atom.radical_electrons > 0)
    )


def feasible_valence(atom: ChemicalAtom, bond_sum: int) -> bool:
    states = allowed_valences(atom)
    if states is None:
        return True
    return atom_valence_target(atom, bond_sum) in states


def charge_sum(atoms: Iterable[ChemicalAtom]) -> int | None:
    values = [atom.formal_charge for atom in atoms]
    return (
        sum(values) if values and all(value is not None for value in values) else None
    )


__all__ = [
    "VALENCES",
    "VALENCES_FOR_CHARGE",
    "CHARGE_BY_VALENCE",
    "allowed_valences",
    "inferred_charge",
    "atom_valence_target",
    "feasible_valence",
    "charge_sum",
]
