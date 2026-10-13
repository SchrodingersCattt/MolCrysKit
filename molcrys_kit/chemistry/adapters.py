"""Convenience adapters for array and batched bond-order perception.

The chemistry solver works on immutable :class:`ChemicalEntity` records.  This
module keeps the common NumPy/ASE boundary small and explicit: coordinates and
atomic numbers are copied into one finite entity, and the regular
``infer_bond_orders`` engine is then used.  Array inputs are never annotated or
mutated in place.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ase.data import atomic_numbers as _atomic_number_table
from ase.data import chemical_symbols

from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    FiniteChemicalEntity,
)


def _symbols(values: Sequence[Any]) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in values:
        if isinstance(value, str):
            symbol = value
        else:
            number = int(value)
            if number <= 0 or number >= len(chemical_symbols):
                raise ValueError(f"unsupported atomic number: {number}")
            symbol = chemical_symbols[number]
        if not symbol:
            raise ValueError(f"unsupported element: {value!r}")
        symbols.append(symbol)
    return tuple(symbols)


def _numbers(values: Sequence[Any], symbols: Sequence[str]) -> np.ndarray:
    """Normalize element symbols/numbers for the geometric candidate detector."""
    if all(isinstance(value, str) for value in values):
        try:
            return np.asarray(
                [_atomic_number_table[str(value)] for value in values], dtype=int
            )
        except KeyError as exc:
            raise ValueError(f"unsupported element: {exc.args[0]!r}") from exc
    return np.asarray(values, dtype=int)


def _entity_from_arrays(
    positions: Any,
    atomic_numbers: Sequence[Any],
    bonds: Any = None,
    *,
    atom_ids: Sequence[str] | None = None,
    bond_orders: Sequence[float | None] | None = None,
    cell: Any = None,
    pbc: Any = (False, False, False),
    entity_id: str = "array",
) -> FiniteChemicalEntity:
    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("positions must be finite")
    symbols = _symbols(tuple(atomic_numbers))
    if len(symbols) != len(coordinates):
        raise ValueError("atomic_numbers must have one value per position")
    ids = (
        tuple(atom_ids)
        if atom_ids is not None
        else tuple(f"a{i}" for i in range(len(symbols)))
    )
    if len(ids) != len(symbols) or len(set(ids)) != len(ids):
        raise ValueError("atom_ids must be unique and match positions")
    atoms = tuple(
        ChemicalAtom(atom_id=ids[i], element=symbols[i]) for i in range(len(ids))
    )

    if bonds is None:
        # Connectivity is deliberately explicit at this boundary.  Importing
        # the geometric candidate detector lazily keeps chemistry importable
        # without pulling the structure package into model code.
        from ..structures.bond import infer_bond_pairs

        matrix = None if cell is None else np.asarray(cell, dtype=float)
        inferred = infer_bond_pairs(
            coordinates,
            _numbers(atomic_numbers, symbols),
            cell=matrix,
            pbc=pbc,
        )
        pairs = inferred.pairs.tolist()
        if matrix is not None and any(bool(value) for value in pbc):
            inverse = np.linalg.inv(matrix)
            shifts = []
            for (first, second), vector in zip(pairs, inferred.vectors):
                raw = (
                    np.asarray(vector, dtype=float)
                    - (coordinates[second] - coordinates[first])
                ) @ inverse
                shifts.append(tuple(int(round(value)) for value in raw))
        else:
            shifts = [(0, 0, 0)] * len(pairs)
    else:
        if hasattr(bonds, "pairs"):
            bonds = bonds.pairs
        pairs = []
        shifts = []
        source_orders: list[float | None] = []
        for raw in bonds:
            if isinstance(raw, ChemicalBond):
                first, second = raw.atom1_id, raw.atom2_id
                shift = tuple(int(v) for v in raw.atom2_image_shift)
                source_orders.append(raw.order)
            elif isinstance(raw, dict):
                first, second = raw["atom1_id"], raw["atom2_id"]
                shift = tuple(int(v) for v in raw.get("atom2_image_shift", (0, 0, 0)))
                source_orders.append(raw.get("order"))
            else:
                values = tuple(raw)
                if len(values) < 2:
                    raise ValueError("bond pairs must contain two atom indices")
                first, second = int(values[0]), int(values[1])
                shift = (
                    tuple(int(v) for v in values[2:5])
                    if len(values) >= 5
                    else (0, 0, 0)
                )
                source_orders.append(None)
            if first in ids:
                first = ids.index(first)
            if second in ids:
                second = ids.index(second)
            if not (0 <= int(first) < len(ids) and 0 <= int(second) < len(ids)):
                raise ValueError("bond endpoint is outside the atom array")
            pairs.append((int(first), int(second)))
            shifts.append(shift)
    if bond_orders is not None and len(bond_orders) != len(pairs):
        raise ValueError("bond_orders must match bonds")
    records = tuple(
        ChemicalBond(
            ids[first],
            ids[second],
            order=(
                float(source_orders[index])
                if bond_orders is None and source_orders[index] is not None
                else None
                if bond_orders is None or bond_orders[index] is None
                else float(bond_orders[index])
            ),
            kind=BondKind.UNKNOWN,
            atom2_image_shift=shifts[index],
        )
        for index, (first, second) in enumerate(pairs)
    )
    matrix = None
    if cell is not None:
        matrix_array = np.asarray(cell, dtype=float)
        if matrix_array.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        matrix = tuple(tuple(float(value) for value in row) for row in matrix_array)
    embedding = Embedding(
        coordinates_A=tuple(
            (ids[index], tuple(float(v) for v in coordinates[index]))
            for index in range(len(ids))
        ),
        cell_A=matrix,
    )
    return FiniteChemicalEntity(
        entity_id=str(entity_id), atoms=atoms, bonds=records, embedding=embedding
    )


def infer_bond_orders_array(
    positions: Any,
    atomic_numbers: Sequence[Any],
    bonds: Any = None,
    *,
    atom_ids: Sequence[str] | None = None,
    bond_orders: Sequence[float | None] | None = None,
    cell: Any = None,
    pbc: Any = (False, False, False),
    total_charge: int | None = None,
    options: Any = None,
    strict: bool | None = None,
    entity_id: str = "array",
):
    """Infer bond orders from coordinate and element arrays.

    ``bonds`` may be an ``(M, 2)`` index array, an ``(M, 5)`` array with a
    lattice image shift, or an iterable of ``ChemicalBond``-like mappings. If
    omitted, the standard geometric candidate detector supplies connectivity.
    The returned object is exactly the regular solver's ``BondOrderResult``.
    """
    entity = _entity_from_arrays(
        positions,
        atomic_numbers,
        bonds,
        atom_ids=atom_ids,
        bond_orders=bond_orders,
        cell=cell,
        entity_id=entity_id,
        pbc=pbc,
    )
    from . import BondOrderOptions, infer_bond_orders

    if strict is not None:
        if options is None:
            options = BondOrderOptions(strict=bool(strict))
        elif getattr(options, "strict", None) != bool(strict):
            raise ValueError("strict conflicts with options.strict")
    return infer_bond_orders(
        entity, total_charge=total_charge, cell=cell, options=options
    )


def infer_bond_orders_batch(
    structures: Iterable[Any],
    atomic_numbers: Sequence[Any] | None = None,
    bonds: Any = None,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Run :func:`infer_bond_orders_array` independently for each frame.

    ``structures`` may be an iterable of ``(positions, numbers[, bonds])``
    tuples, or a stack of position arrays when ``atomic_numbers`` is shared.
    Results are returned in input order and no input array is modified.
    """
    items = list(structures)
    results = []
    for item in items:
        # Entity batches are useful when a crystal has already been annotated;
        # route those through the same pure façade without rebuilding records.
        from .models import FiniteChemicalEntity, PeriodicChemicalEntity

        if isinstance(item, (FiniteChemicalEntity, PeriodicChemicalEntity)):
            from . import infer_bond_orders

            results.append(infer_bond_orders(item, **kwargs))
            continue
        if isinstance(item, dict):
            positions = item.get("positions")
            numbers = item.get(
                "atomic_numbers", item.get("numbers", item.get("symbols"))
            )
            if positions is None or numbers is None:
                raise TypeError("batch mapping requires positions and atomic_numbers")
            frame_bonds = item.get("bonds", bonds)
            results.append(
                infer_bond_orders_array(positions, numbers, frame_bonds, **kwargs)
            )
            continue
        if (
            isinstance(item, (tuple, list))
            and len(item) >= 2
            and np.asarray(item[0]).ndim == 2
        ):
            positions, numbers = item[:2]
            frame_bonds = item[2] if len(item) >= 3 else bonds
        else:
            if atomic_numbers is None:
                raise TypeError("atomic_numbers is required for a position-array batch")
            positions, numbers, frame_bonds = item, atomic_numbers, bonds
        results.append(
            infer_bond_orders_array(positions, numbers, frame_bonds, **kwargs)
        )
    return tuple(results)


# Both spellings are kept as lightweight aliases because callers commonly use
# either plural ``arrays`` or ``batch`` for this stateless boundary.
infer_bond_orders_arrays = infer_bond_orders_array
infer_bond_orders_from_arrays = infer_bond_orders_array


__all__ = [
    "infer_bond_orders_array",
    "infer_bond_orders_arrays",
    "infer_bond_orders_from_arrays",
    "infer_bond_orders_batch",
]
