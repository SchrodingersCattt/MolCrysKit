"""Auditable chemical replacement and packing transfer in ordered full cells.

This is a placement layer, not a chemistry inference engine. Complete replacement
molecules carry the existing CrystalMolecule chemistry, bonds and atom arrays.
Roles and integral molecular charges are explicit caller assertions, checked
against known atom charges. Only fully occupied, finite, contiguous molecules
are accepted. Outputs use full-cell P1 semantics; orbit completeness describes
site selection and does not certify preservation of a space group.

Sampling is a bounded seeded construction heuristic, not an optimizer or proof
of impossibility. Failed requests return no candidate structure. Relaxation
protocols describe degrees of freedom for an external driver; no potential or
optimizer is selected or run here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial.transform import Rotation

from ..chemistry.models import ChemicalBond, Embedding
from ..chemistry.topology import analyze_periodic_topology
from ..constants import get_atomic_radius, has_atomic_radius
from ..constants.config import (
    CHEMICAL_EDIT_ORIENTATION_SAMPLES,
    KEY_ATOM_ID,
    KEY_FORMAL_CHARGE,
    KEY_FORMAL_CHARGE_KNOWN,
    KEY_OCCUPANCY,
    MIN_PERIODIC_CELL_VOLUME_A3,
    PARTIAL_OCCUPANCY_TOLERANCE,
)
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import (
    CrystalMolecule,
    _refresh_contiguous_bond_geometry,
    _strip_stale_frac_arrays,
)
from .forward import MolecularTemplate

__all__ = [
    "ChemicalAtomMapping",
    "ChemicalEdit",
    "CrystalEditSpec",
    "CrystalInventory",
    "PlacementPolicy",
    "MoleculeInsertion",
    "StoichiometryPolicy",
    "PackingTransferPlan",
    "CounterfactualStatus",
    "CounterfactualResult",
    "RelaxationProtocol",
    "apply_chemical_edit",
    "transfer_packing",
    "periodic_clash_diagnostics",
    "write_cf0",
]


class CounterfactualStatus(str, Enum):
    """Construction status; success makes no stability or realizability claim."""

    CONSTRUCTED = "CONSTRUCTED"
    INCOMPATIBLE = "INCOMPATIBLE"
    CLASH = "CLASH"


@dataclass(frozen=True)
class CrystalInventory:
    """One role and declared integral charge per molecule, in crystal order.

    Optional orbit IDs are caller-declared site groups (e.g. saved forward pose
    IDs). They select complete groups, not a new symmetry inference algorithm.
    """

    roles: tuple[str, ...]
    charges: tuple[int, ...]
    orbit_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "charges", tuple(self.charges))
        object.__setattr__(self, "orbit_ids", tuple(self.orbit_ids))
        if not self.roles or any(not isinstance(r, str) or not r for r in self.roles):
            raise ValueError("roles must contain nonempty strings")
        if len(self.roles) != len(self.charges):
            raise ValueError("one charge is required per role")
        if any(
            isinstance(q, bool) or not isinstance(q, (int, np.integer))
            for q in self.charges
        ):
            raise ValueError("molecular charges must be integers")
        if self.orbit_ids and (
            len(self.orbit_ids) != len(self.roles)
            or any(not isinstance(oid, str) or not oid for oid in self.orbit_ids)
        ):
            raise ValueError("one nonempty string orbit ID is required per molecule")


@dataclass(frozen=True)
class ChemicalAtomMapping:
    """Injective (parent index, replacement index) retained-identity pairs.

    Elements and known isotope/stereo identities must agree. Unmapped source
    atoms are deleted; unmapped replacement atoms get new IDs. A mapping need
    not preserve bonds: chemistry is supplied by the complete replacement.
    This type is distinct from symmetry_path.AtomCorrespondence.
    """

    pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        pairs = tuple(tuple(p) for p in self.pairs)
        if any(
            len(p) != 2
            or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in p
            )
            for p in pairs
        ):
            raise ValueError("mapping pairs must contain two integer indices")
        if len({p[0] for p in pairs}) != len(pairs) or len(
            {p[1] for p in pairs}
        ) != len(pairs):
            raise ValueError("atom mapping must be injective")
        object.__setattr__(self, "pairs", tuple(sorted(pairs)))


@dataclass(frozen=True)
class PlacementPolicy:
    """Explicit anchor and orientation policy, with summed-radius clash test.

    Anchors are atom indices or ``com``. ``sample`` tries seeded uniform proper
    rotations about the anchor, including a mapped initial orientation when
    available. ``mapped`` requires three non-collinear retained atom pairs and
    uses a proper least-squares rotation about the declared anchor. No reflection
    or internal distortion is applied. COM requires explicit opt-in here.
    """

    source_anchor: int | str
    target_anchor: int | str
    orientation: str = "sample"
    samples: int = CHEMICAL_EDIT_ORIENTATION_SAMPLES
    clash_scale: float = 1.0

    def __post_init__(self):
        for anchor in (self.source_anchor, self.target_anchor):
            if not (
                (
                    isinstance(anchor, (int, np.integer))
                    and not isinstance(anchor, bool)
                    and anchor >= 0
                )
                or anchor == "com"
            ):
                raise ValueError("anchor must be a nonnegative atom index or 'com'")
        if self.orientation not in {"sample", "mapped"}:
            raise ValueError("orientation must be 'sample' or 'mapped'")
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples < 1
        ):
            raise ValueError("samples must be a positive integer")
        if not np.isfinite(self.clash_scale) or self.clash_scale <= 0:
            raise ValueError("clash_scale must be finite and positive")


@dataclass(frozen=True)
class ChemicalEdit:
    """Complete prebuilt chemistry; source formula/charge guards catch wrong sites."""

    replacement: CrystalMolecule
    source_formula: str
    source_charge: int
    target_charge: int
    placement: PlacementPolicy
    mapping: ChemicalAtomMapping = field(default_factory=ChemicalAtomMapping)

    def __post_init__(self):
        if not isinstance(self.replacement, CrystalMolecule):
            raise TypeError("replacement must be a CrystalMolecule")
        for q in (self.source_charge, self.target_charge):
            if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
                raise ValueError("charges must be integral")
        object.__setattr__(self, "replacement", _clone(self.replacement))


@dataclass(frozen=True)
class CrystalEditSpec:
    """One edit on a role, explicit sites, or all members of selected orbits.

    ``orbit`` expands seed indices using inventory.orbit_ids or the forward
    builder's forward_pose_index. Missing orbit evidence is refused. ``sites``
    is an explicit local-defect operation and makes no completeness claim.
    """

    edit: ChemicalEdit
    scope: str = "role"
    role: str | None = None
    indices: tuple[int, ...] = ()
    expected_charge: int | None = None
    expected_formula: dict[str, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "indices", tuple(self.indices))
        if self.scope not in {"role", "sites", "orbit"}:
            raise ValueError("scope must be role, sites or orbit")
        if self.scope == "role" and (not self.role or self.indices):
            raise ValueError("role scope requires a role and no indices")
        if self.scope != "role" and (not self.indices or self.role is not None):
            raise ValueError("sites/orbit scope requires indices and no role")


@dataclass(frozen=True)
class MoleculeInsertion:
    """One composition molecule placed at a caller anchor, with no template site.

    ``fractional_anchor`` is where that molecule's placement anchor lands, in
    fractional coordinates of the template cell (row vector times the lattice).
    Coordinates outside ``[0, 1)`` are stored and applied as supplied, so the
    anchor sits in the lattice image those numbers name. The periodic clash
    check includes that image. The role ``PlacementPolicy`` supplies the
    molecule anchor and orientation. ``mapped`` orientation is refused: an
    insertion has no template frame.
    """

    composition_index: int
    fractional_anchor: tuple[float, float, float]

    def __post_init__(self):
        index = self.composition_index
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or index < 0
        ):
            raise ValueError(
                "insertion composition index must be a nonnegative integer"
            )
        anchor = np.asarray(self.fractional_anchor, float).reshape(-1)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError(
                "insertion anchor must be three finite fractional coordinates"
            )
        object.__setattr__(self, "composition_index", int(index))
        object.__setattr__(self, "fractional_anchor", tuple(float(v) for v in anchor))


@dataclass(frozen=True)
class StoichiometryPolicy:
    """Vacancies and insertions that make a changed reduced inventory explicit.

    ``vacant_sites`` are template molecule indices omitted from the output.
    They are absent molecules, not partial-occupancy sites. ``insertions`` are
    composition molecules that have no template counterpart. Every other
    composition molecule must map onto every remaining template site by a
    complete role-preserving bijection. The policy does not choose sites, expand
    the cell, or accept a multiplicity-only mismatch.
    """

    vacant_sites: tuple[int, ...] = ()
    insertions: tuple[MoleculeInsertion, ...] = ()

    def __post_init__(self):
        sites = []
        for site in self.vacant_sites:
            if (
                isinstance(site, bool)
                or not isinstance(site, (int, np.integer))
                or site < 0
            ):
                raise ValueError("vacant sites must be nonnegative integer indices")
            sites.append(int(site))
        if len(sites) != len(set(sites)):
            raise ValueError("vacant sites must be unique")
        insertions = tuple(self.insertions)
        if any(not isinstance(item, MoleculeInsertion) for item in insertions):
            raise TypeError("insertions must contain MoleculeInsertion objects")
        if len({item.composition_index for item in insertions}) != len(insertions):
            raise ValueError("each composition molecule can be inserted once")
        object.__setattr__(self, "vacant_sites", tuple(sorted(sites)))
        object.__setattr__(self, "insertions", insertions)


@dataclass(frozen=True)
class PackingTransferPlan:
    """Transfer composition molecules to a template's full-cell site lattice.

    Policies are keyed by role. Default pairing is by role then molecule order.
    Explicit site_mapping contains (composition index, template index) pairs
    and must be a complete role-preserving bijection. Roles and their counts
    are arbitrary caller input. Reduced role stoichiometry is compared separately
    from full-cell multiplicity. Changed reduced counts construct only through
    ``stoichiometry_policy``. A multiplicity-only difference still requires
    commensurate cells; no supercell is built implicitly. Different chemistry
    at equal counts is supported.
    """

    composition_inventory: CrystalInventory
    template_inventory: CrystalInventory
    placements: dict[str, PlacementPolicy]
    mappings: dict[str, ChemicalAtomMapping] = field(default_factory=dict)
    site_mapping: tuple[tuple[int, int], ...] = ()
    expected_charge: int | None = None
    expected_formula: dict[str, int] | None = None
    stoichiometry_policy: StoichiometryPolicy | None = None

    def __post_init__(self):
        object.__setattr__(self, "placements", dict(self.placements))
        object.__setattr__(self, "mappings", dict(self.mappings))
        object.__setattr__(
            self, "site_mapping", ChemicalAtomMapping(self.site_mapping).pairs
        )
        if any(not isinstance(p, PlacementPolicy) for p in self.placements.values()):
            raise TypeError("placements must contain PlacementPolicy objects")
        if any(not isinstance(m, ChemicalAtomMapping) for m in self.mappings.values()):
            raise TypeError("mappings must contain ChemicalAtomMapping objects")
        if self.stoichiometry_policy is not None and not isinstance(
            self.stoichiometry_policy, StoichiometryPolicy
        ):
            raise TypeError("stoichiometry_policy must be a StoichiometryPolicy")


@dataclass(frozen=True)
class RelaxationProtocol:
    """Machine-readable external-driver hook; cell mask uses ASE Voigt order.

    CF1 fixes molecular COM and orientation while freeing internal geometry.
    CF2 frees rigid poses at fixed cell, freezing internal geometry. CF3 frees
    all degrees of freedom. Drivers must enforce these collective constraints;
    Cartesian FixAtoms alone cannot express CF1 or CF2.
    """

    stage: str = "CF0"

    def __post_init__(self):
        if self.stage not in {"CF0", "CF1", "CF2", "CF3"}:
            raise ValueError("stage must be CF0, CF1, CF2 or CF3")

    def constraints(self, crystal: MolecularCrystal) -> dict[str, Any]:
        groups = []
        offset = 0
        for m in crystal.molecules:
            groups.append(list(range(offset, offset + len(m))))
            offset += len(m)
        internal, translation, orientation, cell = {
            "CF0": (False, False, False, False),
            "CF1": (True, False, False, False),
            "CF2": (False, True, True, False),
            "CF3": (True, True, True, True),
        }[self.stage]
        return {
            "stage": self.stage,
            "molecule_atom_indices": groups,
            "internal_free": internal,
            "translation_free": translation,
            "orientation_free": orientation,
            "cell_free_mask": [cell] * 6,
            "symmetry_policy": "P1",
            "pose_reference": "mass_weighted_COM_and_proper_Eckart_frame",
            "requires_collective_constraint_driver": self.stage in {"CF1", "CF2"},
        }


@dataclass(frozen=True)
class CounterfactualResult:
    """One all-or-nothing result and one JSON-serializable provenance record."""

    status: CounterfactualStatus
    crystal: MolecularCrystal | None
    inventory: CrystalInventory | None
    provenance: dict[str, Any]


def _clone(molecule):
    result = molecule.copy()
    result.info = copy.deepcopy(molecule.info)
    return result


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        # CIF readers represent unknown numeric metadata as NaN. Preserve it
        # explicitly while keeping the provenance valid, portable JSON.
        return {"nonfinite_float": value.hex()}
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: _json(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, CrystalMolecule):
        return _molecule_payload(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _molecule_payload(m):
    graph = getattr(m, "_graph", None)
    return {
        "symbols": m.get_chemical_symbols(),
        "arrays": _json(m.arrays),
        "chemical_entity": _json(getattr(m, "chemical_entity", None)),
        "forward_bonds": _json(m.info.get("forward_bonds", [])),
        "forward_pose_index": m.info.get("forward_pose_index"),
        "bonds": []
        if graph is None
        else [[int(i), int(j), _json(d)] for i, j, d in sorted(graph.edges(data=True))],
    }


def _hash(crystal, inventory):
    payload = {
        "lattice": _json(crystal.lattice),
        "pbc": _json(crystal.pbc),
        "molecules": [_molecule_payload(m) for m in crystal.molecules],
        "inventory": _json(inventory),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _formula(crystal):
    return dict(
        sorted(
            Counter(
                s for m in crystal.molecules for s in m.get_chemical_symbols()
            ).items()
        )
    )


def _local_legacy_records(m):
    """Normalize the parser's global legacy endpoints for a molecule copy."""
    records = copy.deepcopy(m.info.get("bond_records", []))
    global_to_local = {int(g): i for i, g in enumerate(m.info.get("atom_indices", []))}
    if (
        records
        and global_to_local
        and all(
            int(r[k]) in global_to_local for r in records for k in ("left", "right")
        )
    ):
        for record in records:
            for key in ("left", "right"):
                record[key] = global_to_local[int(record[key])]
    return records


def _validate_molecule(m, charge):
    if not len(m) or not np.isfinite(m.positions).all():
        raise ValueError(
            "molecules must be nonempty and have finite contiguous coordinates"
        )
    if not np.allclose(
        m.arrays.get(KEY_OCCUPANCY, np.ones(len(m))),
        1,
        atol=PARTIAL_OCCUPANCY_TOLERANCE,
        rtol=0,
    ):
        raise ValueError("partial occupancy requires an ordered realization first")
    known = m.arrays.get(KEY_FORMAL_CHARGE_KNOWN)
    if known is not None and np.all(known):
        charges = m.arrays.get(KEY_FORMAL_CHARGE)
        if charges is None or not np.isclose(np.sum(charges), charge):
            raise ValueError(
                "declared molecular charge disagrees with known atom charges"
            )
    entity = getattr(m, "chemical_entity", None)
    if entity is not None and getattr(entity, "net_charge", None) is not None:
        if entity.net_charge != charge:
            raise ValueError("declared molecular charge disagrees with chemical entity")
    if entity is not None and getattr(entity, "dimension", 0) != 0:
        raise ValueError("periodic bonded frameworks are unsupported")
    if entity is not None:
        identities = m.arrays.get(KEY_ATOM_ID)
        if identities is None or set(map(str, identities)) != {
            a.atom_id for a in entity.atoms
        }:
            raise ValueError("chemical entity requires matching stable atom IDs")
        atom_by_id = {a.atom_id: a for a in entity.atoms}
        if any(
            atom_by_id[str(atom_id)].element != symbol
            for atom_id, symbol in zip(identities, m.get_chemical_symbols())
        ):
            raise ValueError("chemical entity elements disagree with molecule arrays")
    records = m.info.get("forward_bonds") or _local_legacy_records(m)
    if records:
        ids = tuple(str(i) for i in range(len(m)))
        bonds = tuple(
            ChemicalBond(
                str(r["left"]),
                str(r["right"]),
                atom2_image_shift=tuple(
                    r.get("image_shift", r.get("right_image_shift", (0, 0, 0)))
                ),
            )
            for r in records
        )
    elif entity is not None:
        ids = tuple(a.atom_id for a in entity.atoms)
        bonds = entity.bonds
    elif getattr(m, "_graph", None) is not None:
        ids = tuple(str(i) for i in range(len(m)))
        bonds = tuple(
            ChemicalBond(
                str(i), str(j), atom2_image_shift=tuple(d.get("image_shift", (0, 0, 0)))
            )
            for i, j, d in m._graph.edges(data=True)
        )
    else:
        return
    # Reuse exact quotient-graph topology: a boundary bond is not itself
    # evidence of a framework. Only a nonzero translation cycle is.
    if analyze_periodic_topology(ids, bonds).rank:
        raise ValueError("periodic bonded frameworks are unsupported")


def _validate(crystal, inventory):
    lattice = np.asarray(crystal.lattice, float)
    if (
        lattice.shape != (3, 3)
        or not np.isfinite(lattice).all()
        or abs(np.linalg.det(lattice)) < MIN_PERIODIC_CELL_VOLUME_A3
    ):
        raise ValueError("a finite nonsingular lattice is required")
    if len(crystal.molecules) != len(inventory.roles):
        raise ValueError("inventory length differs from molecule count")
    for m, charge in zip(crystal.molecules, inventory.charges):
        _validate_molecule(m, charge)


def periodic_clash_diagnostics(
    crystal: MolecularCrystal, scale: float = 1.0
) -> tuple[dict, ...]:
    """Summed-covalent-radius contacts, including self-images and skewed cells.

    Same-molecule, zero-image atom pairs are internal geometry and excluded.
    All other images within the radius cutoff are checked by ASE's neighbor
    list, respecting pbc. Entries identify local atoms and lattice image shift.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("clash scale must be finite and positive")
    symbols, positions, owners, local = [], [], [], []
    for mi, m in enumerate(crystal.molecules):
        symbols.extend(m.get_chemical_symbols())
        positions.extend(m.positions)
        owners.extend([mi] * len(m))
        local.extend(range(len(m)))
    if not symbols:
        return ()
    atoms = Atoms(symbols, positions=positions, cell=crystal.lattice, pbc=crystal.pbc)
    missing = sorted({s for s in symbols if not has_atomic_radius(s)})
    if missing:
        raise ValueError(f"clash radii unavailable for elements: {missing}")
    radii = np.asarray([get_atomic_radius(s) for s in symbols]) * scale
    left, right, shifts, distances = neighbor_list(
        "ijSd", atoms, radii, self_interaction=False
    )
    contacts = []
    for i, j, shift, distance in zip(left, right, shifts, distances):
        # Keep one of each reverse pair, including same-atom image pairs.
        key = (int(i), int(j), *map(int, shift))
        reverse = (int(j), int(i), *map(int, -shift))
        if key > reverse or (owners[i] == owners[j] and not np.any(shift)):
            continue
        contacts.append(
            {
                "left_molecule": owners[i],
                "right_molecule": owners[j],
                "left_atom": local[i],
                "right_atom": local[j],
                "image_shift": shift.tolist(),
                "distance_A": float(distance),
                "limit_A": float(radii[i] + radii[j]),
            }
        )
    return tuple(
        sorted(
            contacts,
            key=lambda d: (
                d["left_molecule"],
                d["right_molecule"],
                d["left_atom"],
                d["right_atom"],
                d["image_shift"],
            ),
        )
    )


def _identity(molecule, index, key):
    values = molecule.arrays.get(key)
    value = None if values is None else values[index]
    if value in (None, "", ".", "?", "unknown", 0):
        value = None
    entity = getattr(molecule, "chemical_entity", None)
    if value is None and entity is not None and key in {"isotope", "stereochemistry"}:
        atom_id = str(molecule.arrays[KEY_ATOM_ID][index])
        value = getattr(next(a for a in entity.atoms if a.atom_id == atom_id), key)
    return value


def _mapping(source, target, mapping):
    for i, j in mapping.pairs:
        if not (0 <= i < len(source) and 0 <= j < len(target)):
            raise ValueError("atom mapping index outside molecule")
        if source[i].symbol != target[j].symbol:
            raise ValueError("retained atom mapping changes element identity")
        for key in ("isotope", "stereochemistry", "cip_code"):
            if _identity(source, i, key) != _identity(target, j, key):
                raise ValueError(f"retained atom mapping changes or loses {key}")


def _reidentify(molecule, new_ids, entity_id):
    """Remap existing chemistry records, without inferring or changing chemistry."""
    entity = molecule.chemical_entity
    if entity is not None:
        rename = dict(zip(map(str, molecule.arrays[KEY_ATOM_ID]), new_ids))
        molecule.chemical_entity = replace(
            entity,
            entity_id=entity_id,
            atoms=tuple(replace(a, atom_id=rename[a.atom_id]) for a in entity.atoms),
            bonds=tuple(
                replace(
                    b,
                    atom1_id=rename[b.atom1_id],
                    atom2_id=rename[b.atom2_id],
                    atom2_image_shift=(0, 0, 0),
                )
                for b in entity.bonds
            ),
            embedding=Embedding(
                tuple(
                    (atom_id, tuple(map(float, position)))
                    for atom_id, position in zip(new_ids, molecule.positions)
                )
            ),
        )
    # ASE updates existing arrays in place; remove a shorter string dtype
    # before assigning long lineage IDs to prevent silent truncation.
    molecule.arrays.pop(KEY_ATOM_ID, None)
    molecule.set_array(KEY_ATOM_ID, np.asarray(new_ids, dtype=str))


def _anchor(m, anchor):
    if isinstance(anchor, (int, np.integer)) and anchor >= len(m):
        raise ValueError("placement anchor outside molecule")
    # Reuse the forward builder's authoritative anchor convention.
    return MolecularTemplate(m).anchor(anchor)


def _rotation(source, target, mapping, policy):
    if len(mapping.pairs) < 3:
        if policy.orientation == "mapped":
            raise ValueError(
                "mapped orientation requires three non-collinear atom pairs"
            )
        return None
    a = np.array([target.positions[j] for i, j in mapping.pairs]) - _anchor(
        target, policy.target_anchor
    )
    b = np.array([source.positions[i] for i, j in mapping.pairs]) - _anchor(
        source, policy.source_anchor
    )
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        if policy.orientation == "mapped":
            raise ValueError("mapped orientation has a degenerate frame")
        return None
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _place(source, target, policy, mapping, rotation, destination=None):
    _mapping(source, target, mapping)
    if destination is None:
        destination = _anchor(source, policy.source_anchor)
    m = _clone(target)
    m.positions = (
        m.positions - _anchor(target, policy.target_anchor)
    ) @ rotation + destination
    _strip_stale_frac_arrays(m)
    # Parent crystallographic membership is invalid after chemistry/pose edits.
    for key in ("sym_op_index", "asym_id", "site_symmetry_order", "image_shift"):
        m.arrays.pop(key, None)
    for key in list(m.info):
        if key.startswith("forward_") and key not in {
            "forward_bonds",
            "forward_connectivity",
        }:
            m.info.pop(key)
    if m.info.get("bond_records"):
        records = _local_legacy_records(m)
        for record in records:
            record["right_image_shift"] = [0, 0, 0]
            record["vector"] = (
                m.positions[int(record["right"])] - m.positions[int(record["left"])]
            ).tolist()
        m.info["bond_records"] = records
    for record in m.info.get("forward_bonds", ()):
        record["image_shift"] = [0, 0, 0]
    m.set_cell(np.zeros((3, 3)))
    m.set_pbc(False)
    m.info.pop("atom_indices", None)
    m.arrays.pop("molecule_index", None)
    _refresh_contiguous_bond_geometry(m)
    return m


def _record(parent, inventory, seed, spec):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    edit_spec = _json(spec)
    edit_hash = hashlib.sha256(
        json.dumps(edit_spec, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {
        "schema": "molcrys.chemical_edit.v1",
        "edit_hash": edit_hash,
        "parent_hash": _hash(parent, inventory),
        "seed": int(seed),
        "edit_spec": edit_spec,
        "composition_change": False,
        "attempts": 0,
        "clashes": [],
        "symmetry_policy": "P1",
        "bond_geometry_policy": "contiguous_zero_image",
        "validation_limits": [
            "charges are caller assertions unless atom charges are known",
            "prebuilt chemistry is preserved, not inferred or certified",
            "finite sampling is not a proof of feasibility or stability",
        ],
    }


def _refuse(record, reason, status=CounterfactualStatus.INCOMPATIBLE):
    record.update(status=status.value, reason=reason)
    return CounterfactualResult(status, None, None, record)


def _construct(
    parent,
    inventory,
    jobs,
    output_inventory,
    record,
    expected_charge,
    expected_formula,
    *,
    omit_indices=(),
    insertions=(),
):
    rng = np.random.default_rng(record["seed"])
    frames = [
        _rotation(source, target, mapping, policy)
        for _, source, target, policy, mapping in jobs
    ]
    # Insertions have no template frame. Identity is the first sample.
    insertion_frames = [np.eye(3) for _ in insertions]
    budget = max(
        [p.samples if p.orientation == "sample" else 1 for _, _, _, p, _ in jobs]
        + [p.samples if p.orientation == "sample" else 1 for _, _, p, _ in insertions],
        default=1,
    )
    omit = {int(i) for i in omit_indices}
    kept = [i for i in range(len(parent.molecules)) if i not in omit]
    output_of = {old: new for new, old in enumerate(kept)}
    best = None
    best_score = None
    best_attempt = None
    sampled_rotations = {}
    insertion_rotations = {}
    for attempt in range(budget):
        molecules = [_clone(m) for m in parent.molecules]
        identity_records = []
        for n, (index, source, target, policy, mapping) in enumerate(jobs):
            rotation = frames[n]
            if policy.orientation == "sample":
                # Each role policy bounds its own orientation draws, even if
                # another role has a larger global construction budget.
                if attempt < policy.samples:
                    if attempt > 0 or rotation is None:
                        rotation = Rotation.random(random_state=rng).as_matrix()
                    sampled_rotations[n] = rotation
                else:
                    rotation = sampled_rotations[n]
            molecule = _place(source, target, policy, mapping, rotation)
            old_ids = source.arrays[KEY_ATOM_ID]
            new_ids = [
                f"edit:{record['parent_hash']}:{record['edit_hash']}:{index}:{j}"
                for j in range(len(target))
            ]
            for i, j in mapping.pairs:
                new_ids[j] = str(old_ids[i])
            _reidentify(
                molecule,
                new_ids,
                f"edit:{record['parent_hash']}:{record['edit_hash']}:{index}",
            )
            if index not in output_of:
                raise ValueError("cannot edit a vacant template site")
            molecules[index] = molecule
            retained = [
                {"source_index": i, "target_index": j, "atom_id": str(old_ids[i])}
                for i, j in mapping.pairs
            ]
            identity_records.append(
                {
                    "site": index,
                    "retained": retained,
                    "deleted": [
                        str(old_ids[i])
                        for i in range(len(source))
                        if i not in {p[0] for p in mapping.pairs}
                    ],
                    "created": [
                        new_ids[j]
                        for j in range(len(target))
                        if j not in {p[1] for p in mapping.pairs}
                    ],
                    "rotation_row_matrix": rotation.tolist(),
                    "output_index": output_of[index],
                }
            )
        for site in sorted(omit):
            source = parent.molecules[site]
            identity_records.append(
                {
                    "site": int(site),
                    "vacancy": True,
                    "retained": [],
                    "deleted": [str(atom_id) for atom_id in source.arrays[KEY_ATOM_ID]],
                    "created": [],
                    "output_index": None,
                }
            )
        inserted_molecules = []
        for n, (lineage, target, policy, anchor) in enumerate(insertions):
            rotation = insertion_frames[n]
            if policy.orientation == "sample":
                if attempt < policy.samples:
                    if attempt > 0 or rotation is None:
                        rotation = Rotation.random(random_state=rng).as_matrix()
                    insertion_rotations[n] = rotation
                else:
                    rotation = insertion_rotations[n]
            molecule = _place(
                target,
                target,
                policy,
                ChemicalAtomMapping(),
                rotation,
                destination=anchor,
            )
            new_ids = [
                f"edit:{record['parent_hash']}:{record['edit_hash']}:insert:{lineage}:{j}"
                for j in range(len(target))
            ]
            _reidentify(
                molecule,
                new_ids,
                f"edit:{record['parent_hash']}:{record['edit_hash']}:insert:{lineage}",
            )
            inserted_molecules.append(molecule)
            identity_records.append(
                {
                    "site": None,
                    "insertion": True,
                    "composition_index": lineage,
                    "retained": [],
                    "deleted": [],
                    "created": new_ids,
                    "rotation_row_matrix": rotation.tolist(),
                    "output_index": len(kept) + n,
                }
            )
        candidate = MolecularCrystal(
            parent.lattice.copy(),
            [molecules[i] for i in kept] + inserted_molecules,
            pbc=tuple(parent.pbc),
        )
        _validate(candidate, output_inventory)
        formula, charge = _formula(candidate), sum(output_inventory.charges)
        if charge != expected_charge:
            raise ValueError(
                "charge inventory differs from expected charge; declare a charge-changing edit explicitly"
            )
        if expected_formula is not None and formula != expected_formula:
            raise ValueError("formula inventory differs from expected formula")
        # A uniform conservative scale handles mixed-role transfer policies.
        scale = max(
            [p.clash_scale for _, _, _, p, _ in jobs]
            + [p.clash_scale for _, _, p, _ in insertions],
            default=1.0,
        )
        contacts = periodic_clash_diagnostics(candidate, scale)
        record["attempts"] = attempt + 1
        score = (len(contacts), sum(c["limit_A"] - c["distance_A"] for c in contacts))
        if best_score is None or score < best_score:
            best_score = score
            best = contacts
            best_attempt = attempt + 1
        if not contacts:
            record.update(
                status=CounterfactualStatus.CONSTRUCTED.value,
                reason=None,
                clashes=[],
                atom_changes=identity_records,
                formula=formula,
                charge=int(charge),
                role_counts=dict(Counter(output_inventory.roles)),
                result_hash=_hash(candidate, output_inventory),
                relaxation=RelaxationProtocol().constraints(candidate),
            )
            candidate.metadata["chemical_edit"] = copy.deepcopy(record)
            return CounterfactualResult(
                CounterfactualStatus.CONSTRUCTED, candidate, output_inventory, record
            )
    record["clashes"] = list(best or ())
    record["best_attempt"] = best_attempt
    return _refuse(
        record,
        "orientation sampling exhausted with periodic clashes",
        CounterfactualStatus.CLASH,
    )


def apply_chemical_edit(
    parent: MolecularCrystal,
    inventory: CrystalInventory,
    spec: CrystalEditSpec,
    *,
    seed: int,
) -> CounterfactualResult:
    """Apply one role/site/orbit edit atomically; parent and spec remain unchanged."""
    record = _record(parent, inventory, seed, spec)
    try:
        _validate(parent, inventory)
        if spec.scope == "role":
            indices = [i for i, r in enumerate(inventory.roles) if r == spec.role]
        else:
            indices = sorted(set(spec.indices))
            if any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < len(parent.molecules)
                for i in indices
            ):
                raise ValueError("site index outside parent")
            if spec.scope == "orbit":
                ids = inventory.orbit_ids or tuple(
                    m.info.get("forward_pose_index") for m in parent.molecules
                )
                if any(v is None or v == "" for v in ids):
                    raise ValueError(
                        "orbit-complete selection requires explicit orbit IDs or forward pose evidence"
                    )
                selected = {ids[i] for i in indices}
                indices = [i for i, oid in enumerate(ids) if oid in selected]
        if not indices:
            raise ValueError("edit selection is empty")
        edit = spec.edit
        _validate_molecule(edit.replacement, edit.target_charge)
        charges = list(inventory.charges)
        jobs = []
        for i in indices:
            source = parent.molecules[i]
            if (
                source.get_chemical_formula() != edit.source_formula
                or charges[i] != edit.source_charge
            ):
                raise ValueError(f"source formula/charge guard failed at site {i}")
            charges[i] = edit.target_charge
            jobs.append((i, source, edit.replacement, edit.placement, edit.mapping))
        record["selected_sites"] = indices
        record["composition_change"] = any(
            Counter(parent.molecules[i].get_chemical_symbols())
            != Counter(edit.replacement.get_chemical_symbols())
            or edit.source_charge != edit.target_charge
            for i in indices
        )
        out = CrystalInventory(inventory.roles, tuple(charges), inventory.orbit_ids)
        expected = (
            sum(inventory.charges)
            if spec.expected_charge is None
            else spec.expected_charge
        )
        return _construct(
            parent, inventory, jobs, out, record, expected, spec.expected_formula
        )
    except ValueError as exc:
        return _refuse(record, str(exc))


def _compare_transfer_inventories(composition, template, source_inv, template_inv):
    """Compare arbitrary compositions independently of full-cell multiplicity.

    The GCD of role counts defines the common reduction factor for each cell.
    Cross multiplication compares element and charge inventories exactly, even
    when a reduced formula would have fractional element counts.
    """
    source_counts = Counter(source_inv.roles)
    template_counts = Counter(template_inv.roles)
    source_units = gcd(*source_counts.values())
    template_units = gcd(*template_counts.values())
    source_reduced = {
        role: count // source_units for role, count in sorted(source_counts.items())
    }
    template_reduced = {
        role: count // template_units for role, count in sorted(template_counts.items())
    }
    source_formula = _formula(composition)
    template_formula = _formula(template)
    chemistry_changed = {
        element: count * template_units for element, count in source_formula.items()
    } != {
        element: count * source_units for element, count in template_formula.items()
    } or sum(source_inv.charges) * template_units != sum(
        template_inv.charges
    ) * source_units
    stoichiometry_changed = source_reduced != template_reduced
    return {
        "role_counts": {
            "composition": dict(source_counts),
            "template": dict(template_counts),
        },
        "reduced_role_counts": {
            "composition": source_reduced,
            "template": template_reduced,
        },
        "cell_multiplicities": {
            "composition": source_units,
            "template": template_units,
        },
        "site_count_change": source_counts != template_counts,
        "stoichiometry_change": stoichiometry_changed,
        "composition_change": stoichiometry_changed or chemistry_changed,
    }


def _resolve_stoichiometry_policy(source_inv, template_inv, plan):
    """Return the occupied (composition, template) bijection for an explicit policy."""
    policy = plan.stoichiometry_policy
    n_source = len(source_inv.roles)
    n_template = len(template_inv.roles)
    vacant = set(policy.vacant_sites)
    inserted = {item.composition_index for item in policy.insertions}
    if any(j >= n_template for j in vacant):
        raise ValueError("vacant site index outside template")
    if any(i >= n_source for i in inserted):
        raise ValueError("insertion composition index outside composition")
    remaining_template = [j for j in range(n_template) if j not in vacant]
    remaining_source = [i for i in range(n_source) if i not in inserted]
    pairs = list(plan.site_mapping)
    if pairs:
        if (
            sorted(i for i, _ in pairs) != remaining_source
            or sorted(j for _, j in pairs) != remaining_template
        ):
            raise ValueError(
                "site mapping must bijection the non-inserted composition molecules "
                "and the non-vacant template sites"
            )
    else:
        pairs = []
        roles = sorted(
            {source_inv.roles[i] for i in remaining_source}
            | {template_inv.roles[j] for j in remaining_template}
        )
        for role in roles:
            left = [i for i in remaining_source if source_inv.roles[i] == role]
            right = [j for j in remaining_template if template_inv.roles[j] == role]
            if len(left) != len(right):
                raise ValueError(
                    f"stoichiometry policy leaves {len(left)} {role!r} composition "
                    f"molecules and {len(right)} template sites"
                )
            pairs.extend(zip(left, right))
    for i, j in pairs:
        if source_inv.roles[i] != template_inv.roles[j]:
            raise ValueError("site mapping must preserve roles")
    for index in remaining_source:
        role = source_inv.roles[index]
        if role not in plan.placements:
            raise ValueError(f"explicit placement policy missing for role {role}")
    for item in policy.insertions:
        role = source_inv.roles[item.composition_index]
        if role not in plan.placements:
            raise ValueError(f"explicit placement policy missing for role {role}")
        if plan.placements[role].orientation == "mapped":
            raise ValueError(
                f"insertion of role {role} has no template frame for mapped orientation"
            )
    return pairs


def _transfer_jobs(composition, template, source_inv, template_inv, plan, pairs):
    jobs = []
    for i, j in sorted(pairs, key=lambda p: p[1]):
        role = source_inv.roles[i]
        if role != template_inv.roles[j]:
            raise ValueError("site mapping must preserve roles")
        if role not in plan.placements:
            raise ValueError(f"explicit placement policy missing for role {role}")
        jobs.append(
            (
                j,
                template.molecules[j],
                composition.molecules[i],
                plan.placements[role],
                plan.mappings.get(role, ChemicalAtomMapping()),
            )
        )
    return jobs


def _anchor_outside_unit_cell(anchor):
    """True when a fractional anchor names a lattice image other than [0, 1)."""
    return any(coordinate < 0.0 or coordinate >= 1.0 for coordinate in anchor)


def _broken_orbit_ids(orbit_ids, vacant):
    """Declared orbits that lost some, but not all, of their template sites.

    A fully removed orbit is absent from the output and leaves complete orbits
    untouched. An empty ``orbit_ids`` tuple means the template declared none.
    """
    if not orbit_ids:
        return ()
    members = {}
    for index, orbit_id in enumerate(orbit_ids):
        members.setdefault(orbit_id, set()).add(index)
    removed = set(vacant)
    return tuple(
        orbit_id
        for orbit_id, sites in members.items()
        if sites & removed and not sites <= removed
    )


def transfer_packing(
    composition: MolecularCrystal,
    template: MolecularCrystal,
    plan: PackingTransferPlan,
    *,
    seed: int,
) -> CounterfactualResult:
    """Place composition ions on the template lattice.

    Changed reduced stoichiometry constructs only when
    ``plan.stoichiometry_policy`` names the vacant template sites and the
    inserted composition molecules. Without that policy the result is a
    refusal and contains no geometry. A multiplicity-only mismatch is refused
    with ``CELL_MULTIPLICITY_MAPPING_REQUIRED`` before a supplied policy is
    considered. Each vacancy records its deleted template atom IDs in
    ``atom_changes``. Output orbit IDs are kept for complete retained orbits
    and cleared when a vacancy leaves an orbit only partly occupied.
    """
    source_inv, template_inv = plan.composition_inventory, plan.template_inventory
    record = _record(template, template_inv, seed, plan)
    record["composition_hash"] = _hash(composition, source_inv)
    try:
        _validate(composition, source_inv)
        _validate(template, template_inv)
        comparison = _compare_transfer_inventories(
            composition, template, source_inv, template_inv
        )
        record["inventory_comparison"] = comparison
        record["composition_change"] = comparison["composition_change"]
        omit = ()
        insertion_specs = ()
        if comparison["stoichiometry_change"]:
            if plan.stoichiometry_policy is None:
                record["refusal_code"] = "STOICHIOMETRY_POLICY_REQUIRED"
                raise ValueError(
                    "role stoichiometry changes require an explicit vacancy/insertion policy; "
                    "name vacant template sites and inserted composition molecules"
                )
            record["refusal_code"] = "STOICHIOMETRY_POLICY_INVALID"
            pairs = _resolve_stoichiometry_policy(source_inv, template_inv, plan)
            record.pop("refusal_code")
            policy = plan.stoichiometry_policy
            omit = policy.vacant_sites
            kept = [j for j in range(len(template.molecules)) if j not in set(omit)]
            composition_at = {j: i for i, j in pairs}
            roles = [source_inv.roles[composition_at[j]] for j in kept]
            charges = [source_inv.charges[composition_at[j]] for j in kept]
            broken_orbits = _broken_orbit_ids(template_inv.orbit_ids, omit)
            # One ID per molecule or none. A partial orbit cannot keep its
            # label while intact orbits in the same output keep theirs.
            if template_inv.orbit_ids and not policy.insertions and not broken_orbits:
                orbits = tuple(template_inv.orbit_ids[index] for index in kept)
            else:
                orbits = ()
            lattice = np.asarray(template.lattice, float)
            insertion_specs = tuple(
                (
                    item.composition_index,
                    composition.molecules[item.composition_index],
                    plan.placements[source_inv.roles[item.composition_index]],
                    np.asarray(item.fractional_anchor, float) @ lattice,
                )
                for item in policy.insertions
            )
            for item in policy.insertions:
                roles.append(source_inv.roles[item.composition_index])
                charges.append(source_inv.charges[item.composition_index])
            record["vacant_sites"] = list(policy.vacant_sites)
            record["insertions"] = [
                {
                    "composition_index": item.composition_index,
                    "fractional_anchor": list(item.fractional_anchor),
                    "output_index": len(kept) + n,
                }
                for n, item in enumerate(policy.insertions)
            ]
            limits = [
                "vacancies omit template molecules; they are not partial-occupancy sites",
                "insertion anchors are caller coordinates, not optimized positions",
            ]
            if any(
                _anchor_outside_unit_cell(item.fractional_anchor)
                for item in policy.insertions
            ):
                limits.append(
                    "an insertion anchor outside [0, 1) is stored as supplied "
                    "and places the molecule in that lattice image"
                )
            if broken_orbits:
                limits.append(
                    "a vacancy that empties only part of an orbit clears output orbit IDs"
                )
            record["validation_limits"] = list(record["validation_limits"]) + limits
            out = CrystalInventory(tuple(roles), tuple(charges), orbits)
        elif comparison["site_count_change"]:
            # Equal reduced stoichiometry with a different full-cell count is not
            # a vacancy or insertion, even when the caller supplied a policy.
            record["refusal_code"] = "CELL_MULTIPLICITY_MAPPING_REQUIRED"
            raise ValueError(
                "equal reduced role stoichiometry has different full-cell multiplicities; "
                "provide commensurate full cells for a complete site bijection; "
                "implicit expansion or removal is unsupported"
            )
        elif plan.stoichiometry_policy is not None:
            record["refusal_code"] = "STOICHIOMETRY_POLICY_INVALID"
            raise ValueError(
                "an explicit stoichiometry policy applies only when reduced role counts differ"
            )
        else:
            pairs = list(plan.site_mapping)
            if not pairs:
                pairs = [
                    (i, j)
                    for role in sorted(set(source_inv.roles))
                    for i, j in zip(
                        [n for n, r in enumerate(source_inv.roles) if r == role],
                        [n for n, r in enumerate(template_inv.roles) if r == role],
                    )
                ]
            if sorted(i for i, j in pairs) != list(
                range(len(composition.molecules))
            ) or sorted(j for i, j in pairs) != list(range(len(template.molecules))):
                raise ValueError("site mapping must be a complete bijection")
            charges = list(template_inv.charges)
            for i, j in pairs:
                charges[j] = source_inv.charges[i]
            out = CrystalInventory(
                template_inv.roles, tuple(charges), template_inv.orbit_ids
            )
        jobs = _transfer_jobs(
            composition, template, source_inv, template_inv, plan, pairs
        )
        record["site_mapping"] = _json(pairs)
        expected = (
            sum(source_inv.charges)
            if plan.expected_charge is None
            else plan.expected_charge
        )
        formula = (
            _formula(composition)
            if plan.expected_formula is None
            else plan.expected_formula
        )
        return _construct(
            template,
            template_inv,
            jobs,
            out,
            record,
            expected,
            formula,
            omit_indices=omit,
            insertions=insertion_specs,
        )
    except ValueError as exc:
        return _refuse(record, str(exc))


def write_cf0(
    result: CounterfactualResult, directory: str | Path, *, extxyz: bool = True
) -> dict[str, Path]:
    """Write provenance for any outcome, plus P1 CIF/ExtXYZ for constructed CF0.

    The JSON sidecar is authoritative for edit mapping, roles and declared
    molecular charges; a bare CIF cannot encode all of these semantics.
    """
    from ..io.output import write_cif
    from ..io.extxyz import write_extxyz

    directory = Path(directory)
    if any(
        (directory / name).exists()
        for name in ("provenance.json", "cf0.cif", "cf0.extxyz")
    ):
        raise FileExistsError("CF0 output directory already contains edit artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"provenance": directory / "provenance.json"}
    if result.crystal is not None:
        if _hash(result.crystal, result.inventory) != result.provenance["result_hash"]:
            raise ValueError("result was mutated after construction")
        paths["cif"] = directory / "cf0.cif"
        write_cif(result.crystal, str(paths["cif"]), representation="p1")
        if extxyz:
            paths["extxyz"] = directory / "cf0.extxyz"
            write_extxyz(result.crystal, str(paths["extxyz"]))
    paths["provenance"].write_text(
        json.dumps(_json(result.provenance), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    return paths
