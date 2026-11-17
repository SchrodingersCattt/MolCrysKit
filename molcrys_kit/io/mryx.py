"""Read-only import of stored 3-D sites from Mercury CCDC_XML documents.

The parser owns container/XML decoding; consumers receive ordinary ASE Atoms
or MolecularCrystal objects. Source sites are NOT automatically expanded by
symmetry. Internal element codes, bond-type enumerators, Site.uncertain_value
and disorder handles have no assumed chemistry meanings: raw fields and explicit
diagnostics live in ``atoms.info['mryx']``. In particular, unspecified occupancy
is not converted into an allegedly measured occupancy. Geometry-based molecule
identification in ``read_mryx`` is separate from the stored bond table.

This is a deliberately conservative structural reader, not a Mercury session
loader or round-trip writer. It never evaluates expressions or resolves external
resources. Unsupported variants fail rather than fabricate coordinates.
"""

from __future__ import annotations

import gzip
import io
import operator
import re
import warnings
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import atomic_numbers
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from ..constants.mryx import (
    MRYX_CELL_TOLERANCE,
    MRYX_COORDINATE_TOLERANCE_A,
    MRYX_MAX_COMPRESSED_BYTES,
    MRYX_MAX_XML_BYTES,
    MRYX_MAX_XML_DEPTH,
    MRYX_MAX_XML_NODES,
)
from ..structures.crystal import MolecularCrystal

__all__ = ["read_mryx_atoms", "read_mryx"]


def _text(node, path, *, required=True):
    value = node.findtext(path)
    if value is not None and value.strip():
        return value.strip()
    if required:
        raise ValueError(f"MRYX missing required field: {path}")
    return None


def _number(node, path):
    value = float(_text(node, path))
    if not np.isfinite(value):
        raise ValueError(f"MRYX non-finite numeric field: {path}")
    return value


def _vector(node, size):
    if node is None:
        raise ValueError("MRYX missing coordinate vector or matrix")
    container = (
        node
        if node.tag == "vector" and node.get("type") == "Container"
        else node.find(".//vector[@type='Container']")
    )
    if container is None:
        raise ValueError("MRYX missing numeric vector container")
    values = np.array([float(_text(item, ".")) for item in container.findall("item")])
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"MRYX expected a finite vector of length {size}")
    return values


def _raw(node):
    """JSON-compatible source fields without interpreting their semantics."""
    if node is None:
        return None
    return {
        "tag": node.tag,
        "attributes": dict(node.attrib),
        "text": (node.text or "").strip(),
        "children": [_raw(child) for child in node],
    }


def _read_root(filepath):
    with Path(filepath).open("rb") as handle:
        payload = handle.read(MRYX_MAX_COMPRESSED_BYTES + 1)
    if len(payload) > MRYX_MAX_COMPRESSED_BYTES:
        raise ValueError("MRYX input exceeds the compressed/input byte limit")
    if payload.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
                payload = handle.read(MRYX_MAX_XML_BYTES + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError("MRYX invalid or truncated gzip container") from exc
    if len(payload) > MRYX_MAX_XML_BYTES:
        raise ValueError("MRYX expanded XML exceeds the byte limit")
    # Parse incrementally to cap tree size/depth as well as byte size. A plain
    # <!DOCTYPE CCDC_XML> is permitted; entities and external references are not.
    depth = count = 0
    try:
        context = SafeET.iterparse(
            io.BytesIO(payload),
            events=("start", "end"),
            forbid_entities=True,
            forbid_external=True,
        )
        for event, _ in context:
            if event == "start":
                depth += 1
                count += 1
                if depth > MRYX_MAX_XML_DEPTH or count > MRYX_MAX_XML_NODES:
                    raise ValueError("MRYX XML exceeds the depth/node limit")
            else:
                depth -= 1
        root = context.root
    except (ET.ParseError, DefusedXmlException) as exc:
        raise ValueError("MRYX malformed or unsafe XML") from exc
    if root.tag != "All":
        raise ValueError("MRYX expected an All CCDC_XML root")
    return root


def _cell(structure):
    cell = structure.find("cell/value")
    if cell is None:
        raise ValueError("MRYX structure has no crystal cell")
    lengths = np.array([_number(cell, f"{key}/value") for key in ("a", "b", "c")])
    angles = np.array(
        [_number(cell, f"{key}/value") for key in ("alpha", "beta", "gamma")]
    )
    if np.any(lengths <= 0) or np.any(angles <= 0) or np.any(angles >= np.pi):
        raise ValueError("MRYX invalid cell lengths or radian angles")
    # Positive-definite Gram matrix rejects geometrically impossible cells.
    alpha, beta, gamma = angles
    metric = np.array(
        [
            [1, np.cos(gamma), np.cos(beta)],
            [np.cos(gamma), 1, np.cos(alpha)],
            [np.cos(beta), np.cos(alpha), 1],
        ]
    ) * np.outer(lengths, lengths)
    if not np.isfinite(metric).all() or np.linalg.eigvalsh(metric).min() <= 0:
        raise ValueError("MRYX degenerate or impossible cell")
    matrix_node = cell.find("frac_to_orth")
    if matrix_node is None:
        matrix_node = structure.find("crystal_symmetry_match/value/frac_to_orth")
    if matrix_node is None:
        # A conventional cell is sufficient for fractional coordinates alone.
        matrix = np.asarray(Cell.fromcellpar([*lengths, *np.degrees(angles)]))
    else:
        # CCDC stores column-vector transforms; ASE stores lattice vectors as rows.
        matrix = _vector(matrix_node, 9).reshape(3, 3).T
        if not np.allclose(
            matrix @ matrix.T,
            metric,
            rtol=MRYX_CELL_TOLERANCE,
            atol=MRYX_CELL_TOLERANCE,
        ):
            raise ValueError("MRYX coordinate matrix disagrees with cell parameters")
    if abs(np.linalg.det(matrix)) == 0:
        raise ValueError("MRYX singular coordinate matrix")
    inverse_node = cell.find("orth_to_frac")
    if inverse_node is None:
        inverse_node = structure.find("crystal_symmetry_match/value/orth_to_frac")
    if inverse_node is not None:
        inverse = _vector(inverse_node, 9).reshape(3, 3)
        if not np.allclose(
            inverse @ matrix.T,
            np.eye(3),
            rtol=MRYX_CELL_TOLERANCE,
            atol=MRYX_CELL_TOLERANCE,
        ):
            raise ValueError("MRYX inverse coordinate matrices disagree")
    return cell, matrix, matrix_node is not None


def _structure(item, index):
    structures = item.findall(".//value[@type='ConcreteCrystalStructure']")
    if len(structures) != 1:
        raise ValueError("MRYX expected one ConcreteCrystalStructure per graphics item")
    structure = structures[0]
    molecule = structure.find("molecule/value[@type='Molecule3d']")
    if molecule is None:
        raise ValueError("MRYX structure has no supported Molecule3d")
    cell, matrix, has_transform = _cell(structure)
    symbols, labels, positions, raw_atoms = [], [], [], []
    serials, codes = set(), {}
    for handle in molecule.findall("atoms/item"):
        atom = handle.find("value[@type='Atom3d']")
        if atom is None:
            raise ValueError("MRYX unresolved or unsupported atom handle")
        label = _text(atom, "label")
        # Only conventional crystallographic element-prefixed labels are
        # supported without a published internal element-code mapping.
        match = re.fullmatch(r"([A-Z][a-z]?)(?:[0-9_].*)?", label)
        symbol = match.group(1) if match else None
        if symbol not in atomic_numbers or symbol == "X":
            raise ValueError(
                "MRYX cannot infer an element from a nonstandard atom label"
            )
        code = _text(atom, "element/code", required=False)
        if code is not None and codes.setdefault(code, symbol) != symbol:
            raise ValueError(
                "MRYX conflicting elements for the same internal element code"
            )
        serial = _text(handle, "serial_number", required=False)
        if serial is not None:
            if serial in serials:
                raise ValueError("MRYX duplicate atom handle serial number")
            serials.add(serial)
        site = atom.find("site/value")
        if site is None:
            raise ValueError("MRYX atom has no 3-D site")
        frac_node, orth_node = site.find("fractional"), site.find("orthogonal")
        fractional = None if frac_node is None else _vector(frac_node, 3)
        orthogonal = None if orth_node is None else _vector(orth_node, 3)
        if orthogonal is None and fractional is None:
            raise ValueError("MRYX atom has no 3-D coordinates")
        if orthogonal is not None and not has_transform:
            # Orthogonal axes need not equal the conventional lattice frame.
            # Never combine an arbitrary Cartesian orientation with a guessed cell.
            raise ValueError(
                "MRYX orthogonal coordinates require a frac_to_orth matrix"
            )
        derived = None if fractional is None else fractional @ matrix
        if (
            orthogonal is not None
            and derived is not None
            and not np.allclose(
                orthogonal, derived, rtol=0, atol=MRYX_COORDINATE_TOLERANCE_A
            )
        ):
            raise ValueError("MRYX fractional and orthogonal coordinates disagree")
        position = derived if orthogonal is None else orthogonal
        if not np.isfinite(position).all():
            raise ValueError("MRYX coordinate conversion produced non-finite positions")
        symbols.append(symbol)
        labels.append(label)
        positions.append(position)
        raw_atoms.append(_raw(handle))
    if not symbols:
        raise ValueError("MRYX structure has no stored atom sites")
    for bond in molecule.findall("bonds/item"):
        left, right = int(_text(bond, "node1")), int(_text(bond, "node2"))
        if not (0 <= left < len(symbols) and 0 <= right < len(symbols)):
            raise ValueError("MRYX stored bond endpoint is out of range")
    for ref in structure.findall(
        "disorder_assemblies/item/value/groups/item/value/atoms/item"
    ):
        if _text(ref, "serial_number") not in serials:
            raise ValueError("MRYX disorder group references an unknown atom handle")
    diagnostics = [
        "Elements inferred from conventional atom labels; internal element codes are not atomic numbers.",
        "Stored sites only: symmetry is preserved as metadata, not expanded; this may be an asymmetric unit, not a full cell.",
        "Occupancy is unspecified: unit weights used by downstream models are placeholders, not measured occupancies.",
        "Bond types, disorder, ADPs and uncertain_value fields are raw provenance, not interpreted chemistry.",
    ]
    atoms = Atoms(symbols=symbols, positions=positions, cell=matrix, pbc=True)
    atoms.set_array("label", np.array(labels, dtype=str))
    atoms.info["mryx"] = {
        "structure_index": index,
        "label": item.findtext(
            "value/graphics_crystal_memento/graphics_object/label", ""
        ),
        "site_scope": "stored_sites",
        "occupancy_semantics": "unspecified",
        "diagnostics": diagnostics,
        "cell": _raw(cell),
        "crystal_symmetry_match": _raw(structure.find("crystal_symmetry_match")),
        "atoms": raw_atoms,
        "bonds": _raw(molecule.find("bonds")),
        "disorder_assemblies": _raw(structure.find("disorder_assemblies")),
    }
    return atoms


def read_mryx_atoms(filepath: str | Path, index: int | str = 0) -> Atoms | list[Atoms]:
    """Read stored Mercury sites as ASE Atoms, retaining raw structural metadata.

    ``index=0`` selects the first structure in document order; integers (including
    negative indices) select one structure, and ``index=':'`` returns all. A
    UserWarning and ``info['mryx']['diagnostics']`` describe conservative import
    semantics. No symmetry expansion, verified occupancy, source bond-order
    decoding, session-state restoration or MRYX writing is provided. Gzip and
    plain CCDC_XML input are bounded; malformed/unsafe documents raise ValueError.
    """
    if index != ":":
        if isinstance(index, bool):
            raise ValueError("MRYX index must be an integer or ':'")
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise ValueError("MRYX index must be an integer or ':'") from exc
    root = _read_root(filepath)
    items = root.findall("graphics_crystal/item")
    if not items:
        raise ValueError("MRYX contains no graphics crystal structures")
    # Validate all structures even when selecting one, so malformed siblings
    # cannot become silently different results in count/read/all entry points.
    frames = [_structure(item, i) for i, item in enumerate(items)]
    if index == ":":
        result = frames
    else:
        if not -len(frames) <= index < len(frames):
            raise ValueError("MRYX structure index is out of range")
        result = frames[index]
    warnings.warn(
        "MRYX conservative import: " + " ".join(frames[0].info["mryx"]["diagnostics"]),
        UserWarning,
        stacklevel=2,
    )
    return result


def read_mryx(
    filepath: str | Path,
    index: int | str = 0,
    *,
    bond_thresholds=None,
    max_atoms=None,
    bond_scale: float = 1.0,
) -> MolecularCrystal | list[MolecularCrystal]:
    """Read Mercury stored sites into the standard MolecularCrystal model.

    Selection/limitations match :func:`read_mryx_atoms`. Molecules and bonds are
    identified by the existing geometry pipeline, NOT by decoding Mercury bond
    enumerators. Raw input metadata is retained in ``crystal.metadata['mryx']``.
    Existing writers can export this stored-site geometry to CIF/POSCAR/ExtXYZ;
    such output is not a reconstructed full symmetry cell or Mercury round trip.
    """

    def convert(atoms):
        crystal = MolecularCrystal.from_ase(
            atoms,
            bond_thresholds=bond_thresholds,
            max_atoms=max_atoms,
            bond_scale=bond_scale,
        )
        crystal.metadata.update(atoms.info)
        return crystal

    frames = read_mryx_atoms(filepath, index=index)
    return (
        [convert(atoms) for atoms in frames]
        if isinstance(frames, list)
        else convert(frames)
    )
