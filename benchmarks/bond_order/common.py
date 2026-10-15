"""Shared, dependency-light helpers for the bond-order benchmark.

The benchmark deliberately treats candidate/comparator engines as optional.  An
unavailable engine is represented in the result file and is never replaced by
an invented prediction.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    p = Path(path) if path else MANIFEST_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def verify_manifest(manifest: dict[str, Any]) -> tuple[bool, str]:
    recorded = manifest.get("manifest_sha256")
    if not recorded:
        return False, "manifest_sha256 is missing"
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if actual != recorded:
        return (
            False,
            f"manifest checksum mismatch: expected {recorded}, calculated {actual}",
        )
    ids = [c.get("case_id") for c in manifest.get("cases", [])]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        return False, "case IDs are missing or duplicated"
    for case in manifest.get("cases", []):
        case_hash = case.get("fixture_sha256")
        unsigned_case = {k: v for k, v in case.items() if k != "fixture_sha256"}
        calculated_case = hashlib.sha256(canonical_json(unsigned_case)).hexdigest()
        if not case_hash or case_hash != calculated_case:
            return False, f"fixture checksum mismatch for {case.get('case_id')}"
    return True, actual


def fixture_entity(case: dict[str, Any]):
    """Build a chemistry-domain entity without ASE/NetworkX.

    Import is delayed so ``prepare`` and report tooling still work in a minimal
    environment where optional chemistry dependencies are not installed.
    """
    from molcrys_kit.chemistry.models import (
        BondKind,
        ChemicalAtom,
        ChemicalBond,
        Embedding,
        FiniteChemicalEntity,
    )

    atoms = []
    coordinates = []
    for atom in case["atoms"]:
        atoms.append(
            ChemicalAtom(
                atom_id=atom["id"],
                element=atom["element"],
                formal_charge=atom.get("formal_charge"),
                explicit_hydrogens=atom.get("explicit_hydrogens"),
            )
        )
        if "coordinates_A" in atom:
            coordinates.append(
                (atom["id"], tuple(float(x) for x in atom["coordinates_A"]))
            )
    embedding = Embedding(tuple(coordinates)) if coordinates else None
    bonds = []
    for edge in case["bonds"]:
        try:
            kind = BondKind(edge.get("input_kind", "unknown"))
        except ValueError:
            kind = BondKind.UNKNOWN
        bonds.append(
            ChemicalBond(
                atom1_id=edge["a"],
                atom2_id=edge["b"],
                # Gold ``order`` is truth, never an input assertion.  A fixture
                # may opt into a source-explicit order with ``input_order``.
                order=edge.get("input_order"),
                kind=kind,
                aromatic=bool(edge.get("input_aromatic", False)),
                atom2_image_shift=tuple(edge.get("image_shift", (0, 0, 0))),
            )
        )
    return FiniteChemicalEntity(
        entity_id=case["case_id"],
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        embedding=embedding,
    )


def _edge_key(
    a: str,
    b: str,
    image_shift: tuple[int, int, int] | list[int] = (0, 0, 0),
    occurrence: int = 1,
) -> str:
    """Return a canonical, periodic-safe edge identity.

    Zero-image edges retain the historical ``a--b`` spelling.  A non-zero
    image is part of the identity and reverse traversal negates that image;
    repeated records receive a deterministic ``#N`` suffix.
    """
    shift = tuple(int(value) for value in image_shift)
    left, right = str(a), str(b)
    if shift != (0, 0, 0) and (right, left, tuple(-value for value in shift)) < (
        left,
        right,
        shift,
    ):
        left, right, shift = right, left, tuple(-value for value in shift)
    key = f"{left}--{right}"
    if shift != (0, 0, 0):
        key += "@[" + ",".join(str(value) for value in shift) + "]"
    if occurrence > 1:
        key += f"#{occurrence}"
    return key


def _bond_records(candidate: Any) -> list[Any] | None:
    """Return ``candidate`` when it is a sequence of bond records.

    ``BondOrderResult.bond_evidence`` is a provenance tuple, not an edge list.
    Treating it as bonds drops every predicted order while the run stays ``ok``.
    """
    if not isinstance(candidate, (list, tuple)) or not candidate:
        return None
    first = candidate[0]
    if isinstance(first, dict) and ("atom1_id" in first or "a" in first):
        return list(candidate)
    if hasattr(first, "atom1_id") and hasattr(first, "atom2_id"):
        return list(candidate)
    return None


def _extract_bonds(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("bonds", "predicted_bonds", "bond_evidence"):
            found = _bond_records(value.get(key))
            if found is not None:
                return found
        if "bond_orders" in value:
            return value["bond_orders"]
    entity = getattr(value, "entity", None)
    if entity is not None:
        found = _bond_records(getattr(entity, "bonds", None))
        if found is not None:
            return found
    for attr in ("bonds", "bond_evidence"):
        found = _bond_records(getattr(value, attr, None))
        if found is not None:
            return found
    components = getattr(value, "components", None)
    if components:
        found = _bond_records(getattr(components[0], "bonds", None))
        if found is not None:
            return found
    return None


def normalise_prediction(value: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Turn common engine result shapes into a lossless JSON prediction."""
    if isinstance(value, dict) and "bond_orders" in value:
        out = {"bond_orders": {}, "bond_kinds": {}, "aromatic_bonds": []}
        out["bond_orders"].update(
            {str(k): v for k, v in value.get("bond_orders", {}).items()}
        )
        out["bond_kinds"].update(
            {str(k): v for k, v in value.get("bond_kinds", {}).items()}
        )
        out["aromatic_bonds"] = list(value.get("aromatic_bonds", []))
        for key in (
            "formal_charges",
            "implicit_hydrogens",
            "attached_hydrogens",
            "resonance_class",
            "status",
        ):
            if key in value:
                out[key] = value[key]
        return out
    bonds = _extract_bonds(value)
    if bonds is None:
        raise ValueError("engine result does not expose bonds/entity.bonds")
    out: dict[str, Any] = {"bond_orders": {}, "bond_kinds": {}, "aromatic_bonds": []}
    for edge in bonds:
        if isinstance(edge, dict):
            a, b = (
                edge.get("atom1_id", edge.get("a")),
                edge.get("atom2_id", edge.get("b")),
            )
            order, kind, aromatic = (
                edge.get("order"),
                edge.get("kind", "unknown"),
                edge.get("aromatic", False),
            )
        else:
            a, b = getattr(edge, "atom1_id", None), getattr(edge, "atom2_id", None)
            order, kind, aromatic = (
                getattr(edge, "order", None),
                getattr(edge, "kind", "unknown"),
                getattr(edge, "aromatic", False),
            )
            kind = getattr(kind, "value", kind)
        if a is None or b is None:
            continue
        shift = (
            tuple(int(value) for value in edge.get("atom2_image_shift", (0, 0, 0)))
            if isinstance(edge, dict)
            else tuple(int(value) for value in getattr(edge, "atom2_image_shift", (0, 0, 0)))
        )
        base = _edge_key(a, b, shift)
        occurrence = sum(key == base or key.startswith(base + "#") for key in out["bond_orders"]) + 1
        key = _edge_key(a, b, shift, occurrence)
        out["bond_orders"][key] = order
        out["bond_kinds"][key] = getattr(kind, "value", kind)
        if aromatic:
            out["aromatic_bonds"].append(key)
    atoms = None
    if hasattr(value, "entity"):
        atoms = getattr(value.entity, "atoms", None)
    if atoms is None and hasattr(value, "atoms"):
        atoms = value.atoms
    if atoms:
        out["formal_charges"] = {a.atom_id: a.formal_charge for a in atoms}
        attached: dict[str, int] = {a.atom_id: 0 for a in atoms}
        for edge in bonds:
            if isinstance(edge, dict):
                left = edge.get("atom1_id", edge.get("a"))
                right = edge.get("atom2_id", edge.get("b"))
            else:
                left = getattr(edge, "atom1_id", None)
                right = getattr(edge, "atom2_id", None)
            by_id = {a.atom_id: a for a in atoms}
            if left in by_id and by_id[left].element in {"H", "D"} and right in attached:
                attached[right] += 1
            if right in by_id and by_id[right].element in {"H", "D"} and left in attached:
                attached[left] += 1
        out["attached_hydrogens"] = attached
        out["implicit_hydrogens"] = {
            a.atom_id: (0 if a.implicit_hydrogens is None and attached[a.atom_id] else a.implicit_hydrogens)
            for a in atoms
        }
    resonance = getattr(value, "resonance_class", None)
    if resonance is None:
        classes = getattr(value, "resonance_classes", None)
        if classes:
            resonance = classes[0] if isinstance(classes, (list, tuple)) else classes
    if resonance is not None:
        out["resonance_class"] = getattr(
            resonance, "class_id", getattr(resonance, "value", resonance)
        )
    status = getattr(value, "status", None)
    if status is not None:
        out["status"] = getattr(status, "value", status)
    return out


def environment(*, processes: int = 1, command: str | None = None) -> dict[str, Any]:
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        git_commit = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "processes": processes,
        "command": command or " ".join(sys.argv),
        "git_commit": git_commit,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def call_mck(case: dict[str, Any]) -> dict[str, Any]:
    try:
        module = importlib.import_module("molcrys_kit.chemistry")
        fn = getattr(module, "infer_bond_orders")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "unavailable",
            "reason": f"infer_bond_orders API unavailable: {exc}",
        }
    try:
        entity = fixture_entity(case)
        kwargs: dict[str, Any] = {"total_charge": None}
        if "cell_A" in case:
            kwargs["cell"] = case["cell_A"]
        try:
            result = fn(entity, **kwargs)
        except TypeError:
            # Compatibility with early API prototypes; no chemistry fallback is used.
            result = fn(entity)
        prediction = normalise_prediction(result, case)
        result_status = getattr(getattr(result, "status", None), "value", getattr(result, "status", None))
        if result_status == "indeterminate":
            return {
                "status": "incomplete",
                "reason": "MolCrysKit returned INDETERMINATE",
                "prediction": prediction,
            }
        return {"status": "ok", "prediction": prediction}
    except Exception as exc:  # candidate failures are data, not fabricated wins
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def obabel_binary() -> str | None:
    return os.environ.get("OBABEL_BIN") or shutil.which("obabel")


def call_ob_cpp(case: dict[str, Any]) -> dict[str, Any]:
    """Run the pinned CLI when available; otherwise record an explicit gap.

    The native recipe and version pin live in ``openbabel/README.md``.  The
    command-line adapter intentionally marks SMILES-only output as partial: it
    cannot claim atom-ID-complete scoring when Open Babel omitted explicit H.
    """
    binary = obabel_binary()
    if not binary:
        return {
            "status": "unavailable",
            "reason": "Open Babel 3.2.1 executable not found (set OBABEL_BIN)",
        }
    smiles = case.get("smiles")
    if not smiles:
        return {
            "status": "unavailable",
            "reason": "fixture has no line notation for Open Babel adapter",
        }
    try:
        proc = subprocess.run(
            [binary, f"-:{smiles}", "-osdf", "-O", "-"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode:
            return {
                "status": "error",
                "reason": f"obabel exit {proc.returncode}: {proc.stderr.strip()}",
            }
        # Parsing atom-ID mapping from a SMILES-only OB output is intentionally
        # conservative.  Report availability but incomplete comparability.
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        return {
            "status": "incomplete",
            "reason": "native OB output lacks fixture stable-ID mapping; provide source-aware adapter",
            "version": version,
            "raw_stdout": proc.stdout,
        }
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def call_ob_python(case: dict[str, Any]) -> dict[str, Any]:
    try:
        pybel = importlib.import_module("openbabel.pybel")
    except ImportError as exc:
        return {
            "status": "unavailable",
            "reason": f"Open Babel Python bindings unavailable: {exc}",
        }
    smiles = case.get("smiles")
    if not smiles:
        return {
            "status": "unavailable",
            "reason": "fixture has no line notation for Open Babel adapter",
        }
    try:
        mol = pybel.readstring("smi", smiles)
        # Explicitly exercise the pinned comparator operations where bindings
        # expose them. Stable-ID mapping remains incomplete for omitted H.
        if hasattr(mol.OBMol, "PerceiveBondOrders"):
            mol.OBMol.PerceiveBondOrders()
        return {
            "status": "incomplete",
            "reason": "OB Python SMILES adapter cannot prove fixture atom-ID mapping",
            "raw_formula": mol.formula,
        }
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


ENGINE_CALLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "mck": call_mck,
    "ob_cpp": call_ob_cpp,
    "ob_python": call_ob_python,
}


def timed_case(
    engine: str, case: dict[str, Any], repeats: int = 1, timeout_s: float | None = None
) -> dict[str, Any]:
    fn = ENGINE_CALLS[engine]

    def invoke() -> dict[str, Any]:
        if timeout_s is None or timeout_s <= 0 or not hasattr(signal, "setitimer"):
            return fn(case)

        def alarm(_signum, _frame):
            raise TimeoutError(f"case exceeded {timeout_s:g}s timeout")

        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(case)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    runs = []
    result: dict[str, Any] | None = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        try:
            result = invoke()
        except TimeoutError as exc:
            result = {"status": "timeout", "reason": str(exc)}
        elapsed = time.perf_counter_ns() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
        runs.append(
            {"elapsed_ns": elapsed, "status": result.get("status"), "peak_rss_kb": rss}
        )
        if result.get("status") in {"unavailable", "error", "incomplete"}:
            break
    assert result is not None
    result = dict(result)
    result["raw_runs"] = runs
    result["elapsed_ns"] = runs[-1]["elapsed_ns"]
    return result
