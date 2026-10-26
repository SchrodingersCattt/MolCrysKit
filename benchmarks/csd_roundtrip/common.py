"""Version guards and explicit, coordinate-free public result projection."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LOCAL = ROOT / "_local"
STRATA = (
    "disordered",
    "special_position",
    "multi_ion",
    "multi_component",
    "single_component",
)
STATUSES = {"pass", "mismatch", "unsupported", "indeterminate", "error", "timeout"}


def isotope_symbol(symbol):
    """Normalize explicit hydrogen isotope symbols without guessing masses."""
    return {"D": ("H", 2), "T": ("H", 3)}.get(symbol, (symbol, None))


def normalize_source_isotopes(value):
    """Normalize a private snapshot in memory; the frozen input stays intact."""
    if isinstance(value, dict):
        if "symbol" in value and "xyz" in value:
            symbol, isotope = isotope_symbol(value["symbol"])
            value["symbol"] = symbol
            if isotope is not None:
                value["isotope"] = isotope
        for item in value.values():
            normalize_source_isotopes(item)
    elif isinstance(value, list):
        for item in value:
            normalize_source_isotopes(item)
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


@lru_cache(maxsize=1)
def config():
    return read_json(ROOT / "config.json")


def harness_hash():
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "config.json"]
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def verify_target():
    expected = config()["target_commit"]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if actual != expected:
        changes = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                expected,
                "HEAD",
                "--",
                "molcrys_kit",
                "pyproject.toml",
            ],
            cwd=REPO,
            text=True,
        )
        if changes.strip():
            raise ValueError("target package differs from the frozen experiment")
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "molcrys_kit"], cwd=REPO, text=True
    )
    if dirty.strip():
        raise ValueError("target package has local changes")
    return actual


def verify_manifest(manifest):
    unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("sha256") != digest(unsigned):
        raise ValueError("manifest checksum mismatch")
    if manifest["config_sha256"] != digest(config()):
        raise ValueError("config differs from the frozen manifest")
    refs = [x["refcode"] for x in manifest["cases"]]
    if len(refs) != len(set(refs)):
        raise ValueError("duplicate refcode")


def public_result(raw):
    """Never pass arbitrary errors, arrays, labels or metadata to a report."""
    keys = (
        "refcode",
        "stratum",
        "targeted",
        "tags",
        "status",
        "record_match",
        "geometry_roundtrip_match",
        "molecular_completeness",
        "n_components",
        "n_reference_molecules",
        "n_reference_atoms",
        "missing_atoms",
        "input_sha256",
        "source_sha256",
        "elapsed_s",
        "manifest_sha256",
        "harness_sha256",
        "stages",
        "diagnostics",
    )
    result = {
        k: raw[k] for k in keys if k in raw and k not in {"diagnostics", "stages"}
    }
    result["stages"] = {
        name: {
            k: v
            for k, v in stage.items()
            if k
            in {
                "status",
                "code",
                "max_A",
                "rmsd_A",
                "n_atoms",
                "n_molecules",
                "n_operations",
                "detected_spacegroup",
                "source_spacegroup",
                "length_error_A",
                "angle_error_deg",
                "missing_annotations",
                "error_type",
                "unknown_annotations",
                "n_alternatives",
            }
            and (
                isinstance(v, str)
                or (isinstance(v, (int, float)) and math.isfinite(v))
                or (
                    k in {"missing_annotations", "unknown_annotations"}
                    and isinstance(v, list)
                    and all(
                        x in {"bond_order", "formal_charge", "isotope", "stereo"}
                        for x in v
                    )
                )
            )
        }
        for name, stage in raw.get("stages", {}).items()
    }
    checks = {
        "hard_clash",
        "intermolecular_clash",
        "isolated_atoms",
        "hydrogen_presence",
        "formula_consistency",
        "bond_distances",
    }
    result["diagnostics"] = {
        key: {
            k: v
            for k, v in value.items()
            if k in checks and v in {"passed", "failed", "skipped"}
        }
        for key, value in raw.get("diagnostics", {}).items()
        if key in {"source_sanity", "output_sanity"} and isinstance(value, dict)
    }
    for key in ("source_sanity_error", "output_sanity_error"):
        value = raw.get("diagnostics", {}).get(key)
        if isinstance(value, str) and value.isidentifier():
            result["diagnostics"][key] = value
    if result.get("status") not in STATUSES:
        raise ValueError("invalid result status")
    return result
