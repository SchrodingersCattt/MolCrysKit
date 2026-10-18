"""Calibrate ring-geometry thresholds from exported CSD observations.

The aromaticity resolver deliberately does not depend on the CCDC Python API.
The API is licensed software and is not available in the normal MolCrysKit
runtime or CI environments.  A CSD job can therefore export one JSON record
per ring edge (and one summary record per ring), then pass those records to
``calibrate_aromaticity_geometry``.  The returned JSON-compatible mapping is
the hand-off consumed by the calibration report and can be reviewed before it
is promoted to the runtime defaults.

Two kinds of observations are expected:

* ``bond_class == "single"`` records provide the element-pair reference
  lengths.  They should come from source-confirmed single bonds in the
  CSD sample, rather than distances inferred by MolCrysKit itself.
* ``ring_class`` records provide whole-ring geometry metrics.  The classes are
  ``"aromatic"``, ``"saturated"``, and ``"conjugated_non_aromatic"``.  Every
  ring record also carries a ``references_A`` mapping.  This records the
  single-bond references used to normalise its metrics, so a snapshot cannot
  promote thresholds calculated with one set of references alongside another.
  Planarity separation is calibrated only from the aromatic and saturated
  classes; conjugated non-aromatic planarity is reported for review but does
  not set a runtime boundary.

The calibrator reports the coverage and uncertainty bands instead of silently
filling a missing element pair.  A caller can keep the conservative PR #179
thresholds when coverage is insufficient, which is the intended behaviour for
the default installation without a licensed CSD export.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ELEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("C", "C"),
    ("C", "N"),
    ("C", "O"),
    ("C", "S"),
    ("N", "N"),
    ("N", "O"),
    ("N", "S"),
    ("O", "O"),
    ("O", "S"),
    ("S", "S"),
)

# CSD bond lengths are reported in Angstroms.  A promoted reference and the
# median from its source rows are therefore compared with an absolute
# tolerance at the scale of the measurements, rather than requiring JSON
# serialization and floating-point arithmetic to agree to machine precision.
REFERENCE_TOLERANCE_A = 1e-4

RING_CLASSES: tuple[str, ...] = (
    "aromatic",
    "saturated",
    "conjugated_non_aromatic",
)


class CalibrationError(ValueError):
    """Raised when observations cannot support a safe calibration."""


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    """Minimum evidence required before a snapshot may replace defaults."""

    min_pair_observations: int = 20
    min_ring_observations: int = 25
    min_metric_separation: float = 0.01
    min_alternation_separation: float = 0.01


def _pair(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        fields = value.replace("/", "-").split("-")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        fields = list(value)
    else:
        raise CalibrationError(f"invalid element pair: {value!r}")
    if len(fields) != 2 or any(not isinstance(item, str) or not item for item in fields):
        raise CalibrationError(f"invalid element pair: {value!r}")
    return tuple(sorted((fields[0], fields[1])))  # type: ignore[return-value]


def _number(record: Mapping[str, object], key: str, *, positive: bool = False) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    if positive and result == 0.0:
        return None
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CalibrationError("cannot calculate a quantile from no observations")
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "q05": _quantile(values, 0.05),
        "median": _quantile(values, 0.50),
        "q95": _quantile(values, 0.95),
    }


def calibrate_aromaticity_geometry(
    records: Iterable[Mapping[str, object]],
    *,
    source: Mapping[str, object] | None = None,
    expected_pairs: Sequence[tuple[str, str]] = ELEMENT_PAIRS,
    policy: CalibrationPolicy = CalibrationPolicy(),
) -> dict[str, object]:
    """Build a reviewable geometry calibration snapshot.

    ``records`` may contain both edge and ring rows.  Invalid rows are ignored
    and counted in ``quality.invalid_rows``; a missing element pair or a
    distribution with no safe separation raises :class:`CalibrationError`
    rather than producing a deceptively precise threshold.
    """

    expected_pairs = tuple(_pair(pair) for pair in expected_pairs)
    pair_values: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in expected_pairs
    }
    ring_metrics: dict[str, list[dict[str, float]]] = {
        ring_class: [] for ring_class in RING_CLASSES
    }
    reference_blocks: list[dict[tuple[str, str], float]] = []
    pending_rings: list[tuple[str, dict[str, float], dict[tuple[str, str], float]]] = []
    invalid_rows = 0
    total_rows = 0

    def parse_references(value: object) -> dict[tuple[str, str], float] | None:
        if not isinstance(value, Mapping):
            return None
        parsed: dict[tuple[str, str], float] = {}
        try:
            for key, raw in value.items():
                pair = _pair(key)
                if pair not in pair_values:
                    return None
                if isinstance(raw, Mapping):
                    raw = raw.get("median")
                number = float(raw)
                if not math.isfinite(number) or number <= 0:
                    return None
                parsed[pair] = number
        except (CalibrationError, TypeError, ValueError):
            return None
        if set(parsed) != set(pair_values):
            return None
        return parsed

    for record in records:
        total_rows += 1
        if not isinstance(record, Mapping):
            invalid_rows += 1
            continue
        row_invalid = False

        # A dedicated metadata row is accepted in addition to references_A on
        # each ring row.  It is useful for JSONL exporters that write one
        # header record followed by many rings.
        if "references_A" in record:
            parsed_references = parse_references(record.get("references_A"))
            if parsed_references is None:
                row_invalid = True
            else:
                reference_blocks.append(parsed_references)

        recognized = False
        if record.get("bond_class") == "single":
            recognized = True
            try:
                pair = _pair(record.get("pair"))
            except CalibrationError:
                row_invalid = True
            else:
                value = _number(record, "length_A", positive=True)
                if pair in pair_values and value is not None:
                    pair_values[pair].append(value)
                else:
                    row_invalid = True

        ring_class = record.get("ring_class")
        # Keep accepting the old spelling as a compatibility alias, while all
        # emitted snapshots use the explicit three-way class split.
        if ring_class == "non_aromatic":
            ring_class = "conjugated_non_aromatic"
        if ring_class in RING_CLASSES:
            recognized = True
            metrics = {
                key: _number(record, key)
                for key in ("planarity", "bond_metric", "alternation_gap")
            }
            ring_references = parse_references(record.get("references_A"))
            if ring_references is None and reference_blocks:
                ring_references = reference_blocks[-1]
            if any(value is None for value in metrics.values()) or ring_references is None:
                row_invalid = True
            else:
                pending_rings.append(
                    (ring_class, metrics, ring_references)  # type: ignore[arg-type]
                )
        elif ring_class is not None:
            row_invalid = True

        if not recognized and "references_A" not in record:
            row_invalid = True
        if row_invalid:
            invalid_rows += 1

    missing_pairs = [
        pair for pair, values in pair_values.items()
        if len(values) < policy.min_pair_observations
    ]
    if missing_pairs:
        formatted = ", ".join("-".join(pair) for pair in missing_pairs)
        raise CalibrationError(
            "insufficient source-confirmed single-bond coverage for: " + formatted
        )
    pair_summary = {
        "-".join(pair): _summary(values)
        for pair, values in sorted(pair_values.items())
    }
    expected_references = {
        pair: float(summary["median"])
        for pair, summary in (
            (_pair(key), value) for key, value in pair_summary.items()
        )
    }
    if not reference_blocks:
        # Ring rows are required to carry the block, but a dedicated metadata
        # row is also allowed.  Either way, no snapshot is emitted without it.
        raise CalibrationError("missing references_A block")
    for references in reference_blocks:
        if any(
            not math.isclose(
                references[pair], expected, rel_tol=0.0, abs_tol=REFERENCE_TOLERANCE_A
            )
            for pair, expected in expected_references.items()
        ):
            raise CalibrationError(
                "references_A does not match the promoted element-pair medians"
            )
    for ring_class, metrics, references in pending_rings:
        if any(
            not math.isclose(
                references[pair], expected, rel_tol=0.0, abs_tol=REFERENCE_TOLERANCE_A
            )
            for pair, expected in expected_references.items()
        ):
            raise CalibrationError(
                "ring references_A does not match the promoted element-pair medians"
            )
        ring_metrics[ring_class].append(metrics)

    missing_metrics = [
        ring_class
        for ring_class in RING_CLASSES
        if len(ring_metrics[ring_class]) < policy.min_ring_observations
    ]
    if missing_metrics:
        classes = ", ".join(missing_metrics)
        raise CalibrationError(
            f"insufficient ring metrics (n < {policy.min_ring_observations}): {classes}"
        )

    aromatic_planarity = [item["planarity"] for item in ring_metrics["aromatic"]]
    saturated_planarity = [item["planarity"] for item in ring_metrics["saturated"]]
    conjugated_planarity = [
        item["planarity"] for item in ring_metrics["conjugated_non_aromatic"]
    ]
    aromatic_metric = [item["bond_metric"] for item in ring_metrics["aromatic"]]
    saturated_metric = [item["bond_metric"] for item in ring_metrics["saturated"]]
    conjugated_metric = [
        item["bond_metric"] for item in ring_metrics["conjugated_non_aromatic"]
    ]
    aromatic_alternation = [item["alternation_gap"] for item in ring_metrics["aromatic"]]
    saturated_alternation = [item["alternation_gap"] for item in ring_metrics["saturated"]]
    conjugated_alternation = [
        item["alternation_gap"] for item in ring_metrics["conjugated_non_aromatic"]
    ]
    non_aromatic_metric = saturated_metric + conjugated_metric
    non_aromatic_alternation = saturated_alternation + conjugated_alternation

    # The two tails define the abstention band.  A sample in this interval is
    # not assigned either class by the runtime resolver.
    aromatic_metric_upper = _quantile(aromatic_metric, 0.95)
    saturated_metric_lower = _quantile(saturated_metric, 0.05)
    if saturated_metric_lower - aromatic_metric_upper < policy.min_metric_separation:
        raise CalibrationError("aromatic and saturated bond metrics overlap")
    aromatic_alternation_upper = _quantile(aromatic_alternation, 0.95)
    conjugated_alternation_lower = _quantile(conjugated_alternation, 0.05)
    if conjugated_alternation_lower - aromatic_alternation_upper < policy.min_alternation_separation:
        raise CalibrationError("aromatic and conjugated alternation gaps overlap")
    aromatic_planarity_upper = _quantile(aromatic_planarity, 0.95)
    saturated_planarity_lower = _quantile(saturated_planarity, 0.05)
    if saturated_planarity_lower - aromatic_planarity_upper < policy.min_metric_separation:
        raise CalibrationError("aromatic and saturated planarity overlaps")

    return {
        "schema_version": 1,
        "status": "calibrated",
        "source": dict(source or {}),
        "quality": {
            "rows": total_rows,
            "invalid_rows": invalid_rows,
            "aromatic_rings": len(ring_metrics["aromatic"]),
            "saturated_rings": len(ring_metrics["saturated"]),
            "conjugated_non_aromatic_rings": len(ring_metrics["conjugated_non_aromatic"]),
            # Compatibility aggregate for consumers of the initial schema.
            "non_aromatic_rings": len(ring_metrics["saturated"])
            + len(ring_metrics["conjugated_non_aromatic"]),
            "pair_coverage": {
                "-".join(pair): len(values) for pair, values in sorted(pair_values.items())
            },
        },
        "element_pair_references_A": pair_summary,
        "references_A": {
            "-".join(pair): value for pair, value in sorted(expected_references.items())
        },
        "geometry_bands": {
            "planarity": {
                "aromatic": _summary(aromatic_planarity),
                "saturated": _summary(saturated_planarity),
                "conjugated_non_aromatic": _summary(conjugated_planarity),
                "aromatic_max": aromatic_planarity_upper,
                "saturated_min": saturated_planarity_lower,
                "uncertain": [
                    aromatic_planarity_upper,
                    saturated_planarity_lower,
                ],
            },
            "bond_metric": {
                "aromatic": _summary(aromatic_metric),
                "saturated": _summary(saturated_metric),
                "conjugated_non_aromatic": _summary(conjugated_metric),
                "non_aromatic": _summary(non_aromatic_metric),
                "aromatic_max": aromatic_metric_upper,
                "saturated_min": saturated_metric_lower,
                "uncertain": [aromatic_metric_upper, saturated_metric_lower],
            },
            "alternation_gap": {
                "aromatic": _summary(aromatic_alternation),
                "saturated": _summary(saturated_alternation),
                "conjugated_non_aromatic": _summary(conjugated_alternation),
                "non_aromatic": _summary(non_aromatic_alternation),
                "aromatic_max": aromatic_alternation_upper,
                "conjugated_non_aromatic_min": conjugated_alternation_lower,
                "non_aromatic_min": conjugated_alternation_lower,
                "uncertain": [aromatic_alternation_upper, conjugated_alternation_lower],
            },
        },
    }


def read_records(path: str | Path) -> list[Mapping[str, object]]:
    """Read JSON or JSON-lines ring observations exported from a CSD job."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".jsonl":
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
    if not isinstance(value, list):
        raise CalibrationError("calibration input must be a JSON list or JSONL file")
    return value


def write_snapshot(snapshot: Mapping[str, object], path: str | Path) -> None:
    """Write a deterministic, human-reviewable calibration snapshot."""

    Path(path).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def runtime_overrides(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Return the ``CHEMISTRY_PERCEPTION_CONFIG`` overrides from a snapshot.

    This is intentionally explicit rather than an import-time environment
    hook.  A calibrated file is a reviewed scientific input, so applications
    should choose when to promote it and should retain the snapshot alongside
    their provenance records.  The returned keys match the runtime config
    names used by :mod:`molcrys_kit.chemistry.aromaticity`.
    """

    if snapshot.get("schema_version") != 1:
        raise CalibrationError("unsupported or missing calibration schema_version")
    if snapshot.get("status") != "calibrated":
        raise CalibrationError("only a calibrated snapshot can override runtime thresholds")
    references = snapshot.get("element_pair_references_A")
    promoted_references = snapshot.get("references_A")
    bands = snapshot.get("geometry_bands")
    if (
        not isinstance(references, Mapping)
        or not isinstance(promoted_references, Mapping)
        or not isinstance(bands, Mapping)
    ):
        raise CalibrationError("snapshot is missing references or geometry bands")
    pair_lengths: dict[tuple[str, str], float] = {}
    for key, summary in references.items():
        pair = _pair(key)
        if pair not in ELEMENT_PAIRS:
            raise CalibrationError(f"unexpected element-pair reference {key!r}")
        if not isinstance(summary, Mapping):
            raise CalibrationError(f"invalid summary for element pair {key!r}")
        median = summary.get("median")
        try:
            median_value = float(median)
        except (TypeError, ValueError):
            raise CalibrationError(f"invalid median for element pair {key!r}") from None
        if not math.isfinite(median_value) or median_value <= 0:
            raise CalibrationError(f"invalid median for element pair {key!r}")
        pair_lengths[pair] = median_value
    missing_pairs = set(ELEMENT_PAIRS).difference(pair_lengths)
    if missing_pairs:
        formatted = ", ".join("-".join(pair) for pair in sorted(missing_pairs))
        raise CalibrationError("snapshot is missing element-pair references: " + formatted)
    exported_lengths: dict[tuple[str, str], float] = {}
    for key, value in promoted_references.items():
        pair = _pair(key)
        if pair not in ELEMENT_PAIRS:
            raise CalibrationError(f"unexpected references_A element pair {key!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise CalibrationError(f"invalid references_A value for element pair {key!r}") from None
        if not math.isfinite(number) or number <= 0:
            raise CalibrationError(f"invalid references_A value for element pair {key!r}")
        exported_lengths[pair] = number
    if set(exported_lengths) != set(pair_lengths) or any(
        not math.isclose(
            exported_lengths[pair], value, rel_tol=0.0, abs_tol=REFERENCE_TOLERANCE_A
        )
        for pair, value in pair_lengths.items()
    ):
        raise CalibrationError("references_A does not match element-pair medians")

    try:
        planarity = float(bands["planarity"]["aromatic_max"])
        planarity_min = float(bands["planarity"]["saturated_min"])
        metric = bands["bond_metric"]
        alternation = bands["alternation_gap"]
        aromatic_metric = float(metric["aromatic_max"])
        saturated_metric = float(metric["saturated_min"])
        alternation_limit = float(alternation["aromatic_max"])
        alternation_min = float(alternation["conjugated_non_aromatic_min"])
    except (KeyError, TypeError, ValueError):
        raise CalibrationError("snapshot has incomplete geometry bands") from None
    values = (
        planarity,
        planarity_min,
        aromatic_metric,
        saturated_metric,
        alternation_limit,
        alternation_min,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise CalibrationError("snapshot has non-finite geometry thresholds")
    if planarity_min <= planarity:
        raise CalibrationError("snapshot planarity bands are not separated")
    if saturated_metric <= aromatic_metric:
        raise CalibrationError("snapshot bond metric bands are not separated")
    if alternation_min <= alternation_limit:
        raise CalibrationError("snapshot alternation bands are not separated")
    return {
        "AROMATIC_RING_PLANARITY_MAX": planarity,
        "AROMATIC_RING_BOND_METRIC_AROMATIC_MAX": aromatic_metric,
        "AROMATIC_RING_BOND_METRIC_SATURATED_MIN": saturated_metric,
        "AROMATIC_RING_BOND_ALTERNATION_MAX": alternation_limit,
        "AROMATIC_RING_SINGLE_BOND_LENGTHS": pair_lengths,
    }


__all__ = [
    "CalibrationError",
    "CalibrationPolicy",
    "ELEMENT_PAIRS",
    "RING_CLASSES",
    "calibrate_aromaticity_geometry",
    "read_records",
    "runtime_overrides",
    "write_snapshot",
]
