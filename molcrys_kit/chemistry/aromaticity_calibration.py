"""Calibrate ring-geometry thresholds from exported CSD observations.

The aromaticity resolver deliberately does not depend on the CCDC Python API.
The API is licensed software and is not available in the normal MolCrysKit
runtime or CI environments.  A CSD job can therefore export one JSON record
per ring edge (and one summary record per ring), then pass those records to
``calibrate_aromaticity_geometry``.  The returned JSON-compatible mapping is
the hand-off consumed by the calibration report and can be reviewed before it
is promoted to the runtime defaults.

Two classes of observations are expected:

* ``bond_class == "single"`` records provide the element-pair reference
  lengths.  They should come from source-confirmed single bonds in the
  CSD sample, rather than distances inferred by MolCrysKit itself.
* ``ring_class`` records provide whole-ring geometry metrics.  The class is
  ``"aromatic"`` or ``"non_aromatic"`` and the fields are the same metrics
  calculated by :mod:`molcrys_kit.chemistry.aromaticity`.

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
    minimum = 0.0 if not positive else 0.0
    if not math.isfinite(result) or result < minimum:
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

    pair_values: dict[tuple[str, str], list[float]] = {
        _pair(pair): [] for pair in expected_pairs
    }
    aromatic_metrics: list[Mapping[str, object]] = []
    non_aromatic_metrics: list[Mapping[str, object]] = []
    invalid_rows = 0
    total_rows = 0

    for record in records:
        total_rows += 1
        if not isinstance(record, Mapping):
            invalid_rows += 1
            continue
        if record.get("bond_class") == "single":
            try:
                pair = _pair(record.get("pair"))
            except CalibrationError:
                invalid_rows += 1
            else:
                value = _number(record, "length_A", positive=True)
                if pair in pair_values and value is not None:
                    pair_values[pair].append(value)
                else:
                    invalid_rows += 1
        ring_class = record.get("ring_class")
        if ring_class == "aromatic":
            aromatic_metrics.append(record)
        elif ring_class in {"non_aromatic", "saturated"}:
            non_aromatic_metrics.append(record)

    missing_pairs = [pair for pair, values in pair_values.items() if len(values) < policy.min_pair_observations]
    if missing_pairs:
        formatted = ", ".join("-".join(pair) for pair in missing_pairs)
        raise CalibrationError(
            "insufficient source-confirmed single-bond coverage for: " + formatted
        )
    if len(aromatic_metrics) < policy.min_ring_observations:
        raise CalibrationError("insufficient aromatic ring observations")
    if len(non_aromatic_metrics) < policy.min_ring_observations:
        raise CalibrationError("insufficient non-aromatic ring observations")

    pair_summary = {
        "-".join(pair): _summary(values)
        for pair, values in sorted(pair_values.items())
    }
    aromatic_planarity = [
        value for record in aromatic_metrics if (value := _number(record, "planarity")) is not None
    ]
    non_aromatic_planarity = [
        value for record in non_aromatic_metrics if (value := _number(record, "planarity")) is not None
    ]
    aromatic_metric = [
        value for record in aromatic_metrics if (value := _number(record, "bond_metric")) is not None
    ]
    non_aromatic_metric = [
        value for record in non_aromatic_metrics if (value := _number(record, "bond_metric")) is not None
    ]
    aromatic_alternation = [
        value
        for record in aromatic_metrics
        if (value := _number(record, "alternation_gap")) is not None
    ]
    non_aromatic_alternation = [
        value
        for record in non_aromatic_metrics
        if (value := _number(record, "alternation_gap")) is not None
    ]
    required_metrics = {
        "aromatic planarity": aromatic_planarity,
        "non-aromatic planarity": non_aromatic_planarity,
        "aromatic bond metric": aromatic_metric,
        "non-aromatic bond metric": non_aromatic_metric,
        "aromatic alternation gap": aromatic_alternation,
        "non-aromatic alternation gap": non_aromatic_alternation,
    }
    missing_metrics = [
        name
        for name, values in required_metrics.items()
        if len(values) < policy.min_ring_observations
    ]
    if missing_metrics:
        raise CalibrationError("missing ring metrics: " + ", ".join(missing_metrics))

    # The two tails define the abstention band.  A sample in this interval is
    # not assigned either class by the runtime resolver.
    aromatic_metric_upper = _quantile(aromatic_metric, 0.95)
    non_aromatic_metric_lower = _quantile(non_aromatic_metric, 0.05)
    if non_aromatic_metric_lower - aromatic_metric_upper < policy.min_metric_separation:
        raise CalibrationError("aromatic and non-aromatic bond metrics overlap")
    aromatic_alternation_upper = _quantile(aromatic_alternation, 0.95)
    non_aromatic_alternation_lower = _quantile(non_aromatic_alternation, 0.05)
    if non_aromatic_alternation_lower - aromatic_alternation_upper < policy.min_alternation_separation:
        raise CalibrationError("aromatic and non-aromatic alternation gaps overlap")

    return {
        "schema_version": 1,
        "status": "calibrated",
        "source": dict(source or {}),
        "quality": {
            "rows": total_rows,
            "invalid_rows": invalid_rows,
            "aromatic_rings": len(aromatic_metrics),
            "non_aromatic_rings": len(non_aromatic_metrics),
            "pair_coverage": {
                "-".join(pair): len(values) for pair, values in sorted(pair_values.items())
            },
        },
        "element_pair_references_A": pair_summary,
        "geometry_bands": {
            "planarity": {
                "aromatic": _summary(aromatic_planarity),
                "non_aromatic": _summary(non_aromatic_planarity),
                "aromatic_max": _quantile(aromatic_planarity, 0.99),
                "non_aromatic_min": _quantile(non_aromatic_planarity, 0.05),
                "uncertain": [
                    _quantile(aromatic_planarity, 0.99),
                    _quantile(non_aromatic_planarity, 0.05),
                ],
            },
            "bond_metric": {
                "aromatic": _summary(aromatic_metric),
                "non_aromatic": _summary(non_aromatic_metric),
                "aromatic_max": aromatic_metric_upper,
                "saturated_min": non_aromatic_metric_lower,
                "uncertain": [aromatic_metric_upper, non_aromatic_metric_lower],
            },
            "alternation_gap": {
                "aromatic": _summary(aromatic_alternation),
                "non_aromatic": _summary(non_aromatic_alternation),
                "aromatic_max": aromatic_alternation_upper,
                "non_aromatic_min": non_aromatic_alternation_lower,
                "uncertain": [aromatic_alternation_upper, non_aromatic_alternation_lower],
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

    if snapshot.get("status") != "calibrated":
        raise CalibrationError("only a calibrated snapshot can override runtime thresholds")
    references = snapshot.get("element_pair_references_A")
    bands = snapshot.get("geometry_bands")
    if not isinstance(references, Mapping) or not isinstance(bands, Mapping):
        raise CalibrationError("snapshot is missing references or geometry bands")
    pair_lengths: dict[tuple[str, str], float] = {}
    for key, summary in references.items():
        pair = _pair(key)
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

    try:
        planarity = float(bands["planarity"]["aromatic_max"])
        metric = bands["bond_metric"]
        alternation = bands["alternation_gap"]
        aromatic_metric = float(metric["aromatic_max"])
        saturated_metric = float(metric["saturated_min"])
        alternation_limit = float(alternation["aromatic_max"])
    except (KeyError, TypeError, ValueError):
        raise CalibrationError("snapshot has incomplete geometry bands") from None
    values = (planarity, aromatic_metric, saturated_metric, alternation_limit)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise CalibrationError("snapshot has non-finite geometry thresholds")
    if saturated_metric <= aromatic_metric:
        raise CalibrationError("snapshot bond metric bands are not separated")
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
    "calibrate_aromaticity_geometry",
    "read_records",
    "runtime_overrides",
    "write_snapshot",
]
