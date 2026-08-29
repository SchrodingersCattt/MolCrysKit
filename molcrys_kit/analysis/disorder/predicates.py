"""Small predicates for disorder-aware downstream consumers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from molcrys_kit.constants.config import PARTIAL_OCCUPANCY_TOLERANCE


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_minor_site(site: Mapping[str, Any]) -> bool:
    """Return whether a raw site dictionary represents a minor disorder image."""
    flag = site.get("_is_minor")
    if flag is not None:
        return bool(flag)

    occ = _as_float(site.get("occ", site.get("occupancy")))
    dg = site.get("dg", site.get("disorder_group", ".")) or "."
    da = site.get("da", site.get("disorder_assembly", ".")) or "."

    if isinstance(dg, str) and dg.strip().startswith("-"):
        return True
    if occ is None or occ >= 1.0 - PARTIAL_OCCUPANCY_TOLERANCE:
        return False
    if (
        str(dg).strip() == "."
        and str(da).strip() == "."
        and occ < 0.5 - PARTIAL_OCCUPANCY_TOLERANCE
    ):
        return True
    return False


__all__ = ["is_minor_site"]
