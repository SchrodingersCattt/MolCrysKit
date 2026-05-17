"""Provenance records for ordered replicas generated from disorder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DisorderProvenance:
    """Source-site audit trail for one ordered disorder replica."""

    kept_indices: List[int]
    dropped_indices: List[int]
    method: str
    coupled: bool
    sym_op_indices: Optional[List[int]] = None


__all__ = ["DisorderProvenance"]
