"""Provenance records for ordered replicas generated from disorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .symmetry_audit import ParentSymmetryAudit


@dataclass(frozen=True)
class DisorderProvenance:
    """Source-site audit trail for one ordered disorder replica."""

    kept_indices: List[int]
    dropped_indices: List[int]
    method: str
    coupled: bool
    sym_op_indices: Optional[List[int]] = None
    symmetry_audit: Optional[ParentSymmetryAudit] = None
    parent_space_group_number: Optional[int] = None
    parent_space_group_symbol: Optional[str] = None
    respects_parent_group: Optional[bool] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kept_indices": list(self.kept_indices),
            "dropped_indices": list(self.dropped_indices),
            "method": self.method,
            "coupled": bool(self.coupled),
            "sym_op_indices": (
                None if self.sym_op_indices is None else list(self.sym_op_indices)
            ),
            "parent_space_group_number": self.parent_space_group_number,
            "parent_space_group_symbol": self.parent_space_group_symbol,
            "respects_parent_group": self.respects_parent_group,
            "notes": list(self.notes),
            "symmetry_audit": (
                None if self.symmetry_audit is None else self.symmetry_audit.to_dict()
            ),
        }
        return payload


__all__ = ["DisorderProvenance"]
