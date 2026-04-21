"""
Conflict-edge priority table and a single helper for adding/promoting
edges in the disorder exclusion graph.

The exclusion graph stores, on each edge, a `conflict_type` attribute
describing *why* two atoms cannot coexist.  Multiple passes may
discover the same pair of atoms and want to attach different types
(e.g. one pass labels the pair as a generic geometric clash, a later
pass realises it is actually a coordination-valence conflict).  Rather
than scatter ad-hoc "if existing is geometric, upgrade to X" snippets
throughout the codebase (the previous approach), we centralise the
ordering here.

Higher number = stronger semantic claim = wins over lower-numbered
types.  When a pass tries to add an edge whose new type is weaker than
the existing one, the existing edge is preserved unchanged.

Priority rationale (highest to lowest):

* ``logical_alternative`` (100):  PART/assembly labels in the CIF
  declare the two atoms as alternatives at the same site.  This is the
  crystallographer's explicit statement and overrides everything else.
* ``symmetry_clash`` (90):  symmetry images of the same fragment that
  collapse onto one another (ghost overlap).  Geometrically derived
  but with explicit symmetry provenance.
* ``explicit`` (80):  assembly-based exclusion (different disorder
  groups in the same assembly).  Crystallographer-asserted exclusion.
* ``valence`` (70):  the pair would over-coordinate one of the two
  atoms relative to its expected valence.  Chemistry-aware coordination
  conflict.
* ``valence_geometry`` (60):  the pair would force a geometrically
  unreasonable bond angle / distance combination.  Chemistry-aware
  geometric conflict.
* ``geometric`` (30):  pure distance-based clash, no chemistry.
* ``implicit_sp`` (20):  weak partial-occupancy overlap from special
  positions identified by proximity clustering.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


CONFLICT_PRIORITY: dict[str, int] = {
    "logical_alternative": 100,
    "symmetry_clash": 90,
    "explicit": 80,
    "valence": 70,
    "valence_geometry": 60,
    "geometric": 30,
    "implicit_sp": 20,
}


def add_or_promote_edge(
    graph: nx.Graph, u: int, v: int, new_type: str, **attrs: Any
) -> None:
    """Add a conflict edge or upgrade the existing edge's type.

    If ``(u, v)`` already exists in ``graph``, its ``conflict_type`` is
    only overwritten when ``new_type`` has a strictly higher priority
    than the current one (per :data:`CONFLICT_PRIORITY`).  Any extra
    ``attrs`` (e.g. ``distance=...``) are written when the edge is
    created or when its type is promoted; weaker promotions leave the
    edge untouched.

    Unknown conflict types are treated as priority 0 (always lose to a
    known one).  This keeps the helper future-proof but logs no
    silent mismatch.
    """
    if not graph.has_edge(u, v):
        graph.add_edge(u, v, conflict_type=new_type, **attrs)
        return

    cur_type = graph[u][v].get("conflict_type", "")
    cur_pri = CONFLICT_PRIORITY.get(cur_type, 0)
    new_pri = CONFLICT_PRIORITY.get(new_type, 0)
    if new_pri <= cur_pri:
        return

    graph[u][v]["conflict_type"] = new_type
    for k, val in attrs.items():
        graph[u][v][k] = val
