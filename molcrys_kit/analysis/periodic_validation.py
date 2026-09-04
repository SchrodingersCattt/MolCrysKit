"""Library validation for periodic geometry bundles."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from ..structures.periodic_geometry import PeriodicBundle, PeriodicEdge, PeriodicGraph

def validate_periodic_bundle(bundle: PeriodicBundle|Any, metadata: Mapping[str,Any]|None=None, *, tolerance: float=1e-6):
    atoms=bundle.atoms if isinstance(bundle,PeriodicBundle) else bundle; graph=bundle.graph if isinstance(bundle,PeriodicBundle) else None; cell=atoms.cell.array
    if graph is None and metadata is not None and metadata.get("periodic_graph"):
        raw=metadata["periodic_graph"]
        graph=PeriodicGraph(tuple(raw["nodes"]),tuple(PeriodicEdge(e["left_node"],e["right_node"],tuple(e["right_image_shift"]),e.get("rule_id",""),bool(e.get("closure",False))) for e in raw["edges"]),raw.get("closure","translation"))
    if cell.shape!=(3,3) or abs(np.linalg.det(cell))<1e-12: raise ValueError("periodic bundle requires a non-singular 3x3 cell")
    if not np.all(np.isfinite(atoms.positions)): raise ValueError("periodic bundle contains non-finite coordinates")
    required={"atom_id","chain_id","fragment_id","repeat_id"}; missing=sorted(required-set(atoms.arrays))
    if missing: raise ValueError(f"periodic bundle is missing required arrays: {', '.join(missing)}")
    if metadata is not None and metadata.get("atom_count") not in (None,len(atoms)): raise ValueError("sidecar atom_count does not match ExtXYZ")
    frac=atoms.positions@np.linalg.inv(cell)
    for axis,periodic in enumerate(atoms.pbc):
        if not periodic and np.any((frac[:,axis]<-tolerance)|(frac[:,axis]>1+tolerance)): raise ValueError("coordinates leave a non-periodic cell direction")
    if graph is not None and graph.cycle_rank<1: raise ValueError("periodic graph has no closure cycle")
    return {"ok":True,"atom_count":int(len(atoms)),"array_names":sorted(atoms.arrays),"cell_volume_A3":float(abs(np.linalg.det(cell))),"pbc":[bool(x) for x in atoms.pbc],"cycle_rank":graph.cycle_rank if graph is not None else None,"winding_cycles":[list(v) for v in graph.winding_cycles()] if graph is not None else []}

__all__=["validate_periodic_bundle"]
