"""Library validation for periodic geometry bundles."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from ..structures.periodic_geometry import PeriodicBundle, PeriodicEdge, PeriodicGraph

def validate_periodic_bundle(bundle: PeriodicBundle|Any, metadata: Mapping[str,Any]|None=None, *, tolerance: float=1e-6):
    atoms=bundle.atoms if isinstance(bundle,PeriodicBundle) else bundle; graph=bundle.graph if isinstance(bundle,PeriodicBundle) else None; cell=atoms.cell.array
    if graph is None and metadata is not None and metadata.get("periodic_graph"):
        raw=metadata["periodic_graph"]
        graph=PeriodicGraph(tuple(raw["nodes"]),tuple(PeriodicEdge(e["left_node"],e["right_node"],tuple(e["right_image_shift"]),e.get("rule_id",""),bool(e.get("closure",False)),e.get("left_port"),e.get("right_port")) for e in raw["edges"]),raw.get("closure","translation"))
    if cell.shape!=(3,3) or abs(np.linalg.det(cell))<1e-12: raise ValueError("periodic bundle requires a non-singular 3x3 cell")
    if not np.all(np.isfinite(atoms.positions)): raise ValueError("periodic bundle contains non-finite coordinates")
    required={"atom_id","chain_id","fragment_id","repeat_id"}; missing=sorted(required-set(atoms.arrays))
    if missing: raise ValueError(f"periodic bundle is missing required arrays: {', '.join(missing)}")
    if metadata is not None and metadata.get("atom_count") not in (None,len(atoms)): raise ValueError("sidecar atom_count does not match structure")
    if len(set(np.asarray(atoms.arrays["atom_id"]).tolist())) != len(atoms): raise ValueError("periodic bundle atom_id values must be unique")
    if metadata is not None and metadata.get("atom_records"):
        records=metadata["atom_records"]
        if len(records)!=len(atoms): raise ValueError("sidecar atom_records do not match structure atom count")
        expected_symbols=[record.get("symbol") for record in records]
        if all(symbol is not None for symbol in expected_symbols) and atoms.get_chemical_symbols()!=expected_symbols: raise ValueError("periodic bundle symbols do not match sidecar atom order")
        for name in required:
            expected=np.asarray([record[name] for record in records],dtype="U64" if name=="fragment_id" else int)
            if not np.array_equal(np.asarray(atoms.arrays[name]),expected): raise ValueError(f"periodic bundle array {name!r} does not match sidecar")
    frac=atoms.positions@np.linalg.inv(cell)
    for axis,periodic in enumerate(atoms.pbc):
        if not periodic and np.any((frac[:,axis]<-tolerance)|(frac[:,axis]>1+tolerance)): raise ValueError("coordinates leave a non-periodic cell direction")
    if graph is not None:
        if graph.cycle_rank<1: raise ValueError("periodic graph has no closure cycle")
        for edge in graph.edges:
            if (edge.left_node == edge.right_node and edge.right_image_shift == (0, 0, 0)
                    and edge.left_port is not None and edge.left_port == edge.right_port):
                raise ValueError("periodic graph contains a degenerate zero-winding self-loop")
            if any(not atoms.pbc[axis] and edge.right_image_shift[axis] != 0 for axis in range(3)):
                raise ValueError("periodic graph crosses a non-periodic cell direction")
    return {"ok":True,"atom_count":int(len(atoms)),"array_names":sorted(atoms.arrays),"cell_volume_A3":float(abs(np.linalg.det(cell))),"pbc":[bool(x) for x in atoms.pbc],"cycle_rank":graph.cycle_rank if graph is not None else None,"winding_cycles":[list(v) for v in graph.winding_cycles()] if graph is not None else []}

__all__=["validate_periodic_bundle"]
