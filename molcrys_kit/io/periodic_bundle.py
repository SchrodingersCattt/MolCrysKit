"""Paired ExtXYZ/JSON I/O for geometry-native periodic bundles."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import ase.io
import numpy as np
from ..structures.periodic_geometry import PeriodicBundle, PeriodicGraph

def _default(value: Any):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating)): return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def _graph(graph: PeriodicGraph):
    return {"nodes":list(graph.nodes),"edges":[{"left_node":e.left_node,"right_node":e.right_node,"right_image_shift":list(e.right_image_shift),"rule_id":e.rule_id,"closure":e.closure} for e in graph.edges],"closure":graph.closure,"cycle_rank":graph.cycle_rank,"winding_cycles":[list(v) for v in graph.winding_cycles()]}

def write_periodic_bundle(bundle: PeriodicBundle, output: str|Path, *, overwrite: bool=False):
    path=Path(output)
    if path.suffix.lower() not in {".xyz",".extxyz"}: path.mkdir(parents=True,exist_ok=True); xyz=path/"structure.extxyz"
    else: xyz=path; xyz.parent.mkdir(parents=True,exist_ok=True)
    sidecar=xyz.with_suffix(".json")
    if not overwrite and (xyz.exists() or sidecar.exists()): raise FileExistsError(f"bundle output exists: {xyz}")
    ase.io.write(xyz,bundle.atoms,format="extxyz",write_info=True,write_results=False)
    payload=dict(bundle.metadata); payload.update({"files":{"extxyz":xyz.name,"extxyz_sha256":_sha256(xyz)},"atom_count":len(bundle.atoms),"instances":[{"instance_id":i.instance_id,"template_id":i.template_id,"chain_id":i.chain_id,"repeat_id":i.repeat_id,"rotation":[list(row) for row in i.rotation],"translation":list(i.translation)} for i in bundle.instances],"periodic_graph":_graph(bundle.graph),"validation":bundle.validation})
    sidecar.write_text(json.dumps(payload,indent=2,sort_keys=True,default=_default)+"\n",encoding="utf-8")
    return xyz,sidecar

def read_periodic_bundle(extxyz: str|Path, sidecar: str|Path|None=None):
    xyz=Path(extxyz); sidecar_path=Path(sidecar) if sidecar is not None else xyz.with_suffix(".json")
    atoms=ase.io.read(xyz,format="extxyz",index=0); payload=json.loads(sidecar_path.read_text(encoding="utf-8")); expected=payload.get("files",{}).get("extxyz_sha256")
    if expected and expected != _sha256(xyz): raise ValueError("periodic bundle checksum mismatch: ExtXYZ changed after sidecar creation")
    return atoms,payload

__all__=["read_periodic_bundle","write_periodic_bundle"]
