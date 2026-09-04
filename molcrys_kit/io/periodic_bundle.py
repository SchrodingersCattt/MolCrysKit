"""Paired structure/JSON I/O for geometry-native periodic bundles."""
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

_FORMATS={"cif":"cif","extxyz":"extxyz","xyz":"xyz","poscar":"vasp"}
_SUFFIXES={".cif":"cif",".extxyz":"extxyz",".xyz":"xyz",".poscar":"poscar",".vasp":"poscar"}

def _normalise_format(value: str|None):
    key=(value or "").lower().replace("-","")
    aliases={"extendedxyz":"extxyz","vasp":"poscar","contcar":"poscar"}
    key=aliases.get(key,key)
    if key not in _FORMATS: raise ValueError(f"unsupported periodic bundle format {value!r}; expected one of {tuple(_FORMATS)}")
    return key

def _output_path(output: str|Path, format_name: str|None):
    path=Path(output)
    inferred=_SUFFIXES.get(path.suffix.lower())
    selected=_normalise_format(format_name) if format_name is not None else (inferred or "cif")
    if inferred is not None and format_name is not None and selected != inferred:
        raise ValueError(f"output suffix {path.suffix!r} conflicts with format {selected!r}")
    is_file=bool(inferred) or (path.exists() and path.is_file())
    if is_file: return path,selected
    return path/f"structure.{ 'vasp' if selected=='poscar' else selected }",selected

def write_periodic_bundle(bundle: PeriodicBundle, output: str|Path, *, format: str|None=None, overwrite: bool=False):
    structure,format_name=_output_path(output,format)
    structure.parent.mkdir(parents=True,exist_ok=True)
    sidecar=structure.with_suffix(".json")
    if not overwrite and (structure.exists() or sidecar.exists()): raise FileExistsError(f"bundle output exists: {structure}")
    if format_name=="extxyz":
        ase.io.write(structure,bundle.atoms,format="extxyz",write_info=True,write_results=False)
    else:
        ase.io.write(structure,bundle.atoms,format=_FORMATS[format_name])
    digest=_sha256(structure)
    files={"structure":structure.name,"format":format_name,"structure_sha256":digest}
    if format_name=="extxyz": files.update({"extxyz":structure.name,"extxyz_sha256":digest})
    payload=dict(bundle.metadata); payload.update({"files":files,"atom_count":len(bundle.atoms),"instances":[{"instance_id":i.instance_id,"template_id":i.template_id,"chain_id":i.chain_id,"repeat_id":i.repeat_id,"rotation":[list(row) for row in i.rotation],"translation":list(i.translation)} for i in bundle.instances],"periodic_graph":_graph(bundle.graph),"validation":bundle.validation})
    sidecar.write_text(json.dumps(payload,indent=2,sort_keys=True,default=_default)+"\n",encoding="utf-8")
    return structure,sidecar

def read_periodic_bundle(structure: str|Path, sidecar: str|Path|None=None):
    structure=Path(structure); sidecar_path=Path(sidecar) if sidecar is not None else structure.with_suffix(".json")
    payload=json.loads(sidecar_path.read_text(encoding="utf-8")); files=payload.get("files",{})
    format_name=_normalise_format(files.get("format") or _SUFFIXES.get(structure.suffix.lower()) or "extxyz")
    atoms=ase.io.read(structure,format=_FORMATS[format_name],index=0); expected=files.get("structure_sha256") or files.get("extxyz_sha256")
    if expected and expected != _sha256(structure): raise ValueError("periodic bundle checksum mismatch: structure changed after sidecar creation")
    return atoms,payload

__all__=["read_periodic_bundle","write_periodic_bundle"]
