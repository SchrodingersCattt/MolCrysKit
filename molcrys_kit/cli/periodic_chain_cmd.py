"""CLI commands for periodic geometry bundles."""
from __future__ import annotations
import json
from pathlib import Path
import click
from ..analysis.periodic_validation import validate_periodic_bundle
from ..io.periodic_bundle import read_periodic_bundle, write_periodic_bundle
from ..operations.periodic_chain import build_periodic_chains
from ..structures.periodic_geometry import BoundaryPort, ChainSpec, ConnectionRule, FragmentTemplate, ScrewSpec

def _template(item):
    ports=tuple(BoundaryPort(p["port_id"],tuple(p["position"]),tuple(p.get("faces",())),tuple(p["direction"]) if p.get("direction") is not None else None,tuple(p.get("rule_ids",()))) for p in item.get("ports",()))
    return FragmentTemplate(item["template_id"],tuple(item["symbols"]),tuple(tuple(x) for x in item["positions"]),ports,tuple(tuple(x) for x in item.get("explicit_connections",())),item.get("metadata",{}))

def _rule(item):
    return ConnectionRule(item["rule_id"],item["left_template"],item["left_port"],item["right_template"],item["right_port"],tuple(tuple(x) for x in item.get("allowed_image_shifts",((0,0,0),))),tuple(item["distance_range"]) if item.get("distance_range") is not None else None,tuple(item["angle_range_deg"]) if item.get("angle_range_deg") is not None else None,tuple(item["dihedral_range_deg"]) if item.get("dihedral_range_deg") is not None else None)

@click.group("build")
def build_group():
    """Build geometry-native structures."""

@build_group.command("chain")
@click.argument("config",type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("-o","--output",required=True,type=click.Path(path_type=Path))
@click.option("--format", "format_name", type=click.Choice(("cif", "poscar", "xyz", "extxyz"), case_sensitive=False), default=None, help="Structure format; defaults to CIF when output is a directory.")
@click.option("--overwrite",is_flag=True)
def build_chain(config: Path, output: Path, format_name: str|None, overwrite: bool):
    payload=json.loads(config.read_text(encoding="utf-8")); templates={x["template_id"]:_template(x) for x in payload["templates"]}; rules=tuple(_rule(x) for x in payload.get("rules",()))
    raw=payload["spec"]; screw=ScrewSpec(**raw["screw"]) if raw.get("screw") else None
    spec=ChainSpec(tuple(raw["sequence"]),raw.get("chain_count",1),raw.get("closure","translation"),tuple(raw["target_winding"]) if raw.get("target_winding") is not None else None,tuple(tuple(x) for x in raw["instance_centers"]) if raw.get("instance_centers") is not None else None,screw,raw.get("seed",0),raw.get("max_backtracks",64),raw.get("min_distance",0.8),raw.get("tolerance",1e-6),tuple(tuple(x) for x in raw["chain_centers"]) if raw.get("chain_centers") is not None else None)
    structure,sidecar=write_periodic_bundle(build_periodic_chains(templates,rules,payload["cell"],payload.get("pbc",(True,True,True)),spec),output,format=format_name,overwrite=overwrite); click.echo(f"Wrote {structure}"); click.echo(f"Wrote {sidecar}")

@click.command("validate-periodic-bundle")
@click.argument("input",type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--json","as_json",is_flag=True)
def validate_bundle(input: Path, as_json: bool):
    atoms,metadata=read_periodic_bundle(input); report=validate_periodic_bundle(atoms,metadata); click.echo(json.dumps(report,indent=2) if as_json else f"OK: {report['atom_count']} atoms")

def register_periodic_chain_commands(group: click.Group):
    build_group.add_command(build_chain); group.add_command(build_group); group.add_command(validate_bundle)
