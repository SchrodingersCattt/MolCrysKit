"""CLI commands for periodic geometry bundles."""

from __future__ import annotations
import json
from pathlib import Path
import click
from ..analysis.periodic_validation import validate_periodic_bundle
from ..io.periodic_bundle import read_periodic_bundle, write_periodic_bundle
from ..operations.periodic_chain import build_periodic_chains
from ..structures.periodic_geometry import (
    BoundaryPort,
    ChainSpec,
    ConnectionRule,
    FragmentTemplate,
    ScrewSpec,
)


def load_chain_request(config: str | Path):
    """Load and decode a JSON periodic-chain request for CLI and tests."""
    path = Path(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level request must be a JSON object")
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("request.templates must be a non-empty list")
    raw_spec = payload.get("spec")
    if not isinstance(raw_spec, dict):
        raise ValueError("request.spec must be an object")
    templates = {}
    for item in raw_templates:
        if not isinstance(item, dict):
            raise ValueError("every request.templates item must be an object")
        template = _template(item)
        if template.template_id in templates:
            raise ValueError(f"duplicate template_id {template.template_id!r}")
        templates[template.template_id] = template
    rules = tuple(_rule(item) for item in payload.get("rules", ()))
    screw = ScrewSpec(**raw_spec["screw"]) if raw_spec.get("screw") else None
    spec = ChainSpec(
        sequence=tuple(raw_spec["sequence"]),
        chain_count=raw_spec.get("chain_count", 1),
        closure=raw_spec.get("closure", "translation"),
        target_winding=(
            tuple(raw_spec["target_winding"])
            if raw_spec.get("target_winding") is not None
            else None
        ),
        instance_centers=(
            tuple(tuple(item) for item in raw_spec["instance_centers"])
            if raw_spec.get("instance_centers") is not None
            else None
        ),
        screw=screw,
        seed=raw_spec.get("seed", 0),
        max_backtracks=raw_spec.get("max_backtracks", 64),
        min_distance=raw_spec.get("min_distance", 0.8),
        tolerance=raw_spec.get("tolerance", 1e-6),
        chain_centers=(
            tuple(tuple(item) for item in raw_spec["chain_centers"])
            if raw_spec.get("chain_centers") is not None
            else None
        ),
    )
    return payload, templates, rules, spec


def _template(item):
    ports = tuple(
        BoundaryPort(
            p["port_id"],
            tuple(p["position"]),
            tuple(p.get("faces", ())),
            tuple(p["direction"]) if p.get("direction") is not None else None,
            tuple(p.get("rule_ids", ())),
        )
        for p in item.get("ports", ())
    )
    return FragmentTemplate(
        item["template_id"],
        tuple(item["symbols"]),
        tuple(tuple(x) for x in item["positions"]),
        ports,
        tuple(tuple(x) for x in item.get("explicit_connections", ())),
        item.get("metadata", {}),
    )


def _rule(item):
    return ConnectionRule(
        item["rule_id"],
        item["left_template"],
        item["left_port"],
        item["right_template"],
        item["right_port"],
        tuple(tuple(x) for x in item.get("allowed_image_shifts", ((0, 0, 0),))),
        tuple(item["distance_range"])
        if item.get("distance_range") is not None
        else None,
        tuple(item["angle_range_deg"])
        if item.get("angle_range_deg") is not None
        else None,
        tuple(item["dihedral_range_deg"])
        if item.get("dihedral_range_deg") is not None
        else None,
    )


@click.group("build")
def build_group():
    """Build geometry-native structures."""


@build_group.command("chain")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--format",
    "format_name",
    type=click.Choice(("cif", "poscar", "xyz", "extxyz"), case_sensitive=False),
    default=None,
    help="Structure format; defaults to CIF when output is a directory.",
)
@click.option("--overwrite", is_flag=True)
def build_chain(config: Path, output: Path, format_name: str | None, overwrite: bool):
    try:
        payload, templates, rules, spec = load_chain_request(config)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise click.ClickException(
            f"Invalid chain request {config}: {error}"
        ) from error
    structure, sidecar = write_periodic_bundle(
        build_periodic_chains(
            templates,
            rules,
            payload["cell"],
            payload.get("pbc", (True, True, True)),
            spec,
        ),
        output,
        format=format_name,
        overwrite=overwrite,
    )
    click.echo(f"Wrote {structure}")
    click.echo(f"Wrote {sidecar}")


@click.command("validate-periodic-bundle")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True)
def validate_bundle(input: Path, as_json: bool):
    atoms, metadata = read_periodic_bundle(input)
    report = validate_periodic_bundle(atoms, metadata)
    click.echo(
        json.dumps(report, indent=2) if as_json else f"OK: {report['atom_count']} atoms"
    )


def register_periodic_chain_commands(group: click.Group):
    build_group.add_command(build_chain)
    group.add_command(build_group)
    group.add_command(validate_bundle)


__all__ = ["load_chain_request", "register_periodic_chain_commands"]
