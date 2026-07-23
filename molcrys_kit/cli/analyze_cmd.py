"""Analysis commands for the MolCrysKit CLI."""

from __future__ import annotations

from pathlib import Path

import click

from molcrys_kit.analysis import enumerate_bfdh_facets, find_polyhedra
from molcrys_kit.analysis.interactions import interaction_profile

from ._common import load_crystal, rows_to_json


def _miller_text(value) -> str:
    return "(" + " ".join(str(int(x)) for x in value) + ")"


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-index", type=int, default=2, show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def bfdh(input: Path, max_index: int, top_n: int, as_json: bool) -> None:
    """Rank low-index facets using BFDH morphology."""
    if max_index < 1:
        raise click.UsageError("--max-index must be >= 1.")
    if top_n < 1:
        raise click.UsageError("--top-n must be >= 1.")
    facets = enumerate_bfdh_facets(load_crystal(input), max_index=max_index, top_n=top_n)
    if as_json:
        click.echo(rows_to_json(facets))
        return

    click.echo("miller       d_hkl      importance")
    click.echo("-----------------------------------")
    for facet in facets:
        importance = getattr(facet, "morphological_importance", getattr(facet, "importance", ""))
        click.echo(f"{_miller_text(facet.miller_index):<12} {facet.d_hkl:>8.4f} {importance}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def interactions(input: Path, as_json: bool) -> None:
    """Summarize weak interactions and continuous scores."""
    profile = interaction_profile(load_crystal(input))
    if as_json:
        click.echo(rows_to_json(profile))
        return

    click.echo("Interaction profile")
    for key, value in profile.__dict__.items():
        if isinstance(value, (int, float, str, bool)):
            click.echo(f"  {key}: {value}")
        elif isinstance(value, list):
            click.echo(f"  {key}: {len(value)}")
        else:
            click.echo(f"  {key}: {value}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--central", required=True, help="Central atom symbol or moiety string.")
@click.option("--ligand", required=True, help="Ligand atom symbol or moiety string.")
@click.option("--level", type=click.Choice(["atom", "molecule"]), default="atom", show_default=True)
@click.option("--cutoff", type=float, default=None, help="Coordination cutoff in Angstrom.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def polyhedra(input: Path, central: str, ligand: str, level: str, cutoff: float | None, as_json: bool) -> None:
    """Enumerate coordination polyhedra."""
    if cutoff is not None and cutoff <= 0:
        raise click.UsageError("--cutoff must be positive.")
    rows = find_polyhedra(load_crystal(input), central=central, ligand=ligand, level=level, cutoff=cutoff)
    if as_json:
        click.echo(rows_to_json(rows))
        return

    click.echo(f"Found {len(rows)} polyhedra")
    for idx, row in enumerate(rows):
        center = row.get("center_index", row.get("central_index", idx))
        cn = row.get("coordination_number", row.get("cn", "?"))
        shape = row.get("shape", row.get("best_shape", ""))
        click.echo(f"  [{idx}] center={center} cn={cn} shape={shape}")


@click.command("sanity-check")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--checks", type=str, default=None, help="Comma-separated list of checks to run (default: all).")
@click.option("--hard-clash-scale", type=float, default=None, help="Scale factor for hard clash detection.")
@click.option("--hard-clash-tolerance", type=float, default=None, help="Absolute tolerance for hard clash.")
@click.option("--intermolecular-clash-scale", type=float, default=None, help="Scale factor for intermolecular clash.")
@click.option("--intermolecular-clash-tolerance", type=float, default=None, help="Tolerance for intermolecular clash.")
@click.option("--ignore-hh/--no-ignore-hh", default=None, help="Ignore H-H intermolecular clashes.")
@click.option("--max-clashes", type=int, default=None, help="Maximum allowed intermolecular clashes.")
@click.option("--bond-min-factor", type=float, default=None, help="Min factor for bond distance check.")
@click.option("--bond-max-factor", type=float, default=None, help="Max factor for bond distance check.")
@click.option("--isolated-elements", type=str, default=None, help="Comma-separated elements for isolated atom check.")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write JSON report to file.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of table to stdout.")
def sanity_check_cmd(
    input: Path,
    checks: str | None,
    hard_clash_scale: float | None,
    hard_clash_tolerance: float | None,
    intermolecular_clash_scale: float | None,
    intermolecular_clash_tolerance: float | None,
    ignore_hh: bool | None,
    max_clashes: int | None,
    bond_min_factor: float | None,
    bond_max_factor: float | None,
    isolated_elements: str | None,
    output: Path | None,
    as_json: bool,
) -> None:
    """Run structural sanity checks on an ExtXYZ file.

    Supports multi-frame files: each frame is checked independently.
    """
    import json as json_mod

    from molcrys_kit.analysis.sanity_check import sanity_check
    from molcrys_kit.io.extxyz import read_extxyz

    # Parse options
    check_list = [c.strip() for c in checks.split(",")] if checks else None
    iso_elems = set(e.strip() for e in isolated_elements.split(",")) if isolated_elements else None

    # Load frames
    frames = read_extxyz(str(input))
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise click.ClickException(f"No frames found in {input}")

    all_reports: list[dict] = []
    n_pass = 0

    for idx, crystal in enumerate(frames):
        report = sanity_check(
            crystal,
            checks=check_list,
            hard_clash_scale=hard_clash_scale,
            hard_clash_tolerance=hard_clash_tolerance,
            intermolecular_clash_scale=intermolecular_clash_scale,
            intermolecular_clash_tolerance=intermolecular_clash_tolerance,
            ignore_hh=ignore_hh,
            max_clashes=max_clashes,
            bond_distance_min_factor=bond_min_factor,
            bond_distance_max_factor=bond_max_factor,
            isolated_elements=iso_elems,
        )
        if report.passed:
            n_pass += 1
        all_reports.append({"frame": idx, **report.to_dict()})

    # Output
    n_total = len(frames)
    summary_data = {
        "file": str(input),
        "n_frames": n_total,
        "n_passed": n_pass,
        "n_failed": n_total - n_pass,
        "frames": all_reports,
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_mod.dumps(summary_data, indent=2, default=str))
        click.echo(f"Report written to {output}")

    if as_json:
        click.echo(json_mod.dumps(summary_data, indent=2, default=str))
    else:
        click.echo(f"Sanity check: {n_pass}/{n_total} frames passed.")
        # Show per-frame failures (limit output for large files)
        failed_frames = [r for r in all_reports if not r["passed"]]
        if failed_frames:
            show_limit = min(10, len(failed_frames))
            for r in failed_frames[:show_limit]:
                frame_idx = r["frame"]
                failed_checks = [c["name"] for c in r["results"] if not c["passed"]]
                click.echo(f"  ✗ frame {frame_idx}: {', '.join(failed_checks)}")
            if len(failed_frames) > show_limit:
                click.echo(f"  ... and {len(failed_frames) - show_limit} more.")
        else:
            click.echo("  All frames passed.")


def register_analyze_commands(group: click.Group) -> None:
    group.add_command(bfdh)
    group.add_command(interactions)
    group.add_command(polyhedra)
    group.add_command(sanity_check_cmd)
