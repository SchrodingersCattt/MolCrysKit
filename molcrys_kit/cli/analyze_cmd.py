"""Analysis commands for the MolCrysKit CLI."""

from __future__ import annotations

from pathlib import Path

import click

from ._common import load_crystal, rows_to_json


def _miller_text(value) -> str:
    return "(" + " ".join(str(int(x)) for x in value) + ")"


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--symprec", type=float, default=0.1, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def summary(input: Path, symprec: float, as_json: bool) -> None:
    """Summarize composition, cell, symmetry, and disorder."""
    from molcrys_kit.analysis import summarize_structure

    if symprec <= 0:
        raise click.UsageError("--symprec must be positive.")
    report = summarize_structure(
        load_crystal(input), source=str(input), symprec=symprec
    )
    if as_json:
        click.echo(rows_to_json(report))
        return

    cell = report["cell"]
    symmetry = report["symmetry"]
    disorder = report["disorder"]
    click.echo(f"Structure summary: {input}")
    click.echo(
        f"  Reduced site formula: {report['formula']} ({report['n_atoms']} sites)"
    )
    click.echo(
        "  Species: "
        + ", ".join(
            f"{symbol}={count}" for symbol, count in report["species_counts"].items()
        )
    )
    click.echo(f"  PBC: {tuple(report['pbc'])}")
    click.echo(
        "  Cell: "
        + " ".join(
            f"{name}={value:.4f}"
            for name, value in zip("abc", cell["lengths_A"], strict=True)
        )
        + " A; "
        + " ".join(
            f"{name}={value:.2f}"
            for name, value in zip(
                ("alpha", "beta", "gamma"), cell["angles_deg"], strict=True
            )
        )
        + f" deg; V={cell['volume_A3']:.4f} A^3"
    )
    if symmetry and symmetry["status"] == "ok":
        click.echo(
            "  Symmetry: "
            f"{symmetry['space_group_symbol']} "
            f"(#{symmetry['space_group_number']}, {symmetry['crystal_system']})"
        )
        wyckoff = ", ".join(
            f"{'/'.join(site['species'])}:{site['symbol']}"
            for site in symmetry["wyckoff_sites"]
        )
        click.echo(f"  Wyckoff: {wyckoff}")
    elif symmetry:
        click.echo(f"  Symmetry: unavailable ({symmetry['reason']})")
    else:
        reason = (
            "requires full 3D periodicity"
            if any(report["pbc"])
            else "nonperiodic structure"
        )
        click.echo(f"  Symmetry: not evaluated ({reason})")
    click.echo(
        "  Disorder: "
        + (
            f"partial_sites={disorder['partial_occupancy_sites']}, "
            f"groups={disorder['groups']}, assemblies={disorder['assemblies']}"
            if disorder["has_disorder"]
            else "none detected"
        )
    )


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-index", type=int, default=2, show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def bfdh(input: Path, max_index: int, top_n: int, as_json: bool) -> None:
    """Rank low-index facets using BFDH morphology."""
    from molcrys_kit.analysis import enumerate_bfdh_facets

    if max_index < 1:
        raise click.UsageError("--max-index must be >= 1.")
    if top_n < 1:
        raise click.UsageError("--top-n must be >= 1.")
    facets = enumerate_bfdh_facets(
        load_crystal(input), max_index=max_index, top_n=top_n
    )
    if as_json:
        click.echo(rows_to_json(facets))
        return

    click.echo("miller       d_hkl      importance")
    click.echo("-----------------------------------")
    for facet in facets:
        importance = getattr(
            facet, "morphological_importance", getattr(facet, "importance", "")
        )
        click.echo(
            f"{_miller_text(facet.miller_index):<12} {facet.d_hkl:>8.4f} {importance}"
        )


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def interactions(input: Path, as_json: bool) -> None:
    """Summarize weak interactions and continuous scores."""
    from molcrys_kit.analysis.interactions import interaction_profile

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


@click.command("bond-orders")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json", "as_json", is_flag=True, help="Print the auditable result as JSON."
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail when a complete, unambiguous assignment is unavailable.",
)
@click.option(
    "--charge",
    type=int,
    default=None,
    help="Require this total formal charge per component.",
)
def bond_orders(input: Path, as_json: bool, strict: bool, charge: int | None) -> None:
    """Infer covalent bond orders and formal charges for INPUT.

    Perception runs on a copy after connectivity has been annotated, so this
    reporting command never changes the loaded structure.  It calls the same
    immutable entity solver exposed through ``mck.chemistry``.
    """
    import json as json_mod

    from molcrys_kit.chemistry import (
        BondOrderOptions,
        annotate_chemistry,
        infer_bond_orders,
    )

    try:
        # ``MolecularCrystal.copy`` preserves stable atom IDs while isolating
        # the chemistry attachment performed by annotate_chemistry.
        crystal = load_crystal(input).copy()
        chemistry = annotate_chemistry(crystal)
        options = BondOrderOptions(strict=strict)
        reports = [
            infer_bond_orders(
                component,
                total_charge=charge,
                options=options,
            )
            for component in chemistry.components
        ]
    except (TypeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    statuses = [_status_value(report) for report in reports]
    payload = {
        "file": str(input),
        "status": "indeterminate"
        if not statuses
        else (statuses[0] if len(set(statuses)) == 1 else "mixed"),
        "components": [
            report.to_dict() if hasattr(report, "to_dict") else report
            for report in reports
        ],
    }
    if as_json:
        click.echo(json_mod.dumps(payload, indent=2, default=_json_default))
        return

    click.echo(f"Bond-order perception: {input}")
    click.echo(f"  Components: {len(reports)}")
    for index, report in enumerate(reports):
        entity = getattr(report, "entity", None)
        status = _status_value(report)
        bonds = getattr(entity, "bonds", ()) if entity is not None else ()
        orders = ", ".join(
            f"{bond.atom1_id}-{bond.atom2_id}:{bond.order:g}"
            for bond in bonds
            if getattr(bond, "order", None) is not None
        )
        click.echo(f"  [{index}] status={status}; bonds={orders or 'none'}")


def _status_value(value) -> str:
    status = getattr(value, "status", value)
    return str(getattr(status, "value", status))


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


@click.command("cell-integrity")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json", "as_json", is_flag=True, help="Print JSON instead of a summary."
)
def cell_integrity(input: Path, as_json: bool) -> None:
    """Audit lattice handedness, molecule images, and transform provenance."""
    import json as json_mod

    from molcrys_kit.analysis import check_cell_integrity

    if input.suffix.lower() == ".extxyz":
        from molcrys_kit.io.extxyz import read_extxyz

        loaded = read_extxyz(str(input), index=":")
        frames = loaded if isinstance(loaded, list) else [loaded]
    else:
        frames = [load_crystal(input)]
    reports = [
        {"frame": index, **check_cell_integrity(crystal).to_dict()}
        for index, crystal in enumerate(frames)
    ]
    payload = {"file": str(input), "n_frames": len(reports), "frames": reports}
    if as_json:
        click.echo(json_mod.dumps(payload, indent=2, default=str))
        return
    n_passed = sum(bool(item["passed"]) for item in reports)
    click.echo(f"Cell integrity: {n_passed}/{len(reports)} frames passed.")
    for item in reports:
        if not item["passed"]:
            failed = [name for name, passed in item["checks"].items() if not passed]
            click.echo(f"  ✗ frame {item['frame']}: {', '.join(failed)}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--central", required=True, help="Central atom symbol or moiety string.")
@click.option("--ligand", required=True, help="Ligand atom symbol or moiety string.")
@click.option(
    "--level",
    type=click.Choice(["atom", "molecule"]),
    default="atom",
    show_default=True,
)
@click.option(
    "--cutoff", type=float, default=None, help="Coordination cutoff in Angstrom."
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def polyhedra(
    input: Path,
    central: str,
    ligand: str,
    level: str,
    cutoff: float | None,
    as_json: bool,
) -> None:
    """Enumerate coordination polyhedra."""
    if cutoff is not None and cutoff <= 0:
        raise click.UsageError("--cutoff must be positive.")
    from molcrys_kit.analysis import find_polyhedra

    rows = find_polyhedra(
        load_crystal(input), central=central, ligand=ligand, level=level, cutoff=cutoff
    )
    if as_json:
        click.echo(rows_to_json(rows))
        return

    click.echo(f"Found {len(rows)} polyhedra")
    for idx, row in enumerate(rows):
        center = row.get("center_index", row.get("central_index", idx))
        cn = row.get("coordination_number", row.get("cn", "?"))
        shape = row.get("shape", row.get("best_shape", ""))
        click.echo(f"  [{idx}] center={center} cn={cn} shape={shape}")


@click.command("polyhedron-network")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--central", required=True, help="Central moiety string.")
@click.option("--ligand", required=True, help="Ligand moiety string.")
@click.option(
    "--center-kind",
    type=click.Choice(["centroid", "com", "heavy_centroid"]),
    default="centroid",
    show_default=True,
)
@click.option(
    "--cutoff", type=float, default=None, help="Candidate search radius in Angstrom."
)
@click.option(
    "--hard-cutoff", type=float, default=None, help="Hard shell radius in Angstrom."
)
@click.option(
    "--central-species", default=None, help="Replica-local central species id."
)
@click.option("--ligand-species", default=None, help="Replica-local ligand species id.")
@click.option(
    "--central-index",
    type=int,
    multiple=True,
    help="Central molecule index; repeatable.",
)
@click.option(
    "--ligand-index", type=int, multiple=True, help="Ligand molecule index; repeatable."
)
@click.option(
    "--disorder-policy",
    type=click.Choice(["major", "exclusive", "as_refined"]),
    default="major",
    show_default=True,
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print JSON instead of a summary."
)
def polyhedron_network(
    input: Path,
    central: str,
    ligand: str,
    center_kind: str,
    cutoff: float | None,
    hard_cutoff: float | None,
    central_species: str | None,
    ligand_species: str | None,
    central_index: tuple[int, ...],
    ligand_index: tuple[int, ...],
    disorder_policy: str,
    as_json: bool,
) -> None:
    """Report one periodic coordination-polyhedron network for a single centre type."""
    if cutoff is not None and cutoff <= 0:
        raise click.UsageError("--cutoff must be positive.")
    if hard_cutoff is not None and hard_cutoff <= 0:
        raise click.UsageError("--hard-cutoff must be positive.")
    from molcrys_kit.analysis import molecular_polyhedron_network

    report = molecular_polyhedron_network(
        load_crystal(input),
        central,
        ligand,
        center_kind=center_kind,
        cutoff=cutoff,
        hard_cutoff=hard_cutoff,
        central_species=central_species,
        ligand_species=ligand_species,
        central_indices=list(central_index) or None,
        ligand_indices=list(ligand_index) or None,
        disorder_policy=disorder_policy,
    )
    if as_json:
        click.echo(rows_to_json(report))
        return
    click.echo(f"Polyhedron networks: {len(report.networks)}")
    click.echo(f"Configuration: {report.configuration_status}")
    click.echo(f"Centre disorder: {report.centre_disorder}")
    click.echo(f"Ligand disorder: {report.ligand_disorder}")
    for warning in report.warnings:
        click.echo(f"Warning: {warning}")
    for index, network in enumerate(report.networks):
        ranks = [component.rank for component in network.components]
        click.echo(
            f"Network {index}: shells={len(network.shells)} "
            f"contacts={len(network.contacts)} "
            f"components={len(network.components)} ranks={ranks}"
        )
        for shell in network.shells:
            click.echo(
                f"  centre molecule={shell.molecule_index} "
                f"realization={shell.realization_index} "
                f"cn={len(shell.vertices)} species={shell.species_id}"
            )


@click.command("sanity-check")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--checks",
    type=str,
    default=None,
    help="Comma-separated list of checks to run (default: all).",
)
@click.option(
    "--profile",
    type=click.Choice(["complete-crystal", "fragment"]),
    default=None,
    help="Check profile; metadata structure_scope=fragment selects fragment automatically.",
)
@click.option(
    "--skip-check",
    "skip_checks",
    type=click.Choice(
        [
            "hard_clash",
            "intermolecular_clash",
            "isolated_atoms",
            "hydrogen_presence",
            "formula_consistency",
            "bond_distances",
        ]
    ),
    multiple=True,
    help="Skip a check while retaining a machine-readable skipped result; repeatable.",
)
@click.option(
    "--hard-clash-scale",
    type=float,
    default=None,
    help="Scale factor for hard clash detection.",
)
@click.option(
    "--hard-clash-tolerance",
    type=float,
    default=None,
    help="Absolute tolerance for hard clash.",
)
@click.option(
    "--intermolecular-clash-scale",
    type=float,
    default=None,
    help="Scale factor for intermolecular clash.",
)
@click.option(
    "--intermolecular-clash-tolerance",
    type=float,
    default=None,
    help="Tolerance for intermolecular clash.",
)
@click.option(
    "--ignore-hh/--no-ignore-hh",
    default=None,
    help="Ignore H-H intermolecular clashes.",
)
@click.option(
    "--max-clashes",
    type=int,
    default=None,
    help="Maximum allowed intermolecular clashes.",
)
@click.option(
    "--bond-min-factor",
    type=float,
    default=None,
    help="Min factor for bond distance check.",
)
@click.option(
    "--bond-max-factor",
    type=float,
    default=None,
    help="Max factor for bond distance check.",
)
@click.option(
    "--isolated-elements",
    type=str,
    default=None,
    help="Comma-separated elements for isolated atom check.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON report to file.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print JSON instead of table to stdout."
)
def sanity_check_cmd(
    input: Path,
    checks: str | None,
    profile: str | None,
    skip_checks: tuple[str, ...],
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
    """Run structural sanity checks on a supported structure file.

    Multi-frame ExtXYZ files are checked frame by frame. CIF and POSCAR inputs
    contain one structure.
    """
    import json as json_mod

    from molcrys_kit.analysis.sanity_check import sanity_check

    # Parse options
    check_list = [c.strip() for c in checks.split(",")] if checks else None
    iso_elems = (
        {e.strip() for e in isolated_elements.split(",")} if isolated_elements else None
    )

    # Load frames
    if input.suffix.lower() == ".extxyz":
        from molcrys_kit.io.extxyz import read_extxyz

        loaded = read_extxyz(str(input), index=":")
        frames = loaded if isinstance(loaded, list) else [loaded]
    else:
        frames = [load_crystal(input)]
    if not frames:
        raise click.ClickException(f"No frames found in {input}")

    all_reports: list[dict] = []
    n_pass = 0

    for idx, crystal in enumerate(frames):
        report = sanity_check(
            crystal,
            checks=check_list,
            profile=profile,
            skip_checks=skip_checks,
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


def _load_net_policy(path: Path) -> dict[str, object]:
    import json

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise click.UsageError(
                "YAML net-topology policies require PyYAML, which is not installed."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"policy file is not valid JSON: {exc}") from exc
    from molcrys_kit.nets import policies_from_mapping

    try:
        return policies_from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc


def _node_id(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("node_id"), str):
        return value["node_id"]
    raise click.UsageError("each graph node must be an id or an object with node_id")


def _exact_edge_shift(value: object, index: int):
    """Parse a graph-JSON shift with the same rule as ``exact_shift``.

    Floats and Booleans are rejected. ``int(0.9)`` and ``int(True)`` are not
    applied, because those coercions would identify a different quotient.
    """
    from molcrys_kit.periodic._lattice import exact_shift

    try:
        return exact_shift(value, f"edge {index} image shift")
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc


def _load_net_graph(path: Path):
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"graph file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.UsageError("a net graph file must be a JSON object")
    unknown = set(payload) - {"nodes", "edges", "pbc", "embedding"}
    if unknown:
        raise click.UsageError(f"unknown graph keys: {sorted(unknown)}")
    if "nodes" not in payload or "edges" not in payload:
        raise click.UsageError("a net graph file needs nodes and edges")
    nodes = [_node_id(node) for node in payload["nodes"]]
    edges = []
    for index, edge in enumerate(payload["edges"]):
        if isinstance(edge, dict):
            try:
                source = str(edge["source"])
                target = str(edge["target"])
                raw_shift = edge["image_shift"]
            except (KeyError, TypeError) as exc:
                raise click.UsageError(f"edge {index} is not a quotient edge") from exc
            shift = _exact_edge_shift(raw_shift, index)
            if "edge_id" in edge:
                edges.append((str(edge["edge_id"]), source, target, shift))
            else:
                edges.append((source, target, shift))
            continue
        if not isinstance(edge, list) or len(edge) not in {3, 4}:
            raise click.UsageError(
                "an edge must be [source, target, shift] or [id, source, target, shift]"
            )
        if len(edge) == 3:
            edges.append(
                (str(edge[0]), str(edge[1]), _exact_edge_shift(edge[2], index))
            )
        else:
            edges.append(
                (
                    str(edge[0]),
                    str(edge[1]),
                    str(edge[2]),
                    _exact_edge_shift(edge[3], index),
                )
            )
    pbc = tuple(payload.get("pbc", (True, True, True)))
    from molcrys_kit.periodic import PeriodicEmbedding, PeriodicQuotientGraph

    try:
        graph = PeriodicQuotientGraph.from_edges(nodes, edges, pbc=pbc)
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    embedding_payload = payload.get("embedding")
    if embedding_payload is None:
        return graph, None
    if not isinstance(embedding_payload, dict):
        raise click.UsageError("embedding must be an object")
    try:
        positions = tuple(
            (str(row[0]), tuple(float(value) for value in row[1]))
            for row in embedding_payload["positions_A"]
        )
        lattice = tuple(
            tuple(float(value) for value in row)
            for row in embedding_payload["lattice_A"]
        )
        fractional = embedding_payload.get("fractional")
        fractional_rows = None
        if fractional is not None:
            fractional_rows = tuple(
                (str(row[0]), tuple(float(value) for value in row[1]))
                for row in fractional
            )
        embedding = PeriodicEmbedding(positions, lattice, fractional=fractional_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise click.UsageError(f"embedding could not be read: {exc}") from exc
    return graph, embedding


@click.command("net-topology")
@click.argument(
    "input",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON quotient graph. Mutually exclusive with a crystal file.",
)
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON policy. YAML is accepted only when PyYAML is installed.",
)
@click.option(
    "--nodes",
    type=click.Choice(
        ["atom_sites", "molecular_units", "coordination_centres", "explicit_nodes"]
    ),
    default=None,
    help="Crystal node policy. Overrides the policy file.",
)
@click.option(
    "--edges",
    type=click.Choice(
        ["supplied", "coordination_shell", "contacts", "hydrogen_bonds", "user_edges"]
    ),
    default=None,
    help="Crystal edge policy. Overrides the policy file.",
)
@click.option("--cutoff", type=float, default=None, help="Contact cutoff in Angstrom.")
@click.option(
    "--disorder",
    type=click.Choice(["major", "as_refined", "explicit"]),
    default=None,
    help="Which refined sites may be occupied together.",
)
@click.option(
    "--independent-assemblies",
    is_flag=True,
    default=False,
    help="Allow one major group from each disorder assembly.",
)
@click.option("--ring-length", type=int, default=None, help="Cover-cycle length cap.")
@click.option(
    "--coordination-depth",
    type=int,
    default=None,
    help="Coordination-sequence depth on the periodic cover.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the typed JSON report.")
def net_topology(
    input: Path | None,
    graph_path: Path | None,
    policy_path: Path | None,
    nodes: str | None,
    edges: str | None,
    cutoff: float | None,
    disorder: str | None,
    independent_assemblies: bool,
    ring_length: int | None,
    coordination_depth: int | None,
    as_json: bool,
) -> None:
    """Report periodic-net topology, linking, and duality status."""
    import json
    from dataclasses import replace

    crystal_options = [nodes, edges, cutoff, disorder, independent_assemblies or None]
    if input is not None and graph_path is not None:
        raise click.UsageError("pass a crystal file or --graph, not both.")
    if input is None and graph_path is None:
        raise click.UsageError("pass a crystal file or --graph.")
    if graph_path is not None and any(option is not None for option in crystal_options):
        raise click.UsageError(
            "node, edge, cutoff, and disorder options apply to a crystal file."
        )
    if cutoff is not None and cutoff <= 0:
        raise click.UsageError("--cutoff must be positive.")
    if ring_length is not None and ring_length < 0:
        raise click.UsageError("--ring-length must be zero or positive.")
    if coordination_depth is not None and coordination_depth < 0:
        raise click.UsageError("--coordination-depth must be zero or positive.")

    parsed = (
        _load_net_policy(policy_path)
        if policy_path is not None
        else {
            "nodes": None,
            "edges": None,
            "disorder": None,
            "partition": None,
            "limits": {},
        }
    )
    from molcrys_kit.nets import (
        NetAnalysisLimits,
        analyze_periodic_net_system,
        build_periodic_site_graph,
        limits_from_mapping,
    )
    from molcrys_kit.nets.policies import DisorderPolicy, EdgePolicy, NodePolicy

    node_policy = parsed["nodes"] or NodePolicy()
    edge_policy = parsed["edges"] or EdgePolicy()
    disorder_policy = parsed["disorder"] or DisorderPolicy()
    if nodes is not None:
        node_policy = replace(node_policy, kind=nodes)
    if edges is not None:
        edge_policy = replace(edge_policy, kind=edges)
    if cutoff is not None:
        edge_policy = replace(edge_policy, cutoff_A=cutoff)
    if disorder is not None:
        disorder_policy = replace(disorder_policy, mode=disorder)
    if independent_assemblies:
        disorder_policy = replace(disorder_policy, independent_assemblies=True)
    try:
        limits = limits_from_mapping(NetAnalysisLimits.from_config(), parsed["limits"])
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    if ring_length is not None:
        limits = replace(limits, ring_length_limit=ring_length)
    if coordination_depth is not None:
        limits = replace(limits, coordination_depth=coordination_depth)

    if graph_path is not None:
        graph, embedding = _load_net_graph(graph_path)
        report = analyze_periodic_net_system(
            [(graph, embedding)],
            limits=limits,
            partition_policy=parsed["partition"],
        )
    else:
        report = analyze_periodic_net_system(
            build_periodic_site_graph(
                load_crystal(input),
                node_policy,
                edge_policy,
                disorder_policy,
                limits,
            ),
            limits=limits,
            partition_policy=parsed["partition"],
        )
    payload = report.to_dict()
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"schema: {payload['schema']}")
    click.echo(f"disorder: {payload['disorder_status']} ({payload['disorder_reason']})")
    click.echo(f"primary: {payload['primary_proposal_id']}")
    if payload["primary_is_not_unique"]:
        click.echo("primary partition is not unique")
    if payload["truncated"]:
        click.echo("discovery truncated")
    for note in payload["ambiguities"]:
        click.echo(f"ambiguity: {note}")
    for proposal in payload["proposals"]:
        if proposal["proposal_id"] != payload["primary_proposal_id"]:
            continue
        for candidate in proposal["candidates"]:
            topology = candidate["topology"]
            click.echo(
                f"{candidate['candidate_id']}: "
                f"periodic_rank={candidate['periodic_rank']} "
                f"quotient_cycle_rank={candidate['quotient_cycle_rank']} "
                f"topology={topology['status']} "
                f"proof={topology['proof']}"
            )


def register_analyze_commands(group: click.Group) -> None:
    group.add_command(summary)
    group.add_command(bfdh)
    group.add_command(interactions)
    group.add_command(bond_orders)
    group.add_command(cell_integrity)
    group.add_command(polyhedra)
    group.add_command(polyhedron_network)
    group.add_command(net_topology)
    group.add_command(sanity_check_cmd)
