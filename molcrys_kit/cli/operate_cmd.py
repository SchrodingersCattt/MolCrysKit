"""Structure-operation commands for the MolCrysKit CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import click

from molcrys_kit.analysis.disorder import generate_ordered_replicas_from_disordered_sites
from molcrys_kit.io import write_xyz_with_freeze
from molcrys_kit.operations import (
    ClusterCarver,
    LigandTopologyOverflowError,
    add_hydrogens,
    create_supercell,
    generate_slabs_with_terminations,
    generate_topological_slab,
    generate_vacancy,
    interpolate_crystal,
    remove_solvents,
)

from ._common import echo_paths, load_crystal, write_crystal_sequence, write_structure


def _parse_seed(seed_element: str | None, seed_index: tuple[int, ...] | None):
    if seed_element is not None and seed_index:
        raise click.UsageError("Specify --seed-element OR --seed-index, not both.")
    if seed_element is not None:
        return seed_element
    if seed_index:
        return list(seed_index)
    raise click.UsageError("Specify a seed via --seed-element ELEMENT or --seed-index I [I ...].")


def _parse_cut_cc_bonds(text: str | None) -> List[Tuple[int, int]]:
    """Parse ``i,j;k,l`` into parent-index edge pairs."""
    if text is None or not text.strip():
        return []
    out: List[Tuple[int, int]] = []
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise click.UsageError("--cut-cc-bonds expects 'i,j;k,l' parent-index pairs.")
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise click.UsageError("--cut-cc-bonds entries must be integer pairs.") from exc
    return out


def _parse_cap_bond_lengths(entries: tuple[str, ...]) -> dict[str, float]:
    cap_overrides: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise click.UsageError(f"--cap-bond-length expects ELEM=DIST, got {entry!r}.")
        elem, dist_str = entry.split("=", 1)
        try:
            cap_overrides[elem.strip()] = float(dist_str)
        except ValueError as exc:
            raise click.UsageError(f"Invalid cap distance in {entry!r}.") from exc
    return cap_overrides


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output file or file stem.")
@click.option("--method", type=click.Choice(["optimal", "random", "enumerate"]), default="optimal", show_default=True)
@click.option("--count", type=int, default=1, show_default=True, help="Number of structures for random/enumerate modes.")
@click.option("--seed", type=int, default=None, help="Random seed for random mode.")
@click.option("--coupled", is_flag=True, help="Couple symmetry-expanded copies of the same disorder assembly.")
def disorder(input: Path, output: Path, method: str, count: int, seed: int | None, coupled: bool) -> None:
    """Resolve CIF disorder into ordered replica structures."""
    replicas = generate_ordered_replicas_from_disordered_sites(
        str(input), generate_count=count, method=method, random_seed=seed, coupled=coupled
    )
    crystals = [item[0] if isinstance(item, tuple) else item for item in replicas]
    echo_paths(write_crystal_sequence(crystals, output))


@click.command(name="add-h")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--target-elements", multiple=True, help="Element symbols to hydrogenate; repeatable.")
@click.option("--optimize-torsion", is_flag=True, help="Enable torsion optimization during placement.")
@click.option("--no-formula-moiety", is_flag=True, help="Disable CIF formula-moiety H-count correction.")
def add_h(input: Path, output: Path, target_elements: tuple[str, ...], optimize_torsion: bool, no_formula_moiety: bool) -> None:
    """Add missing hydrogen atoms."""
    crystal = load_crystal(input)
    result = add_hydrogens(
        crystal,
        target_elements=list(target_elements) or None,
        optimize_torsion=optimize_torsion,
        use_formula_moiety=not no_formula_moiety,
    )
    write_structure(result, output)
    click.echo(f"Wrote {output}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--miller", nargs=3, type=int, required=True, metavar="H K L", help="Miller indices.")
@click.option("--layers", type=int, default=None, help="Explicit number of unit planes.")
@click.option("--min-thickness", type=float, default=None, help="Minimum slab thickness in Angstrom.")
@click.option("--vacuum", type=float, default=10.0, show_default=True, help="Vacuum thickness in Angstrom.")
@click.option("--terminations", default="single", show_default=True, help="single, tasker_preferred, all, or a termination index.")
def slab(input: Path, output: Path, miller: tuple[int, int, int], layers: int | None, min_thickness: float | None, vacuum: float, terminations: str) -> None:
    """Generate a topology-preserving surface slab."""
    crystal = load_crystal(input)
    if terminations == "single":
        result = generate_topological_slab(crystal, miller, layers=layers, min_thickness=min_thickness, vacuum=vacuum)
        write_structure(result, output)
        click.echo(f"Wrote {output}")
        return

    term_selection = terminations
    indices = None
    if terminations.isdigit():
        term_selection = "by_index"
        indices = [int(terminations)]
    if term_selection == "tasker":
        term_selection = "tasker_preferred"
    results = generate_slabs_with_terminations(
        crystal,
        miller_index=miller,
        layers=layers,
        min_slab_size=min_thickness,
        min_vacuum_size=vacuum,
        term_selection=term_selection,
        termination_indices=indices,
    )
    written = []
    stem = output.with_suffix("")
    suffix = output.suffix or ".cif"
    for slab_crystal, info in results:
        path = stem.parent / f"{stem.name}_term{info.termination_index}{suffix}"
        write_structure(slab_crystal, path)
        written.append(path)
    echo_paths(written)


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output stem; writes <stem>__group<k>.xyz plus JSON sidecar.")
@click.option("--mode", type=click.Choice(["bond_shells", "rcut"]), default="bond_shells", show_default=True)
@click.option("--seed-element", type=str, default=None, help="Seed on every atom of this element.")
@click.option("--seed-index", type=int, multiple=True, help="Explicit zero-based global atom seed index; repeatable.")
@click.option("--max-atoms", type=int, default=500, show_default=True, help="Hard safety cap for bond_shells mode.")
@click.option("--cut-cc-bonds", type=str, default=None, metavar="I,J;K,L", help="Parent-index C-C bonds to truncate.")
@click.option("--rcut", type=float, default=None, help="Radial cutoff in Angstrom for rcut mode.")
@click.option("--freeze-shell", type=click.Choice(["0", "1", "2"]), default="1", show_default=True)
@click.option("--cap-distance", type=float, default=None, help="Uniform cap distance in Angstrom.")
@click.option("--cap-bond-length", multiple=True, metavar="ELEM=DIST", help="Override one X-H cap length; repeatable.")
@click.option("--seed-merge-radius", type=float, default=0.0, show_default=True, help="Group adjacent seeds within this radius.")
@click.option("--convention-reference", type=str, default="", help="Free-text citation/protocol note for the sidecar JSON.")
@click.option("--no-stop-at-non-seed-metals", is_flag=True, help="Disable implicit metal-boundary rule.")
def cluster(
    input: Path,
    output: Path,
    mode: str,
    seed_element: str | None,
    seed_index: tuple[int, ...],
    max_atoms: int,
    cut_cc_bonds: str | None,
    rcut: float | None,
    freeze_shell: str,
    cap_distance: float | None,
    cap_bond_length: tuple[str, ...],
    seed_merge_radius: float,
    convention_reference: str,
    no_stop_at_non_seed_metals: bool,
) -> None:
    """Carve finite, H-capped QM cluster models."""
    seed = _parse_seed(seed_element, seed_index)
    if mode == "rcut" and rcut is None:
        raise click.UsageError("--rcut is required when --mode rcut.")
    cut_edges = _parse_cut_cc_bonds(cut_cc_bonds)
    cap_overrides = _parse_cap_bond_lengths(cap_bond_length)

    crystal = load_crystal(input)
    carver = ClusterCarver(crystal, seed_merge_radius=seed_merge_radius)
    try:
        if mode == "bond_shells":
            clusters = carver.carve_bond_shells(
                seed,
                max_atoms=max_atoms,
                cut_cc_bonds=cut_edges,
                freeze_shell=int(freeze_shell),
                cap_distance=cap_distance,
                cap_bond_lengths=cap_overrides or None,
                parent_label=os.path.abspath(input),
                convention_reference=convention_reference,
                stop_at_non_seed_metals=not no_stop_at_non_seed_metals,
            )
        else:
            clusters = carver.carve_rcut(
                seed,
                rcut=rcut,
                freeze_shell=int(freeze_shell),
                cap_distance=cap_distance,
                cap_bond_lengths=cap_overrides or None,
                parent_label=os.path.abspath(input),
                convention_reference=convention_reference,
            )
    except LigandTopologyOverflowError as exc:
        if exc.candidates:
            formatted = ";".join(f"{a},{b}" for a, b in exc.candidates)
            raise click.ClickException(f"{exc}\nAll candidate bonds: --cut-cc-bonds \"{formatted}\"") from exc
        raise click.ClickException(str(exc)) from exc

    out_dir = output.parent
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    for idx, carved in enumerate(clusters):
        xyz_path = f"{output}__group{idx}.xyz"
        sidecar = write_xyz_with_freeze(carved, xyz_path)
        click.echo(
            f"[group {idx}] mode={carved.provenance.mode} natoms={len(carved)} "
            f"frozen={len(carved.frozen_local_indices)} caps={len(carved.cap_local_indices)} "
            f"-> {xyz_path}\n            sidecar: {sidecar}"
        )


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--scale", nargs=3, type=int, required=True, metavar="A B C", help="Supercell replication factors.")
def supercell(input: Path, output: Path, scale: tuple[int, int, int]) -> None:
    """Create a supercell."""
    result = create_supercell(load_crystal(input), scale)
    write_structure(result, output)
    click.echo(f"Wrote {output}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--species", nargs=2, multiple=True, metavar="SPECIES_ID COUNT", help="Species/count pair; repeatable.")
@click.option("--seed-index", type=int, default=None, help="Seed molecule index.")
@click.option("--method", default="spatial_cluster", show_default=True)
@click.option("--random-seed", type=int, default=None)
def vacancy(input: Path, output: Path, species: tuple[tuple[str, str], ...], seed_index: int | None, method: str, random_seed: int | None) -> None:
    """Generate a vacancy by removing a molecule cluster."""
    species_list = [{"species_id": sid, "count": int(count)} for sid, count in species] or None
    result = generate_vacancy(load_crystal(input), species_list=species_list, seed_index=seed_index, method=method, random_seed=random_seed)
    write_structure(result, output)
    click.echo(f"Wrote {output}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--targets", multiple=True, required=True, help="Solvent species identifiers to remove; repeatable.")
def desolvate(input: Path, output: Path, targets: tuple[str, ...]) -> None:
    """Remove solvent species from a crystal."""
    result = remove_solvents(load_crystal(input), list(targets))
    write_structure(result, output)
    click.echo(f"Wrote {output}")


@click.command()
@click.argument("start", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("end", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output file (.extxyz for bundle) or filename stem.")
@click.option("--method", type=click.Choice(["se3_screw", "com_so3", "slerp"]), default="se3_screw", show_default=True)
@click.option("--n-images", type=int, default=11, show_default=True)
@click.option("--include-endpoints/--exclude-endpoints", default=True, show_default=True)
def interpolate(start: Path, end: Path, output: Path, method: str, n_images: int, include_endpoints: bool) -> None:
    """Interpolate crystal images between two endpoints."""
    frames = interpolate_crystal(load_crystal(start), load_crystal(end), method=method, n_images=n_images, include_endpoints=include_endpoints)
    echo_paths(write_crystal_sequence(frames, output))


def register_operate_commands(group: click.Group) -> None:
    group.add_command(disorder)
    group.add_command(add_h)
    group.add_command(slab)
    group.add_command(cluster)
    group.add_command(supercell)
    group.add_command(vacancy)
    group.add_command(desolvate)
    group.add_command(interpolate)
