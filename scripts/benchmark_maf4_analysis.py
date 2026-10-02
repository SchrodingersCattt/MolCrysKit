"""Report reproducible disorder-analysis stages for the MAF-4 fixture.

The benchmark intentionally reports timings instead of enforcing a wall-clock
threshold: the result hashes and counts are the regression contract, while
timings are used to compare comparable machines and algorithm changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Keep direct ``python scripts/…`` execution on the checkout under test rather
# than accidentally importing an unrelated globally installed MolCrysKit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder  # noqa: E402
from molcrys_kit.analysis.disorder.solver import DisorderSolver  # noqa: E402
from molcrys_kit.io.cif import scan_cif_disorder  # noqa: E402


def _hash_payload(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _graph_contract_hash(graph) -> str:
    edges = [
        (int(left), int(right), str(data.get("conflict_type", "")))
        for left, right, data in graph.edges(data=True)
    ]
    return _hash_payload(sorted(edges))


def _crystal_contract_hash(crystal) -> str:
    sites = [
        {
            "id": record.site_id,
            "symbol": record.symbol,
            "molecule": record.molecule_index,
            "position": [round(value, 8) for value in record.cartesian_position_A],
        }
        for record in crystal.get_site_records()
    ]
    bonds = [
        {
            "left": record.left_global_index,
            "right": record.right_global_index,
            "shift": list(record.right_image_shift),
        }
        for record in crystal.get_bond_records()
    ]
    return _hash_payload({"sites": sites, "bonds": bonds})


def benchmark(path: str | Path) -> dict:
    """Run the MAF-4 disorder pipeline and return stage timings/contracts."""
    path = str(path)
    stages: dict[str, float] = {}

    started = time.perf_counter()
    info = scan_cif_disorder(path)
    stages["scan_cif_disorder_s"] = time.perf_counter() - started

    started = time.perf_counter()
    builder = DisorderGraphBuilder(info, info.lattice_matrix)
    graph = builder.build()
    stages["graph_build_s"] = time.perf_counter() - started

    started = time.perf_counter()
    solver = DisorderSolver(info, graph, info.lattice_matrix)
    structures = solver.solve(num_structures=1, method="optimal")
    stages["solve_and_reconstruct_s"] = time.perf_counter() - started
    result = structures[0]

    return {
        "input": path,
        "expanded_atom_count": len(info.labels),
        "graph_edge_count": graph.number_of_edges(),
        "output_atom_count": result.get_total_nodes(),
        "output_molecule_count": len(result.molecules),
        "stages": stages,
        "graph_contract_hash": _graph_contract_hash(graph),
        "result_contract_hash": _crystal_contract_hash(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/data/cif/MAF-4.cif",
        help="MAF-4 CIF path (default: tests/data/cif/MAF-4.cif)",
    )
    parser.add_argument("--output", type=Path, help="Write JSON to this path as well as stdout.")
    args = parser.parse_args()
    payload = json.dumps(benchmark(args.path), indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
