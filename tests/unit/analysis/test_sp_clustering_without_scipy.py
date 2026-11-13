"""Special-position clustering uses NumPy only and matches SciPy's results."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from molcrys_kit.analysis.disorder.graph import _complete_linkage_maxclust

ROOT = Path(__file__).resolve().parents[3]
DAP4 = ROOT / "tests" / "data" / "cif" / "DAP-4.cif"


def _partition(labels):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values())


def _scipy_partition(dist, n_clusters):
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    spatial = pytest.importorskip("scipy.spatial.distance")
    z = hierarchy.linkage(spatial.squareform(dist, checks=False), method="complete")
    return _partition(hierarchy.fcluster(z, t=n_clusters, criterion="maxclust"))


def _random_dist(rng, n):
    points = rng.normal(size=(n, 3))
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8, 12, 24])
def test_matches_scipy_on_random_geometry(n):
    rng = np.random.default_rng(1000 + n)
    for _ in range(25):
        dist = _random_dist(rng, n)
        for n_clusters in range(1, n + 1):
            assert _partition(_complete_linkage_maxclust(dist, n_clusters)) == (
                _scipy_partition(dist, n_clusters)
            )


@pytest.mark.parametrize("n_clusters", [1, 2, 3, 4, 6])
def test_matches_scipy_with_tied_distances(n_clusters):
    # Symmetry-generated special-position copies sit on regular lattices, so
    # many pair distances are exactly equal.
    grid = np.array([[i, j, 0.0] for i in range(3) for j in range(4)])
    dist = np.linalg.norm(grid[:, None] - grid[None, :], axis=-1)
    assert _partition(_complete_linkage_maxclust(dist, n_clusters)) == _scipy_partition(
        dist, n_clusters
    )
    ring = np.array(
        [[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    )
    dist = np.linalg.norm(ring[:, None] - ring[None, :], axis=-1)
    assert _partition(_complete_linkage_maxclust(dist, n_clusters)) == _scipy_partition(
        dist, n_clusters
    )


def test_label_edge_cases():
    dist = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert _partition(_complete_linkage_maxclust(dist, 1)) == [[0, 1]]
    assert _partition(_complete_linkage_maxclust(dist, 2)) == [[0], [1]]
    assert _partition(_complete_linkage_maxclust(dist, 5)) == [[0], [1]]


_SCRIPT = textwrap.dedent(
    """
    import json, sys
    class _Blocker:
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("scipy", "ase", "pymatgen", "spglib"):
                raise ImportError(name)
    sys.meta_path.insert(0, _Blocker())
    from molcrys_kit.analysis.disorder import generate_ordered_replicas_from_disordered_sites as gen
    replicas, kept = gen({path!r}, generate_count=1, method="optimal", return_kept_indices=True)[0]
    print("RESULT" + json.dumps({{"kept": sorted(int(i) for i in kept), "n": len(replicas.molecules)}}))
    """
)


def _summary(result):
    replicas, kept = result[0]
    return {"kept": sorted(int(i) for i in kept), "n": len(replicas.molecules)}


def _generate():
    from molcrys_kit.analysis.disorder import (
        generate_ordered_replicas_from_disordered_sites as gen,
    )

    return gen(str(DAP4), generate_count=1, method="optimal", return_kept_indices=True)


def test_dap4_resolution_matches_scipy_reference_clustering(monkeypatch):
    pytest.importorskip("ase")
    from molcrys_kit.analysis.disorder import graph

    numpy_result = _summary(_generate())

    def scipy_reference(dist, n_clusters):
        return np.asarray(
            _scipy_labels(dist, n_clusters)
        )

    monkeypatch.setattr(graph, "_complete_linkage_maxclust", scipy_reference)
    assert _summary(_generate()) == numpy_result


def _scipy_labels(dist, n_clusters):
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    spatial = pytest.importorskip("scipy.spatial.distance")
    z = hierarchy.linkage(spatial.squareform(dist, checks=False), method="complete")
    return hierarchy.fcluster(z, t=n_clusters, criterion="maxclust")


def test_dap4_resolution_runs_without_scipy():
    pytest.importorskip("ase")  # reference comparison needs the desktop stack
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(path=str(DAP4))],
        capture_output=True,
        text=True,
        timeout=170,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    line = next(x for x in completed.stdout.splitlines() if x.startswith("RESULT"))
    assert json.loads(line[len("RESULT") :]) == _summary(_generate())