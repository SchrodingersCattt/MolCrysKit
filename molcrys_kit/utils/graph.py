"""
Graph utilities for molecular crystal topology operations.

This module provides general-purpose graph functions used across
MolCrysKit — stoichiometry analysis, molecule identification, and
any future code that needs fast graph comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def graph_invariant(graph: "nx.Graph", node_attr: str = "symbol") -> tuple:
    """Compute a hashable topological invariant for fast isomorphism pre-check.

    Two graphs can only be isomorphic if their invariants are equal.
    The converse is not guaranteed (hash collision), so a full
    isomorphism test is still needed when invariants match.

    The invariant is a tuple of:

    1. ``(n_nodes, n_edges)``
    2. Sorted degree sequence
    3. Sorted ``(node_attribute, degree)`` pair counts

    The node attribute key defaults to ``"symbol"`` (chemical element),
    which is how :class:`~molcrys_kit.structures.molecule.CrystalMolecule`
    builds its ``.graph``.  Pass *node_attr* to override.

    Complexity: O(N log N) where N = number of nodes.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with optional node attributes.
    node_attr : str, optional
        Node attribute key to use for the element-degree signature.
        Defaults to ``"symbol"``.

    Returns
    -------
    tuple
        Hashable invariant.  Equal invariants are a necessary (but not
        sufficient) condition for isomorphism.

    Examples
    --------
    >>> import networkx as nx
    >>> g1 = nx.path_graph(3)
    >>> g2 = nx.path_graph(3)
    >>> graph_invariant(g1) == graph_invariant(g2)
    True
    >>> g3 = nx.cycle_graph(3)
    >>> graph_invariant(g1) == graph_invariant(g3)
    False
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    degrees = sorted(d for _, d in graph.degree())

    # Element-degree signature: count of each (attribute, degree) pair.
    attr_deg_counts: dict[tuple, int] = {}
    for node, deg in graph.degree():
        attr_val = graph.nodes[node].get(node_attr, "?")
        key = (attr_val, deg)
        attr_deg_counts[key] = attr_deg_counts.get(key, 0) + 1
    attr_deg_sig = tuple(sorted(attr_deg_counts.items()))

    return (n_nodes, n_edges, tuple(degrees), attr_deg_sig)
