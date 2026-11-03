"""Domain-neutral periodic quotient multigraph.

Existing fragment, chemistry, and coordination-shell graphs stay domain
objects. Adapters copy their exact integer connectivity into one immutable
multigraph and keep ports, bond records, and shell contacts beside it.

Periodic rank is the dimension of the cycle-translation subgroup. Quotient
cycle rank counts finite loops as well as lattice windings. This package does
not name nets, detect interpenetration, or construct dual tilings.

See ``docs/periodic-graph.md`` for orientation, gauge, basis, and migration
conventions.
"""

from .adapters import (
    CallerGraphProvenance,
    ChemicalGraphProvenance,
    FragmentEdgeProvenance,
    FragmentGraphProvenance,
    GraphAttribute,
    PeriodicGraphAdaptation,
    PeriodicGraphProvenance,
    PolyhedronGraphProvenance,
    QuotientGraphProvenance,
    adapt_caller_graph,
    adapt_chemical_entity,
    adapt_fragment_graph,
    adapt_periodic_graph,
    adapt_polyhedron_network,
    fragment_graph_from_provenance,
    to_periodic_quotient_graph,
)
from .algorithms import (
    ClosedWalkTranslation,
    CoverComponentFamily,
    PeriodicGraphAnalysis,
    QuotientComponent,
    TranslationSubgroup,
    analyze_periodic_graph,
    closed_walk_translations,
    cover_component_families,
    cover_multiplicity,
    periodic_rank,
    quotient_components,
    quotient_cycle_rank,
    translation_subgroups,
)
from .graph import (
    PeriodicEmbedding,
    PeriodicQuotientEdge,
    PeriodicQuotientGraph,
    PeriodicQuotientNode,
    PeriodicQuotientView,
    reversed_edge,
)
from .transforms import (
    EmbeddedQuotientGraph,
    SupercellQuotientGraph,
    SupercellReplication,
    apply_node_gauge,
    apply_unimodular_basis,
    equivalent_supercell,
    relabel_nodes,
    replace_lattice,
    shift_embedding_origin,
)

__all__ = [
    "CallerGraphProvenance",
    "ChemicalGraphProvenance",
    "ClosedWalkTranslation",
    "CoverComponentFamily",
    "EmbeddedQuotientGraph",
    "FragmentEdgeProvenance",
    "FragmentGraphProvenance",
    "GraphAttribute",
    "PeriodicEmbedding",
    "PeriodicGraphAdaptation",
    "PeriodicGraphAnalysis",
    "PeriodicGraphProvenance",
    "PeriodicQuotientEdge",
    "PeriodicQuotientGraph",
    "PeriodicQuotientNode",
    "PeriodicQuotientView",
    "PolyhedronGraphProvenance",
    "QuotientComponent",
    "QuotientGraphProvenance",
    "SupercellQuotientGraph",
    "SupercellReplication",
    "TranslationSubgroup",
    "adapt_caller_graph",
    "adapt_chemical_entity",
    "adapt_fragment_graph",
    "adapt_periodic_graph",
    "adapt_polyhedron_network",
    "analyze_periodic_graph",
    "apply_node_gauge",
    "apply_unimodular_basis",
    "closed_walk_translations",
    "cover_component_families",
    "cover_multiplicity",
    "equivalent_supercell",
    "fragment_graph_from_provenance",
    "periodic_rank",
    "quotient_components",
    "quotient_cycle_rank",
    "relabel_nodes",
    "replace_lattice",
    "reversed_edge",
    "shift_embedding_origin",
    "to_periodic_quotient_graph",
    "translation_subgroups",
]
