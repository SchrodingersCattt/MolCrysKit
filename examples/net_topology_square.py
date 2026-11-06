"""Classify one two-periodic quotient without naming a prototype."""

from molcrys_kit.nets import analyze_periodic_net_system
from molcrys_kit.periodic import PeriodicQuotientGraph

graph = PeriodicQuotientGraph.from_edges(
    ("a",),
    [("ax", "a", "a", (1, 0, 0)), ("ay", "a", "a", (0, 1, 0))],
)
report = analyze_periodic_net_system(graph).to_dict()
candidate = next(
    item
    for proposal in report["proposals"]
    if proposal["proposal_id"] == report["primary_proposal_id"]
    for item in proposal["candidates"]
)
print(report["schema"])
print(
    candidate["periodic_rank"],
    candidate["topology"]["status"],
    candidate["topology"]["proof"],
)
