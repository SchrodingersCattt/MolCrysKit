"""Small reproducible benchmark entry point for periodic-chain construction."""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np

# Make the benchmark runnable directly from a source checkout, without relying
# on whichever molcrys-kit version happens to be installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molcrys_kit.operations.periodic_chain import build_periodic_chains
from molcrys_kit.structures.periodic_geometry import BoundaryPort, ChainSpec, ConnectionRule, FragmentTemplate

def run(atom_counts=(1000, 10000, 100000)):
    template=FragmentTemplate("unit",("C",),((0.,0.,0.),),(BoundaryPort("join",(0.,0.,0.)),))
    rule=ConnectionRule("join","unit","join","unit", "join", allowed_image_shifts=((0,0,0),(1,0,0)), distance_range=(0.,20.))
    rows=[]
    for count in atom_counts:
        cell=np.diag([max(100.,float(count)),100.,100.])
        repeats=max(2,int(count))
        t0=time.perf_counter(); bundle=build_periodic_chains({"unit":template},(rule,),cell,(True,True,True),ChainSpec(("unit",)*repeats, target_winding=(1,0,0), min_distance=.5)); elapsed=time.perf_counter()-t0
        rows.append({"requested_atoms":count,"constructed_atoms":len(bundle.atoms),"seconds":elapsed,"graph_edges":len(bundle.graph.edges)})
    return rows

if __name__ == "__main__":
    for row in run(): print(row)
