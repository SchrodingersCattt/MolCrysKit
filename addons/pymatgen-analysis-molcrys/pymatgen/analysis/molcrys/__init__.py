"""Pymatgen analysis namespace package for MolCrysKit.

Install with ``pip install pymatgen-analysis-molcrys``, then::

    from pymatgen.analysis.molcrys import TopologicalSlabGenerator
"""

from .slab import TopologicalSlabGenerator, generate_topological_slab

__all__ = [
    "TopologicalSlabGenerator",
    "generate_topological_slab",
]
