# MolCrysKit API Reference

This document serves as a placeholder for the MolCrysKit API reference. The API reference provides detailed information about the classes, functions, and methods available in the MolCrysKit package.

## Modules

### molcrys_kit.structures
- [CrystalMolecule](../molcrys_kit/structures/molecule.py): Molecular representation with graph connectivity
- [MolecularCrystal](../molcrys_kit/structures/crystal.py): Crystal structure composed of molecules
- [Atom](../molcrys_kit/structures/atom.py): Individual atom representation

### molcrys_kit.operations
- [add_hydrogens](../molcrys_kit/operations/hydrogen_completion.py): Function to add hydrogen atoms based on geometric rules
- [HydrogenCompleter](../molcrys_kit/operations/hydrogen_completion.py): Class for adding hydrogen atoms to molecular crystals
- [generate_topological_slab](../molcrys_kit/operations/surface.py): Function to generate surface slabs preserving molecular topology
- [TopologicalSlabGenerator](../molcrys_kit/operations/surface.py): Class for generating surface slabs while preserving molecular topology
- [generate_vacancy](../molcrys_kit/operations/defects.py): Function to generate vacancies by removing molecular clusters

### molcrys_kit.analysis
- [DisorderSolver](../molcrys_kit/analysis/disorder/solver.py): Class for solving disorder problems using graph algorithms
- [DisorderGraphBuilder](../molcrys_kit/analysis/disorder/graph.py): Class for building exclusion graphs from disorder data
- [ChemicalEnvironment](../molcrys_kit/analysis/chemical_env.py): Class for analyzing chemical environments in molecular crystals

### molcrys_kit.io
- [read_mol_crystal](../molcrys_kit/io/cif.py): Function to read molecular crystals from CIF files
- [write_cif](../molcrys_kit/io/output.py): Function to write crystal structures to CIF files
- [identify_molecules](../molcrys_kit/io/cif.py): Function to identify individual molecules in a crystal structure

## Detailed API Documentation

For detailed API documentation, please refer to the docstrings in the source code files.