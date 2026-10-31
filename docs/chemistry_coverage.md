# Chemistry coverage and release gates

This matrix is normative for the experimental `molcrys_kit.chemistry` API.
An entry marked **indeterminate** is an explicit refusal path, not an inferred
answer. MolCrysKit does not call RDKit, OPSIN, InChI, or another chemistry
engine in production. Development comparisons may find disagreements, but a
human decision against a fixed IUPAC/IUCr source is required before changing a
golden result.

## Chemical graph and crystal interpretation

| Rule family | State | Current behavior |
| --- | --- | --- |
| Stable atom identity and embedding separation | implemented | Immutable atom IDs survive geometry-only transforms; embeddings reference IDs. |
| CIF isotope, charge, bond, name, uncertainty, disorder, absolute-structure retention | implemented | Explicit source values retain raw text and parsed value/SU where applicable. |
| 0D–3D connected-component rank | implemented | Periodic candidate edges are classified by translation-subgroup rank. |
| Bond order and formal charge | partial | Hückel aromaticity, source precedence, charge-frontier tree dynamic programming, exact radical counts, periodic winding checks, and geometry-backed H-inventory boundaries cover the section 4.1 organic micro-families. Metals, mixed valence, and underdetermined radicals stay provisional. Open Babel and corpus gates remain unrun on this branch and are tracked in issue #176. |
| Oxidation state and coordination geometry | indeterminate | Fields exist; no general solver is released. |
| Disorder ensemble consensus | indeterminate | Ordered replicas retain provenance, but cross-replica chemical consensus is not yet aggregated. |

## CIP and stereochemistry

| Rule family / stereogenic unit | State | Current behavior |
| --- | --- | --- |
| Sequence Rule 1a | implemented | Recursive atomic-number comparison. |
| Sequence Rule 1b | implemented | Duplicate nodes for rings and multiple bonds. |
| Sequence Rule 2 | implemented | Isotope mass ordering. |
| Sequence Rules 3–5, pseudoasymmetry, descriptor-dependent priority | indeterminate | No label is forced when these rules may resolve a tie. |
| Tetrahedral R/S | implemented | Coordinate assignment with planar/degenerate refusal. |
| Double-bond E/Z | implemented | Endpoint CIP ordering plus projected 3D side test. |
| Cumulene, axial/atropisomeric, planar, helical | indeterminate | Not released. |
| Square-planar, trigonal-bipyramidal, octahedral descriptors | indeterminate | Not released. |
| Polymer tacticity descriptors | indeterminate | Not released. |
| Crystal enantiomer aggregation | partial | Finite entities are compared by self-contained graph isomorphism; E/Z remains invariant under mirroring. Other stereogenic-unit families are excluded visibly. |
| Absolute configuration vs experimental absolute structure | implemented | Coordinate labels and CIF Flack/Hooft/Rogers/Parsons evidence are separate; no numeric threshold creates a verdict. |

## Linear notation

| Dialect | State | Current behavior |
| --- | --- | --- |
| OpenSMILES 1.0 | partial | Deterministic finite covalent subset: branches, rings, aromatic bonds, isotopes, charges, and common bond orders. Unsupported semantics fail loudly or select MCK-LN automatically. |
| BigSMILES | indeterminate | The current polymer model lacks typed bonding descriptors; explicit requests fail rather than emitting invalid BigSMILES. |
| MCK-LN 1 | implemented | Versioned non-JSON extension with parser/generator round trips for finite, periodic, polymer, and nested multicomponent entities, stable IDs, embeddings, and periodic/coordination semantics. |

`LineNotation.lossless` states whether the selected dialect preserved every
field in its declared scope. `dialect="auto"` never chooses OpenSMILES when
MolCrysKit-specific chemical semantics would be discarded.

## Naming

| Standard family | State | Current behavior |
| --- | --- | --- |
| Blue Book 2013 | partial | Shared substitutive grammar for acyclic parents (multiplied alcohols, ketones, aldehydes, and carboxylic acids, ene/diene vowel rules, haloalkanes, N-substituted formamides, sulfur links, hydroperoxides, and carbon- or oxygen-centred radicals), Hantzsch–Widman oxygen heterocycles, cycloalkanols, quinones and cyclohexadiene diols, phenoxides, aryl sulfates, and finite peptide-like conjugates. Retained parents (water, benzene, phenol, formamide) stay preferred only when they are in the reviewed set. Full fused-ring and bis-peptide coverage is still outside the grammar. |
| Red Book 2005 | partial | Deterministic composition/dimensionality descriptions; general additive coordination naming is not released. |
| Purple Book 2008 | partial | Single named repeat-unit `poly(...)` result is provisional; end groups and typed connections remain required. |
| Full PIN selection | indeterminate outside rows above | `preferred` is never asserted for fallback composition descriptions. |
| Certified published reference names | implemented (closed registry) | The ten reviewed SMILES fixtures carry their complete source graph, formula, and fixed PubChem IUPAC name; strict conversion accepts only those exact normalized source strings after valence and formula validation. This registry does not claim general PIN coverage. |
| Name → structure | partial | Canonical spellings emitted by the substitutive grammar are accepted, together with documented aliases (`1,2-dihydroxybenzene` for `benzene-1,2-diol`, `oxolane` / `1-oxacyclopentane`, `chloroethane` / `1-chloroethane`, ungrouped replacement prefixes, and the legacy multiplicative-unsaturation vowel). Malformed stems such as `hex-2-one` fail closed. Unsupported graphs stay composition descriptions. |

Forward and inverse naming share one substitutive grammar. It covers the
earlier neutral parents and, beyond them, multiplied and mixed acyclic
functional groups, unsaturated acids, haloalkanes, N-substituted formamides,
sulfides, sulfoxides, sulfones, hydroperoxides and peroxyl radicals, Hantzsch–Widman
oxygen rings, cycloalkanols, quinones, phenoxide and phenoxyl ions, aryl
sulfates, and finite conjugates such as S-substituted cysteines and glutathione.
A bis(glutathione) carbonyl is still outside the grammar and stays a composition
description. Preferred-name status is unchanged: generated names are general
IUPAC names unless they are in the reviewed preferred set.

`name_entity()` may return a composition description. `smiles_to_iupac(strict=True)`
requires a name that parses back to the same constitution, charge, and radical
count. Strict conversion applies OpenSMILES default valences to unbracketed
organic-subset atoms, including the halogens (`Cl` is hydrogen chloride). A
bracket atom stores an explicit hydrogen count; a deficit below the lowest
closed-shell valence for that element and charge is recorded as
`radical_electrons`. A valence that falls between allowed states, such as an
unsupported sulfur environment, is left indeterminate rather than hydrogenated
or assigned a guessed radical. A Kekulé six-membered ring with an alternating
single/double pattern is normalized to the aromatic representation before
naming, so `C1=CC=CC=C1` and `c1ccccc1` share the benzene result. Saturated
ketones keep the `an` ending (`hexan-2-one`). Acyclic polyenes keep the
linking vowel in both directions (`hepta-2,4-diene`, `buta-1,3-diene`).
Repeated skeletal-replacement prefixes are grouped (`2,8-dioxa`). Retained
parents are used for pyridine, naphthalene, furan, thiophene, and pyrrole.
Benzene polyols use the hydroxy suffix (`benzene-1,2-diol`). A single
substituent on ethane omits the locant (`chloroethane`). Nuclide labels omit
a redundant locant (`(13C)methane`) and cite isotopic hydrogen by count
(`(2H4)methane`).

Explicit `@`/`@@` centres are labeled `R`/`S` only when the CIP digraph
assigns them. A carbocyclic aromatic bond contributes one duplicate node, as
a double bond does, and a ring digit is ordered where it is written.
Heteroatom-containing mancude rings are ranked by P-92.1.4.4 with averaged
duplicate atomic numbers; this digraph does not apply that average, so a
heteroaryl ligand is not claimed to be fully CIP-ranked. A centre the digraph
cannot rank stays unlabeled: no cycle-cutting atomic-number order is
published as `R`/`S`, and the naming result warns that the token was left
unpublished. Phenyl β-D-glucuronide is therefore
`(2S,3S,4S,5R,6S)-3,4,5-trihydroxy-6-phenoxyoxane-2-carboxylic acid`.
A compound prefix keeps its multiplying prefix in alphabetical order, so
`(2,4-dimethylphenoxy)` is cited before `trihydroxy`. Unbranched polyenes
cite each double-bond configuration (`(2E,4E)-hexa-2,4-diene`). Repeated
identical nuclides are grouped (`(1,2-13C2)ethane`). Quinoline is the
retained parent of that fused benzopyridine.
Generated names are general IUPAC names, not preferred names.
The `strict=False` compatibility path deliberately preserves the lower-level
parser's unresolved hydrogen fields before one-way naming, so the same
unbracketed input may remain a composition description there.  Empty or
malformed OpenSMILES is a syntax error; strict conversion reports it as
`NamingIndeterminateError` while non-strict conversion retains
`LineNotationError`.  Non-strict conversion also intentionally does not enforce
the strict valence gate; an over-valent input can therefore remain a
composition description rather than being rejected.

Every `NamingResult` carries the result kind, standard/version, status, rule
trace, warnings, and alternatives. `strict=True` rejects provisional or
indeterminate results.

## Golden corpus policy

- `tests/data/chemistry_golden/` contains human-reviewed truth records in
  TOML and the reviewed SMILES reference corpus in JSON; these are deliberately
  not an external-engine dump.
- Stereo coordinates and invariance transformations live beside the engine in
  `tests/unit/chemistry/test_stereo.py` until the coordinate corpus schema is
  frozen.
- A new implemented row requires a positive golden, an indeterminate boundary
  case, and at least one identity-preserving transform/reordering test.
- This epic is **not release-complete** while any required rule-family row is
  marked partial or indeterminate. Such rows are acceptable on the feature
  branch only because public results expose their scope and strict refusal.
