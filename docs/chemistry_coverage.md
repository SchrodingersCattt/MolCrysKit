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
| Bond order and formal charge | partial | Bounded main-group valence solver retains competitive alternatives. Metals, mixed valence, and delocalized systems may be provisional. |
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
| Blue Book 2013 | partial | Parent hydrides, straight-chain alkanes/alcohols/carboxylic acids, benzene/phenol with simple substituents, and N-(hydroxyphenyl)alkanamides. |
| Red Book 2005 | partial | Deterministic composition/dimensionality descriptions; general additive coordination naming is not released. |
| Purple Book 2008 | partial | Single named repeat-unit `poly(...)` result is provisional; end groups and typed connections remain required. |
| Full PIN selection | indeterminate outside rows above | `preferred` is never asserted for fallback composition descriptions. |
| Name → structure | partial | Exact canonical forms emitted by the self-contained reversible subset parser are accepted; general IUPAC names, synonyms, stereochemical names, and unsupported entity classes fail closed. |

The reversible conversion API covers neutral finite covalent entities only:
water, azane, C1–C12 straight-chain hydrocarbons, corresponding alcohols and
carboxylic acids, benzene/phenol with simple halogen, methyl, and hydroxy
substituents, and the supported `N-(hydroxyphenyl)alkanamide` forms.  Use
`name_entity()` when a one-way composition description is acceptable; use
`smiles_to_iupac(strict=True)` when a name must round-trip through
`iupac_to_smiles()`.  Strict conversion applies OpenSMILES default valences to
unbracketed organic-subset atoms (`CCO` is therefore accepted); bracket atoms
such as `[C]` retain their explicitly requested hydrogen semantics.
The `strict=False` compatibility path deliberately preserves the lower-level
parser's unresolved hydrogen fields before one-way naming, so the same
unbracketed input may remain a composition description there.  Empty or
malformed OpenSMILES is a syntax error; strict conversion reports it as
`NamingIndeterminateError` while non-strict conversion retains
`LineNotationError`.

Every `NamingResult` carries the result kind, standard/version, status, rule
trace, warnings, and alternatives. `strict=True` rejects provisional or
indeterminate results.

## Golden corpus policy

- `tests/data/chemistry_golden/` contains human-reviewed truth records in
  TOML; it is deliberately not an external-engine dump.
- Stereo coordinates and invariance transformations live beside the engine in
  `tests/unit/chemistry/test_stereo.py` until the coordinate corpus schema is
  frozen.
- A new implemented row requires a positive golden, an indeterminate boundary
  case, and at least one identity-preserving transform/reordering test.
- This epic is **not release-complete** while any required rule-family row is
  marked partial or indeterminate. Such rows are acceptable on the feature
  branch only because public results expose their scope and strict refusal.
