# CSD molecular-crystal round-trip benchmark

This experiment tests the forward-construction contract of issue #172 and
records evidence for issue #198. It changes no production API. The target
package tree is frozen at `9bf0ffb0fb321d1084dbc444250b5f186af3c873`.

## Corpus and provenance

`manifest.json` freezes 1000 exact refcodes (200 per stratum) plus IBPRAC,
BENZEN02, HXMTAM10 and IQUHUY. Seed 42 shuffles the sorted CSD identifier
population. The targeted entries are excluded from random selection. Strata
have precedence: disorder/partial occupancy, ordered special position,
ordered multi-ion, remaining ordered multi-component, remaining single
component. Tags retain overlaps. A multi-ion label requires at least two
components with nonzero CCDC-supplied formal charge. Special positions are
classified independently using the complete component centroid and CCDC
operations, modulo translations of the original conventional cell.

Eligibility is 3D, carbon-containing, finite/non-polymeric structures.
Metals, solvent components and missing coordinates are retained. Family
suffixes identify distinct entries and are not collapsed. Screening errors
and exclusions are counted before freezing. No frozen failure is replaced.
Fixed quotas describe this test set; the aggregate is not a population-
weighted estimate for the entire CSD.

The live experiment uses CCDC API 3.7.2, CSD 6.01 and the active February,
May and September 2026 updates (1,469,017 entries at retrieval). The frozen
manifest records database metadata checksums. `source_environment.json`
records the actual database pool, not just every file found on disk.
Run environment files record Python, package versions, worker count and
the benchmark source hash. CSD access and coordinate redistribution remain
subject to the applicable CCDC licence.

## Match contract

No geometry optimization, missing-atom generation, rigid alignment, origin
fitting or cell scaling is performed. Different parsers' Cartesian axes are
normalized using their explicit lattice matrices, `X @ inv(L_parser) @ L_ref`;
this is a representation conversion, not a fitted rotation or basis search.

Allowed differences are molecule/atom ordering, chemically valid graph
automorphisms, and one common integer lattice translation per whole molecule.
Independent atom wrapping cannot rescue a broken molecule. Matching uses a
geometrically constrained graph isomorphism and a global one-to-one molecular
assignment. A centroid-distance prefilter is a necessary condition only; the
full atom and graph check remains authoritative. Ionic components and solvent
are never removed to make a match succeed.

The required tolerances are in `config.json`:

| Property | Absolute tolerance |
|---|---:|
| Maximum mapped atomic distance | 1e-4 Å |
| Corresponding cell lengths | 1e-6 Å |
| Cell angles | 1e-5 degrees |
| Operation translations modulo integer lattice vectors | 1e-7 fractional |
| Occupancies and molecular state weights | 1e-5 |

Known connectivity, bond orders, charges and stereochemistry must survive.
The CCDC API used here does not expose a reliable isotope field; its source
truth is explicitly unknown, and isotope preservation is tested synthetically.
Unknown annotations are never silently replaced by a known identity claim.
The target template API accepts untyped edges: inability to provide/preserve
known CSD bond orders is scored as a capability gap rather than a geometric
failure. Atom-identity mismatches are still checked when bond-order support
is missing. CIF readback chemistry is compared to the original CSD graph,
not the constructor's own graph.

`record_match` requires every applicable gate to pass. `molecular_completeness`
separately reports atoms present in the chemical model without coordinates;
these are counted, not placed. `geometry_roundtrip_match` is a deliberately
narrower ordered-structure score: export, in-memory geometry/cell/symmetry,
P1 and ASU independent geometry/composition, source-operation provenance,
and MolCrysKit readback must all pass; chemical annotation gates are excluded
only from this explicitly named subsidiary score.

## Independent references and disorder

Dataset preparation uses the licensed CCDC interpreter separately from the
MolCrysKit interpreter. It reuses the maintained exporter disorder lookup,
including normalization of zero-based CCDC group IDs. The benchmark exporter
writes 12-decimal cell/site/occupancy values and preserves group/assembly and
charge fields. Original CCDC components, graphs, missing atoms and unique
unit-cell packing are retained privately. CIF-derived CCDC components supply
the ordered poses; original CSD chemistry is a separate truth companion.
One representative of each molecular orbit is selected independently of the
target constructor. A fixed nontrivial SO(3) rotation changes template-local
coordinates, with the corresponding proper pose rotation restoring the
original measured geometry.

For disorder, CCDC's selected-molecule packing is not used as an atomic-average
oracle: it can choose one state or join alternatives into a connected union.
Original disordered sites are independently expanded, retaining chemical and
assembly/group identity and original atomic occupancies. Raw CIF site loops
are read with pymatgen without occupancy rescaling. This checks export
conservation even when the chemical union is not an ordered molecule.

The constructor is attempted using CCDC's selected complete molecular state
and its source chemistry, anchored at the centroid of the ordered scaffold
(or the component centroid when no scaffold exists). Alternative conformers
and their source group selections remain in the private companion. This
tests the target's symmetry-derived disorder path. Arbitrary positional,
compositional, unequal-weight or non-symmetry-related alternatives that cannot
be represented by that path are failures/limitations, not silently inferred
inverse models. The source atom occupancies are never replaced by group
weights, including H atoms for which the two API fields can differ.

Disorder gates check the in-memory marginal inventory, P1 average readback
with pymatgen and CCDC, source-operation provenance, known chemical identity,
local whole-molecule exclusivity, true-ASU output and reconstructable sidecar
semantics. Absent cross-site correlations remain `unknown_from_source`;
enumeration is not experimental evidence of independent or coupled sites.
At most 256 explicit source combinations and 256 generated local selections
are examined; truncation remains visible. A JSON file's existence or a list
of alternative IDs does not establish lossless model restoration.

## Running and recording

Run from a checkout containing the frozen target package tree, with its normal
Python dependencies installed. The CCDC interpreter must be the vendor's
licensed environment. In PowerShell:

```powershell
cmd /c '"D:\CCDC\ccdc-software\csd-python-api\miniconda\Scripts\activate.bat" && python -B -m benchmarks.csd_roundtrip.ccdc_adapter prepare'
cmd /c '"D:\CCDC\ccdc-software\csd-python-api\miniconda\Scripts\activate.bat" && python -B -m benchmarks.csd_roundtrip.ccdc_adapter export'
python -B -m benchmarks.csd_roundtrip.run --run-id baseline --workers 4 --ccdc-activate D:\CCDC\ccdc-software\csd-python-api\miniconda\Scripts\activate.bat
python -B -m benchmarks.csd_roundtrip.report --run-id baseline
```

The committed manifest is already frozen: `prepare` intentionally refuses to
overwrite it. To reproduce it, use a new benchmark copy and the same CSD pool;
to rerun existing cases, use `export` then `run`. `--targeted` runs only the
four targeted entries; `--refcodes IBPRAC HXMTAM10` selects frozen cases for
diagnosis. Use a new run ID after changing the harness. Resume rejects mixed
manifest/configuration, harness or cached-source hashes. The complete package
tree must still equal the target even if benchmark-only commits are newer.

Each evaluation case runs in a separate process with a 300-second deadline;
population screening and reusable source-cache preparation are timed outside
that evaluation deadline. Completed stages are checkpointed before subsequent
native calls. Timeout/crash results retain checkpoints and remain in the
denominator. Child process trees are terminated on timeout. The CCDC readback
helper exits immediately after atomically writing its complete snapshot to
avoid an observed Windows CCDC native-finalizer stall after successful work.
That helper does not turn failed parses into successful snapshots.

Statuses are `pass`, `mismatch`, `unsupported`, `indeterminate`, `error`, and
`timeout`. Earlier failures can prevent later gates from running; each stage's
observed denominator is visible. Sanity checks are diagnostics: pre-existing
source problems do not become reconstruction failures merely because they
survive unchanged. Higher accidental symmetry is reported by spglib at
1e-3 Å separately from verification of the requested operations at 1e-4 Å.

Git tracks code, configuration, refcodes, versions, checksums, numeric metrics,
JSONL/CSV results and Markdown reports. All coordinates, CIFs, molecule
templates, atom maps, sidecars, raw exception traces and native logs live in
ignored `_local/`. Public JSON uses an explicit projection; it never serializes
arbitrary source objects or traceback text. Published examples use refcodes
and commands, not licensed coordinates.

## Tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -p pytest_timeout tests/unit/benchmarks/test_csd_roundtrip.py tests/unit/operations/test_forward.py tests/unit/io/test_forward_fixtures.py -q
python -B -m pytest -p pytest_timeout tests -q
ruff check benchmarks/csd_roundtrip tests/unit/benchmarks/test_csd_roundtrip.py
```

The benchmark unit tests use only synthetic structures. They do not import
CCDC, use a licence, or retrieve structures in CI. On Windows, the four existing
forward-fixture checksum tests require byte-identical Git blobs; automatic
CRLF conversion can change their hashes without changing their chemistry.
