# Bond-order benchmark package

This directory is a self-contained, conservative benchmark harness.  The
micro/gold manifest is frozen in `data/manifest.json` and contains independent,
hand-authored graph truth.  It includes named regressions for the old Cartesian
product limit (`c13h28_long_chain`), first-64 truncation (`first_64_premature_cut`), order invariance, missing charge/H metadata,
correct periodic image vectors, aromatic resonance, and metal coordination.
Null truth is intentional and means “unknown”; it is scored separately from a
wrong assertion.  Open Babel is a comparator pinned to 3.2.1, not an oracle.

Functional coverage for #174 is the chemistry engine and the section 4.1
tests.  Paired Open Babel 3.2.1 runs, the section 5 superiority gates,
latency, throughput, RSS, QM9, and the curated corpora are
[issue #176](https://github.com/SchrodingersCattt/MolCrysKit/issues/176).

```bash
python -m benchmarks.bond_order.prepare --verify-manifest
python -m benchmarks.bond_order.prepare --describe
python -m benchmarks.bond_order.run --suite ci --engine all --output work/bond-order-results
python -m benchmarks.bond_order.score --input work/bond-order-results/ci.json \
  --output work/bond-order-score.json
python -m benchmarks.bond_order.report --input work/bond-order-score.json --strict-gates
```

`run` retains every case, raw timing, status, environment and error.  Missing,
incomplete, crashed or timed-out engines remain visible.  `score --strict-gates`
exits nonzero when a selected MolCrysKit case is missing or incomplete, its
known-truth order gate is empty or incorrect, a prediction asserts an order
where the truth is unknown, the manifest checksum does not match, or the Python
environment was not recorded.  Open Babel superiority, latency,
throughput, RSS, process count, QM9, and the curated corpora stay on the
scorecard as `deferred-superiority` and do not change that exit code.  They
never fabricate a baseline or a win.  The 130,831 QM9 records, 1,000 external
molecules, 100 experimental CIFs, and the seven-process paired comparison are
[issue #176](https://github.com/SchrodingersCattt/MolCrysKit/issues/176).  Those
corpora stay `missing` in the frozen manifest until that issue provisions
source IDs, licenses, and checksums.

Open Babel setup and version verification are documented in
`openbabel/README.md`.  The current adapter intentionally refuses to claim
stable-ID-complete scoring from SMILES-only Open Babel output.  This keeps the
head-to-head result auditable while a source-aware adapter is prepared.

The current adapter records one observed interpreter (`processes_mode=single-process adapter`) even when `--processes N` is requested; this prevents metadata from manufacturing seven-process evidence. Independent process orchestration is part of issue #176 and is not required for the functional `--strict-gates` check.

Fixtures with drawn H atoms record `attached_hydrogens` separately and set
`implicit_hydrogens` to zero for those atoms.  This keeps atom/H conservation
checks from counting the same hydrogen twice.  Periodic edge identities include
their oriented image shift, so repeated unit-cell edges remain independently
scorable.
