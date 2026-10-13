# Bond-order benchmark package (issue #174)

This directory is a self-contained, conservative benchmark harness.  The
micro/gold manifest is frozen in `data/manifest.json` and contains independent,
hand-authored graph truth.  It includes named regressions for the old Cartesian
product limit (`c13h28_long_chain`), first-64 truncation (`first_64_premature_cut`), order invariance, missing charge/H metadata,
correct periodic image vectors, aromatic resonance, and metal coordination.
Null truth is intentional and means “unknown”; it is scored separately from a
wrong assertion.  Open Babel is a comparator pinned to 3.2.1, not an oracle.

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
exits nonzero for any missing comparison, invalid manifest, incomplete result,
unavailable corpus, fewer than seven independent release processes, or failed
superiority gate; it never fabricates a baseline or a win.  The release contract
requires 130,831 QM9 records, 1,000 independently curated external molecules,
and 100 attributed experimental CIFs.  These corpora are explicitly marked
`missing` in the frozen manifest until provisioned with source IDs, licenses and
checksums; a CI micro run cannot stand in for the release suite.

Open Babel setup and version verification are documented in
`openbabel/README.md`.  The current adapter intentionally refuses to claim
stable-ID-complete scoring from SMILES-only Open Babel output.  This keeps the
head-to-head result auditable while a source-aware adapter is prepared.

The current adapter records one observed interpreter (`processes_mode=single-process adapter`) even when `--processes N` is requested; this prevents metadata from manufacturing the required seven-process release evidence. A release run remains incomplete until independent process orchestration is added and observed.
