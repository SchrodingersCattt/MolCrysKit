# Open Babel 3.2.1 native baseline

The comparator is pinned to Open Babel tag `openbabel-3-2-1`, commit
`0e94434fa75c9f61095023e3c12e0d5f2ac035ff`.  It is a benchmark dependency,
never a MolCrysKit production dependency or a ground-truth oracle.

A clean native build (outside the MolCrysKit environment) is:

```bash
git clone --branch openbabel-3-2-1 --depth 1 https://github.com/openbabel/openbabel.git
cd openbabel
# Verify the checkout before configuring:
test "$(git rev-parse HEAD)" = 0e94434fa75c9f61095023e3c12e0d5f2ac035ff
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOPENBABEL_BUILD_TESTS=OFF -DBUILD_GUI=OFF -DRUN_SWIG=OFF
cmake --build build --parallel
# Point the adapter at the resulting executable:
export OBABEL_BIN="$PWD/build/bin/obabel"
```

The adapter calls the same public command-line conversion path used for the
pinned release and records the raw output.  A SMILES-only output cannot prove
the stable fixture atom-ID mapping (especially for omitted hydrogens), so it is
reported as `incomplete`, never silently scored as a win.  A source-aware native
adapter must be supplied before a release comparison can be complete.

For the optional Python comparator, install the bindings built from this exact
checkout and run with the same interpreter.  The benchmark records missing
bindings as `unavailable` and does not install Open Babel automatically.
