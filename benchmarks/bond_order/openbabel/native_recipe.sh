#!/usr/bin/env bash
# Reproducible Open Babel 3.2.1 native comparator build.  This is never
# invoked by MolCrysKit production code and does not modify the source tree.
set -euo pipefail
prefix=${1:-"$PWD/openbabel-3.2.1-build"}
source_dir=${OPENBABEL_SOURCE_DIR:-"$PWD/openbabel-3.2.1-src"}
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch openbabel-3-2-1 --depth 1 https://github.com/openbabel/openbabel.git "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "0e94434fa75c9f61095023e3c12e0d5f2ac035ff" ]]
cmake -S "$source_dir" -B "$prefix" -DCMAKE_BUILD_TYPE=Release \
  -DOPENBABEL_BUILD_TESTS=OFF -DBUILD_GUI=OFF -DRUN_SWIG=OFF
cmake --build "$prefix" --parallel
printf 'OBABEL_BIN=%s\n' "$prefix/bin/obabel"
