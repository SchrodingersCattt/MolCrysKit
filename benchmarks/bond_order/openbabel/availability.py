"""CLI availability probe; no package installation is attempted."""
from __future__ import annotations

import argparse
import json
from . import baseline_metadata, native_binary, python_bindings_available


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    data = {**baseline_metadata(), 'native_binary': native_binary(), 'python_bindings': python_bindings_available()}
    if a.json: print(json.dumps(data, indent=2))
    else:
        print(f"Open Babel {data['version']} commit {data['commit']}")
        print(f"native_binary={data['native_binary'] or 'missing'} python_bindings={data['python_bindings']}")
    return 0

if __name__ == '__main__': raise SystemExit(main())
