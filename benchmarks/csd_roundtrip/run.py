"""Run frozen cases in isolated processes with resumable version guards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import os
import platform
import subprocess
import sys
import time

from .common import (
    LOCAL,
    ROOT,
    config,
    harness_hash,
    public_result,
    read_json,
    verify_manifest,
    verify_target,
    write_json,
)


def compatible_resume(previous, metadata):
    return all(
        previous.get(k) == metadata[k] for k in ("manifest_sha256", "harness_sha256")
    )


def execute(case, metadata, run_dir):
    ref = case["refcode"]
    result_path = run_dir / "private_results" / (ref + ".json")
    if result_path.exists():
        previous = read_json(result_path)
        if not compatible_resume(previous, metadata):
            raise ValueError("resume version mismatch for " + ref)
        source_path = LOCAL / "cases" / ref / "source.json"
        if source_path.exists() and previous.get("source_sha256") not in (
            None,
            read_json(source_path)["source_sha256"],
        ):
            raise ValueError(
                "private source changed since the previous result for " + ref
            )
        if previous.get("finished", True):
            return public_result(previous)
    case_file = run_dir / "requests" / (ref + ".json")
    write_json(case_file, case)
    log = run_dir / "logs" / (ref + ".txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "benchmarks.csd_roundtrip.worker",
                "--case",
                str(case_file),
                "--metadata",
                str(run_dir / "metadata.json"),
                "--result",
                str(result_path),
            ],
            cwd=ROOT.parents[1],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=config()["timeout_s"])
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
            )
            process.wait()
            result = (
                read_json(result_path)
                if result_path.exists()
                else {**case, **metadata, "stages": {}}
            )
            result.update(
                status="timeout",
                record_match=False,
                finished=True,
                elapsed_s=round(time.monotonic() - started, 4),
            )
            result["stages"]["execution"] = {
                "status": "timeout",
                "code": "case_300s_limit",
            }
            write_json(result_path, result)
            return public_result(result)
    if code or not result_path.exists():
        write_json(
            result_path,
            {
                **case,
                **metadata,
                "status": "error",
                "record_match": False,
                "elapsed_s": round(time.monotonic() - started, 4),
                "stages": {
                    "execution": {"status": "error", "code": "worker_process_failed"}
                },
            },
        )
    return public_result(read_json(result_path))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--refcodes", nargs="+")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ccdc-activate", required=True)
    args = p.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        p.error("run-id must be alphanumeric, underscore or hyphen")
    verify_target()
    manifest = read_json(ROOT / "manifest.json")
    verify_manifest(manifest)
    hh = harness_hash()
    run_dir = LOCAL / "runs" / args.run_id
    public_dir = ROOT / "reports" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"manifest_sha256": manifest["sha256"], "harness_sha256": hh}
    meta_file = run_dir / "metadata.json"
    if meta_file.exists() and read_json(meta_file) != metadata:
        raise ValueError("run-id belongs to a different harness or manifest")
    write_json(meta_file, metadata)
    versions = {}
    for name in (
        "molcrys-kit",
        "numpy",
        "scipy",
        "networkx",
        "ase",
        "pymatgen",
        "spglib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(
        public_dir / "environment.json",
        {
            **metadata,
            "target_commit": config()["target_commit"],
            "python": platform.python_version(),
            "platform": platform.system(),
            "packages": versions,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "ccdc": read_json(ROOT / "source_environment.json")
            if (ROOT / "source_environment.json").exists()
            else manifest["environment"],
            "workers": args.workers,
            "timeout_s": config()["timeout_s"],
        },
    )
    os.environ["CSD_ROUNDTRIP_ACTIVATE"] = args.ccdc_activate
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    selected = [
        c
        for c in manifest["cases"]
        if (not args.targeted or c["targeted"])
        and (not args.refcodes or c["refcode"] in args.refcodes)
    ]
    if args.refcodes and set(args.refcodes) - {c["refcode"] for c in selected}:
        raise ValueError("requested refcode is not in the frozen selection")
    results = {}
    output = public_dir / "results.jsonl"
    if output.exists():
        import json

        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not compatible_resume(row, metadata):
                raise ValueError("public resume version mismatch")
            results[row["refcode"]] = row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(execute, case, metadata, run_dir): case for case in selected
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["refcode"]] = result
            import json

            temp = output.with_suffix(".tmp")
            temp.write_text(
                "".join(
                    json.dumps(results[r], sort_keys=True, allow_nan=False) + "\n"
                    for r in sorted(results)
                ),
                encoding="utf-8",
            )
            temp.replace(output)
            print(
                len(results),
                "/",
                len(selected),
                result["refcode"],
                result["status"],
                result.get("elapsed_s"),
                flush=True,
            )
    from .report import generate

    generate(args.run_id)


if __name__ == "__main__":
    main()
