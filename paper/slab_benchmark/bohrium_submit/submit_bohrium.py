"""
Submit multi-system slab benchmark to Bohrium via dpdispatcher.

This is the exact driver used to produce the timing data committed at
``paper/slab_benchmark/results/multi_system_benchmark.json`` (Figure 4a2).

Usage:
    # Set credentials first (no credentials are committed):
    export BOHRIUM_USERNAME="your_bohrium_email"
    export BOHRIUM_PASSWORD="your_bohrium_password"
    export BOHRIUM_PROJECT_ID="your_project_id"

    python submit_bohrium.py

The script will:
1. Upload slab_efficiency.py + slab_examples/ CIFs to Bohrium
2. Submit job with c4_m8_cpu machine
3. Wait for completion and download results

Credentials are read from environment variables:
    BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID
"""

import os
import sys
import time
import json
from pathlib import Path

# ── dpdispatcher imports ───────────────────────────────────────────────────────
from dpdispatcher import Machine, Resources, Task, Submission

# ── config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
WORK_BASE = str(SCRIPT_DIR)  # local work directory

# Bohrium credentials from environment
BOHRIUM_EMAIL = os.environ.get("BOHRIUM_USERNAME", "guomingyu@dp.tech")
BOHRIUM_PASSWORD = os.environ.get("BOHRIUM_PASSWORD", "")
BOHRIUM_PROJECT_ID = int(os.environ.get("BOHRIUM_PROJECT_ID", "27666"))

# Job parameters — image has MolCrysKit + all deps pre-installed
IMAGE = "registry.dp.tech/dptech/dp/native/prod-19853/molcrys-kit:dev260325"
MACHINE_TYPE = "c4_m8_cpu"
PLATFORM = "ali"

# No install needed: image is pre-built with molcrys_kit + ase + pymatgen + matplotlib
# Note: do NOT use "exec > eff.log 2>&1" — that prevents Bohrium from collecting output files.
# Instead write eff.log via python-level redirection inside slab_efficiency.py (already done via
# THIS_DIR/multi_system_benchmark.json). eff.log is written by python script itself.
REMOTE_COMMAND = (
    "set -ex; "
    "echo 'v9-preinstalled-image'; "
    "python -c 'import molcrys_kit; print(\"molcrys_kit OK\")'; "
    "python slab_efficiency.py 2>&1 | tee eff.log"
)

print(f"Bohrium email: {BOHRIUM_EMAIL}")
print(f"Project ID: {BOHRIUM_PROJECT_ID}")
print(f"Image: {IMAGE}")
print(f"Machine: {MACHINE_TYPE}")
print(f"Work base: {WORK_BASE}")

# ── machine config ─────────────────────────────────────────────────────────────
machine_dict = {
    "batch_type": "Bohrium",
    "context_type": "Bohrium",
    "local_root": str(SCRIPT_DIR),
    "remote_root": "/tmp/mck_benchmark",
    "remote_profile": {
        "email": BOHRIUM_EMAIL,
        "password": BOHRIUM_PASSWORD,
        "project_id": BOHRIUM_PROJECT_ID,
        "input_data": {
            "job_type": "container",
            "platform": PLATFORM,
            "scass_type": MACHINE_TYPE,
            "image_address": IMAGE,
            "job_name": "mck_multi_system_benchmark",
            "log_file": "eff.log",
            "backward_files": [
                "eff.log",
                "multi_system_benchmark.json",
                "multi_system_benchmark_figure.pdf",
                "multi_system_benchmark_figure.png",
            ],
        },
    },
}

# ── resources config ───────────────────────────────────────────────────────────
resources_dict = {
    "number_node": 1,
    "cpu_per_node": 4,
    "gpu_per_node": 0,
    "queue_name": MACHINE_TYPE,
    "group_size": 1,
}

# ── task config ────────────────────────────────────────────────────────────────
# forward_files: no zip needed — image has molcrys_kit pre-installed
task_dict = {
    "command": REMOTE_COMMAND,
    "task_work_path": "./",
    "forward_files": [
        "slab_efficiency.py",
        "slab_examples/Acetaminophen_HXACAN.cif",
        "slab_examples/beta-HMX_OCHTET12.cif",
        "slab_examples/DAP-M4.cif",
    ],
    "backward_files": [
        "eff.log",
        "multi_system_benchmark.json",
        "multi_system_benchmark_figure.pdf",
        "multi_system_benchmark_figure.png",
    ],
    "outlog": "dpdispatcher_out.log",
    "errlog": "dpdispatcher_err.log",
}

# ── CRLF fix: patch gen_script to always use Unix line endings ─────────────────
# dpdispatcher on Windows generates \r\n in .sub files, which breaks bash on Linux
from dpdispatcher.machine import Machine as _Machine

_orig_gen_script = _Machine.gen_script

def _patched_gen_script(self, job):
    script = _orig_gen_script(self, job)
    return script.replace('\r\n', '\n').replace('\r', '\n')

_Machine.gen_script = _patched_gen_script

# ── submit ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("Submitting benchmark to Bohrium")
    print("="*60)

    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task = Task(**task_dict)

    submission = Submission(
        work_base=WORK_BASE,
        machine=machine,
        resources=resources,
        task_list=[task],
        forward_common_files=[],
        backward_common_files=[],
    )

    print("\nRunning submission (this will block until job completes)...")
    print("You can check job status at https://bohrium.dp.tech/")
    print()

    submission.run_submission()

    print("\n" + "="*60)
    print("Job completed! Results downloaded to:")
    print(f"  {WORK_BASE}/eff.log")
    print(f"  {WORK_BASE}/multi_system_benchmark.json")
    print(f"  {WORK_BASE}/multi_system_benchmark_figure.pdf")
    print("="*60)


if __name__ == "__main__":
    main()
