#!/usr/bin/env python3
"""Run docker build and capture all output."""
import subprocess, sys, os

os.chdir("/aisi-nas/guomingyu/personal/MolCrysKit")

cmd = [
    "docker", "build", "--no-cache",
    "-f", "Dockerfile.local",
    "-t", "molcryskit:latest",
    "."
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")
