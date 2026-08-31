#!/usr/bin/env python3
"""
Merge cp310 wheels into the main wheels/ directory.
Remove cp312 wheels that have a cp310 replacement, then copy in the cp310 ones.
"""
import os, shutil, re

wheels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wheels")
cp310_dir = "/tmp/molcryskit_wheels_cp310"

def pkg_name(filename):
    """Extract normalised package name from wheel filename."""
    return re.split(r"[-_]", filename)[0].lower().replace("-", "_")

# Build map of package name -> cp310 wheel filename
cp310_wheels = {}
for f in os.listdir(cp310_dir):
    if f.endswith(".whl"):
        cp310_wheels[pkg_name(f)] = f

print(f"cp310 wheels available: {len(cp310_wheels)}")

# Remove cp312 wheels that have a replacement
removed = []
for f in list(os.listdir(wheels_dir)):
    if "cp312" in f and f.endswith(".whl"):
        name = pkg_name(f)
        if name in cp310_wheels:
            os.remove(os.path.join(wheels_dir, f))
            removed.append(f)
            print(f"  Removed: {f}")

print(f"Removed {len(removed)} cp312 wheels")

# Copy cp310 wheels in
copied = []
for name, f in cp310_wheels.items():
    dst = os.path.join(wheels_dir, f)
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(cp310_dir, f), dst)
        copied.append(f)
        print(f"  Copied: {f}")
    else:
        print(f"  Already present: {f}")

print(f"\nCopied {len(copied)} cp310 wheels")
print(f"Total wheels in {wheels_dir}: {len(os.listdir(wheels_dir))}")
print("\nRemaining cp312 wheels:")
for f in sorted(os.listdir(wheels_dir)):
    if "cp312" in f:
        print(f"  {f}")
