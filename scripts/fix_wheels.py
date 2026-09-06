#!/usr/bin/env python3
"""
Find all wheels in ./wheels/ that are incompatible with Python 3.10,
then download compatible replacements.
"""
import subprocess, os, sys, re, zipfile, email

wheels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wheels")

def get_wheel_metadata(whl_path):
    """Extract requires-python from wheel metadata."""
    try:
        with zipfile.ZipFile(whl_path) as z:
            for name in z.namelist():
                if name.endswith('/METADATA') or name.endswith('.dist-info/METADATA'):
                    content = z.read(name).decode('utf-8', errors='replace')
                    for line in content.splitlines():
                        if line.startswith('Requires-Python:'):
                            return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return None

def is_compatible_with_310(requires_python):
    """Check if requires-python spec allows Python 3.10."""
    if not requires_python:
        return True
    from packaging.specifiers import SpecifierSet
    try:
        return "3.10" in SpecifierSet(requires_python)
    except Exception:
        return True

# Scan all wheels
print("Scanning wheels/ for Python 3.10 compatibility...")
incompatible = []
for f in sorted(os.listdir(wheels_dir)):
    if not f.endswith('.whl'):
        continue
    path = os.path.join(wheels_dir, f)
    req_py = get_wheel_metadata(path)
    if req_py:
        compat = is_compatible_with_310(req_py)
        if not compat:
            pkg_name = re.split(r'[-_]', f)[0].lower()
            print(f"  INCOMPATIBLE: {f}  (Requires-Python: {req_py})")
            incompatible.append((pkg_name, f))

if not incompatible:
    print("All wheels are compatible with Python 3.10!")
    sys.exit(0)

print(f"\nFound {len(incompatible)} incompatible wheels. Downloading replacements...")
for pkg_name, old_file in incompatible:
    # Remove old
    os.remove(os.path.join(wheels_dir, old_file))
    print(f"  Removed: {old_file}")
    
    # Download compatible version
    result = subprocess.run(
        ["pip", "download", f"{pkg_name}<100", "--dest", wheels_dir,
         "--no-deps", "--python-version", "3.10",
         "--platform", "manylinux_2_17_x86_64",
         "--implementation", "cp", "--abi", "cp310",
         "--only-binary=:all:"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        saved = [l for l in result.stdout.splitlines() if "Saved" in l or "already" in l.lower()]
        print(f"  Downloaded: {saved[0].strip() if saved else 'OK'}")
    else:
        # Try pure python fallback
        result2 = subprocess.run(
            ["pip", "download", f"{pkg_name}<100", "--dest", wheels_dir,
             "--no-deps", "--python-version", "3.10",
             "--only-binary=:all:"],
            capture_output=True, text=True
        )
        if result2.returncode == 0:
            saved = [l for l in result2.stdout.splitlines() if "Saved" in l or "already" in l.lower()]
            print(f"  Downloaded (relaxed): {saved[0].strip() if saved else 'OK'}")
        else:
            print(f"  FAILED to download {pkg_name}: {result2.stderr[-200:]}")

print("\nDone!")
