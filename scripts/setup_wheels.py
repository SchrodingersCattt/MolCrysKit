#!/usr/bin/env python3
"""Copy wheels from /tmp/molcryskit_wheels to ./wheels as a real directory."""
import os
import shutil

src = "/tmp/molcryskit_wheels"
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wheels")

# Remove symlink or existing dir
if os.path.islink(dst):
    os.unlink(dst)
    print(f"Removed symlink: {dst}")
elif os.path.isdir(dst):
    shutil.rmtree(dst)
    print(f"Removed existing dir: {dst}")

# Copy as real directory
shutil.copytree(src, dst)
count = len(os.listdir(dst))
print(f"Copied {count} files to {dst}")
print("wheels is symlink:", os.path.islink(dst))
print("wheels is dir:", os.path.isdir(dst))
