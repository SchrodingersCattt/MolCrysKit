# Docker Guide

Two Dockerfiles are provided for different environments:

| File | Base image | Use case |
|------|-----------|----------|
| `Dockerfile` | `python:3.10-slim` | Local use, reviewers, CI |
| `Dockerfile.bohrium` | `registry.dp.tech/dptech/ubuntu:ubuntu24.04-py3.12` | Bohrium cloud platform |

Both install MolCrysKit directly from the GitHub archive. A local Docker build context is still used to start the build, but the package, notebook assets, and helper scripts are fetched from the selected GitHub ref inside the image build.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows)
  or the Docker Engine (Linux).

## Quick Start (General Use)

```bash
# 1. Clone the repository and enter the package directory
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit

# 2. Build the image (≈ 5–10 min on first run; subsequent builds use the cache)
docker build -t molcryskit:latest .

# 3. Run the smoke test to confirm everything works
docker run --rm molcryskit:latest python /opt/molcryskit/scripts/docker_smoke_test.py

# 4. Start the Jupyter notebook server
docker run -it --rm -p 8888:8888 molcryskit:latest
# Then open http://localhost:8888 in your browser.
# Example CIF files are available at /workspace/notebook/example/ inside the container.
```

## One-Step Build + Test Helper

```bash
# From the MolCrysKit/ directory:
bash scripts/docker-test.sh
```

This script builds the image and runs the smoke test automatically, reporting
`ALL CHECKS PASSED` on success.

## Bohrium Cloud Platform

```bash
# Build with the Bohrium-specific Dockerfile
docker build -f Dockerfile.bohrium -t molcryskit-bohrium:latest .

# Pin to an immutable Git tag instead of the moving main branch
# (recommended for archival/reviewer reproducibility)
docker build -f Dockerfile.bohrium \
    --build-arg MOLCRYSKIT_REF=refs/tags/v0.4.0 \
    -t molcryskit-bohrium:v0.4.0 .
```

The Bohrium image uses `pip install` from the GitHub archive zip (no `git clone`
required) and does not include Jupyter — Bohrium provides its own notebook
environment.

## Permanent Image Publication with GHCR

The Bohrium registry is convenient for cloud execution, but it should not be the
only archival location because image retention is controlled by the platform and
project namespace. For a stable public anchor, publish immutable release images
to GitHub Container Registry (GHCR).

The workflow [`publish-ghcr.yml`](../.github/workflows/publish-ghcr.yml) pushes
[`Dockerfile`](../Dockerfile) images to `ghcr.io/<owner>/molcryskit`:

- pushing a Git tag such as `v0.4.0` publishes `ghcr.io/<owner>/molcryskit:v0.4.0`
- stable release tags also receive `latest`
- manual dispatch can publish a development snapshot from a chosen Git ref

Recommended archival pattern:

```bash
# 1. Create and push an immutable release tag
git tag v0.4.0
git push origin v0.4.0

# 2. GitHub Actions publishes the image automatically to GHCR
#    ghcr.io/<owner>/molcryskit:v0.4.0
```

For Bohrium, keep using [`Dockerfile.bohrium`](../Dockerfile.bohrium) as the
platform-specific runtime image, but cite the GitHub repository and GHCR image
as the permanent public anchor.

## Mounting Your Own Data

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v /path/to/your/cif/files:/workspace/my_data \
    molcryskit:latest
```

Your files will be accessible at `/workspace/my_data/` inside the container.
