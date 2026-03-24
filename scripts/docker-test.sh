#!/usr/bin/env bash
# scripts/docker-test.sh
# ======================
# Build the MolCrysKit Docker image and run the bundled smoke test.
# Usage:  bash scripts/docker-test.sh [IMAGE_TAG]
#
# Requirements: Docker must be installed and the daemon must be running.
# Run this script from the MolCrysKit/ directory (repo root):
#   cd MolCrysKit && bash scripts/docker-test.sh

set -euo pipefail

IMAGE="${1:-molcryskit:latest}"
# scripts/ lives one level below the repo root (MolCrysKit/).
# The Docker build context must be the repo root so Dockerfile can be found.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================"
echo " MolCrysKit Docker Build & Smoke Test"
echo "============================================"
echo "Image  : ${IMAGE}"
echo "Context: ${REPO_ROOT}"
echo ""

# -- Step 1: Build ----------------------------------------------------------
echo ">>> Step 1/2: Building Docker image..."
docker build -t "${IMAGE}" "${REPO_ROOT}"
echo ""

# -- Step 2: Smoke test -----------------------------------------------------
echo ">>> Step 2/2: Running smoke test inside container..."
# Temporarily disable errexit so we can capture the exit code and print a
# human-readable summary before propagating it.
set +e
docker run --rm "${IMAGE}" python /opt/molcryskit/scripts/docker_smoke_test.py
STATUS=$?
set -e

echo ""
if [ "${STATUS}" -eq 0 ]; then
    echo "============================================"
    echo " ALL CHECKS PASSED"
    echo "============================================"
    echo ""
    echo "To start the Jupyter notebook server run:"
    echo "  docker run -it --rm -p 8888:8888 ${IMAGE}"
    echo "Then open http://localhost:8888 in your browser."
else
    echo "============================================"
    echo " SMOKE TEST FAILED (exit code ${STATUS})"
    echo "============================================"
fi

exit "${STATUS}"
