# MolCrysKit Docker Image
# ========================
# Provides a reproducible, offline-runnable environment for MolCrysKit
# and the bundled example notebooks / CIF files.
#
# Build (run from the MolCrysKit/ directory):
#   docker build -t molcryskit:latest .
#
# Pin to a specific release tag (recommended for reproducibility):
#   docker build --build-arg MOLCRYSKIT_REF=v0.1.0 -t molcryskit:v0.1.0 .
#
# Run - interactive shell:
#   docker run -it --rm molcryskit:latest bash
#
# Run - Jupyter notebook server (open http://localhost:8888 in your browser):
#   docker run -it --rm -p 8888:8888 molcryskit:latest
#
# Run - built-in smoke test:
#   docker run --rm molcryskit:latest python /opt/molcryskit/scripts/docker_smoke_test.py
#
# For Bohrium cloud use, see Dockerfile.bohrium instead.

FROM python:3.10-slim

LABEL maintainer="Ming-Yu Guo <guomy26@mail2.sysu.edu.cn>"
LABEL description="MolCrysKit: A Topology-Aware Toolkit for Molecular Crystal Preprocessing"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/SchrodingersCattt/MolCrysKit"
LABEL org.opencontainers.image.licenses="MIT"

# -- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -- Python build tools -----------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# -- MolCrysKit + all runtime dependencies from GitHub archive --------------
# pip resolves all dependencies declared in pyproject.toml automatically.
# The [vis] extra adds nglview and py3Dmol for 3-D visualisation in the notebook.
ARG MOLCRYSKIT_REF=main
RUN pip install --no-cache-dir \
    "molcrys-kit[vis] @ https://github.com/SchrodingersCattt/MolCrysKit/archive/refs/heads/${MOLCRYSKIT_REF}.zip"

# -- Jupyter ----------------------------------------------------------------
# jupyter-server >=2 (shipped with notebook >=7) uses ServerApp, not NotebookApp.
# Pin notebook<7 to keep the classic interface and the NotebookApp config below.
RUN pip install --no-cache-dir \
    "notebook>=6.4,<7" \
    jupyter \
    ipykernel \
    matplotlib

# -- Notebook + scripts from same GitHub ref --------------------------------
# The pip install above installs the Python package only (molcrys_kit/).
# Download the full archive to extract notebook/ and scripts/ as well.
RUN mkdir -p /opt/molcryskit && \
    curl -sL \
      "https://github.com/SchrodingersCattt/MolCrysKit/archive/refs/heads/${MOLCRYSKIT_REF}.tar.gz" \
    | tar xz \
        --wildcards \
        --strip-components=1 \
        -C /opt/molcryskit \
        "*/notebook" \
        "*/scripts"

# -- Workspace setup --------------------------------------------------------
WORKDIR /workspace
RUN cp -r /opt/molcryskit/notebook ./notebook

# -- Jupyter configuration --------------------------------------------------
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'"               >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True"             >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False"          >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''"                    >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''"                 >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/workspace'"   >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", \
     "--notebook-dir=/workspace"]
