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
# Run – interactive shell:
#   docker run -it --rm molcryskit:latest bash
#
# Run – Jupyter notebook server (open http://localhost:8888 in your browser):
#   docker run -it --rm -p 8888:8888 molcryskit:latest
#
# Run – built-in smoke test:
#   docker run --rm molcryskit:latest python /opt/molcryskit/docker_smoke_test.py

FROM python:3.10-slim

LABEL maintainer="Ming-Yu Guo <guomy26@mail2.sysu.edu.cn>"
LABEL description="MolCrysKit: A Topology-Aware Toolkit for Molecular Crystal Preprocessing"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/SchrodingersCattt/MolCrysKit"
LABEL org.opencontainers.image.licenses="MIT"

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    "numpy>=1.18.0" \
    "scipy>=1.4.0" \
    "networkx>=2.5.0"

RUN pip install --no-cache-dir "ase>=3.26.0"
RUN pip install --no-cache-dir pymatgen

# jupyter-server >=2 (shipped with notebook >=7) uses ServerApp, not NotebookApp.
# Pin notebook<7 to keep the classic interface and the NotebookApp config below.
RUN pip install --no-cache-dir \
    "notebook>=6.4,<7" \
    jupyter \
    ipykernel \
    matplotlib \
    nglview \
    py3Dmol

# ── Clone & Install MolCrysKit from GitHub ───────────────────────────────────
# ARG lets callers pin to a release tag or commit SHA:
#   docker build --build-arg MOLCRYSKIT_REF=v0.1.0 ...
ARG MOLCRYSKIT_REF=main
RUN git clone --depth=1 --branch ${MOLCRYSKIT_REF} \
        https://github.com/SchrodingersCattt/MolCrysKit.git \
        /opt/molcryskit \
    && cd /opt/molcryskit \
    && pip install --no-cache-dir . \
    && rm -rf /opt/molcryskit/.git

# ── Bundled smoke test ────────────────────────────────────────────────────────
# docker_smoke_test.py will be part of the repo once pushed to GitHub.
# Until then, COPY it from local build context.
# TODO: remove this COPY once docker_smoke_test.py is in the GitHub repo.
COPY docker_smoke_test.py /opt/molcryskit/docker_smoke_test.py

# ── Notebook + example CIF files ─────────────────────────────────────────────
# notebook/ (containing molcryskit.ipynb and example/) will be part of the repo
# once pushed to GitHub. Until then, COPY from local build context.
# TODO: remove this COPY once notebook/ is in the GitHub repo, and instead use:
#   RUN cp -r /opt/molcryskit/notebook /workspace/notebook
WORKDIR /workspace
COPY notebook/ ./notebook/

# ── Jupyter configuration ─────────────────────────────────────────────────────
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
