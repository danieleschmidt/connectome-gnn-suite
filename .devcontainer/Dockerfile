# Development container for Connectome-GNN-Suite
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        graphviz \
        graphviz-dev \
        libgraphviz-dev \
        pkg-config \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libhdf5-dev \
        libfftw3-dev \
        libeigen3-dev \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit (for GPU development)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get -y install cuda-toolkit-11-8 || echo "CUDA installation failed (normal in CPU-only environments)"

# Install Python dependencies
COPY requirements-dev.txt /tmp/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements-dev.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    && rm /tmp/requirements-dev.txt

# Install additional development tools
RUN pip install \
    jupyterlab \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyter-dash \
    ipywidgets \
    pre-commit \
    commitizen

# Install neuroimaging tools
RUN pip install \
    nibabel \
    nilearn \
    dipy \
    fury \
    pybids

# Install visualization and plotting libraries
RUN pip install \
    plotly \
    dash \
    kaleido \
    bokeh \
    altair \
    seaborn \
    matplotlib \
    brain-space \
    netneurotools

# Set up Jupyter Lab
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter lab build

# Create workspace directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV JUPYTER_ENABLE_LAB=yes
ENV SHELL=/bin/zsh

# Expose ports for development
EXPOSE 8888 8050 3000

# Default command
CMD ["zsh"]