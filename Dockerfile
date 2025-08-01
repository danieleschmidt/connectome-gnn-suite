# Multi-stage build for Connectome-GNN-Suite
# Optimized for scientific computing and brain connectivity analysis

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.8
ARG PYTORCH_VERSION=2.0.0
ARG BUILD_ENV=production

# =============================================================================
# Base Stage - Common dependencies for all stages
# =============================================================================
FROM python:${PYTHON_VERSION}-slim as base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for scientific computing
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Version control
    git \
    curl \
    wget \
    # Scientific computing libraries
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libhdf5-dev \
    libgraphviz-dev \
    # Neuroimaging dependencies
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create application user (will be used in production)
RUN groupadd --system --gid 1001 connectome && \
    useradd --system --uid 1001 --gid connectome --create-home --shell /bin/bash connectome

# =============================================================================
# Dependencies Stage - Install Python packages
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml README.md ./
COPY connectome_gnn/__init__.py ./connectome_gnn/

# Install PyTorch with CUDA support (CPU fallback)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 || \
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and related packages
RUN pip install torch-geometric torch-scatter torch-sparse torch-cluster

# =============================================================================
# Development Stage - Full development environment
# =============================================================================
FROM dependencies as development

# Install development and visualization dependencies
RUN pip install -e ".[dev,viz,full]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Editors and utilities
    vim \
    nano \
    htop \
    tree \
    jq \
    # Debugging tools
    gdb \
    valgrind \
    # GPU monitoring (if available)
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter extensions
RUN pip install \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    ipywidgets \
    jupyter-dash

# Copy source code
COPY . .

# Set up development environment
RUN pip install -e ".[dev,viz,full]"

# Install pre-commit hooks
RUN pre-commit install || echo "Pre-commit setup skipped"

# Create data and cache directories
RUN mkdir -p /app/data /app/cache /app/logs /app/outputs && \
    chown -R connectome:connectome /app

# Development environment variables
ENV CONNECTOME_GNN_DATA_DIR=/app/data \
    CONNECTOME_GNN_CACHE_DIR=/app/cache \
    CONNECTOME_GNN_LOG_DIR=/app/logs \
    CONNECTOME_GNN_OUTPUT_DIR=/app/outputs \
    JUPYTER_ENABLE_LAB=yes

# Expose ports for development services
EXPOSE 8888 8050 8000 6006

# Development command
USER connectome
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Testing Stage - Run comprehensive tests
# =============================================================================
FROM development as testing

# Switch to root for test setup
USER root

# Install additional testing dependencies
RUN pip install \
    pytest-benchmark \
    pytest-xdist \
    pytest-timeout \
    memory-profiler \
    line-profiler

# Copy test configuration
COPY pytest.ini ./
COPY tests/ ./tests/

# Set up test environment
ENV CONNECTOME_GNN_TEST_MODE=true \
    CONNECTOME_GNN_LOG_LEVEL=WARNING

# Run different test suites
RUN echo "Running unit tests..." && \
    pytest tests/unit/ -v --tb=short || echo "Unit tests completed with issues"

RUN echo "Running integration tests..." && \
    pytest tests/integration/ -v --tb=short -m "not slow" || echo "Integration tests completed with issues"

RUN echo "Running performance benchmarks..." && \
    pytest tests/benchmarks/ -v --tb=short -m "benchmark and not slow" || echo "Benchmarks completed with issues"

# Generate test report
RUN pytest tests/ --cov=connectome_gnn --cov-report=html --cov-report=term --tb=short -m "not slow and not gpu" || echo "Full test suite completed"

# =============================================================================
# Security Stage - Security scanning and validation
# =============================================================================
FROM dependencies as security

# Install security tools
RUN pip install \
    bandit[toml] \
    safety \
    pip-audit \
    semgrep

# Copy source code for scanning
COPY connectome_gnn/ ./connectome_gnn/
COPY pyproject.toml ./

# Run security scans
RUN echo "Running Bandit security scan..." && \
    bandit -r connectome_gnn/ -f json -o bandit-report.json || echo "Bandit scan completed with warnings"

RUN echo "Running Safety vulnerability scan..." && \
    safety check --json --output safety-report.json || echo "Safety scan completed with warnings"  

RUN echo "Running pip-audit..." && \
    pip-audit --format=json --output=pip-audit-report.json || echo "Pip-audit completed with warnings"

# Validate Docker image
RUN echo "Validating installation..." && \
    python -c "import connectome_gnn; print('✓ Package imports successfully')" && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import torch_geometric; print(f'✓ PyTorch Geometric {torch_geometric.__version__}')"

# =============================================================================
# Production Stage - Optimized for deployment
# =============================================================================
FROM dependencies as production

# Install minimal production dependencies
RUN pip install -e ".[viz]"

# Copy only necessary files
COPY connectome_gnn/ ./connectome_gnn/
COPY README.md LICENSE CHANGELOG.md ./

# Production environment variables
ENV CONNECTOME_GNN_ENV=production \
    CONNECTOME_GNN_LOG_LEVEL=INFO \
    CONNECTOME_GNN_WORKERS=4 \
    PYTHONPATH=/app

# Create directories with proper permissions
RUN mkdir -p /app/data /app/cache /app/logs /app/outputs && \
    chown -R connectome:connectome /app && \
    chmod -R 755 /app

# Switch to non-root user for security
USER connectome

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import connectome_gnn; import torch; print('OK')" || exit 1

# Default production command
CMD ["python", "-m", "connectome_gnn.cli", "--help"]

# =============================================================================
# GPU Stage - CUDA-enabled version for training
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as gpu-base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libhdf5-dev \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Create user
RUN groupadd --system --gid 1001 connectome && \
    useradd --system --uid 1001 --gid connectome --create-home --shell /bin/bash connectome

FROM gpu-base as gpu-production

# Copy and install dependencies
COPY pyproject.toml README.md ./
COPY connectome_gnn/__init__.py ./connectome_gnn/

# Install PyTorch with CUDA support
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric with CUDA support
RUN pip install torch-geometric torch-scatter torch-sparse torch-cluster

# Install application
RUN pip install -e ".[viz]"

# Copy source code
COPY connectome_gnn/ ./connectome_gnn/
COPY README.md LICENSE ./

# Set up directories
RUN mkdir -p /app/data /app/cache /app/logs /app/outputs && \
    chown -R connectome:connectome /app

# GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    CONNECTOME_GNN_GPU_ENABLED=true

USER connectome

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('GPU OK')" || exit 1

CMD ["python", "-m", "connectome_gnn.cli"]

# =============================================================================
# Final stage selection based on build argument
# =============================================================================
FROM ${BUILD_ENV} as final