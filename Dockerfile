# Multi-stage build for Connectome-GNN-Suite
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install -e ".[dev,viz,full]"

# Copy source code
COPY . .

# Install pre-commit hooks
RUN pre-commit install || true

# Expose Jupyter and development server ports
EXPOSE 8888 8000

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

# Copy requirements and install production dependencies
COPY pyproject.toml ./
RUN pip install -e ".[viz]"

# Copy source code
COPY connectome_gnn/ ./connectome_gnn/
COPY README.md LICENSE ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash connectome
RUN chown -R connectome:connectome /app
USER connectome

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import connectome_gnn; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "connectome_gnn"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests
RUN pytest --cov=connectome_gnn --cov-report=term-missing

# Security scanning stage  
FROM development as security

# Install security tools
RUN pip install bandit safety pip-audit

# Run security scans
RUN bandit -r connectome_gnn/ || true
RUN safety check || true
RUN pip-audit || true