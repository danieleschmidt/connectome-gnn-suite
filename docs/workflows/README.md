# GitHub Actions Workflows

This directory contains documentation and templates for CI/CD workflows.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and type checking on every PR and push.

**Triggers**:
- Pull requests to `main`
- Pushes to `main` branch
- Manual workflow dispatch

**Jobs**:
- **test**: Run pytest with coverage across Python 3.9, 3.10, 3.11
- **lint**: Run black, isort, flake8, mypy
- **security**: Run bandit security linting

**Implementation Notes**:
- Use matrix strategy for multiple Python versions
- Cache dependencies for faster builds
- Upload coverage reports to codecov.io
- Fail fast on linting errors

### 2. Documentation Build (`docs.yml`)

**Purpose**: Build and deploy documentation to GitHub Pages.

**Triggers**:
- Pushes to `main` branch
- PR to `main` (build only, no deploy)

**Jobs**:
- **build-docs**: Use Sphinx to build HTML documentation
- **deploy**: Deploy to gh-pages branch (main only)

### 3. Release (`release.yml`)

**Purpose**: Automated package building and PyPI publishing.

**Triggers**:
- GitHub release creation
- Manual workflow dispatch with version tag

**Jobs**:
- **build**: Build source and wheel distributions
- **publish**: Upload to PyPI (requires secrets)
- **github-release**: Update release with artifacts

### 4. Security Scanning (`security.yml`)

**Purpose**: Regular security vulnerability scanning.

**Triggers**:
- Weekly schedule (Sunday 2 AM UTC)
- Manual workflow dispatch

**Jobs**:
- **dependency-scan**: Use pip-audit for dependency vulnerabilities
- **code-scan**: Use CodeQL for code analysis
- **container-scan**: Scan any Docker images

## Workflow Templates

See individual files in this directory for complete workflow implementations:

- `ci-template.yml` - Complete CI workflow
- `docs-template.yml` - Documentation deployment
- `release-template.yml` - Automated releases
- `security-template.yml` - Security scanning

## Setup Instructions

1. Copy desired template files to `.github/workflows/`
2. Remove `-template` suffix from filenames
3. Update repository-specific configurations:
   - Python versions
   - Package name
   - PyPI repository name
   - Secret names

## Required Secrets

Configure these in GitHub repository settings:

- `PYPI_API_TOKEN`: PyPI upload token
- `CODECOV_TOKEN`: Codecov.io upload token (optional)

## Branch Protection

Recommended branch protection rules for `main`:

- Require pull request reviews (1 reviewer)  
- Require status checks to pass
- Require up-to-date branches
- Include administrators
- Required status checks:
  - `test (3.9)`
  - `test (3.10)` 
  - `test (3.11)`
  - `lint`
  - `security`

## Monitoring

Monitor workflow health through:

- GitHub Actions dashboard
- Failed workflow notifications
- Code coverage trends
- Security scan results

For detailed implementation guides, see the template files in this directory.