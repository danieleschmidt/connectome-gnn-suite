# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation, as these cannot be automated due to GitHub App permission limitations.

## GitHub Actions Workflows

### Required Manual Action

The workflow templates are provided in `docs/workflows/` but must be manually copied to `.github/workflows/` by a repository maintainer with appropriate permissions.

### Step-by-Step Setup

1. **Create Workflows Directory**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy and Rename Workflow Templates**
   ```bash
   # Copy each template file and remove the -template suffix
   cp docs/workflows/ci-template.yml .github/workflows/ci.yml
   cp docs/workflows/docs-template.yml .github/workflows/docs.yml
   cp docs/workflows/release-template.yml .github/workflows/release.yml
   cp docs/workflows/security-template.yml .github/workflows/security.yml
   cp docs/workflows/autonomous-template.yml .github/workflows/autonomous.yml
   ```

3. **Configure Repository Secrets**
   
   Navigate to Settings > Secrets and Variables > Actions and add:
   - `PYPI_API_TOKEN`: PyPI upload token for package publishing
   - `CODECOV_TOKEN`: (Optional) Codecov.io upload token for coverage reporting

4. **Set up Branch Protection Rules**
   
   Navigate to Settings > Branches and configure protection for `main`:
   - ✅ Require pull request reviews (1 reviewer minimum)
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date
   - ✅ Include administrators
   - Required status checks:
     - `test (3.9)`
     - `test (3.10)`
     - `test (3.11)`
     - `lint`
     - `security`

## GitHub Repository Settings

### Security Configuration

1. **Enable Security Features**
   - Navigate to Settings > Security & analysis
   - ✅ Enable Dependency graph
   - ✅ Enable Dependabot alerts
   - ✅ Enable Dependabot security updates
   - ✅ Enable Code scanning alerts
   - ✅ Enable Secret scanning alerts

2. **Configure Dependabot**
   - The `.github/dependabot.yml` file should be created automatically
   - Review and customize update schedules if needed

### Pages Configuration (Optional)

For automatic documentation deployment:
1. Navigate to Settings > Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `/ (root)`

### Environment Configuration (Optional)

For production deployments:
1. Navigate to Settings > Environments
2. Create environments: `development`, `staging`, `production`
3. Configure protection rules and secrets per environment

## Required Permissions

The following permissions are required for full SDLC functionality:

### GitHub App Permissions (if using app-based access)
- ✅ Contents: Read and Write (for code changes)
- ✅ Pull requests: Read and Write (for automated PRs)
- ✅ Issues: Read and Write (for issue management)
- ✅ Actions: Read (for workflow status)
- ✅ Security events: Write (for security scanning)
- ✅ Administration: Read (for repository settings - if available)

### Personal Access Token Permissions (alternative)
If using a personal access token instead of GitHub App:
- `repo` (Full control of private repositories)
- `workflow` (Update GitHub Action workflows)
- `write:packages` (for package publishing)

## Verification Steps

After manual setup, verify the implementation:

1. **Test CI Pipeline**
   ```bash
   # Create a test branch and push changes
   git checkout -b test-ci-setup
   echo "# Test" >> README.md
   git commit -am "test: verify CI pipeline"
   git push origin test-ci-setup
   
   # Create PR and verify all checks pass
   ```

2. **Test Security Scanning**
   - Verify security workflows run on schedule
   - Check Security tab for any identified issues
   - Review dependency vulnerability alerts

3. **Test Documentation Build**
   - Verify docs build successfully
   - Check GitHub Pages deployment (if enabled)

4. **Test Release Process**
   - Create a test release tag
   - Verify package builds and uploads successfully

## Automation Readiness Checklist

Once manual setup is complete, verify these automated capabilities:

- ✅ CI runs on every PR and push to main
- ✅ Security scans run weekly and on PR
- ✅ Documentation builds and deploys automatically
- ✅ Releases are automated when tags are created
- ✅ Dependabot provides automated dependency updates
- ✅ Code quality checks prevent poor code from merging
- ✅ Branch protection enforces review requirements

## Troubleshooting

### Common Issues

1. **Workflow Permissions Errors**
   - Verify repository has appropriate permissions enabled
   - Check if organization settings restrict workflow permissions

2. **Secret Access Issues**
   - Ensure secrets are configured at repository level
   - Verify secret names match those used in workflows

3. **Branch Protection Conflicts**
   - Ensure required status checks match workflow job names
   - Verify branch names match protection rules

4. **Documentation Deployment Issues**
   - Check GitHub Pages source configuration
   - Verify `gh-pages` branch permissions

### Support Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Security Features Documentation](https://docs.github.com/en/code-security)

## Maintenance

Regular maintenance tasks:
- Review and update workflow dependencies monthly
- Update Python version matrix as new versions are released
- Review security scan results and address findings
- Update branch protection rules as team processes evolve

---

**Note**: This manual setup is only required once. Once configured, the SDLC will operate automatically according to the defined workflows and policies.