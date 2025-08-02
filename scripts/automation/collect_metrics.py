#!/usr/bin/env python3
"""
Automated metrics collection script for Connectome-GNN-Suite.
Collects various project metrics and updates the project metrics file.
"""

import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collect and update project metrics."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.metrics_file = repo_root / ".terragon" / "project-metrics.json"
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or create default structure."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics structure."""
        return {
            "metadata": {
                "projectName": "connectome-gnn-suite",
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
            },
            "codeQualityMetrics": {},
            "securityMetrics": {},
            "performanceMetrics": {},
            "automationMetrics": {},
        }
    
    def _run_command(self, cmd: str, capture_output: bool = True) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"Error running command '{cmd}': {e}")
        return None
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git-related metrics."""
        metrics = {}
        
        # Recent commits
        commits_last_week = self._run_command(
            "git rev-list --count --since='1 week ago' HEAD"
        )
        if commits_last_week:
            metrics["commitsPerWeek"] = int(commits_last_week)
        
        # Contributors
        contributors = self._run_command(
            "git shortlog -sn --all | wc -l"
        )
        if contributors:
            metrics["totalContributors"] = int(contributors)
        
        # Last commit date
        last_commit = self._run_command(
            "git log -1 --format=%ci"
        )
        if last_commit:
            metrics["lastCommitDate"] = last_commit
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage (if pytest-cov is available)
        coverage_output = self._run_command(
            "python -m pytest --cov=connectome_gnn --cov-report=json -q tests/ 2>/dev/null"
        )
        if coverage_output:
            coverage_file = self.repo_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        metrics["testCoverage"] = {
                            "current": round(coverage_data.get("totals", {}).get("percent_covered", 0)),
                            "target": 80,
                            "trend": "stable"
                        }
                except Exception:
                    pass
        
        # Count Python files
        python_files = self._run_command(
            "find connectome_gnn/ -name '*.py' | wc -l"
        )
        if python_files:
            metrics["pythonFiles"] = int(python_files)
        
        # Count test files
        test_files = self._run_command(
            "find tests/ -name 'test_*.py' | wc -l"
        )
        if test_files:
            metrics["testFiles"] = int(test_files)
        
        # Lines of code
        loc = self._run_command(
            "find connectome_gnn/ -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $1}'"
        )
        if loc:
            metrics["linesOfCode"] = int(loc)
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {}
        
        # Check for security reports
        security_reports_dir = self.repo_root / "security-reports"
        if security_reports_dir.exists():
            # Parse bandit report if available
            bandit_report = security_reports_dir / "bandit-report.json"
            if bandit_report.exists():
                try:
                    with open(bandit_report) as f:
                        bandit_data = json.load(f)
                        metrics["vulnerabilities"] = {
                            "high": len([r for r in bandit_data.get("results", []) 
                                       if r.get("issue_severity") == "HIGH"]),
                            "medium": len([r for r in bandit_data.get("results", []) 
                                         if r.get("issue_severity") == "MEDIUM"]),
                            "low": len([r for r in bandit_data.get("results", []) 
                                      if r.get("issue_severity") == "LOW"])
                        }
                except Exception:
                    pass
        
        # Last security scan timestamp
        metrics["lastSecurityScan"] = datetime.now(timezone.utc).isoformat()
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # Count dependencies from pyproject.toml
        pyproject_file = self.repo_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file) as f:
                    content = f.read()
                    # Simple count of dependencies in the dependencies section
                    deps_section = False
                    dep_count = 0
                    for line in content.split('\n'):
                        if line.strip().startswith('dependencies = ['):
                            deps_section = True
                        elif deps_section and line.strip() == ']':
                            deps_section = False
                        elif deps_section and line.strip().startswith('"'):
                            dep_count += 1
                    metrics["totalDependencies"] = dep_count
            except Exception:
                pass
        
        return metrics
    
    def collect_automation_metrics(self) -> Dict[str, Any]:
        """Collect automation-related metrics."""
        metrics = {}
        
        # Check for workflow files
        workflows_dir = self.repo_root / ".github" / "workflows"
        if workflows_dir.exists():
            workflow_count = len(list(workflows_dir.glob("*.yml")))
            metrics["cicdWorkflows"] = workflow_count
        else:
            # Check for template workflows
            template_workflows = self.repo_root / "docs" / "workflows"
            if template_workflows.exists():
                template_count = len(list(template_workflows.glob("*-template.yml")))
                metrics["cicdWorkflowTemplates"] = template_count
        
        # Check for Docker configurations
        if (self.repo_root / "Dockerfile").exists():
            metrics["containerization"] = True
        
        if (self.repo_root / "docker-compose.yml").exists():
            metrics["dockerCompose"] = True
        
        # Check for pre-commit configuration
        if (self.repo_root / ".pre-commit-config.yaml").exists():
            metrics["preCommitHooks"] = True
        
        return metrics
    
    def collect_project_structure_metrics(self) -> Dict[str, Any]:
        """Collect project structure metrics."""
        metrics = {}
        
        # Documentation files
        docs_files = [
            "README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", 
            "CODE_OF_CONDUCT.md", "SECURITY.md", "PROJECT_CHARTER.md"
        ]
        
        existing_docs = sum(1 for doc in docs_files if (self.repo_root / doc).exists())
        metrics["documentationCompleteness"] = round((existing_docs / len(docs_files)) * 100)
        
        # Configuration files
        config_files = [
            "pyproject.toml", ".gitignore", ".editorconfig", 
            ".dockerignore", "pytest.ini"
        ]
        
        existing_configs = sum(1 for config in config_files if (self.repo_root / config).exists())
        metrics["configurationCompleteness"] = round((existing_configs / len(config_files)) * 100)
        
        return metrics
    
    def update_metrics(self) -> None:
        """Update all metrics and save to file."""
        print("🔍 Collecting project metrics...")
        
        # Update timestamp
        self.metrics["metadata"]["lastUpdated"] = datetime.now(timezone.utc).isoformat()
        
        # Collect various metrics
        git_metrics = self.collect_git_metrics()
        code_metrics = self.collect_code_quality_metrics()
        security_metrics = self.collect_security_metrics()
        dependency_metrics = self.collect_dependency_metrics()
        automation_metrics = self.collect_automation_metrics()
        structure_metrics = self.collect_project_structure_metrics()
        
        # Update metrics sections
        self.metrics.setdefault("teamProductivityMetrics", {}).update(git_metrics)
        self.metrics.setdefault("codeQualityMetrics", {}).update(code_metrics)
        self.metrics.setdefault("securityMetrics", {}).update(security_metrics)
        self.metrics.setdefault("dependencyMetrics", {}).update(dependency_metrics)
        self.metrics.setdefault("automationMetrics", {}).update(automation_metrics)
        self.metrics.setdefault("projectStructureMetrics", {}).update(structure_metrics)
        
        # Calculate repository maturity score
        maturity_score = self._calculate_maturity_score()
        self.metrics["repositoryMaturity"]["current"] = maturity_score
        
        # Save updated metrics
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Metrics updated successfully")
        print(f"📊 Repository maturity score: {maturity_score}%")
    
    def _calculate_maturity_score(self) -> int:
        """Calculate overall repository maturity score."""
        scores = []
        
        # Documentation score
        doc_score = self.metrics.get("projectStructureMetrics", {}).get("documentationCompleteness", 0)
        scores.append(doc_score)
        
        # Configuration score
        config_score = self.metrics.get("projectStructureMetrics", {}).get("configurationCompleteness", 0)
        scores.append(config_score)
        
        # Test coverage score
        test_coverage = self.metrics.get("codeQualityMetrics", {}).get("testCoverage", {}).get("current", 0)
        scores.append(test_coverage)
        
        # Automation score
        automation_features = 0
        automation_data = self.metrics.get("automationMetrics", {})
        
        if automation_data.get("containerization"):
            automation_features += 20
        if automation_data.get("dockerCompose"):
            automation_features += 20
        if automation_data.get("preCommitHooks"):
            automation_features += 20
        if automation_data.get("cicdWorkflows", 0) > 0 or automation_data.get("cicdWorkflowTemplates", 0) > 0:
            automation_features += 40
        
        scores.append(automation_features)
        
        # Security score (assume high if no vulnerabilities)
        vulnerabilities = self.metrics.get("securityMetrics", {}).get("vulnerabilities", {})
        if isinstance(vulnerabilities, dict):
            total_vulns = sum(vulnerabilities.values())
            security_score = max(0, 100 - (total_vulns * 10))  # Deduct 10 points per vulnerability
        else:
            security_score = 85  # Default score
        scores.append(security_score)
        
        # Calculate weighted average
        return min(100, round(sum(scores) / len(scores)))
    
    def display_summary(self) -> None:
        """Display metrics summary."""
        print("\n📊 Project Metrics Summary")
        print("=" * 50)
        
        maturity = self.metrics.get("repositoryMaturity", {}).get("current", 0)
        print(f"Repository Maturity: {maturity}%")
        
        test_coverage = self.metrics.get("codeQualityMetrics", {}).get("testCoverage", {}).get("current", 0)
        print(f"Test Coverage: {test_coverage}%")
        
        doc_completeness = self.metrics.get("projectStructureMetrics", {}).get("documentationCompleteness", 0)
        print(f"Documentation: {doc_completeness}%")
        
        vulnerabilities = self.metrics.get("securityMetrics", {}).get("vulnerabilities", {})
        if isinstance(vulnerabilities, dict):
            total_vulns = sum(vulnerabilities.values())
            print(f"Security Vulnerabilities: {total_vulns}")
        
        last_updated = self.metrics.get("metadata", {}).get("lastUpdated", "Unknown")
        print(f"Last Updated: {last_updated}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--repo-root", type=Path, default=".", help="Repository root path")
    parser.add_argument("--summary", action="store_true", help="Display summary after collection")
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        print(f"Error: Repository root {repo_root} does not exist")
        sys.exit(1)
    
    collector = MetricsCollector(repo_root)
    collector.update_metrics()
    
    if args.summary:
        collector.display_summary()


if __name__ == "__main__":
    main()