#!/usr/bin/env python3
"""
Repository health check script for Connectome-GNN-Suite.
Performs comprehensive health checks and generates reports.
"""

import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse


class RepositoryHealthChecker:
    """Comprehensive repository health checking."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": "unknown",
            "score": 0,
            "checks": {}
        }
    
    def _run_command(self, cmd: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_root
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            return 1, "", str(e)
    
    def check_file_exists(self, filepath: Path, required: bool = True) -> Dict[str, Any]:
        """Check if a file exists and return status."""
        exists = filepath.exists()
        return {
            "exists": exists,
            "required": required,
            "path": str(filepath),
            "status": "pass" if exists or not required else "fail"
        }
    
    def check_documentation_health(self) -> Dict[str, Any]:
        """Check documentation completeness and quality."""
        print("📚 Checking documentation health...")
        
        required_docs = [
            "README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md", "SECURITY.md", "PROJECT_CHARTER.md"
        ]
        
        optional_docs = [
            "ARCHITECTURE.md", "ROADMAP.md", "SETUP_REQUIRED.md"
        ]
        
        doc_checks = {}
        score = 0
        max_score = len(required_docs) * 2 + len(optional_docs)  # Required docs worth 2 points
        
        # Check required documentation
        for doc in required_docs:
            check = self.check_file_exists(self.repo_root / doc, required=True)
            doc_checks[doc] = check
            if check["status"] == "pass":
                score += 2
        
        # Check optional documentation
        for doc in optional_docs:
            check = self.check_file_exists(self.repo_root / doc, required=False)
            doc_checks[doc] = check
            if check["status"] == "pass":
                score += 1
        
        # Check README quality
        readme_quality = self._check_readme_quality()
        doc_checks["readme_quality"] = readme_quality
        if readme_quality["status"] == "pass":
            score += 5  # Bonus for good README
            max_score += 5
        
        # Check docs/ directory structure
        docs_dir = self.repo_root / "docs"
        if docs_dir.exists():
            subdirs = ["guides", "adr", "workflows", "monitoring", "runbooks"]
            docs_structure = {}
            for subdir in subdirs:
                exists = (docs_dir / subdir).exists()
                docs_structure[subdir] = exists
                if exists:
                    score += 1
            doc_checks["docs_structure"] = docs_structure
            max_score += len(subdirs)
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100) if max_score > 0 else 0,
            "status": "pass" if score / max_score >= 0.8 else "warning" if score / max_score >= 0.6 else "fail",
            "checks": doc_checks
        }
    
    def _check_readme_quality(self) -> Dict[str, Any]:
        """Check README.md quality and completeness."""
        readme_file = self.repo_root / "README.md"
        if not readme_file.exists():
            return {"status": "fail", "reason": "README.md not found"}
        
        try:
            with open(readme_file) as f:
                content = f.read()
            
            required_sections = [
                "installation", "usage", "example", "quick start",
                "getting started", "api", "documentation"
            ]
            
            sections_found = 0
            for section in required_sections:
                if section.lower() in content.lower():
                    sections_found += 1
            
            # Check for code examples
            has_code_examples = "```" in content
            
            # Check length (should be substantial)
            word_count = len(content.split())
            
            quality_score = 0
            if sections_found >= 3:
                quality_score += 1
            if has_code_examples:
                quality_score += 1
            if word_count >= 500:
                quality_score += 1
            if len(content.split('\n')) >= 50:  # Substantial content
                quality_score += 1
            
            return {
                "status": "pass" if quality_score >= 3 else "warning" if quality_score >= 2 else "fail",
                "sections_found": sections_found,
                "has_code_examples": has_code_examples,
                "word_count": word_count,
                "quality_score": quality_score
            }
            
        except Exception as e:
            return {"status": "fail", "reason": f"Error reading README: {e}"}
    
    def check_code_quality_health(self) -> Dict[str, Any]:
        """Check code quality configuration and metrics."""
        print("🔍 Checking code quality health...")
        
        config_files = {
            "pyproject.toml": True,
            ".pre-commit-config.yaml": True,
            ".editorconfig": True,
            ".gitignore": True,
            "pytest.ini": False
        }
        
        checks = {}
        score = 0
        max_score = 0
        
        # Check configuration files
        for config_file, required in config_files.items():
            check = self.check_file_exists(self.repo_root / config_file, required)
            checks[config_file] = check
            max_score += 2 if required else 1
            if check["status"] == "pass":
                score += 2 if required else 1
        
        # Check if pre-commit is installed
        exit_code, _, _ = self._run_command("pre-commit --version")
        checks["pre_commit_installed"] = {
            "status": "pass" if exit_code == 0 else "fail",
            "installed": exit_code == 0
        }
        max_score += 1
        if exit_code == 0:
            score += 1
        
        # Check if tests exist
        tests_dir = self.repo_root / "tests"
        test_files = list(tests_dir.glob("**/test_*.py")) if tests_dir.exists() else []
        checks["test_coverage"] = {
            "tests_dir_exists": tests_dir.exists(),
            "test_files_count": len(test_files),
            "status": "pass" if len(test_files) >= 5 else "warning" if len(test_files) > 0 else "fail"
        }
        max_score += 2
        if len(test_files) >= 5:
            score += 2
        elif len(test_files) > 0:
            score += 1
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100) if max_score > 0 else 0,
            "status": "pass" if score / max_score >= 0.8 else "warning" if score / max_score >= 0.6 else "fail",
            "checks": checks
        }
    
    def check_security_health(self) -> Dict[str, Any]:
        """Check security configuration and practices."""
        print("🔒 Checking security health...")
        
        checks = {}
        score = 0
        max_score = 0
        
        # Check for security configuration files
        security_files = ["SECURITY.md", ".github/dependabot.yml"]
        for sec_file in security_files:
            check = self.check_file_exists(self.repo_root / sec_file, required=True)
            checks[sec_file.replace("/", "_")] = check
            max_score += 1
            if check["status"] == "pass":
                score += 1
        
        # Check for security tools in development dependencies
        pyproject_file = self.repo_root / "pyproject.toml"
        if pyproject_file.exists():
            with open(pyproject_file) as f:
                content = f.read()
            
            security_tools = ["bandit", "safety", "pip-audit"]
            tools_found = []
            for tool in security_tools:
                if tool in content:
                    tools_found.append(tool)
                    score += 1
            checks["security_tools"] = {
                "tools_found": tools_found,
                "status": "pass" if len(tools_found) >= 2 else "warning" if len(tools_found) > 0 else "fail"
            }
            max_score += len(security_tools)
        
        # Check for secrets in repository (basic check)
        potential_secrets = self._check_for_potential_secrets()
        checks["secrets_check"] = potential_secrets
        max_score += 2
        if potential_secrets["status"] == "pass":
            score += 2
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100) if max_score > 0 else 0,
            "status": "pass" if score / max_score >= 0.8 else "warning" if score / max_score >= 0.6 else "fail",
            "checks": checks
        }
    
    def _check_for_potential_secrets(self) -> Dict[str, Any]:
        """Basic check for potential secrets in the repository."""
        # Simple patterns that might indicate secrets
        secret_patterns = [
            "password", "secret", "key", "token", "api_key",
            "private_key", "access_key", "auth"
        ]
        
        suspicious_files = []
        
        # Check common files that shouldn't contain secrets
        files_to_check = [
            "README.md", "pyproject.toml", "requirements.txt", 
            "requirements-dev.txt", ".env.example"
        ]
        
        for file_path in files_to_check:
            full_path = self.repo_root / file_path
            if full_path.exists():
                try:
                    with open(full_path) as f:
                        content = f.read().lower()
                    
                    for pattern in secret_patterns:
                        if pattern in content and "example" not in content and "placeholder" not in content:
                            suspicious_files.append({
                                "file": file_path,
                                "pattern": pattern
                            })
                except Exception:
                    pass
        
        return {
            "status": "pass" if len(suspicious_files) == 0 else "warning",
            "suspicious_files": suspicious_files,
            "message": "No potential secrets found" if len(suspicious_files) == 0 else f"Found {len(suspicious_files)} potential secrets"
        }
    
    def check_automation_health(self) -> Dict[str, Any]:
        """Check automation and CI/CD configuration."""
        print("🤖 Checking automation health...")
        
        checks = {}
        score = 0
        max_score = 0
        
        # Check for workflow templates (since actual workflows might not exist due to permissions)
        workflows_dir = self.repo_root / "docs" / "workflows"
        if workflows_dir.exists():
            templates = list(workflows_dir.glob("*-template.yml"))
            checks["workflow_templates"] = {
                "count": len(templates),
                "templates": [t.name for t in templates],
                "status": "pass" if len(templates) >= 3 else "warning" if len(templates) > 0 else "fail"
            }
            max_score += 3
            score += min(len(templates), 3)
        
        # Check for containerization
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        docker_score = 0
        for docker_file in docker_files:
            if (self.repo_root / docker_file).exists():
                docker_score += 1
        
        checks["containerization"] = {
            "docker_files_present": docker_score,
            "total_docker_files": len(docker_files),
            "status": "pass" if docker_score == len(docker_files) else "warning" if docker_score > 0 else "fail"
        }
        max_score += len(docker_files)
        score += docker_score
        
        # Check for Makefile
        makefile_check = self.check_file_exists(self.repo_root / "Makefile", required=False)
        checks["makefile"] = makefile_check
        max_score += 1
        if makefile_check["status"] == "pass":
            score += 1
        
        # Check for development environment setup
        dev_env_files = [".devcontainer/devcontainer.json", ".vscode/settings.json"]
        dev_env_score = 0
        for dev_file in dev_env_files:
            if (self.repo_root / dev_file).exists():
                dev_env_score += 1
        
        checks["dev_environment"] = {
            "dev_files_present": dev_env_score,
            "status": "pass" if dev_env_score >= 1 else "fail"
        }
        max_score += 2
        score += dev_env_score
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100) if max_score > 0 else 0,
            "status": "pass" if score / max_score >= 0.8 else "warning" if score / max_score >= 0.6 else "fail",
            "checks": checks
        }
    
    def check_dependency_health(self) -> Dict[str, Any]:
        """Check dependency management and security."""
        print("📦 Checking dependency health...")
        
        checks = {}
        score = 0
        max_score = 5
        
        # Check if dependencies are properly specified
        pyproject_file = self.repo_root / "pyproject.toml"
        if pyproject_file.exists():
            score += 2
            checks["dependency_file"] = {"status": "pass", "file": "pyproject.toml"}
        else:
            checks["dependency_file"] = {"status": "fail", "file": "missing"}
        
        # Check for development dependencies
        if pyproject_file.exists():
            with open(pyproject_file) as f:
                content = f.read()
            
            if "[project.optional-dependencies]" in content or "dev" in content:
                score += 1
                checks["dev_dependencies"] = {"status": "pass"}
            else:
                checks["dev_dependencies"] = {"status": "fail"}
        
        # Check for pinned versions (should have some constraints)
        if pyproject_file.exists():
            with open(pyproject_file) as f:
                content = f.read()
            
            # Look for version constraints
            version_patterns = [">=", "==", "~=", "<", ">"]
            has_constraints = any(pattern in content for pattern in version_patterns)
            
            if has_constraints:
                score += 1
                checks["version_constraints"] = {"status": "pass"}
            else:
                checks["version_constraints"] = {"status": "warning"}
        
        # Check for dependency scanning tools
        exit_code, _, _ = self._run_command("safety --version")
        if exit_code == 0:
            score += 1
            checks["security_scanning"] = {"status": "pass", "tool": "safety"}
        else:
            checks["security_scanning"] = {"status": "warning", "tool": "missing"}
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100),
            "status": "pass" if score / max_score >= 0.8 else "warning" if score / max_score >= 0.6 else "fail",
            "checks": checks
        }
    
    def calculate_overall_health(self) -> None:
        """Calculate overall repository health score."""
        category_weights = {
            "documentation": 0.25,
            "code_quality": 0.25,
            "security": 0.25,
            "automation": 0.15,
            "dependencies": 0.10
        }
        
        weighted_score = 0
        for category, weight in category_weights.items():
            if category in self.report["checks"]:
                category_score = self.report["checks"][category]["percentage"]
                weighted_score += category_score * weight
        
        self.report["score"] = round(weighted_score)
        
        # Determine overall health status
        if self.report["score"] >= 85:
            self.report["overall_health"] = "excellent"
        elif self.report["score"] >= 70:
            self.report["overall_health"] = "good"
        elif self.report["score"] >= 55:
            self.report["overall_health"] = "fair"
        else:
            self.report["overall_health"] = "poor"
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        print("🏥 Starting repository health check...")
        
        # Run all health checks
        self.report["checks"]["documentation"] = self.check_documentation_health()
        self.report["checks"]["code_quality"] = self.check_code_quality_health()
        self.report["checks"]["security"] = self.check_security_health()
        self.report["checks"]["automation"] = self.check_automation_health()
        self.report["checks"]["dependencies"] = self.check_dependency_health()
        
        # Calculate overall health
        self.calculate_overall_health()
        
        return self.report
    
    def save_report(self, output_file: Optional[Path] = None) -> None:
        """Save health check report to file."""
        if output_file is None:
            output_file = self.repo_root / "health-check-report.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"📄 Health check report saved to {output_file}")
    
    def print_summary(self) -> None:
        """Print health check summary."""
        print("\n" + "="*60)
        print("📊 REPOSITORY HEALTH CHECK SUMMARY")
        print("="*60)
        
        print(f"Overall Health: {self.report['overall_health'].upper()} ({self.report['score']}%)")
        print()
        
        for category, results in self.report["checks"].items():
            status_emoji = {
                "pass": "✅",
                "warning": "⚠️",
                "fail": "❌"
            }
            
            emoji = status_emoji.get(results["status"], "❓")
            print(f"{emoji} {category.replace('_', ' ').title()}: {results['percentage']}% ({results['status']})")
        
        print("\n" + "="*60)
        
        # Recommendations based on health
        if self.report["score"] < 70:
            print("🔧 RECOMMENDATIONS:")
            for category, results in self.report["checks"].items():
                if results["status"] in ["warning", "fail"]:
                    print(f"  • Improve {category.replace('_', ' ')}")
            print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Repository health check")
    parser.add_argument("--repo-root", type=Path, default=".", 
                       help="Repository root path")
    parser.add_argument("--output", type=Path, 
                       help="Output file for health report")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        print(f"Error: Repository root {repo_root} does not exist")
        sys.exit(1)
    
    checker = RepositoryHealthChecker(repo_root)
    report = checker.run_health_check()
    
    if not args.quiet:
        checker.print_summary()
    
    checker.save_report(args.output)
    
    # Exit with error code if health is poor
    if report["score"] < 50:
        sys.exit(1)


if __name__ == "__main__":
    main()