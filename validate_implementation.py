#!/usr/bin/env python3
"""Implementation validation script for Connectome-GNN-Suite."""

import ast
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any


class ImplementationValidator:
    """Validates the implementation without requiring heavy dependencies."""
    
    def __init__(self):
        self.results = {
            'syntax_checks': {},
            'structure_checks': {},
            'completeness_checks': {},
            'quality_metrics': {}
        }
    
    def validate_syntax(self, file_paths: List[str]) -> Dict[str, bool]:
        """Validate Python syntax for given files."""
        results = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Parse AST to validate syntax
                ast.parse(code)
                results[file_path] = True
                print(f"✅ {file_path}: Syntax valid")
                
            except SyntaxError as e:
                results[file_path] = False
                print(f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
            except FileNotFoundError:
                results[file_path] = False
                print(f"❌ {file_path}: File not found")
            except Exception as e:
                results[file_path] = False
                print(f"❌ {file_path}: Error - {e}")
        
        return results
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate project structure and completeness."""
        required_structure = {
            'connectome_gnn/': {
                '__init__.py': True,
                'cli.py': True,
                'training.py': True,
                'advanced_training.py': True,
                'utils.py': True,
                'data/': {
                    '__init__.py': True,
                    'dataset.py': True,
                    'processor.py': True,
                    'hcp_loader.py': True
                },
                'models/': {
                    '__init__.py': True,
                    'base.py': True,
                    'hierarchical.py': True,
                    'temporal.py': True,
                    'multimodal.py': True,
                    'population.py': True
                },
                'tasks/': {
                    '__init__.py': True,
                    'base.py': True,
                    'node_level.py': True,
                    'edge_level.py': True,
                    'graph_level.py': True
                },
                'visualization/': {
                    '__init__.py': True,
                    'brain_plots.py': True,
                    'explainer.py': True
                },
                'benchmarks/': {
                    '__init__.py': True,
                    'subject_benchmark.py': True
                }
            },
            'tests/': {
                '__init__.py': True,
                'conftest.py': True,
                'test_imports.py': True
            },
            'pyproject.toml': True,
            'README.md': True
        }
        
        def check_structure(structure_dict: Dict, base_path: str = "") -> Dict[str, bool]:
            results = {}
            
            for item, expected in structure_dict.items():
                item_path = Path(base_path) / item
                
                if isinstance(expected, dict):
                    # Directory
                    if item_path.is_dir():
                        results[str(item_path)] = True
                        # Recursively check contents
                        sub_results = check_structure(expected, str(item_path))
                        results.update(sub_results)
                    else:
                        results[str(item_path)] = False
                else:
                    # File
                    results[str(item_path)] = item_path.exists()
            
            return results
        
        structure_results = check_structure(required_structure)
        
        # Print results
        print("\n📁 Project Structure Validation:")
        for path, exists in structure_results.items():
            status = "✅" if exists else "❌"
            print(f"{status} {path}")
        
        return structure_results
    
    def count_lines_of_code(self, directory: str = "connectome_gnn") -> Dict[str, int]:
        """Count lines of code in the project."""
        stats = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'files_count': 0
        }
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        stats['files_count'] += 1
                        stats['total_lines'] += len(lines)
                        
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                stats['blank_lines'] += 1
                            elif stripped.startswith('#'):
                                stats['comment_lines'] += 1
                            else:
                                stats['code_lines'] += 1
                                
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        return stats
    
    def analyze_code_complexity(self, file_paths: List[str]) -> Dict[str, Dict[str, int]]:
        """Analyze code complexity metrics."""
        complexity_results = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                tree = ast.parse(code)
                
                # Count different AST node types
                node_counts = {
                    'classes': 0,
                    'functions': 0,
                    'imports': 0,
                    'if_statements': 0,
                    'for_loops': 0,
                    'while_loops': 0,
                    'try_blocks': 0
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        node_counts['classes'] += 1
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        node_counts['functions'] += 1
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        node_counts['imports'] += 1
                    elif isinstance(node, ast.If):
                        node_counts['if_statements'] += 1
                    elif isinstance(node, ast.For):
                        node_counts['for_loops'] += 1
                    elif isinstance(node, ast.While):
                        node_counts['while_loops'] += 1
                    elif isinstance(node, ast.Try):
                        node_counts['try_blocks'] += 1
                
                complexity_results[file_path] = node_counts
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                complexity_results[file_path] = {}
        
        return complexity_results
    
    def validate_implementation_completeness(self) -> Dict[str, float]:
        """Validate completeness of implementation."""
        completeness_scores = {}
        
        # Check model completeness
        model_files = [
            'connectome_gnn/models/base.py',
            'connectome_gnn/models/hierarchical.py',
            'connectome_gnn/models/temporal.py', 
            'connectome_gnn/models/multimodal.py',
            'connectome_gnn/models/population.py'
        ]
        
        required_classes_per_file = {
            'connectome_gnn/models/base.py': ['BaseConnectomeModel', 'ConnectomeGNNLayer'],
            'connectome_gnn/models/hierarchical.py': ['HierarchicalBrainGNN'],
            'connectome_gnn/models/temporal.py': ['TemporalConnectomeGNN'],
            'connectome_gnn/models/multimodal.py': ['MultiModalBrainGNN'],
            'connectome_gnn/models/population.py': ['PopulationGraphGNN']
        }
        
        for file_path, expected_classes in required_classes_per_file.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_classes = 0
                for class_name in expected_classes:
                    if f"class {class_name}" in content:
                        found_classes += 1
                
                completeness_scores[file_path] = found_classes / len(expected_classes)
                
            except FileNotFoundError:
                completeness_scores[file_path] = 0.0
        
        return completeness_scores
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation."""
        print("🔍 Running Connectome-GNN-Suite Implementation Validation")
        print("=" * 60)
        
        # Get all Python files to check
        python_files = []
        for root, dirs, files in os.walk("connectome_gnn"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # 1. Syntax validation
        print("\\n🐍 Validating Python Syntax...")
        syntax_results = self.validate_syntax(python_files)
        syntax_score = sum(syntax_results.values()) / len(syntax_results) * 100
        
        # 2. Structure validation
        print("\\n📁 Validating Project Structure...")
        structure_results = self.validate_project_structure()
        structure_score = sum(structure_results.values()) / len(structure_results) * 100
        
        # 3. Code metrics
        print("\\n📊 Analyzing Code Metrics...")
        code_stats = self.count_lines_of_code()
        print(f"Files: {code_stats['files_count']}")
        print(f"Total lines: {code_stats['total_lines']:,}")
        print(f"Code lines: {code_stats['code_lines']:,}")
        print(f"Comment lines: {code_stats['comment_lines']:,}")
        print(f"Blank lines: {code_stats['blank_lines']:,}")
        
        # 4. Complexity analysis
        print("\\n🧮 Analyzing Code Complexity...")
        complexity_results = self.analyze_code_complexity(python_files[:5])  # Sample
        
        total_classes = sum(result.get('classes', 0) for result in complexity_results.values())
        total_functions = sum(result.get('functions', 0) for result in complexity_results.values())
        print(f"Classes implemented: {total_classes}")
        print(f"Functions implemented: {total_functions}")
        
        # 5. Implementation completeness
        print("\\n✅ Checking Implementation Completeness...")
        completeness_results = self.validate_implementation_completeness()
        avg_completeness = sum(completeness_results.values()) / len(completeness_results) * 100
        
        for file_path, score in completeness_results.items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            print(f"{status} {file_path}: {score*100:.1f}% complete")
        
        # Overall assessment
        print("\\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Syntax Validation: {syntax_score:.1f}%")
        print(f"Structure Validation: {structure_score:.1f}%")
        print(f"Implementation Completeness: {avg_completeness:.1f}%")
        print(f"Total Lines of Code: {code_stats['code_lines']:,}")
        print(f"Classes Implemented: {total_classes}")
        print(f"Functions Implemented: {total_functions}")
        
        overall_score = (syntax_score + structure_score + avg_completeness) / 3
        
        if overall_score >= 90:
            status = "🌟 EXCELLENT"
        elif overall_score >= 80:
            status = "✅ GOOD"
        elif overall_score >= 70:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ NEEDS WORK"
        
        print(f"\\nOverall Score: {overall_score:.1f}% - {status}")
        
        # Save results
        results = {
            'syntax_validation': {
                'score': syntax_score,
                'details': syntax_results
            },
            'structure_validation': {
                'score': structure_score,
                'details': structure_results
            },
            'code_metrics': code_stats,
            'complexity_analysis': complexity_results,
            'completeness_validation': {
                'score': avg_completeness,
                'details': completeness_results
            },
            'overall_score': overall_score
        }
        
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 Detailed results saved to: validation_results.json")
        
        return results


if __name__ == "__main__":
    validator = ImplementationValidator()
    results = validator.run_full_validation()