.PHONY: help install install-dev test lint format clean docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=connectome_gnn --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	pre-commit run --all-files

format: ## Format code
	black connectome_gnn/ tests/
	isort connectome_gnn/ tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	python -m build

upload-test: ## Upload to TestPyPI
	twine upload --repository testpypi dist/*

upload: ## Upload to PyPI
	twine upload dist/*

# Security and Quality
security: ## Run security scans
	bandit -r connectome_gnn/ -f json -o bandit_report.json || true
	safety check --json --output safety_report.json || true
	pip-audit --format=json --output=audit_report.json || true

quality: ## Run code quality analysis
	flake8 connectome_gnn/ tests/ --statistics
	mypy connectome_gnn/
	@echo "Code quality analysis complete"

# Terragon Autonomous System
terragon-discover: ## Run autonomous value discovery
	@echo "🤖 Running autonomous value discovery..."
	@python -c "
	import json
	import subprocess
	from datetime import datetime
	
	# Run security scans
	subprocess.run(['make', 'security'], check=False)
	
	# Simple value discovery logic
	items = []
	
	# Check for missing tests
	result = subprocess.run(['find', 'connectome_gnn/', '-name', '*.py', '-not', '-path', '*/test*'], 
	                       capture_output=True, text=True)
	py_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
	
	result = subprocess.run(['find', 'tests/', '-name', 'test_*.py'], 
	                       capture_output=True, text=True)
	test_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
	
	if py_files > test_files * 2:
	    items.append({
	        'id': 'TEST-001',
	        'title': 'Add missing test coverage',
	        'score': 75.0,
	        'category': 'testing'
	    })
	
	print(f'🔍 Discovered {len(items)} value opportunities')
	
	# Update metrics
	try:
	    with open('.terragon/value-metrics.json', 'r') as f:
	        metrics = json.load(f)
	    
	    metrics['lastUpdated'] = datetime.now().isoformat()
	    metrics['continuousValueMetrics']['totalItemsDiscovered'] = len(items)
	    
	    with open('.terragon/value-metrics.json', 'w') as f:
	        json.dump(metrics, f, indent=2)
	    
	    print('📊 Updated value metrics')
	except:
	    print('⚠️  Could not update metrics')
	"

terragon-status: ## Show autonomous system status
	@echo "🤖 Terragon Autonomous SDLC Status"
	@echo "=================================="
	@if [ -f .terragon/value-metrics.json ]; then \
		python -c "
		import json
		with open('.terragon/value-metrics.json') as f:
		    metrics = json.load(f)
		print(f'📊 Repository Maturity: {metrics[\"repositoryMaturity\"][\"current\"]}%')
		print(f'🔍 Items Discovered: {metrics[\"continuousValueMetrics\"][\"totalItemsDiscovered\"]}')
		print(f'✅ Items Completed: {metrics[\"continuousValueMetrics\"][\"totalItemsCompleted\"]}')
		print(f'📅 Last Updated: {metrics[\"lastUpdated\"]}')
		"; \
	else \
		echo "❌ Terragon system not initialized"; \
	fi

terragon-init: ## Initialize Terragon autonomous system
	@echo "🤖 Initializing Terragon Autonomous SDLC..."
	@mkdir -p .terragon/logs .terragon/cache
	@if [ ! -f .terragon/config.yaml ]; then echo "⚠️  Run make install-dev first to create config"; fi
	@echo "✅ Terragon system initialized"

all-checks: ## Run all quality and security checks
	make lint
	make test-cov
	make security
	make quality
	@echo "🎉 All checks completed successfully!"