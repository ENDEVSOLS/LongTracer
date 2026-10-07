.PHONY: lint typecheck test test-cov build clean install benchmark

benchmark:
	python benchmarks/evaluator/run.py --split heldout --out benchmarks/evaluator/reports/

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

typecheck:
	mypy longtracer/ --ignore-missing-imports

test:
	pytest -v --tb=short

test-cov:
	pytest --cov=longtracer --cov-report=term-missing

build:
	python -m build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	pip install -e ".[all]"
	pip install ruff mypy pytest pytest-cov hypothesis build twine
