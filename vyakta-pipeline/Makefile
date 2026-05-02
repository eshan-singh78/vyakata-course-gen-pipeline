.PHONY: install lint format test clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src || true

format:
	ruff format src tests

test:
	pytest -q

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
