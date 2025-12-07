test:
	uv run pytest

cov:
	uv run pytest --cov=llamda tests

cov-report:
	uv run pytest --cov=llamda --cov-report=html tests

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format --check .

format-fix:
	uv run ruff format .

typecheck:
	uv run mypy llamda

.PHONY: dist
dist:
	rm -rf dist/*
	python3 -m build
	twine upload dist/*
