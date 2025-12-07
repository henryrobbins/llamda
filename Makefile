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
	uv run ruff format .

format-fix:
	uv run ruff format --fix .

typecheck:
	uv run mypy llamda

typecheck-fix:
	uv run mypy --fix-syntax llamda

.PHONY: dist
dist:
	rm -rf dist/*
	python3 -m build
	twine upload dist/*
