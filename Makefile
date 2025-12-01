test:
	pytest

cov:
	pytest --cov=llamda tests

cov-report:
	pytest --cov=llamda --cov-report=html tests

.PHONY: dist
dist:
	rm -rf dist/*
	python3 -m build
	twine upload dist/*
