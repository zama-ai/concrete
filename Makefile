SHELL:=/bin/bash

setup_env:
	poetry install
	poetry run python -m pip install -U pip wheel setuptools
.PHONY: setup_env

sync_env:
	poetry install --remove-untracked
	make setup_env
.PHONY: sync_env

python_format:
	poetry run env bash ./script/source_format/format_python.sh --dir hdk --dir tests
.PHONY: python_format

check_python_format:
	poetry run env bash ./script/source_format/format_python.sh --dir hdk --dir tests --check
.PHONY: check_python_format

pylint:
	poetry run pylint --rcfile=pylintrc hdk tests
.PHONY: pylint

conformance: python_format
.PHONY: conformance

pcc: check_python_format pylint mypy_ci
.PHONY: pcc

pytest:
	poetry run pytest --cov=hdk -vv --cov-report=xml tests/
.PHONY: pytest

# Not a huge fan of ignoring missing imports, but some packages do not have typing stubs
mypy:
	poetry run mypy -p hdk --ignore-missing-imports
.PHONY: mypy

# Friendly target to run mypy without ignoring missing stubs and still have errors messages
# Allows to see which stubs we are missing
mypy_ns:
	poetry run mypy -p hdk
.PHONY: mypy_ns

mypy_test:
	find ./tests/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports
.PHONY: mypy_test

mypy_ci: mypy mypy_test
.PHONY: mypy_ci

pytest_and_coverage: pytest coverage
.PHONY: pytest_and_coverage

coverage:
	@if [[ "$$BB" == "" ]]; then BB=origin/main; fi && poetry run diff-cover coverage.xml --fail-under 100 --html-report coverage.html --compare-branch $$BB
.PHONY: coverage

docs:
	cd docs && poetry run make html
.PHONY: docs
