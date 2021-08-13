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

pcc: check_python_format pylint mypy_ci pydocstyle
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

docker_build:
	docker build -t hdk:mlir -f docker/Dockerfile .
.PHONY: docker_build

docker_rebuild:
	docker build --no-cache -t hdk:mlir -f docker/Dockerfile .
.PHONY: docker_rebuild

docker_start:
	@# the slash before pwd is for Windows
	docker run --rm -it --volume /"$$(pwd)":/hdk hdk:mlir
.PHONY: docker_start

docker_build_and_start: docker_build docker_start
.PHONY: docker_build_and_start

docker_bas: docker_build_and_start
.PHONY: docker_bas

docs:
	@# Generate the auto summary of documentations
	poetry run sphinx-apidoc -o docs/_apidoc hdk

	@# Docs
	cd docs && poetry run make html
.PHONY: docs

clean_docs:
	rm -rf docs/_apidoc docs/_build
.PHONY: clean_docs

open_docs:
	@# This is macOS only. On other systems, one would use `start` or `xdg-open`
	open docs/_build/html/index.html
.PHONY: open_docs

build_and_open_docs: clean_docs docs open_docs
.PHONY: build_and_open_docs

pydocstyle:
	@# From http://www.pydocstyle.org/en/stable/error_codes.html
	poetry run pydocstyle hdk --convention google --add-ignore=D1,D202
.PHONY: pydocstyle
