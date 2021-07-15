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

pcc: check_python_format pylint
.PHONY: pcc

pytest:
	poetry run pytest --cov=hdk -vv --cov-report=xml tests/
.PHONY: pytest

docs:
	cd docs && poetry run make html
.PHONY: docs
