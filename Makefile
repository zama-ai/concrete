setup_env:
	poetry install
	poetry run python -m pip install -U pip wheel setuptools
.PHONY: setup_env

sync_env:
	poetry install --remove-untracked
	make setup_env
.PHONY: sync_env

python_format:
	poetry run env bash ./script/source_format/format_python.sh --dir hdk
.PHONY: python_format

check_python_format:
	poetry run env bash ./script/source_format/format_python.sh --dir hdk --check
.PHONY: check_python_format

pylint:
	poetry run pylint --rcfile=pylintrc hdk
.PHONY: pylint

conformance: python_format
.PHONY: conformance

pcc: check_python_format pylint
.PHONY: pcc
