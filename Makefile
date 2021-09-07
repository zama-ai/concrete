SHELL:=/bin/bash

DEV_DOCKER_IMG:=concretefhe-dev
DEV_DOCKERFILE:=docker/Dockerfile.concretefhe-dev
SRC_DIR:=concrete

setup_env:
	poetry install
	poetry run python -m pip install -U pip wheel setuptools
	poetry run python -m pip install -r torch_requirements.txt \
		-f https://download.pytorch.org/whl/torch_stable.html
.PHONY: setup_env

sync_env:
	poetry install --remove-untracked
	$(MAKE) setup_env
.PHONY: sync_env

python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks
.PHONY: python_format

check_python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks --check
.PHONY: check_python_format

check_strip_nb:
	poetry run python ./script/nbmake_utils/notebook_sanitize.py examples --check
.PHONY: strip_nb

pylint:
	$(MAKE) --keep-going pylint_src pylint_tests pylint_benchmarks
.PHONY: pylint

pylint_src:
	poetry run pylint --rcfile=pylintrc $(SRC_DIR)
.PHONY: pylint_src

pylint_tests:
	@# Disable duplicate code detection in tests
	poetry run pylint --disable=R0801 --rcfile=pylintrc tests
.PHONY: pylint_tests

pylint_benchmarks:
	@# Disable duplicate code detection in benchmarks
	poetry run pylint --disable=R0801 --rcfile=pylintrc benchmarks
.PHONY: pylint_benchmarks

flake8:
	poetry run flake8 --max-line-length 100 --per-file-ignores="__init__.py:F401" \
	$(SRC_DIR)/ tests/ benchmarks/
.PHONY: flake8

python_linting: pylint flake8
.PHONY: python_linting

conformance: strip_nb python_format
.PHONY: conformance

pcc:
	@$(MAKE) --keep-going --jobs $(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory pcc_internal
.PHONY: pcc

pcc_internal: check_python_format check_strip_nb python_linting mypy_ci pydocstyle
.PHONY: pcc_internal

pytest:
	poetry run pytest -svv \
	--cov=$(SRC_DIR) --cov-report=term-missing:skip-covered --cov-report=xml tests/
.PHONY: pytest

# Not a huge fan of ignoring missing imports, but some packages do not have typing stubs
mypy:
	poetry run mypy -p $(SRC_DIR) --ignore-missing-imports
.PHONY: mypy

# Friendly target to run mypy without ignoring missing stubs and still have errors messages
# Allows to see which stubs we are missing
mypy_ns:
	poetry run mypy -p $(SRC_DIR)
.PHONY: mypy_ns

mypy_test:
	find ./tests/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports
.PHONY: mypy_test

mypy_benchmark:
	find ./benchmarks/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports
.PHONY: mypy_benchmark

# The plus indicates that make will be called by the command and allows to share the context with
# the parent make execution. We serialize calls to these targets as they may overwrite each others
# cache which can cause issues.
mypy_ci:
	$(MAKE) --keep-going mypy mypy_test mypy_benchmark
.PHONY: mypy_ci

pytest_and_coverage: pytest coverage
.PHONY: pytest_and_coverage

coverage:
	@if [[ "$$BB" == "" ]]; then BB=origin/main; fi && \
	poetry run diff-cover coverage.xml --fail-under 100 \
	--html-report coverage.html --compare-branch $$BB
.PHONY: coverage

docker_build:
	docker build --pull -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .
.PHONY: docker_build

docker_rebuild:
	docker build --pull --no-cache -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .
.PHONY: docker_rebuild

docker_start:
	@# the slash before pwd is for Windows
	docker run --rm -it \
	-p 8888:8888 \
	--env DISPLAY=host.docker.internal:0 \
	--volume /"$$(pwd)":/src \
	$(DEV_DOCKER_IMG)
.PHONY: docker_start

docker_build_and_start: docker_build docker_start
.PHONY: docker_build_and_start

docker_bas: docker_build_and_start
.PHONY: docker_bas

docs: clean_docs
	@# Generate the auto summary of documentations
	poetry run sphinx-apidoc -o docs/_apidoc $(SRC_DIR)

	@# Docs
	cd docs && poetry run $(MAKE) html SPHINXOPTS='-W --keep-going'
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
	poetry run pydocstyle $(SRC_DIR) --convention google --add-ignore=D1,D202
.PHONY: pydocstyle

strip_nb:
	poetry run python ./script/nbmake_utils/notebook_sanitize.py examples
.PHONY: strip_nb

notebook_timeout:
	poetry run python ./script/nbmake_utils/notebook_test_timeout.py examples
.PHONY: notebook_timeout

pytest_nb:
	poetry run pytest --nbmake examples/*.ipynb
.PHONY: pytest_nb

benchmark:
	poetry run pytest benchmarks/ --benchmark-save=findings
.PHONY: benchmark

jupyter:
	poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0
.PHONY: jupyter
