SHELL:=/bin/bash

DEV_DOCKER_IMG:=concretefhe-dev
DEV_DOCKERFILE:=docker/Dockerfile.concretefhe-dev
SRC_DIR:=concrete
NOTEBOOKS_DIR:=docs/user/advanced_examples

setup_env:
	poetry run python -m pip install -U pip wheel
	poetry run python -m pip install -U --force-reinstall setuptools
	poetry install
	poetry run python -m pip install -r torch_requirements.txt \
		-f https://download.pytorch.org/whl/torch_stable.html
.PHONY: setup_env

sync_env:
	poetry install --remove-untracked
	$(MAKE) setup_env
.PHONY: sync_env

python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks --dir script
.PHONY: python_format

check_python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks --dir script --check
.PHONY: check_python_format

check_finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py $(NOTEBOOKS_DIR) --check
.PHONY: check_finalize_nb

pylint:
	$(MAKE) --keep-going pylint_src pylint_tests pylint_benchmarks pylint_script
.PHONY: pylint

pylint_src:
	poetry run pylint --rcfile=pylintrc $(SRC_DIR)
.PHONY: pylint_src

pylint_tests:
	@# Disable duplicate code detection in tests
	poetry run pylint --disable=R0801 --rcfile=pylintrc tests
.PHONY: pylint_tests

pylint_benchmarks:
	@# Disable duplicate code detection, docstring requirement, too many locals/statements
	poetry run pylint --disable=R0801,R0914,R0915,C0103,C0114,C0115,C0116 \
	--rcfile=pylintrc benchmarks
.PHONY: pylint_benchmarks

pylint_script:
	poetry run pylint --rcfile=pylintrc script
.PHONY: pylint_script

flake8:
	poetry run flake8 --max-line-length 100 --per-file-ignores="__init__.py:F401" \
	$(SRC_DIR)/ tests/ benchmarks/ script/
.PHONY: flake8

python_linting: pylint flake8
.PHONY: python_linting

conformance: finalize_nb python_format
.PHONY: conformance

pcc:
	@$(MAKE) --keep-going --jobs $(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory pcc_internal
.PHONY: pcc

PCC_DEPS := check_python_format check_finalize_nb python_linting mypy_ci pydocstyle shell_lint
PCC_DEPS += check_version_coherence
pcc_internal: $(PCC_DEPS)
.PHONY: pcc_internal

pytest:
	poetry run pytest -svv \
	--cov=$(SRC_DIR) --cov-fail-under=100 \
	--cov-report=term-missing:skip-covered --cov-report=xml tests/
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

mypy_script:
	find ./script/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports
.PHONY: mypy_script

# The plus indicates that make will be called by the command and allows to share the context with
# the parent make execution. We serialize calls to these targets as they may overwrite each others
# cache which can cause issues.
mypy_ci:
	$(MAKE) --keep-going mypy mypy_test mypy_benchmark mypy_script
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

docker_publish_measurements: docker_build
	mkdir -p .benchmarks
	@# Poetry is not installed on the benchmark servers
	@# Thus, we ran `extract_machine_info.py` script using native python
	python script/progress_tracker_utils/extract_machine_info.py
	docker run --rm --volume /"$$(pwd)":/src $(DEV_DOCKER_IMG) \
	/bin/bash ./script/progress_tracker_utils/benchmark_and_publish_findings_in_docker.sh
.PHONY: docker_publish_measurements

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
	poetry run pydocstyle $(SRC_DIR) --convention google --add-ignore=D1,D202 --add-select=D401
.PHONY: pydocstyle

finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py $(NOTEBOOKS_DIR)
.PHONY: finalize_nb

# A warning in a package unrelated to the project made pytest fail with notebooks
# Run notebook tests without warnings as sources are already tested with warnings treated as errors
pytest_nb:
	poetry run pytest -Wignore --nbmake $(NOTEBOOKS_DIR)/*.ipynb
.PHONY: pytest_nb

benchmark:
	poetry run python script/progress_tracker_utils/extract_machine_info.py
	poetry run python script/progress_tracker_utils/measure.py benchmarks
.PHONY: benchmark

jupyter:
	poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0
.PHONY: jupyter

release_docker:
	./docker/build_release_image.sh
.PHONY: release_docker

upgrade_py_deps:
	./script/make_utils/upgrade_deps.sh
.PHONY: upgrade_py_deps

# This is done by hand as pytest-codeblocks was failing with our native extensions.
# See refused PR on the project here: https://github.com/nschloe/pytest-codeblocks/pull/58
test_codeblocks:
	poetry run python ./script/make_utils/test_md_python_code.py --md_dir docs/
.PHONY: test_codeblocks

# From https://stackoverflow.com/a/63523300 for the find command
shell_lint:
	find \( -path "./.venv" -o -path "./.docker_venv" \) -prune -o -type f -name "*.sh" -print | \
	xargs shellcheck
.PHONY: shell_lint

set_version:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi;
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}"
.PHONY: set_version

check_version_coherence:
	poetry run python ./script/make_utils/version_utils.py check-version
.PHONY: check_version_coherence

changelog: check_version_coherence
	PROJECT_VER=($$(poetry version));\
	PROJECT_VER="$${PROJECT_VER[1]}";\
	poetry run python ./script/make_utils/changelog_helper.py > "CHANGELOG_$${PROJECT_VER}.md"
.PHONY: changelog
