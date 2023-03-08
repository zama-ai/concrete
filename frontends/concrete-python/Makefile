SHELL:=/bin/bash

DEV_DOCKER_IMG:=concrete-numpy-dev
DEV_DOCKERFILE:=docker/Dockerfile.dev
DEV_CONTAINER_VENV_VOLUME:=concrete-numpy-internal-venv
DEV_CONTAINER_CACHE_VOLUME:=concrete-numpy-internal-cache
SRC_DIR:=concrete

.PHONY: setup_env # Set up the environment
setup_env:
	poetry run python -m pip install -U pip wheel
	poetry run python -m pip install -U --force-reinstall setuptools
	if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
		poetry install --only dev; \
	else \
		poetry install; \
	fi

.PHONY: sync_env # Synchronise the environment
sync_env:
	if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
		poetry install --remove-untracked --only dev; \
	else \
		poetry install --remove-untracked; \
	fi
	$(MAKE) setup_env

.PHONY: python_format # Apply python formatting
python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir script

.PHONY: check_python_format # Check python format
check_python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir script --check

.PHONY: check_finalize_nb # Sanitize notebooks
check_finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py docs --check

.PHONY: pylint # Run pylint
pylint:
	$(MAKE) --keep-going pylint_src pylint_tests pylint_script

.PHONY: pylint_src # Run pylint on sources
pylint_src:
	poetry run pylint --rcfile=pylintrc $(SRC_DIR)

.PHONY: pylint_tests # Run pylint on tests
pylint_tests:
	@# Disable duplicate code detection (R0801) in tests
	@# Disable unnecessary lambda (W0108) for tests
	find ./tests/ -type f -name "*.py" | xargs poetry run pylint --disable=R0801,W0108 --rcfile=pylintrc

.PHONY: pylint_script # Run pylint on scripts
pylint_script:
	find ./script/ -type f -name "*.py" | xargs poetry run pylint --rcfile=pylintrc

.PHONY: flake8 # Run flake8
flake8:
	poetry run flake8 --max-line-length 100 --per-file-ignores="__init__.py:F401" \
	$(SRC_DIR)/ tests/ script/

.PHONY: ruff
ruff:
	poetry run ruff $(SRC_DIR)/ tests/ script/

.PHONY: python_linting # Run python linters
python_linting: pylint flake8 ruff

.PHONY: conformance # Run command to fix some conformance issues automatically
conformance: finalize_nb python_format supported_functions licenses

.PHONY: pcc # Run pre-commit checks
pcc:
	@$(MAKE) --keep-going --jobs $$(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory pcc_internal

PCC_DEPS := check_python_format check_finalize_nb python_linting mypy_ci pydocstyle shell_lint
PCC_DEPS += check_supported_functions # check_licenses

# Not commented on purpose for make help, since internal
.PHONY: pcc_internal
pcc_internal: $(PCC_DEPS)

# One can reproduce pytest thanks to the --randomly-seed which is given by
# pytest-randomly
.PHONY: pytest # Run pytest
pytest:
	poetry run pytest -svv \
	--global-coverage=.global-coverage.json \
	-n $$(./script/make_utils/ncpus.sh) \
	--cov=$(SRC_DIR) --cov-fail-under=100 \
	--randomly-dont-reorganize \
	--cov-report=term-missing:skip-covered tests/

# Not a huge fan of ignoring missing imports, but some packages do not have typing stubs
.PHONY: mypy # Run mypy
mypy:
	poetry run mypy -p $(SRC_DIR) --ignore-missing-imports

# Friendly target to run mypy without ignoring missing stubs and still have errors messages
# Allows to see which stubs we are missing
.PHONY: mypy_ns # Run mypy (without ignoring missing stubs)
mypy_ns:
	poetry run mypy -p $(SRC_DIR)

.PHONY: mypy_test # Run mypy on test files
mypy_test:
	find ./tests/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports

.PHONY: mypy_script # Run mypy on scripts
mypy_script:
	find ./script/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports

# The plus indicates that make will be called by the command and allows to share the context with
# the parent make execution. We serialize calls to these targets as they may overwrite each others
# cache which can cause issues.
.PHONY: mypy_ci # Run all mypy checks for CI
mypy_ci:
	$(MAKE) --keep-going mypy mypy_test mypy_script

.PHONY: docker_build # Build dev docker
docker_build:
	BUILD_ARGS=; \
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .

.PHONY: docker_rebuild # Rebuild docker
docker_rebuild: docker_clean_volumes
	BUILD_ARGS=; \
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull --no-cache -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .

.PHONY: docker_start # Launch docker
docker_start:
	@# the slash before pwd is for Windows
	docker run --rm -it \
	-p 8888:8888 \
	--env DISPLAY=host.docker.internal:0 \
	--volume /"$$(pwd)":/src \
	--volume $(DEV_CONTAINER_VENV_VOLUME):/home/dev_user/dev_venv \
	--volume $(DEV_CONTAINER_CACHE_VOLUME):/home/dev_user/.cache \
	$(DEV_DOCKER_IMG)

.PHONY: docker_build_and_start # Docker build and start
docker_build_and_start: docker_build docker_start

.PHONY: docker_bas  # Docker build and start
docker_bas: docker_build_and_start

.PHONY: docker_clean_volumes  # Docker clean volumes
docker_clean_volumes:
	docker volume rm -f $(DEV_CONTAINER_VENV_VOLUME)
	docker volume rm -f $(DEV_CONTAINER_CACHE_VOLUME)

.PHONY: docker_cv # Docker clean volumes
docker_cv: docker_clean_volumes

.PHONY: pydocstyle # Launch syntax checker on source code documentation
pydocstyle:
	@# From http://www.pydocstyle.org/en/stable/error_codes.html
	poetry run pydocstyle $(SRC_DIR) --convention google --add-ignore=D1,D200,D202,D212,D402,D417 --add-select=D401

.PHONY: finalize_nb # Sanitize notebooks
finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py docs

# A warning in a package unrelated to the project made pytest fail with notebooks
# Run notebook tests without warnings as sources are already tested with warnings treated as errors
.PHONY: pytest_nb # Launch notebook tests
pytest_nb:
	find docs -name "*.ipynb" | grep -v _build | grep -v .ipynb_checkpoints | xargs poetry run pytest -Wignore --nbmake

.PHONY: jupyter # Launch jupyter notebook
jupyter:
	poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0

.PHONY: release_docker # Build a docker release image
release_docker:
	./docker/build_release_image.sh

.PHONY: upgrade_py_deps # Upgrade python dependencies
upgrade_py_deps:
	./script/make_utils/upgrade_deps.sh

# Keeping this target as it proved useful before we had a proper package, allowed to run code that
# pytest-codeblocks was failing to execute if not installed as a pip package.
# This is done by hand as pytest-codeblocks was failing with our native extensions.
# See refused PR on the project here: https://github.com/nschloe/pytest-codeblocks/pull/58
# Test code blocks using a custom python script in the documentation
.PHONY: test_codeblocks
test_codeblocks:
	poetry run python ./script/make_utils/test_md_python_code.py --md_dir docs/

.PHONY: pytest_codeblocks # Test code blocks using pytest in the documentation
pytest_codeblocks:
	poetry run pytest --codeblocks -svv -n $$(./script/make_utils/ncpus.sh) \
	--randomly-dont-reorganize docs/

# From https://stackoverflow.com/a/63523300 for the find command
.PHONY: shell_lint # Lint all bash scripts
shell_lint:
	find \( -path "./.venv" -o -path "./.docker_venv" \) -prune -o -type f -name "*.sh" -print | \
	xargs shellcheck

.PHONY: set_version_no_commit # Dry run for set_version
set_version_no_commit:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi && \
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}"

.PHONY: set_version # Generate a new version number and update all files with it accordingly
set_version:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi && \
	STASH_COUNT="$$(git stash list | wc -l)" && \
	git stash && \
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}" && \
	git add -u && \
	git commit -m "chore: bump version to $${VERSION}" && \
	NEW_STASH_COUNT="$$(git stash list | wc -l)" && \
	if [[ "$$NEW_STASH_COUNT" != "$$STASH_COUNT" ]]; then \
		git stash pop; \
	fi

.PHONY: changelog # Generate a changelog
changelog:
	PROJECT_VER=($$(poetry version)) && \
	PROJECT_VER="$${PROJECT_VER[1]}" && \
	poetry run python ./script/make_utils/changelog_helper.py > "CHANGELOG_$${PROJECT_VER}.md"

.PHONY: release # Create a new release
release:
	@PROJECT_VER=($$(poetry version)) && \
	PROJECT_VER="$${PROJECT_VER[1]}" && \
	TAG_NAME="v$${PROJECT_VER}" && \
	git fetch --tags --force && \
	git tag -s -a -m "$${TAG_NAME} release" "$${TAG_NAME}" && \
	git push origin "refs/tags/$${TAG_NAME}"


.PHONY: show_scope # Show the accepted types and optional scopes (for git conventional commits)
show_scope:
	@echo "Accepted types and optional scopes:"
	@cat .github/workflows/continuous-integration.yaml | grep feat | grep pattern | cut -f 2- -d ":" | cut -f 2- -d " "

.PHONY: show_type # Show the accepted types and optional scopes (for git conventional commits)
show_type:show_scope

# grep recursively, ignore binary files, print file line, print file name
# exclude dot dirs, exclude pylintrc (would match the notes)
# exclude notebooks (sometimes matches in svg text), match the notes in this directory
.PHONY: todo # List all todo left in the code
todo:
	@NOTES_ARGS=$$(poetry run python ./script/make_utils/get_pylintrc_notes.py \
	--pylintrc-path pylintrc) && \
	grep -rInH --exclude-dir='.[^.]*' --exclude=pylintrc --exclude='*.ipynb' "$${NOTES_ARGS}" .


.PHONY: supported_functions # Update docs with supported functions
supported_functions:
	poetry run python script/doc_utils/gen_supported_ufuncs.py docs/getting-started/compatibility.md

.PHONY: check_supported_functions # Check supported functions (for the doc)
check_supported_functions:
	poetry run python script/doc_utils/gen_supported_ufuncs.py docs/getting-started/compatibility.md --check

.PHONY: licenses # Generate the list of licenses of dependencies
licenses:
	@./script/make_utils/licenses.sh

.PHONY: check_licenses # Check if the licenses of dependencies have changed
check_licenses:
	@TMP_OUT="$$(mktemp)" && \
	if ! poetry run env bash ./script/make_utils/licenses.sh --check > "$${TMP_OUT}"; then \
		cat "$${TMP_OUT}"; \
		rm -f "$${TMP_OUT}"; \
		echo "Error while checking licenses, see log above."; \
		echo "Consider re-running 'make licenses'"; \
		exit 1; \
	else \
		echo "Licenses check OK"; \
	fi
.PHONY: check_licenses

.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1\t\2/' | expand -t30 | sort

.PHONY: pip_audit # Run pip-audit and check if there are known vulnerabilities in our dependencies
pip_audit:
	poetry run pip-audit

.PHONY: clean_local_git # Tell the user how to delete local git branches, except main
clean_local_git:
	@git fetch --all --prune
	@echo "Consider doing: "
	@echo
	@# Don't consider deleting `main` or current branches
	@git branch | grep -v "^*" | grep -v main | xargs echo "git branch -D "
	@echo
