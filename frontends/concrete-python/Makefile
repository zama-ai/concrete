PYTHON=python
PIP=$(PYTHON) -m pip

COMPILER_BUILD_DIRECTORY ?= $(PWD)/../../compilers/concrete-compiler/compiler/build
BINDINGS_DIRECTORY=${COMPILER_BUILD_DIRECTORY}/tools/concretelang/python_packages/concretelang_core/
TFHERS_UTILS_DIRECTORY ?= $(PWD)/tests/tfhers-utils/
CONCRETE_KEYGEN_DIRECTORY ?= $(PWD)/../concrete-rust/concrete-keygen/

OS=undefined
ifeq ($(shell uname), Linux)
	OS=linux
	RUNTIME_LIBRARY?=${COMPILER_BUILD_DIRECTORY}/lib/libConcretelangRuntime.so
else ifeq ($(shell uname), Darwin)
	OS=darwin
	RUNTIME_LIBRARY?=${COMPILER_BUILD_DIRECTORY}/lib/libConcretelangRuntime.dylib
endif

CONCRETE_VERSION?="" # empty mean latest
# E.g. to use a previous version: `make CONCRETE_VERSION="<2.7.0" venv`
# E.g. to use a nightly: `make CONCRETE_VERSION="==2.7.0dev20240801`
# see https://pypi.zama.ai/cpu/concrete-python/index.html for available versions

.ONESHELL:
.SHELLFLAGS=-e -c
.PHONY:
.SILENT:

# =====
# Setup
# =====

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate
	python -m pip install -r requirements.dev.txt
	python -m pip install -r requirements.extra-full.txt
	python -m pip install -r requirements.txt

	@echo
	@echo "The new environment is automatically activated with locals cp & bindings for makefile targets."
	@echo "You can have the same activation in a terminal using:"
	@echo 'eval $$(make cp_activate)'

cp_activate: silent_cp_activate
	@echo "echo 'Activating virtual env and local concrete-python and local compiler bindings' ;"

silent_cp_activate:
	@echo "export PATH=$(PWD)/.venv/bin:$$PATH;"
ifeq (,$(wildcard ${RUNTIME_LIBRARY}))
	@echo "export PYTHONPATH=$(PWD);"
else
	@echo "export LD_PRELOAD=$(RUNTIME_LIBRARY);"
	@echo "export PYTHONPATH=${BINDINGS_DIRECTORY}:$(PWD);"
endif

licenses:
	bash scripts/versioning/summary.sh

# =======
# Testing
# =======

tfhers-utils:
	cd ${TFHERS_UTILS_DIRECTORY} && $(MAKE) build

concrete-rust-keygen:
	cd ${CONCRETE_KEYGEN_DIRECTORY} && $(MAKE) build

pytest: pytest-default

pytest-default: tfhers-utils concrete-rust-keygen
	eval $(shell make silent_cp_activate)
	pytest tests -svv -n auto \
		--cov=concrete.fhe \
		--cov-fail-under=100 \
		--cov-report=term-missing:skip-covered \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

pytest-macos:
	pytest tests -svv -n auto \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m minimal

pytest-single: tfhers-utils concrete-rust-keygen
	eval $(shell make silent_cp_activate)
	# test single precision, mono params
	pytest tests -svv -n auto \
		--precision=single \
		--strategy=mono \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

pytest-multi: tfhers-utils concrete-rust-keygen
	eval $(shell make silent_cp_activate)
	# test multi precision, multi params
	pytest tests -svv -n auto \
		--precision=multi \
		--strategy=multi \
		--cov=concrete.fhe \
		--cov-fail-under=100 \
		--cov-report=term-missing:skip-covered \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

pytest-gpu:
	eval $(shell make silent_cp_activate)
	# test multi precision
	pytest tests -svv -n0 --use_gpu \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m minimal

benchmark:
	eval $(shell make silent_cp_activate)

	export PROGRESS_SAMPLES=3
	export PROGRESS_OUTPUT_INDENT=2

	rm -rf progress.json
	find ./benchmarks/ -name "*.py" -exec python {} \;

benchmark-target:
	eval $(shell make silent_cp_activate)

	export PROGRESS_SAMPLES=3
	export PROGRESS_OUTPUT_INDENT=2

	python "benchmarks/$(TARGET).py"

process-benchmark-results-for-grafana:
	eval $(shell make silent_cp_activate)
	python scripts/benchmark/postprocessor.py \
		--source progress.json \
		--target progress.processed.json \
		--path_to_repository ../..

test-notebooks:
	eval $(shell make silent_cp_activate)
	./scripts/jupyter/jupyter.sh

# ==========
# Formatting
# ==========

format:
	eval $(shell make silent_cp_activate)
	bash scripts/format/formatter.sh \
		--dir concrete \
		--dir examples \
		--dir scripts \
		--dir tests \
		--dir benchmarks

sanitize-notebooks:
	eval $(shell make silent_cp_activate)
	$(PYTHON) scripts/notebook/sanitizer.py docs

conformance: format sanitize-notebooks

# =======
# Linting
# =======

check-format:
	eval $(shell make silent_cp_activate)
	bash scripts/format/formatter.sh --check \
		--dir concrete \
		--dir examples \
		--dir scripts \
		--dir tests \
        --dir benchmarks

check-sanitize-notebooks:
	eval $(shell make silent_cp_activate)
	$(PYTHON) scripts/notebook/sanitizer.py docs --check

mypy:
	eval $(shell make silent_cp_activate)
	mypy concrete examples scripts tests benchmarks --ignore-missing-imports --explicit-package-bases

pydocstyle:
	eval $(shell make silent_cp_activate)
	pydocstyle concrete --convention google --add-ignore=D1,D200,D202,D212,D402 --add-select=D401

pylint:
	eval $(shell make silent_cp_activate)
	pylint --rcfile=.pylintrc concrete
	pylint --rcfile=.pylintrc examples --disable=C0103,C0114,C0115,C0116,E0401,R1721
	pylint --rcfile=.pylintrc scripts
	pylint --rcfile=.pylintrc tests --disable=C0301,W0108
	pylint --rcfile=.pylintrc benchmarks

ruff:
	eval $(shell make silent_cp_activate)

	ruff check concrete/
	ruff check examples/
	ruff check scripts/
	ruff check tests/
	ruff check benchmarks/

check-links:
	@# Check that no links target the main branch, some internal repositories (Concrete ML or Concrete) or our internal GitBook
	bash ./scripts/links/check_internal_links.sh

	@# To avoid some issues with priviledges and linkcheckmd
	find ../../docs/ -name "*.md" -type f | xargs chmod +r

	@# Run linkcheck on mardown files. It is mainly used for web links
	$(PYTHON)  -m linkcheckmd ../../docs -local

	$(PYTHON)  -m linkcheckmd ../../README.md

	@# Check that relative links in mardown files are targeting existing files
	$(PYTHON)  ./scripts/links/local_link_check.py

	@# Check that links to mardown headers in mardown files are targeting existing headers
	$(PYTHON)  ./scripts/links/check_headers.py

	@# For weblinks and internal references
	linkchecker ../../docs --check-extern \
		--no-warnings

pcc: check-format check-sanitize-notebooks mypy pydocstyle pylint ruff check-links

# ============
# Distribution
# ============

clear-whls:
	rm -rf dist

build-whl:
	mkdir -p dist
	$(PIP) wheel --no-deps -w dist .

patch-whl-linux:
	GLIBC_VER=$(shell ldd --version | head -n 1 | grep -o '[^ ]*$$'|head|tr '.' '_'); \
	for PLATFORM in manylinux_$${GLIBC_VER}_x86_64 linux_x86_64; do \
        if $(PYTHON) -m auditwheel repair \
                -w dist \
                        --exclude "libstdc++.so.*" \
                        --exclude "libm.so.*" \
                        --exclude "libc.so.*" \
                        --exclude "libgcc_s.so.*" \
                        --exclude "libcap.so*" \
                        --exclude "libudev.so*" \
                        --plat $$PLATFORM dist/*.whl;\
        then \
			echo Success for $$PLATFORM; \
			break; \
		else \
			echo No repair with $$PLATFORM; \
		fi \
	done

patch-whl-darwin:
	delocate-wheel -v dist/*macosx*.whl

whl: clear-whls build-whl patch-whl-$(OS)
