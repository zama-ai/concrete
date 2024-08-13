PYTHON=python
PIP=$(PYTHON) -m pip

COMPILER_BUILD_DIRECTORY ?= $(PWD)/../../compilers/concrete-compiler/compiler/build
BINDINGS_DIRECTORY=${COMPILER_BUILD_DIRECTORY}/tools/concretelang/python_packages/concretelang_core/
RUNTIME_LIBRARY=${COMPILER_BUILD_DIRECTORY}/lib/libConcretelangRuntime.so

CONCRETE_VERSION?="" # empty mean latest
# E.g. to use a previous version: `make CONCRETE_VERSION="<2.7.0" venv`
# E.g. to use a nightly: `make CONCRETE_VERSION="==2.7.0dev20240801`
# see https://pypi.zama.ai/cpu/concrete-python/index.html for available versions

.ONESHELL:
.PHONY:
.SILENT:

# =====
# Setup
# =====

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate
ifeq (,$(wildcard ${RUNTIME_LIBRARY}))
	$(PIP) install --extra-index-url https://pypi.zama.ai/cpu "concrete-python$(CONCRETE_VERSION)"
endif
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.extra-full.txt
	$(PIP) install -r requirements.txt

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

pytest: pytest-default

pytest-default:
	eval $(shell make silent_cp_activate)
	pytest tests -svv -n auto \
		--cov=concrete.fhe \
		--cov-fail-under=100 \
		--cov-report=term-missing:skip-covered \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

pytest-single:
	eval $(shell make silent_cp_activate)
	# test single precision, mono params
	pytest tests -svv -n auto \
		--precision=single \
		--strategy=mono \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

pytest-multi:
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
	# test single precision
	pytest tests -svv -n0 --use_gpu \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

	# test multi precision
	pytest tests -svv -n0 --use_gpu \
		--precision=multi \
		--cov=concrete.fhe \
		--cov-fail-under=100 \
		--cov-report=term-missing:skip-covered \
		--key-cache "${KEY_CACHE_DIRECTORY}" \
		-m "${PYTEST_MARKERS}"

benchmark:
	eval $(shell make silent_cp_activate)
	export PROGRESS_SAMPLES=3
	export PROGRESS_OUTPUT_INDENT=2

	rm -rf progress.json
	find ./benchmarks/ -name "*.py" | xargs python

process-benchmark-results-for-grafana:
	eval $(shell make silent_cp_activate)
	python scripts/benchmark/postprocessor.py \
		--source progress.json \
		--target progress.processed.json \
		--path_to_repository ../..

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
	mypy concrete examples scripts tests benchmarks --ignore-missing-imports

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
	ruff concrete/ examples/ scripts/ tests/ benchmarks/

pcc: check-format check-sanitize-notebooks mypy pydocstyle pylint ruff

# ============
# Distribution
# ============

OS=undefined
ifeq ($(shell uname), Linux)
	OS=linux
else ifeq ($(shell uname), Darwin)
	OS=darwin
endif

clear-whls:
	rm -rf dist

build-whl:
	mkdir -p dist
	$(PIP) wheel --no-deps -w dist .

patch-whl-linux:
	GLIBC_VER=$(shell ldd --version | head -n 1 | grep -o '[^ ]*$$'|head|tr '.' '_'); \
	for PLATFORM in manylinux_$${GLIBC_VER}_x86_64 linux_x86_64; do \
		if $(PYTHON) -m auditwheel repair -w dist --plat $$PLATFORM dist/*.whl; then \
			echo Success for $$PLATFORM; \
			break; \
		else \
			echo No repair with $$PLATFORM; \
		fi \
	done

patch-whl-darwin:
	delocate-wheel -v dist/*macosx*.whl

whl: clear-whls build-whl patch-whl-$(OS)
