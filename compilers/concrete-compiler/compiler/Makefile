LLVM_PROJECT_DIR=../llvm-project
DOCS_DIR=../../../docs

BUILD_TYPE?=Release

# Build dir is `build` for Release builds and `build_BUILD_TYPE` for
# other build types (e.g. Debug, ReleaseWithDebugInfo, etc.)
ifneq ($(BUILD_TYPE),Release)
	BUILD_DIR?=build_$(BUILD_TYPE)
else
	BUILD_DIR?=build
endif
Python3_EXECUTABLE?=$(shell which python3)
BINDINGS_PYTHON_ENABLED=ON
DATAFLOW_EXECUTION_ENABLED=OFF
TIMING_ENABLED=OFF
CC_COMPILER=
CXX_COMPILER=
CUDA_SUPPORT?=OFF
INSTALL_PREFIX?=$(abspath $(BUILD_DIR))/install
INSTALL_PATH=$(abspath $(INSTALL_PREFIX))/concretecompiler/
MAKEFILE_ROOT_DIR=$(shell pwd)
MINIMAL_TESTS?=OFF

KEYSETCACHEDEV=/tmp/KeySetCache
KEYSETCACHECI ?= ../KeySetCache
KEYSETCACHENAME ?= KeySetCacheV4

HPX_VERSION?=1.9.1
HPX_URL=https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/v$(HPX_VERSION).tar.gz
HPX_DIR?=$(shell pwd)
HPX_TARBALL=hpx-$(HPX_VERSION).tar.gz
HPX_LOCAL_DIR=$(HPX_DIR)/hpx-$(HPX_VERSION)
HPX_INSTALL_DIR?=$(HPX_LOCAL_DIR)/build

ML_BENCH_SUBSET_ID=

# Find OS
OS=undefined
ifeq ($(shell uname), Linux)
	OS=linux
else ifeq ($(shell uname), Darwin)
	OS=darwin
	MINIMAL_TESTS_BOOL=ON
endif

# Set rust flags to activate target cpu features
ifeq ($(shell uname -m), x86_64)
ifeq ($(shell uname), Linux)
export RUSTFLAGS=-Ctarget-feature=+aes,+sse2,+avx,+avx2
else
export RUSTFLAGS=-Ctarget-feature=+aes,+sse2,+avx
endif
else
# https://docs.rs/aes/0.8.2/aes/#configuration-flags
export RUSTFLAGS=-Ctarget-cpu=native --cfg aes_armv8
endif

# Setup find arguments for MacOS
ifeq ($(OS), darwin)
FIND_EXECUTABLE_ARG=-perm +111
else
FIND_EXECUTABLE_ARG=-executable
endif

ARCHITECTURE=undefined
ifeq ($(shell uname -m), arm64)
	ARCHITECTURE=aarch64
else
	ARCHITECTURE=amd64
endif

export PATH := $(abspath $(BUILD_DIR))/bin:$(PATH)

ifeq ($(shell which ccache),)
	CCACHE=OFF
else
	CCACHE=ON
endif

ifeq ($(CCACHE),ON)
		CMAKE_CCACHE_OPTIONS=-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
else
		CMAKE_CCACHE_OPTIONS=
endif

ifneq ($(CC_COMPILER),)
		CC_COMPILER_OPTION=-DCMAKE_C_COMPILER=$(CC_COMPILER)
else
		CC_COMPILER_OPTION=
endif

ifneq ($(CXX_COMPILER),)
		CXX_COMPILER_OPTION=-DCMAKE_CXX_COMPILER=$(CXX_COMPILER)
else
		CXX_COMPILER_OPTION=
endif

# If the build type is Debug, and the lld linked is installed
# then use it
CUSTOM_LINKER_OPTS=
ifneq ($(shell which lld),)
ifeq ($(BUILD_TYPE),Debug)
	CUSTOM_LINKER_OPTS=-DLLVM_USE_LINKER=lld
endif
endif

# don't run parallel python tests if compiler doesn't support it
ifeq ($(DATAFLOW_EXECUTION_ENABLED),ON)
		PYTHON_TESTS_MARKER=""
else
		PYTHON_TESTS_MARKER="not parallel"
endif

# Force linking of libomp with libstdc++ in debug builds, as these
# rely on code from libstdc++ implementing assertions
LIBOMP_LINK_TO_LIBSTDCXX_OPT=
ifeq ($(BUILD_TYPE),Debug)
	LIBOMP_LINK_TO_LIBSTDCXX_OPT=-DLIBOMP_USE_STDCPPLIB=ON
endif

all: concretecompiler python-bindings build-tests build-benchmarks doc

# HPX #####################################################

install-hpx-from-source: $(HPX_LOCAL_DIR)
	mkdir -p $(HPX_LOCAL_DIR)/build
	cd $(HPX_LOCAL_DIR)/build && cmake \
		-DHPX_WITH_MAX_CPU_COUNT="" \
		-DHPX_WITH_FETCH_ASIO=on \
		-DHPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY=ON \
		-DHPX_WITH_MALLOC=system \
		$(CXX_COMPILER_OPTION) $(CC_COMPILER_OPTION) ..
	cd $(HPX_LOCAL_DIR)/build && make -j2

$(HPX_TARBALL):
	mkdir -p $(HPX_DIR)
	cd $(HPX_DIR) && curl -L $(HPX_URL) -o $(HPX_TARBALL)

$(HPX_LOCAL_DIR): $(HPX_TARBALL)
	cd $(HPX_DIR) && tar xzvf $(HPX_TARBALL)

$(BUILD_DIR)/configured.stamp:
	mkdir -p $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -GNinja $(LLVM_PROJECT_DIR)/llvm \
	$(CMAKE_CCACHE_OPTIONS) \
	$(CC_COMPILER_OPTION) \
	$(CXX_COMPILER_OPTION) \
	$(CUSTOM_LINKER_OPTS) \
	-DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
	-DLLVM_BUILD_EXAMPLES=OFF \
	-DLLVM_TARGETS_TO_BUILD="host" \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DLLVM_ENABLE_RTTI=ON \
	-DLLVM_LINK_LLVM_DYLIB=on \
	-DMLIR_ENABLE_BINDINGS_PYTHON=$(BINDINGS_PYTHON_ENABLED) \
	-DCONCRETELANG_BINDINGS_PYTHON_ENABLED=$(BINDINGS_PYTHON_ENABLED) \
	-DCONCRETELANG_DATAFLOW_EXECUTION_ENABLED=$(DATAFLOW_EXECUTION_ENABLED) \
	-DCONCRETELANG_TIMING_ENABLED=$(TIMING_ENABLED) \
	-DHPX_DIR=${HPX_INSTALL_DIR}/lib/cmake/HPX \
	-DLLVM_EXTERNAL_PROJECTS=concretelang \
	-DLLVM_EXTERNAL_CONCRETELANG_SOURCE_DIR=. \
	-DPython3_EXECUTABLE=${Python3_EXECUTABLE} \
	-DCONCRETELANG_CUDA_SUPPORT=${CUDA_SUPPORT} \
	-DCUDAToolkit_ROOT=$(CUDA_PATH) \
	$(LIBOMP_LINK_TO_LIBSTDCXX_OPT)
	touch $@

build-initialized: $(BUILD_DIR)/configured.stamp

reconfigure:
	rm $(BUILD_DIR)/configured.stamp
	rm $(BUILD_DIR)/CMakeCache.txt

doc: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-doc

update-dialect-doc: doc
	cp $(BUILD_DIR)/tools/concretelang/docs/concretelang/*Dialect.md $(DOCS_DIR)/dev/compilation/ && \
	sed -i -e 's/\[TOC\]//' $(DOCS_DIR)/dev/compilation/*Dialect.md

concretecompiler: build-initialized
	cmake --build $(BUILD_DIR) --target concretecompiler

python-bindings: build-initialized
	cmake --build $(BUILD_DIR) --target ConcretelangMLIRPythonModules
	cmake --build $(BUILD_DIR) --target ConcretelangPythonModules

clientlib: build-initialized
	cmake --build $(BUILD_DIR) --target ConcretelangClientLib

serverlib: build-initialized
	cmake --build $(BUILD_DIR) --target ConcretelangServerLib



GITHUB_URL=https://api.github.com/repos/zama-ai/concrete-compiler-internal
GITHUB_URL_LIST_ARTIFACTS="${GITHUB_URL}/actions/artifacts?name=${KEYSETCACHENAME}&per_page=1"
CURL=curl -H"Accept: application/vnd.github.v3+json" -H"authorization: Bearer ${GITHUB_TOKEN}"
keysetcache.zip: REDIRECT_URL = $(shell ${CURL} -s ${GITHUB_URL_LIST_ARTIFACTS} | grep archive_download_url | grep -o 'http[^"]\+')
keysetcache.zip:
	${CURL} --location -o keysetcache.zip ${REDIRECT_URL}
	du -h keysetcache.zip

keysetcache_ci_populated: keysetcache.zip
	unzip keysetcache.zip -d ${KEYSETCACHECI}
	du -sh ${KEYSETCACHECI}
	rm keysetcache.zip

keysetcache_populated: keysetcache.zip
	unzip keysetcache.zip -d ${KEYSETCACHEDEV}
	du -sh ${KEYSETCACHEDEV}
	rm keysetcache.zip


ifeq ($(GTEST_PARALLEL_ENABLED),OFF)
GTEST_PARALLEL_PY=
GTEST_PARALLEL_CMD=
GTEST_PARALLEL_SEPARATOR=
else
GTEST_PARALLEL_PY=./gtest-parallel/gtest-parallel
GTEST_PARALLEL_CMD=$(Python3_EXECUTABLE) $(GTEST_PARALLEL_PY) -d $(BUILD_DIR)
GTEST_PARALLEL_SEPARATOR=--
$(GTEST_PARALLEL_PY):
	git clone https://github.com/google/gtest-parallel.git
endif


# test

build-tests: build-unit-tests build-end-to-end-tests

run-tests: run-check-tests run-unit-tests run-end-to-end-tests run-random-end-to-end-tests-for-each-options run-python-tests

## check-tests

run-check-tests: concretecompiler file-check not
	$(BUILD_DIR)/bin/llvm-lit -v tests/check_tests

## unit-tests

build-unit-tests: build-initialized
	cmake --build $(BUILD_DIR) --target ConcretelangUnitTests

run-unit-tests: build-unit-tests
	find $(BUILD_DIR)/tools/concretelang/tests/unit_tests -name unit_tests_concretelang* $(FIND_EXECUTABLE_ARG) -type f | xargs -n1 ./run_test_bin.sh

## python-tests

run-python-tests: python-bindings concretecompiler
	PYTHONPATH=${PYTHONPATH}:$(BUILD_DIR)/tools/concretelang/python_packages/concretelang_core LD_PRELOAD=$(BUILD_DIR)/lib/libConcretelangRuntime.so pytest -vs -m $(PYTHON_TESTS_MARKER) tests/python

test-compiler-file-output: concretecompiler
	pytest -vs tests/test_compiler_file_output

## end-to-end-tests

build-end-to-end-jit-chunked-int: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_chunked_int

build-end-to-end-jit-test: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_test

build-end-to-end-test: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_test

build-end-to-end-jit-lambda: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_lambda

build-end-to-end-tests: build-end-to-end-jit-chunked-int build-end-to-end-jit-test build-end-to-end-test build-end-to-end-jit-lambda

### end-to-end-tests CPU

FIXTURE_CPU_DIR=tests/end_to_end_fixture/tests_cpu

ifeq ($(MINIMAL_TESTS),ON)
	MINIMAL_TESTS_BOOL=true
else
	MINIMAL_TESTS_BOOL=false
endif
$(FIXTURE_CPU_DIR)/%.yaml: tests/end_to_end_fixture/%_gen.py FORCE
	mkdir -p $(FIXTURE_CPU_DIR)
	$(Python3_EXECUTABLE) $< --minimal=$(MINIMAL_TESTS_BOOL) > $@

$(FIXTURE_CPU_DIR)/bug_report.yaml:
	unzip -o $(FIXTURE_CPU_DIR)/bug_report.zip -d $(FIXTURE_CPU_DIR)

generate-cpu-tests: \
	$(FIXTURE_CPU_DIR)/end_to_end_leveled.yaml \
	$(FIXTURE_CPU_DIR)/end_to_end_apply_lookup_table.yaml \
	$(FIXTURE_CPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml \
	$(FIXTURE_CPU_DIR)/bug_report.yaml \
	$(FIXTURE_CPU_DIR)/end_to_end_round.yaml \
	$(FIXTURE_CPU_DIR)/end_to_end_multi_precision.yaml \
	$(FIXTURE_CPU_DIR)/end_to_end_linalg_enc_enc_matmul_dot.yaml

PARALLEL_END_2_END_TESTS= end_to_end_jit_test end_to_end_jit_lambda
run-end-to-end-tests: $(GTEST_PARALLEL_PY) build-end-to-end-tests generate-cpu-tests
	$(foreach TEST,$(PARALLEL_END_2_END_TESTS), \
	    $(GTEST_PARALLEL_CMD) $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/$(TEST) || exit $$?;)
	$(GTEST_PARALLEL_CMD) $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_test $(GTEST_PARALLEL_SEPARATOR) \
	--retry-failing-tests=5 $(FIXTURE_CPU_DIR)/*.yaml

OPTIONS_TO_TESTS=--simulate --compress-evaluation-keys --compress-input-ciphertexts --optimizer-strategy=dag-mono --optimizer-key-sharing=false --public-keys=compact
run-random-end-to-end-tests-for-each-options: $(GTEST_PARALLEL_PY) build-end-to-end-tests generate-cpu-tests
	$(foreach option,$(OPTIONS_TO_TESTS), \
	  $(GTEST_PARALLEL_CMD) $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_test $(GTEST_PARALLEL_SEPARATOR) \
	  --random-tests=100 $(option) $(FIXTURE_CPU_DIR)/*.yaml || exit $$?;)

### end-to-end-tests GPU

FIXTURE_GPU_DIR=tests/end_to_end_fixture/tests_gpu

$(FIXTURE_GPU_DIR):
	mkdir -p $(FIXTURE_GPU_DIR)

$(FIXTURE_GPU_DIR)/end_to_end_apply_lookup_table.yaml: tests/end_to_end_fixture/end_to_end_apply_lookup_table_gen.py
	$(Python3_EXECUTABLE) $< --bitwidth 1 2 3 4 5 6 7 8 > $@

$(FIXTURE_GPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml: tests/end_to_end_fixture/end_to_end_linalg_apply_lookup_table_gen.py
	$(Python3_EXECUTABLE) $< --bitwidth 1 2 3 4 5 6 7 8 > $@


generate-gpu-tests: $(FIXTURE_GPU_DIR) $(FIXTURE_GPU_DIR)/end_to_end_apply_lookup_table.yaml $(FIXTURE_GPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml

run-end-to-end-tests-gpu: build-end-to-end-test generate-gpu-tests
	$(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_test \
	--backend=gpu \
	$(FIXTURE_GPU_DIR)/*.yaml

## end-to-end-dataflow-tests

build-end-to-end-dataflow-tests: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_auto_parallelization
	cmake --build $(BUILD_DIR) --target end_to_end_jit_distributed
	cmake --build $(BUILD_DIR) --target end_to_end_jit_aes_short

run-end-to-end-dataflow-tests: build-end-to-end-dataflow-tests
	$(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_jit_auto_parallelization
	$(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_jit_distributed

run-end-to-end-distributed-tests: $(GTEST_PARALLEL_PY) build-end-to-end-tests generate-cpu-tests
	srun -n4 -c8 --kill-on-bad-exit=1 $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_jit_distributed
	srun -n4 -c8 --kill-on-bad-exit=1 $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_test \
	  --optimizer-strategy=dag-mono --dataflow-parallelize=1 \
	  $(FIXTURE_CPU_DIR)/*round*.yaml $(FIXTURE_CPU_DIR)/*relu*.yaml $(FIXTURE_CPU_DIR)/*linalg*.yaml

# benchmark

build-benchmarks: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_benchmark

## benchmark CPU

BENCHMARK_CPU_DIR=tests/end_to_end_fixture/benchmarks_cpu

$(BENCHMARK_CPU_DIR):
	mkdir -p $@

$(BENCHMARK_CPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml: tests/end_to_end_fixture/end_to_end_linalg_apply_lookup_table_gen.py
	$(Python3_EXECUTABLE) $< --n-ct 1024 > $@

$(BENCHMARK_CPU_DIR)/end_to_end_round.yaml: tests/end_to_end_fixture/end_to_end_round_gen.py
	$(Python3_EXECUTABLE) $< --shapes 1024 > $@

$(BENCHMARK_CPU_DIR)/%.yaml: tests/end_to_end_fixture/%_gen.py
	$(Python3_EXECUTABLE) $< > $@


BENCHS_CPU = \
	$(BENCHMARK_CPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml \
	$(BENCHMARK_CPU_DIR)/end_to_end_round.yaml

generate-cpu-benchmarks: $(BENCHMARK_CPU_DIR) $(BENCHS_CPU)

SECURITY_TO_BENCH=128
OPTIMIZATION_STRATEGY_TO_BENCH=dag-multi
run-cpu-benchmarks: build-benchmarks generate-cpu-benchmarks
	$(foreach optimizer_strategy,$(OPTIMIZATION_STRATEGY_TO_BENCH),$(foreach security,$(SECURITY_TO_BENCH),$(BUILD_DIR)/bin/end_to_end_benchmark \
		--backend=cpu --security-level=$(security) --optimizer-strategy=$(optimizer_strategy)\
		--benchmark_out=benchmarks_results.json --benchmark_out_format=json \
		$(BENCHMARK_CPU_DIR)/*.yaml || exit $$?;))

FIXTURE_APPLICATION_DIR=tests/end_to_end_fixture/application/

run-cpu-benchmarks-application:
	unzip $(FIXTURE_APPLICATION_DIR)/*.zip -d $(FIXTURE_APPLICATION_DIR)
	$(BUILD_DIR)/bin/end_to_end_benchmark \
		--backend=cpu --benchmark_out=benchmarks_results.json --benchmark_out_format=json \
		$(FIXTURE_APPLICATION_DIR)*.yaml

## benchmark GPU

BENCHMARK_GPU_DIR=tests/end_to_end_fixture/benchmarks_gpu

$(BENCHMARK_GPU_DIR):
	mkdir -p $@

$(BENCHMARK_GPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml: tests/end_to_end_fixture/end_to_end_linalg_apply_lookup_table_gen.py
	$(Python3_EXECUTABLE) $< \
	--bitwidth 1 2 3 4 5 6 7 --n-ct 1 128 1024 2048 8192


generate-gpu-benchmarks: $(BENCHMARK_GPU_DIR) $(BENCHMARK_GPU_DIR)/end_to_end_linalg_apply_lookup_table.yaml

run-gpu-benchmarks: build-benchmarks generate-cpu-benchmarks
	$(BUILD_DIR)/bin/end_to_end_benchmark \
	--backend=gpu \
	--benchmark_out=benchmarks_results.json --benchmark_out_format=json \
	$(BENCHMARK_CPU_DIR)/*.yaml


generate-mlbench:
	mkdir -p tests/end_to_end_benchmarks/mlbench
	rm -rf tests/end_to_end_benchmarks/mlbench/*
	unzip tests/end_to_end_benchmarks/mlbench.zip -d tests/end_to_end_benchmarks/mlbench
	rm -f tests/end_to_end_benchmarks/mlbench/**/*\=*
	find tests/end_to_end_benchmarks/mlbench -name "*.mlir" -exec sed -e '1d' -e 's/ func / func.func /g' -e 's/ linalg.tensor_/ tensor./g' -e '$$d' -i {} \;
	$(Python3_EXECUTABLE) tests/end_to_end_benchmarks/generate_bench_yaml.py tests/end_to_end_benchmarks/mlbench tests/end_to_end_benchmarks/mlbench/end_to_end_mlbench

show-stress-tests-summary:
	@echo '------ Stress tests summary ------'
	@echo
	@echo 'Rates:'
	@cd tests/stress_tests/trace && grep success_rate -R
	@echo
	@echo 'Parameters issues:'
	@cd tests/stress_tests/trace && grep BAD -R || echo 'No issues'

stress-tests: concretecompiler
	pytest -vs tests/stress_tests

# useful for faster cache generation, need pytest-parallel
stress-tests-fast-cache: concretecompiler
	pytest --workers auto -vs tests/stress_tests

# LLVM/MLIR dependencies

all-deps: file-check not

file-check: build-initialized
	cmake --build $(BUILD_DIR) --target FileCheck

not: build-initialized
	cmake --build $(BUILD_DIR) --target not

mlir-cpu-runner: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-cpu-runner

opt: build-initialized
	cmake --build $(BUILD_DIR) --target opt

mlir-opt: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-opt

mlir-translate: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-translate

update-python-version:
	echo "__version__ = \"`git describe --tags --abbrev=0 | grep -e '[0-9].*' -o`\"" > lib/Bindings/Python/version.txt

check-python-format:
	black --check tests/python/ lib/Bindings/Python/concrete/

python-format:
	black tests/python/ lib/Bindings/Python/concrete/

python-lint:
	pylint --rcfile=../pylintrc lib/Bindings/Python/concrete/compiler

# libraries we want to have in the installation that aren't already a deps of other targets
install-deps:
	cmake --build $(BUILD_DIR) --target MLIRCAPIRegisterEverything

ifeq ($(OS), darwin)
# rsync should normally come pre-installed on macOS
# and the --parents only exists for GNU's cp not BSD's cp
HIERARCHY_PRESERVING_COPY=rsync -R
else
HIERARCHY_PRESERVING_COPY=cp --parents
endif

ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

PIP=$(Python3_EXECUTABLE) -m pip
PIP_WHEEL=$(PIP) wheel --no-deps -w $(BUILD_DIR)/wheels .
AUDIT_WHEEL_REPAIR=$(Python3_EXECUTABLE) -m auditwheel repair -w $(BUILD_DIR)/wheels

linux-python-package:
	$(PIP) install wheel auditwheel
	# We need to run it twice: the first will generate the directories, so that
	# the second run can find the packages via find_namespace_packages
	$(PIP_WHEEL)
	$(PIP_WHEEL)
	GLIBC_VER=$(shell ldd --version | head -n 1 | grep -o '[^ ]*$$'|head|tr '.' '_'); \
	for PLATFORM in manylinux_$${GLIBC_VER}_x86_64 linux_x86_64; do \
		if $(AUDIT_WHEEL_REPAIR) $(BUILD_DIR)/wheels/*.whl --plat $$PLATFORM; then \
			echo Success for $$PLATFORM; \
			break; \
		else \
			echo No repair with $$PLATFORM; \
		fi \
	done

darwin-python-package:
	$(PIP) install wheel delocate
	$(PIP_WHEEL)
	delocate-wheel -v $(BUILD_DIR)/wheels/*macosx*.whl

python-package: python-bindings $(OS)-python-package
	@echo The python package is: $(BUILD_DIR)/wheels/*.whl

install: concretecompiler install-deps
	$(info Install prefix set to $(INSTALL_PREFIX))
	$(info Installing under $(INSTALL_PATH))
	mkdir -p $(INSTALL_PATH)/include
	cp -R $(abspath $(BUILD_DIR))/bin $(INSTALL_PATH)
	cp -R $(abspath $(BUILD_DIR))/lib $(INSTALL_PATH)

    # Doing find + grep + while loop is a way to have portable behaviour between macOS and GNU/Linux
    # as with `find . -regex "regex"`, the regex language is not the same / to have the same language, the
    # command changes (macOs: `find -E . -regex`, GNU: `find . -regextype posix-extended "regex")
	cd $(MAKEFILE_ROOT_DIR)/include && \
	    find . | \
	    grep "^.*\.\(h\|hpp\|td\)$$" | \
        while read filepath; do $(HIERARCHY_PRESERVING_COPY) $$filepath $(INSTALL_PATH)/include; done
	cd $(MAKEFILE_ROOT_DIR)/$(LLVM_PROJECT_DIR)/llvm/include && \
	    find . | \
	    grep "^.*\.\(h\|hpp\|td\)$$" | \
        while read filepath; do $(HIERARCHY_PRESERVING_COPY) $$filepath $(INSTALL_PATH)/include; done
	cd $(MAKEFILE_ROOT_DIR)/$(LLVM_PROJECT_DIR)/mlir/include && \
	    find . | \
	    grep "^.*\.\(h\|hpp\|td\)$$" | \
        while read filepath; do $(HIERARCHY_PRESERVING_COPY) $$filepath $(INSTALL_PATH)/include; done

	cd $(abspath $(BUILD_DIR))/include                    && find . -iname '*.inc' -exec $(HIERARCHY_PRESERVING_COPY) {} $(INSTALL_PATH)/include \;
	cd $(abspath $(BUILD_DIR))/tools/concretelang/include && find . -iname '*.inc' -exec $(HIERARCHY_PRESERVING_COPY) {} $(INSTALL_PATH)/include \;
	cd $(abspath $(BUILD_DIR))/tools/mlir/include         && find . -iname '*.inc' -exec $(HIERARCHY_PRESERVING_COPY) {} $(INSTALL_PATH)/include \;

FORCE:

.PHONY: build-initialized \
	build-end-to-end-jit \
	concretecompiler \
	python-bindings \
	add-deps \
	file-check \
	not \
	update-python-version \
	python-lint \
	python-format \
	check-python-format \
	build-tests \
	run-tests \
	run-check-tests \
	build-unit-tests \
	run-unit-tests \
	run-python-tests \
	build-end-to-end-tests \
	build-end-to-end-dataflow-tests \
	run-end-to-end-dataflow-tests \
	run-random-end-to-end-tests-for-each-options \
	opt \
	mlir-opt \
	mlir-cpu-runner \
	mlir-translate \
	reconfigure
