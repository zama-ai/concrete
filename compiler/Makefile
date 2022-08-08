BUILD_DIR=./build
Python3_EXECUTABLE?=
BINDINGS_PYTHON_ENABLED=ON
PARALLEL_EXECUTION_ENABLED=OFF
TIMING_ENABLED=OFF
CC_COMPILER=
CXX_COMPILER=

CONCRETE_OPTIMIZER_DIR ?= $(shell pwd)/concrete-optimizer

KEYSETCACHEDEV=/tmp/KeySetCache
KEYSETCACHECI ?= ../KeySetCache
KEYSETCACHENAME ?= KeySetCacheV1

HPX_VERSION?=1.7.1
HPX_URL=https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/$(HPX_VERSION).tar.gz
HPX_TARBALL=$(shell pwd)/hpx-$(HPX_VERSION).tar.gz
HPX_LOCAL_DIR=$(shell pwd)/hpx-$(HPX_VERSION)
HPX_INSTALL_DIR?=$(HPX_LOCAL_DIR)

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

# don't run parallel python tests if compiler doesn't support it
ifeq ($(PARALLEL_EXECUTION_ENABLED),ON)
		PYTHON_TESTS_MARKER=""
else
		PYTHON_TESTS_MARKER="not parallel"
endif

all: concretecompiler python-bindings build-tests build-benchmarks doc

$(BUILD_DIR)/configured.stamp:
	mkdir -p $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -GNinja ../llvm-project/llvm/ \
	$(CMAKE_CCACHE_OPTIONS) \
	$(CC_COMPILER_OPTION) \
	$(CXX_COMPILER_OPTION) \
	-DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
	-DLLVM_BUILD_EXAMPLES=OFF \
	-DLLVM_TARGETS_TO_BUILD="host" \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DMLIR_ENABLE_BINDINGS_PYTHON=$(BINDINGS_PYTHON_ENABLED) \
	-DCONCRETELANG_BINDINGS_PYTHON_ENABLED=$(BINDINGS_PYTHON_ENABLED) \
	-DCONCRETELANG_PARALLEL_EXECUTION_ENABLED=$(PARALLEL_EXECUTION_ENABLED) \
	-DCONCRETELANG_TIMING_ENABLED=$(TIMING_ENABLED) \
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release \
	-DHPX_DIR=${HPX_INSTALL_DIR}/lib/cmake/HPX \
	-DLLVM_EXTERNAL_PROJECTS=concretelang \
	-DLLVM_EXTERNAL_CONCRETELANG_SOURCE_DIR=. \
	-DPython3_EXECUTABLE=${Python3_EXECUTABLE} \
	-DCONCRETE_OPTIMIZER_DIR=${CONCRETE_OPTIMIZER_DIR}
	touch $@

build-initialized: concrete-optimizer-lib $(BUILD_DIR)/configured.stamp

CONCRETE_OPTIMIZER_SRC = $(shell find $(CONCRETE_OPTIMIZER_DIR)/* -not -path '$(CONCRETE_OPTIMIZER_DIR)/target*')

LIB_CONCRETE_OPTIMIZER_CPP = $(CONCRETE_OPTIMIZER_DIR)/target/libconcrete_optimizer_cpp.a

# we need CONCRETE_OPTIMIZER_DIR here to ensure it is present before computing CONCRETE_OPTIMIZER_SRC
$(LIB_CONCRETE_OPTIMIZER_CPP): $(CONCRETE_OPTIMIZER_SRC)
	make -C $(CONCRETE_OPTIMIZER_DIR)/concrete-optimizer-cpp $@

concrete-optimizer-lib: $(LIB_CONCRETE_OPTIMIZER_CPP)

# HPX

install-hpx-from-source: $(HPX_LOCAL_DIR)
	mkdir -p $(HPX_LOCAL_DIR)/build
	cd $(HPX_LOCAL_DIR)/build && cmake \
		-DHPX_WITH_FETCH_ASIO=on \
		-DHPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY=ON \
		-DHPX_WITH_MALLOC=system ..
	cd $(HPX_LOCAL_DIR)/build && make -j2

$(HPX_TARBALL): 
	curl -L $(HPX_URL) -o $(HPX_TARBALL)

$(HPX_LOCAL_DIR): $(HPX_TARBALL)
	tar xzvf $(HPX_TARBALL)

doc: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-doc

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
GITHUB_URL_LIST_ARTIFACTS="${GITHUB_URL}/actions/artifacts"
CURL=curl -H"Accept: application/vnd.github.v3+json" -H"authorization: Bearer ${GITHUB_TOKEN}"
keysetcache.zip: REDIRECT_URL = $(shell ${CURL} -s ${GITHUB_URL_LIST_ARTIFACTS} | grep -A 10 ${KEYSETCACHENAME} | grep archive_download_url | head -n 1 | grep -o 'http[^"]\+')
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


# test

build-tests: build-unit-tests build-end-to-end-tests

run-tests: run-check-tests run-unit-tests run-end-to-end-tests run-python-tests

## check-tests

run-check-tests: concretecompiler file-check not
	$(BUILD_DIR)/bin/llvm-lit -v tests/

## unit-tests

build-unit-tests:
	cmake --build $(BUILD_DIR) --target ConcretelangUnitTests

run-unit-tests: build-unit-tests
	find $(BUILD_DIR)/tools/concretelang/tests/unit_tests -name unit_tests_concretelang* -executable -type f | xargs -n1 ./run_test_bin.sh

## python-tests

run-python-tests: python-bindings concretecompiler
	PYTHONPATH=${PYTHONPATH}:$(BUILD_DIR)/tools/concretelang/python_packages/concretelang_core LD_PRELOAD=$(BUILD_DIR)/lib/libConcretelangRuntime.so pytest -vs -m $(PYTHON_TESTS_MARKER) tests/python

test-compiler-file-output: concretecompiler
	pytest -vs tests/test_compiler_file_output

## end-to-end-tests

build-end-to-end-tests: build-end-to-end-jit-test build-end-to-end-jit-clear-tensor build-end-to-end-jit-fhe build-end-to-end-jit-encrypted-tensor build-end-to-end-jit-fhelinalg build-end-to-end-jit-lambda

run-end-to-end-tests: build-end-to-end-tests
	find $(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests -name end_to_end* -executable -type f | xargs -n1 ./run_test_bin.sh

build-end-to-end-jit-test: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_test

build-end-to-end-jit-clear-tensor: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_clear_tensor

build-end-to-end-jit-fhe: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_fhe

build-end-to-end-jit-encrypted-tensor: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_encrypted_tensor

build-end-to-end-jit-fhelinalg: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_fhelinalg

build-end-to-end-jit-lambda: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_lambda

## end-to-end-dataflow-tests

build-end-to-end-dataflow-tests: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_auto_parallelization
	cmake --build $(BUILD_DIR) --target end_to_end_jit_distributed
	cmake --build $(BUILD_DIR) --target end_to_end_jit_aes_short

run-end-to-end-dataflow-tests: build-end-to-end-dataflow-tests
	$(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_jit_auto_parallelization
	$(BUILD_DIR)/tools/concretelang/tests/end_to_end_tests/end_to_end_jit_distributed

# benchmark

build-benchmarks: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_benchmark

run-benchmarks: build-benchmarks
	$(BUILD_DIR)/bin/end_to_end_benchmark

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

# Python packages

define build_image_and_copy_wheels
	docker image build -t concrete-compiler-manylinux:$(1) --build-arg python_tag=$(1) -f ../builders/Dockerfile.release_manylinux_2_24_x86_64 ..
	docker container run --rm -v ${PWD}/../wheels:/wheels_volume concrete-compiler-manylinux:$(1) cp -r /wheels/. /wheels_volume/.
endef


package_py37: update_python_version
	$(call build_image_and_copy_wheels,cp37-cp37m)

package_py38: update_python_version
	$(call build_image_and_copy_wheels,cp38-cp38)

package_py39: update_python_version
	$(call build_image_and_copy_wheels,cp39-cp39)

package_py310: update_python_version
	$(call build_image_and_copy_wheels,cp310-cp310)

release_tarballs:
	docker image build -t concrete-compiler-manylinux:linux_x86_64_tarball -f ../builders/Dockerfile.release_tarball_linux_x86_64 ..
	docker container run --rm -v ${PWD}/../tarballs:/tarballs_volume concrete-compiler-manylinux:linux_x86_64_tarball cp -r /tarballs/. /tarballs_volume/.

update_python_version:
	echo "__version__ = \"`git describe --tags --abbrev=0 | grep -e '[0-9].*' -o`\"" > lib/Bindings/Python/version.txt

check_python_format:
	black --check tests/python/ lib/Bindings/Python/concrete/

python_format:
	black tests/python/ lib/Bindings/Python/concrete/

python_lint:
	pylint --rcfile=../pylintrc lib/Bindings/Python/concrete/compiler

.PHONY: build-initialized \
	build-end-to-end-jit \
	concretecompiler \
	python-bindings \
	add-deps \
	file-check \
	not \
	package_py37 \
	package_py38 \
	package_py39 \
	package_py310 \
	release_tarballs \
	update_python_version \
	python_lint \
	python_format \
	check_python_format \
	concrete-optimizer-lib \
	build-tests \
	run-tests \
	run-check-tests \
	build-unit-tests \
	run-unit-tests \
	run-python-tests \
	build-end-to-end-tests \
	build-end-to-end-dataflow-tests \
	run-end-to-end-dataflow-tests
