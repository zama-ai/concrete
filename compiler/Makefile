BUILD_DIR=./build
Python3_EXECUTABLE=
BINDINGS_PYTHON_ENABLED=ON
PARALLEL_EXECUTION_ENABLED=OFF
CC_COMPILER=
CXX_COMPILER=

CONCRETE_OPTIMIZER_DIR ?= $(shell pwd)/concrete-optimizer

KEYSETCACHEDEV=/tmp/KeySetCache
KEYSETCACHECI ?= ../KeySetCache
KEYSETCACHENAME ?= KeySetCacheV1

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
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release \
	-DHPX_DIR=${HPX_INSTALL_DIR}/lib/cmake/HPX \
	-DLLVM_EXTERNAL_PROJECTS=concretelang \
	-DLLVM_EXTERNAL_CONCRETELANG_SOURCE_DIR=. \
	-DPython3_EXECUTABLE=${Python3_EXECUTABLE} \
	-DCONCRETE_OPTIMIZER_DIR=${CONCRETE_OPTIMIZER_DIR}
	touch $@

all: concretecompiler python-bindings build-tests build-benchmarks doc

build-initialized: concrete-optimizer-lib $(BUILD_DIR)/configured.stamp

CONCRETE_OPTIMIZER_SRC = $(shell find $(CONCRETE_OPTIMIZER_DIR)/* -not -path '$(CONCRETE_OPTIMIZER_DIR)/target*')

LIB_CONCRETE_OPTIMIZER_CPP = $(CONCRETE_OPTIMIZER_DIR)/target/libconcrete_optimizer_cpp.a

# we need CONCRETE_OPTIMIZER_DIR here to ensure it is present before computing CONCRETE_OPTIMIZER_SRC
$(LIB_CONCRETE_OPTIMIZER_CPP): $(CONCRETE_OPTIMIZER_SRC)
	make -C $(CONCRETE_OPTIMIZER_DIR)/concrete-optimizer-cpp $@

concrete-optimizer-lib: $(LIB_CONCRETE_OPTIMIZER_CPP)

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

build-benchmarks: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_benchmark

test-check: concretecompiler file-check not
	$(BUILD_DIR)/bin/llvm-lit -v tests/

test-python: python-bindings concretecompiler
	PYTHONPATH=${PYTHONPATH}:$(BUILD_DIR)/tools/concretelang/python_packages/concretelang_core LD_PRELOAD=$(BUILD_DIR)/lib/libConcretelangRuntime.so pytest -vs -m $(PYTHON_TESTS_MARKER) tests/python

test-compiler-file-output: concretecompiler
	pytest -vs tests/test_compiler_file_output

test: test-check test-end-to-end-jit test-python test-compiler-file-output support-unit-test testlib-unit-test

test-dataflow: test-end-to-end-jit-dfr test-end-to-end-jit-auto-parallelization

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

# unit-test

clientlib-unit-test: build-clientlib-unit-test
	$(BUILD_DIR)/bin/clientlib_unit_test

build-clientlib-unit-test:
	cmake --build $(BUILD_DIR) --target clientlib_unit_test

testlib-unit-test: build-testlib-unit-test
	$(BUILD_DIR)/tools/concretelang/tests/TestLib/testlib_unit_test

build-testlib-unit-test: build-initialized
	cmake --build $(BUILD_DIR) --target testlib_unit_test

support-unit-test: build-support-unit-test
	$(BUILD_DIR)/tools/concretelang/tests/Support/support_unit_test

build-support-unit-test: build-initialized
	cmake --build $(BUILD_DIR) --target support_unit_test


# test-end-to-end-jit

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

build-end-to-end-jit-dfr: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_dfr

build-end-to-end-jit-auto-parallelization: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_auto_parallelization

build-end-to-end-jit: build-end-to-end-jit-test build-end-to-end-jit-clear-tensor build-end-to-end-jit-encrypted-tensor build-end-to-end-jit-fhe build-end-to-end-jit-fhelinalg build-end-to-end-jit-lambda

build-tests: build-end-to-end-jit build-support-unit-test build-testlib-unit-test

test-end-to-end-jit-test: build-end-to-end-jit-test
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_test

test-end-to-end-jit-clear-tensor: build-end-to-end-jit-clear-tensor
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_clear_tensor

test-end-to-end-jit-fhe: build-end-to-end-jit-fhe
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_fhe

test-end-to-end-jit-encrypted-tensor: build-end-to-end-jit-encrypted-tensor
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_encrypted_tensor

test-end-to-end-jit-fhelinalg: build-end-to-end-jit-fhelinalg
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_fhelinalg

test-end-to-end-jit-lambda: build-initialized build-end-to-end-jit-lambda
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_lambda

test-end-to-end-jit-dfr: build-end-to-end-jit-dfr
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_dfr

test-end-to-end-jit-auto-parallelization: build-end-to-end-jit-auto-parallelization
	$(BUILD_DIR)/tools/concretelang/tests/unittest/end_to_end_jit_auto_parallelization

test-end-to-end-jit: test-end-to-end-jit-test test-end-to-end-jit-fhe test-end-to-end-jit-clear-tensor test-end-to-end-jit-encrypted-tensor test-end-to-end-jit-fhelinalg test-end-to-end-jit-lambda

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

generate_conv_op:
	python -m mlir.dialects.linalg.opdsl.dump_oplib ops.core_named_ops > ops/LinalgNamedStructuredOps.yaml
	$(BUILD_DIR)/bin/mlir-linalg-ods-yaml-gen ops/LinalgNamedStructuredOps.yaml --o-impl=ops/LinalgOps.cpp --o-ods-decl=ops/LinalgNamedStructuredOps.yamlgen.td

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
	test-check \
	test-end-to-end-jit \
	test-end-to-end-jit-test \
	test-end-to-end-jit-clear-tensor \
	test-end-to-end-jit-encrypted-tensor \
	test-end-to-end-jit-fhelinalg \
	test-python \
	test \
	build-tests \
	add-deps \
	file-check \
	not \
	package_py37 \
	package_py38 \
	package_py39 \
	package_py310 \
	release_tarballs \
	update_python_version \
	generate_conv_op \
	python_lint \
	python_format \
	check_python_format \
	concrete-optimizer-lib
