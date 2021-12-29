BUILD_DIR=./build
Python3_EXECUTABLE=
BINDINGS_PYTHON_ENABLED=ON
PARALLEL_EXECUTION_ENABLED=OFF

export PATH := $(BUILD_DIR)/bin:$(PATH)

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

$(BUILD_DIR)/configured.stamp:
	cmake -B $(BUILD_DIR) -GNinja ../llvm-project/llvm/ \
	$(CMAKE_CCACHE_OPTIONS) \
	-DLLVM_ENABLE_PROJECTS=mlir \
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
	-DPython3_EXECUTABLE=${Python3_EXECUTABLE}
	touch $@

build-initialized: $(BUILD_DIR)/configured.stamp

doc: build-initialized
	cmake --build $(BUILD_DIR) --target mlir-doc

concretecompiler: build-initialized
	cmake --build $(BUILD_DIR) --target concretecompiler

python-bindings: build-initialized
	cmake --build $(BUILD_DIR) --target ConcretelangMLIRPythonModules
	cmake --build $(BUILD_DIR) --target ConcretelangPythonModules

test-check: concretecompiler file-check not
	$(BUILD_DIR)/bin/llvm-lit -v tests/

test-python: python-bindings concretecompiler
	PYTHONPATH=${PYTHONPATH}:$(BUILD_DIR)/tools/concretelang/python_packages/concretelang_core LD_PRELOAD=$(BUILD_DIR)/lib/libConcretelangRuntime.so pytest -vs tests/python

test: test-check test-end-to-end-jit test-python support-unit-test

test-dataflow: test-end-to-end-jit-dfr test-end-to-end-jit-auto-parallelization

# unit-test

support-unit-test: build-support-unit-test
	$(BUILD_DIR)/bin/support_unit_test

build-support-unit-test: build-initialized
	cmake --build $(BUILD_DIR) --target support_unit_test


# test-end-to-end-jit

build-end-to-end-jit-test: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_test

build-end-to-end-jit-clear-tensor: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_clear_tensor

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

build-end-to-end-jit: build-end-to-end-jit-test build-end-to-end-jit-clear-tensor build-end-to-end-jit-encrypted-tensor build-end-to-end-jit-fhelinalg


test-end-to-end-jit-test: build-end-to-end-jit-test
	$(BUILD_DIR)/bin/end_to_end_jit_test

test-end-to-end-jit-clear-tensor: build-end-to-end-jit-clear-tensor
	$(BUILD_DIR)/bin/end_to_end_jit_clear_tensor

test-end-to-end-jit-encrypted-tensor: build-end-to-end-jit-encrypted-tensor
	$(BUILD_DIR)/bin/end_to_end_jit_encrypted_tensor

test-end-to-end-jit-fhelinalg: build-end-to-end-jit-fhelinalg
	$(BUILD_DIR)/bin/end_to_end_jit_fhelinalg

test-end-to-end-jit-lambda: build-initialized build-end-to-end-jit-lambda
	$(BUILD_DIR)/bin/end_to_end_jit_lambda

test-end-to-end-jit-dfr: build-end-to-end-jit-dfr
	$(BUILD_DIR)/bin/end_to_end_jit_dfr

test-end-to-end-jit-auto-parallelization: build-end-to-end-jit-auto-parallelization
	$(BUILD_DIR)/bin/end_to_end_jit_auto_parallelization

test-end-to-end-jit: test-end-to-end-jit-test test-end-to-end-jit-clear-tensor test-end-to-end-jit-encrypted-tensor test-end-to-end-jit-fhelinalg

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

file-check:
	cmake --build $(BUILD_DIR) --target FileCheck
not:
	cmake --build $(BUILD_DIR) --target not

# Python packages

define build_image_and_copy_wheels
	docker image build -t concretefhe-compiler-manylinux:$(1) --build-arg python_tag=$(1) -f ../builders/Dockerfile.release_manylinux_2_24_x86_64 ..
	docker container run --rm -v ${PWD}/../wheels:/wheels_volume concretefhe-compiler-manylinux:$(1) cp -r /wheels/. /wheels_volume/.
endef

package_py38: update_python_version
	$(call build_image_and_copy_wheels,cp38-cp38)

package_py39: update_python_version
	$(call build_image_and_copy_wheels,cp39-cp39)

package_py310: update_python_version
	$(call build_image_and_copy_wheels,cp310-cp310)

release_tarballs:
	docker image build -t concretefhe-compiler-manylinux:linux_x86_64_tarball -f ../builders/Dockerfile.release_tarball_linux_x86_64 ..
	docker container run --rm -v ${PWD}/../tarballs:/tarballs_volume concretefhe-compiler-manylinux:linux_x86_64_tarball cp -r /tarballs/. /tarballs_volume/.

update_python_version:
	echo "__version__ = \"`git describe --tags --abbrev=0 | grep -e '[0-9].*' -o`\"" > lib/Bindings/Python/concrete/version.py

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
	add-deps \
	file-check \
	not \
	package_py38 \
	package_py39 \
	package_py310 \
	release_tarballs \
	update_python_version
