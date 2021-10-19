BUILD_DIR=./build
Python3_EXECUTABLE=


$(BUILD_DIR)/configured.stamp:
	cmake -B $(BUILD_DIR) -GNinja ../llvm-project/llvm/ \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=OFF \
	-DLLVM_TARGETS_TO_BUILD="host" \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DZAMALANG_BINDINGS_PYTHON_ENABLED=ON \
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release \
	-DLLVM_EXTERNAL_PROJECTS=zamalang \
	-DLLVM_EXTERNAL_ZAMALANG_SOURCE_DIR=. \
	-DPython3_EXECUTABLE=${Python3_EXECUTABLE}
	touch $@

build-initialized: $(BUILD_DIR)/configured.stamp

build-end-to-end-jit: build-initialized
	cmake --build $(BUILD_DIR) --target end_to_end_jit_test

zamacompiler: build-initialized
	cmake --build $(BUILD_DIR) --target zamacompiler

python-bindings: build-initialized
	cmake --build $(BUILD_DIR) --target ZamalangMLIRPythonModules ZamalangPythonModules

test-check: zamacompiler file-check not
	$(BUILD_DIR)/bin/llvm-lit -v tests/

test-end-to-end-jit: build-end-to-end-jit
	$(BUILD_DIR)/bin/end_to_end_jit_test

test-python: python-bindings
	PYTHONPATH=${PYTHONPATH}:$(BUILD_DIR)/tools/zamalang/python_packages/zamalang_core LD_PRELOAD=$(BUILD_DIR)/lib/libZamalangRuntime.so pytest -vs tests/python

test: test-check test-end-to-end-jit test-python

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

package_py38:
	$(call build_image_and_copy_wheels,cp38-cp38)

package_py39:
	$(call build_image_and_copy_wheels,cp39-cp39)

package_py310:
	$(call build_image_and_copy_wheels,cp310-cp310)
