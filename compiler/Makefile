build:
	cmake -B build -GNinja ${LLVM_PROJECT}/llvm/ \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=OFF \
	-DLLVM_TARGETS_TO_BUILD="host" \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DZAMALANG_BINDINGS_PYTHON_ENABLED=ON \
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release \
	-DLLVM_EXTERNAL_PROJECTS=zamalang \
	-DLLVM_EXTERNAL_ZAMALANG_SOURCE_DIR=.

build-end-to-end-jit: build
	cmake --build build --target end_to_end_jit_test

zamacompiler: build
	cmake --build build --target zamacompiler

python-bindings: build
	cmake --build build --target ZamalangMLIRPythonModules ZamalangPythonModules

test-check: zamacompiler file-check not
	./build/bin/llvm-lit -v tests/

test-end-to-end-jit: build-end-to-end-jit
	./build/bin/end_to_end_jit_test

test: test-check test-end-to-end-jit

test-python: python-bindings
	PYTHONPATH=${PYTHONPATH}:./build/tools/zamalang/python_packages/zamalang_core:./build/tools/zamalang/python_packages/zamalang_core/mlir/_mlir_libs/ LD_PRELOAD=./build/lib/libZamalangRuntime.so pytest -vs tests/python

# LLVM/MLIR dependencies

all-deps: file-check not

file-check:
	cmake --build build/ --target FileCheck
not:
	cmake --build build/ --target not
