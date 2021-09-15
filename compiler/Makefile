build:
	cmake -B build -GNinja ${LLVM_PROJECT}/llvm/ \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=OFF \
	-DLLVM_TARGETS_TO_BUILD="host" \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DLLVM_DIR=${LLVM_PROJECT}/build/lib/cmake/llvm \
	-DMLIR_DIR=${LLVM_PROJECT}/build/lib/cmake/mlir \
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release \
	-DLLVM_EXTERNAL_PROJECTS=zamalang \
	-DLLVM_EXTERNAL_ZAMALANG_SOURCE_DIR=.

build-end-to-end-jit: build
	cmake --build build --target end_to_end_jit_test

zamacompiler: build
	cmake --build build --target zamacompiler

python-bindings: build
	cmake --build build --target ZamalangBindingsPython

test-check: zamacompiler
	${LLVM_PROJECT}/build/bin/llvm-lit -v tests/

test-end-to-end-jit: build-end-to-end-jit
	./build/bin/end_to_end_jit_test

test: test-check test-end-to-end-jit

test-python: python-bindings
	PYTHONPATH=${PYTHONPATH}:./build/tools/zamalang/python:./build/python LD_PRELOAD=./build/lib/libZamalangRuntime.so pytest -v tests/python