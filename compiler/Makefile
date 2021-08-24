build:
	cmake -B build . \
	-DLLVM_DIR=${LLVM_PROJECT}/build/lib/cmake/llvm \
	-DMLIR_DIR=${LLVM_PROJECT}/build/lib/cmake/mlir \
	-DCONCRETE_FFI_RELEASE=${CONCRETE_PROJECT}/target/release

zamacompiler:
	make -C build/ zamacompiler

test-check: zamacompiler
	${LLVM_PROJECT}/build/bin/llvm-lit -v tests/

test-end-to-end-jit:
	make -C build/ end_to_end_jit_test
	./build/bin/end_to_end_jit_test

test: test-check test-end-to-end-jit

test_python:
	pytest -v tests/python