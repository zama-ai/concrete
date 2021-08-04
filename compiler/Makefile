build:
	cmake -B build . -DLLVM_DIR=${LLVM_PROJECT}/build/lib/cmake/llvm    -DMLIR_DIR=${LLVM_PROJECT}/build/lib/cmake/mlir

zamacompiler:
	make -C build/ zamacompiler

test:
	${LLVM_PROJECT}/build/bin/llvm-lit -v tests/

test_python:
	pytest -v tests/python