import sys
import shutil
import numpy as np
from concrete.compiler import LibrarySupport
from test_simulation import compile_run_assert


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: overflow.py mlir_file.mlir input_int... expected_out_int")
        exit()

    with open(sys.argv[1], "r") as f:
        mlir_input = f.read()

    artifact_dir = "./py_test_lib_compile_and_run"
    engine = LibrarySupport.new(artifact_dir)
    args = list(map(int, sys.argv[2:-1]))
    expected_result = int(sys.argv[-1])
    args_and_shape = []
    for arg in args:
        if isinstance(arg, int):
            args_and_shape.append((arg, None))
        else:  # np.array
            args_and_shape.append((arg.flatten().tolist(), list(arg.shape)))
    compile_run_assert(engine, mlir_input, args_and_shape, expected_result)
    shutil.rmtree(artifact_dir)
