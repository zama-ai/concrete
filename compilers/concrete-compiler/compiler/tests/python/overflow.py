import sys
import shutil
import numpy as np
import json
from concrete.compiler import Compiler, lookup_runtime_lib
from test_simulation import compile_run_assert


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: overflow.py mlir_file.mlir input_int... expected_out_int")
        exit()

    with open(sys.argv[1], "r") as f:
        mlir_input = f.read()

    artifact_dir = "./py_test_lib_compile_and_run"
    engine = Compiler(
        artifact_dir,
        lookup_runtime_lib(),
    )
    args_and_res = json.loads(sys.argv[2])
    args = args_and_res[0]
    expected_results = args_and_res[1]
    compile_run_assert(engine, mlir_input, args, expected_results)
    shutil.rmtree(artifact_dir)
