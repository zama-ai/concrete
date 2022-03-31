import os
import tempfile

import pytest
import numpy as np

from concrete.compiler import ClientSupport, CompilationOptions, JITSupport, KeySetCache

KEY_SET_CACHE_PATH = os.path.join(tempfile.gettempdir(), "KeySetCache")

keyset_cache = KeySetCache.new(KEY_SET_CACHE_PATH)


def compile_and_run(engine, mlir_input, args, expected_result):
    options = CompilationOptions.new("main")
    options.set_auto_parallelize(True)
    compilation_result = engine.compile(mlir_input, options)
    # Client
    client_parameters = engine.load_client_parameters(compilation_result)
    key_set = ClientSupport.key_set(client_parameters, keyset_cache)
    public_arguments = ClientSupport.encrypt_arguments(client_parameters, key_set, args)
    # Server
    server_lambda = engine.load_server_lambda(compilation_result)
    public_result = engine.server_call(server_lambda, public_arguments)
    # Client
    result = ClientSupport.decrypt_result(key_set, public_result)
    # Check result
    assert type(expected_result) == type(result)
    if isinstance(expected_result, int):
        assert result == expected_result
    else:
        assert np.all(result == expected_result)


@pytest.mark.parallel
@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        pytest.param(
            """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
            (5, 7),
            12,
            id="add_eint_int",
        ),
        pytest.param(
            """
            func @main(%arg0: tensor<4x!FHE.eint<7>>, %arg1: tensor<4xi8>) -> !FHE.eint<7>
            {
                %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!FHE.eint<7>>, tensor<4xi8>) -> !FHE.eint<7>
                return %ret : !FHE.eint<7>
            }
            """,
            (
                np.array([1, 2, 3, 4], dtype=np.uint8),
                np.array([4, 3, 2, 1], dtype=np.uint8),
            ),
            20,
            id="dot_eint_int_uint8",
        ),
    ],
)
def test_compile_and_run_parallel(mlir_input, args, expected_result):
    engine = JITSupport.new()
    compile_and_run(engine, mlir_input, args, expected_result)
