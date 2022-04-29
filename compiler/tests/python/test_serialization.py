import pytest
import shutil
import numpy as np
from concrete.compiler import (
    JITSupport,
    LibrarySupport,
    ClientSupport,
    CompilationOptions,
    PublicArguments,
)
from concrete.compiler.client_parameters import ClientParameters
from concrete.compiler.public_result import PublicResult


def assert_result(result, expected_result):
    """Assert that result and expected result are equal.

    result and expected_result can be integers on numpy arrays.
    """
    assert type(expected_result) == type(result)
    if isinstance(expected_result, int):
        assert result == expected_result
    else:
        assert np.all(result == expected_result)


def run_with_serialization(
    engine,
    args,
    compilation_result,
    keyset_cache,
):
    """Execute engine on the given arguments. Performs serialization betwee client/server.

    Perform required loading, encryption, execution, and decryption."""
    # Client
    client_parameters = engine.load_client_parameters(compilation_result)
    serialized_client_parameters = client_parameters.serialize()
    client_parameters = ClientParameters.unserialize(serialized_client_parameters)
    key_set = ClientSupport.key_set(client_parameters, keyset_cache)
    public_arguments = ClientSupport.encrypt_arguments(client_parameters, key_set, args)
    public_arguments_buffer = public_arguments.serialize()
    # Server
    public_arguments = PublicArguments.unserialize(
        client_parameters, public_arguments_buffer
    )
    del public_arguments_buffer
    server_lambda = engine.load_server_lambda(compilation_result)
    public_result = engine.server_call(server_lambda, public_arguments)
    public_result_buffer = public_result.serialize()
    # Client
    public_result = PublicResult.unserialize(client_parameters, public_result_buffer)
    del public_result_buffer
    result = ClientSupport.decrypt_result(key_set, public_result)
    return result


def compile_run_assert_with_serialization(
    engine,
    mlir_input,
    args,
    expected_result,
    keyset_cache,
):
    """Compile run and assert result. Performs serialization betwee client/server.

    Can take both JITSupport or LibrarySupport as engine.
    """
    options = CompilationOptions.new("main")
    compilation_result = engine.compile(mlir_input, options)
    result = run_with_serialization(engine, args, compilation_result, keyset_cache)
    assert_result(result, expected_result)


end_to_end_fixture = [
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<5>, %arg1: i6) -> !FHE.eint<5> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<5>, i6) -> (!FHE.eint<5>)
                return %1: !FHE.eint<5>
            }
            """,
        (5, 7),
        12,
        id="enc_plain_int_args",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<5>)
                return %1: !FHE.eint<5>
            }
            """,
        (5, 7),
        12,
        id="enc_enc_int_args",
    ),
    pytest.param(
        """
            func @main(%arg0: tensor<4x!FHE.eint<5>>, %arg1: tensor<4xi6>) -> !FHE.eint<5>
            {
                %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!FHE.eint<5>>, tensor<4xi6>) -> !FHE.eint<5>
                return %ret : !FHE.eint<5>
            }
            """,
        (
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([4, 3, 2, 1], dtype=np.uint8),
        ),
        20,
        id="enc_plain_ndarray_args",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        """
            func @main(%a0: tensor<4x!FHE.eint<5>>, %a1: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>> {
                %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<5>>, tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
                return %res : tensor<4x!FHE.eint<5>>
            }
            """,
        (
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([7, 0, 1, 5], dtype=np.uint8),
        ),
        np.array([8, 2, 4, 9]),
        id="enc_enc_ndarray_args",
    ),
]


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_jit_compile_and_run_with_serialization(
    mlir_input, args, expected_result, keyset_cache
):
    engine = JITSupport.new()
    compile_run_assert_with_serialization(
        engine, mlir_input, args, expected_result, keyset_cache
    )


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_and_run_with_serialization(
    mlir_input, args, expected_result, keyset_cache
):
    artifact_dir = "./py_test_lib_compile_and_run"
    engine = LibrarySupport.new(artifact_dir)
    compile_run_assert_with_serialization(
        engine, mlir_input, args, expected_result, keyset_cache
    )
    shutil.rmtree(artifact_dir)
