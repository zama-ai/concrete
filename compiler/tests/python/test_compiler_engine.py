import os
import tempfile

import pytest
import numpy as np
from concrete.compiler import JITSupport, LibrarySupport, ClientSupport, KeySetCache

KEY_SET_CACHE_PATH = os.path.join(tempfile.gettempdir(), "KeySetCache")

keyset_cache = KeySetCache.new(KEY_SET_CACHE_PATH)


def compile_and_run(engine, mlir_input, args, expected_result):
    compilation_result = engine.compile(mlir_input)
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


end_to_end_fixture = [
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
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.array(4, dtype=np.uint8), np.array(5, dtype=np.uint8)),
        9,
        id="add_eint_int_with_ndarray_as_scalar",
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.uint8(3), np.uint8(5)),
        8,
        id="add_eint_int_with_np_uint8_as_scalar",
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.uint16(3), np.uint16(5)),
        8,
        id="add_eint_int_with_np_uint16_as_scalar",
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.uint32(3), np.uint32(5)),
        8,
        id="add_eint_int_with_np_uint32_as_scalar",
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.uint64(3), np.uint64(5)),
        8,
        id="add_eint_int_with_np_uint64_as_scalar",
    ),
    pytest.param(
        """
            func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
                %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (73,),
        73,
        id="apply_lookup_table",
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
            np.array([1, 2, 3, 4], dtype=np.uint16),
            np.array([4, 3, 2, 1], dtype=np.uint16),
        ),
        20,
        id="dot_eint_int_uint16",
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
            np.array([1, 2, 3, 4], dtype=np.uint32),
            np.array([4, 3, 2, 1], dtype=np.uint32),
        ),
        20,
        id="dot_eint_int_uint32",
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
            np.array([1, 2, 3, 4], dtype=np.uint64),
            np.array([4, 3, 2, 1], dtype=np.uint64),
        ),
        20,
        id="dot_eint_int_uint64",
    ),
    pytest.param(
        """
            func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
                %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
                return %res : tensor<4x!FHE.eint<6>>
            }
            """,
        (
            np.array([31, 6, 12, 9], dtype=np.uint8),
            np.array([32, 9, 2, 3], dtype=np.uint8),
        ),
        np.array([63, 15, 14, 12]),
        id="add_eint_int_1D",
    ),
    pytest.param(
        """
            func @main(%a0: tensor<4x4x!FHE.eint<6>>, %a1: tensor<4x4xi7>) -> tensor<4x4x!FHE.eint<6>> {
                %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x4x!FHE.eint<6>>, tensor<4x4xi7>) -> tensor<4x4x!FHE.eint<6>>
                return %res : tensor<4x4x!FHE.eint<6>>
            }
            """,
        (
            np.array(
                [[31, 6, 12, 9], [31, 6, 12, 9], [31, 6, 12, 9], [31, 6, 12, 9]],
                dtype=np.uint8,
            ),
            np.array(
                [[32, 9, 2, 3], [32, 9, 2, 3], [32, 9, 2, 3], [32, 9, 2, 3]],
                dtype=np.uint8,
            ),
        ),
        np.array(
            [
                [63, 15, 14, 12],
                [63, 15, 14, 12],
                [63, 15, 14, 12],
                [63, 15, 14, 12],
            ],
            dtype=np.uint8,
        ),
        id="add_eint_int_2D",
    ),
    pytest.param(
        """
            func @main(%a0: tensor<2x2x2x!FHE.eint<6>>, %a1: tensor<2x2x2xi7>) -> tensor<2x2x2x!FHE.eint<6>> {
                %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x2x!FHE.eint<6>>, tensor<2x2x2xi7>) -> tensor<2x2x2x!FHE.eint<6>>
                return %res : tensor<2x2x2x!FHE.eint<6>>
            }
            """,
        (
            np.array(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                dtype=np.uint8,
            ),
            np.array(
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
                dtype=np.uint8,
            ),
        ),
        np.array(
            [[[10, 12], [14, 16]], [[18, 20], [22, 24]]],
            dtype=np.uint8,
        ),
        id="add_eint_int_3D",
    ),
]


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_jit_compile_and_run(mlir_input, args, expected_result):
    engine = JITSupport.new()
    compile_and_run(engine, mlir_input, args, expected_result)


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_and_run(mlir_input, args, expected_result):
    engine = LibrarySupport.new("./py_test_lib_compile_and_run")
    compile_and_run(engine, mlir_input, args, expected_result)


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_reload_and_run(mlir_input, args, expected_result):
    engine = LibrarySupport.new("./test_lib_compile_reload_and_run")
    # Here don't save compilation result, reload
    engine.compile(mlir_input)
    compilation_result = engine.reload()
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


@pytest.mark.parametrize(
    "mlir_input, args",
    [
        pytest.param(
            """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
            (5, 7, 8),
            id="add_eint_int_invalid_arg_number",
        ),
    ],
)
def test_compile_and_run_invalid_arg_number(mlir_input, args):
    engine = JITSupport.new()
    with pytest.raises(
        RuntimeError, match=r"function has arity 2 but is applied to too many arguments"
    ):
        compile_and_run(engine, mlir_input, args, None)


@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        pytest.param(
            """
            func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
                %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
            (73,),
            73,
            id="apply_lookup_table",
        ),
    ],
)
def test_compile_and_run_tlu(mlir_input, args, expected_result):
    engine = JITSupport.new()
    compile_and_run(engine, mlir_input, args, expected_result)


@pytest.mark.parametrize(
    "mlir_input",
    [
        pytest.param(
            """
            func @test(%arg0: tensor<4x!FHE.eint<7>>, %arg1: tensor<4xi8>) -> !FHE.eint<7>
            {
                %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!FHE.eint<7>>, tensor<4xi8>) -> !FHE.eint<7>
                return %ret : !FHE.eint<7>
            }
            """,
            id="not @main",
        ),
    ],
)
def test_compile_invalid(mlir_input):
    engine = JITSupport.new()
    with pytest.raises(
        RuntimeError, match=r"cannot find the function for generate client parameters"
    ):
        engine.compile(mlir_input)
