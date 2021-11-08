import pytest
import numpy as np
from zamalang import CompilerEngine


@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        pytest.param(
            """
            func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
                %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
                return %1: !HLFHE.eint<7>
            }
            """,
            (5, 7),
            12,
            id="add_eint_int",
        ),
        pytest.param(
            """
            func @main(%arg0: tensor<4x!HLFHE.eint<7>>, %arg1: tensor<4xi8>) -> !HLFHE.eint<7>
            {
                %ret = "HLFHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!HLFHE.eint<7>>, tensor<4xi8>) -> !HLFHE.eint<7>
                return %ret : !HLFHE.eint<7>
            }
            """,
            (
                np.array([1, 2, 3, 4], dtype=np.uint8),
                np.array([4, 3, 2, 1], dtype=np.uint8),
            ),
            20,
            id="dot_eint_int",
        ),
        pytest.param(
            """
            func @main(%a0: tensor<4x!HLFHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>> {
                %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<6>>, tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>>
                return %res : tensor<4x!HLFHE.eint<6>>
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
            func @main(%a0: tensor<4x4x!HLFHE.eint<6>>, %a1: tensor<4x4xi7>) -> tensor<4x4x!HLFHE.eint<6>> {
                %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x4x!HLFHE.eint<6>>, tensor<4x4xi7>) -> tensor<4x4x!HLFHE.eint<6>>
                return %res : tensor<4x4x!HLFHE.eint<6>>
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
            func @main(%a0: tensor<2x2x2x!HLFHE.eint<6>>, %a1: tensor<2x2x2xi7>) -> tensor<2x2x2x!HLFHE.eint<6>> {
                %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x2x!HLFHE.eint<6>>, tensor<2x2x2xi7>) -> tensor<2x2x2x!HLFHE.eint<6>>
                return %res : tensor<2x2x2x!HLFHE.eint<6>>
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
    ],
)
def test_compile_and_run(mlir_input, args, expected_result):
    engine = CompilerEngine()
    engine.compile_fhe(mlir_input)
    if isinstance(expected_result, int):
        assert engine.run(*args) == expected_result
    else:
        # numpy array
        assert np.all(engine.run(*args) == expected_result)


@pytest.mark.parametrize(
    "mlir_input, args",
    [
        pytest.param(
            """
            func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
                %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
                return %1: !HLFHE.eint<7>
            }
            """,
            (5, 7, 8),
            id="add_eint_int_invalid_arg_number",
        ),
    ],
)
def test_compile_and_run_invalid_arg_number(mlir_input, args):
    engine = CompilerEngine()
    engine.compile_fhe(mlir_input)
    with pytest.raises(ValueError, match=r"wrong number of arguments"):
        engine.run(*args)


@pytest.mark.parametrize(
    "mlir_input, args, expected_result, tab_size",
    [
        pytest.param(
            """
            func @main(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
                %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<7>, tensor<128xi64>) -> (!HLFHE.eint<7>)
                return %1: !HLFHE.eint<7>
            }
            """,
            (73,),
            73,
            128,
            id="apply_lookup_table",
        ),
    ],
)
def test_compile_and_run_tlu(mlir_input, args, expected_result, tab_size):
    engine = CompilerEngine()
    engine.compile_fhe(mlir_input)
    assert abs(engine.run(*args) - expected_result) / tab_size < 0.1


@pytest.mark.parametrize(
    "mlir_input",
    [
        pytest.param(
            """
            func @test(%arg0: tensor<4x!HLFHE.eint<7>>, %arg1: tensor<4xi8>) -> !HLFHE.eint<7>
            {
                %ret = "HLFHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!HLFHE.eint<7>>, tensor<4xi8>) -> !HLFHE.eint<7>
                return %ret : !HLFHE.eint<7>
            }
            """,
            id="not @main",
        ),
    ],
)
def test_compile_invalid(mlir_input):
    engine = CompilerEngine()
    with pytest.raises(RuntimeError, match=r"Compilation failed:"):
        engine.compile_fhe(mlir_input)
