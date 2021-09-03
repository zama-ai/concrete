import pytest
from zamalang import CompilerEngine


@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        (
            """
            func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
                %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
                return %1: !HLFHE.eint<7>
            }
            """,
            (5, 7),
            12,
        ),
    ],
)
def test_compile_and_run(mlir_input, args, expected_result):
    engine = CompilerEngine()
    engine.compile_fhe(mlir_input)
    assert engine.run(*args) == expected_result


@pytest.mark.parametrize(
    "mlir_input, args, expected_result, tab_size",
    [
        (
            """
            func @main(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
                // 0..128 shifted << 55
                %tlu = std.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<7>, tensor<128xi64>) -> (!HLFHE.eint<7>)
                return %1: !HLFHE.eint<7>
            }
            """,
            (73,),
            73,
            128,
        ),
    ],
)
def test_compile_and_run_tlu(mlir_input, args, expected_result, tab_size):
    engine = CompilerEngine()
    engine.compile_fhe(mlir_input)
    assert abs(engine.run(*args) - expected_result) / tab_size < 0.1
