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
