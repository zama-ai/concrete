import pytest
from zamalang import compiler


VALID_INPUTS = [
    pytest.param(
        """
        func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
            %0 = constant 1 : i3
            %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i3) -> (!HLFHE.eint<2>)
            return %1: !HLFHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<4xi2>) -> !HLFHE.eint<2> {
            %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<4xi2>) -> (!HLFHE.eint<2>)
            return %1: !HLFHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>,
            %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
        {
            %1 = "HLFHE.dot_eint_int"(%arg0, %arg1) :
                (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>
            return %1 : !HLFHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func @main(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
            %tlu = std.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
            %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<7>, tensor<128xi64>) -> (!HLFHE.eint<7>)
            return %1: !HLFHE.eint<7>
        }
        """,
        id="add_eint_int_cst",
    ),
]

INVALID_INPUTS = [
    pytest.param("nothing really mlir", id="english sentence"),
    pytest.param(
        """
        func @test(%arg0: !HLFHE.eint<0>) {
            return
        }
        """,
        id="eint<0>",
    ),
]


@pytest.mark.parametrize("mlir_input", VALID_INPUTS)
def test_valid_mlir_inputs(mlir_input):
    # no need to check that it's correctly parsed, as we already have test for this
    # we just wanna make sure it doesn't raise an error for valid inputs
    compiler.round_trip(mlir_input)


@pytest.mark.parametrize("mlir_input", INVALID_INPUTS)
def test_invalid_mlir_inputs(mlir_input):
    # We need to check that invalud inputs are raising an error
    with pytest.raises(RuntimeError, match=r"mlir parsing failed"):
        compiler.round_trip(mlir_input)
