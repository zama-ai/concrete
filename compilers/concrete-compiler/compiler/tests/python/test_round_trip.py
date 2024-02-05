import platform
import pytest
from concrete import compiler


VALID_INPUTS = [
    pytest.param(
        """
        func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
            %0 = arith.constant 1 : i3
            %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
            return %1: !FHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
            %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
            return %1: !FHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func.func @dot_eint_int(%arg0: tensor<2x!FHE.eint<2>>,
            %arg1: tensor<2xi3>) -> !FHE.eint<2>
        {
            %1 = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                (tensor<2x!FHE.eint<2>>, tensor<2xi3>) -> !FHE.eint<2>
            return %1 : !FHE.eint<2>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
            %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
            %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
            return %1: !FHE.eint<7>
        }
        """,
        id="add_eint_int_cst",
    ),
    pytest.param(
        """
        func.func @main(%a0: tensor<4x!FHE.eint<2>>, %a1: tensor<4xi3>) -> tensor<4x!FHE.eint<2>> {
            %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
            return %1: tensor<4x!FHE.eint<2>>
        }
        """,
        id="add_eint_int_1D",
    ),
]

INVALID_INPUTS = [
    pytest.param("nothing really mlir", id="english sentence"),
    pytest.param(
        """
        func.func @test(%arg0: !FHE.eint<0>) {
            return
        }
        """,
        id="eint<0>",
    ),
    pytest.param(
        """
        func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>> {
            // expected-error @+1 {{'FHELinalg.add_eint_int' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
            %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>>
            return %1 : tensor<2x2x3x4x!FHE.eint<2>>
        }
        """,
        id="incompatible dimensions",
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
    with pytest.raises(RuntimeError, match=r"MLIR parsing failed:"):
        compiler.round_trip(mlir_input)
