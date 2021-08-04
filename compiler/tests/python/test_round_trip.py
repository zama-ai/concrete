import pytest
from zamalang import compiler


VALID_INPUTS = [
    """
    func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
        %0 = constant 1 : i3
        %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i3) -> (!HLFHE.eint<2>)
        return %1: !HLFHE.eint<2>
    }
    """,
    """
    func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: memref<4xi2>) -> !HLFHE.eint<2> {
        %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, memref<4xi2>) -> (!HLFHE.eint<2>)
        return %1: !HLFHE.eint<2>
    }
    """,
    """
    func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<2>>,
        %arg1: memref<2xi3>,
        %arg2: memref<!HLFHE.eint<2>>)
    {
        "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
            (memref<2x!HLFHE.eint<2>>, memref<2xi3>, memref<!HLFHE.eint<2>>) -> ()
        return
    }

    """,
]

INVALID_INPUTS = [
    "nothing really mlir",
    """
    func @test(%arg0: !HLFHE.eint<0>) {
        return
    }
    """,
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
