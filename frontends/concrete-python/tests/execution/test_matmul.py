"""
Tests of execution of matmul operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,bounds",
    [
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 3),
        ),
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 127),
        ),
        pytest.param(
            (1, 2),
            (2, 1),
            (0, 3),
        ),
        pytest.param(
            (3, 3),
            (3, 3),
            (0, 3),
        ),
        pytest.param(
            (2, 1),
            (1, 2),
            (0, 7),
        ),
        pytest.param(
            (2,),
            (2,),
            (0, 7),
        ),
        pytest.param(
            (5, 5),
            (5,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (5, 5),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (5, 5),
            (-127, 127),
        ),
        pytest.param(
            (5,),
            (5, 3),
            (0, 3),
        ),
        pytest.param(
            (5, 3),
            (3,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (4, 5, 3),
            (-5, 5),
        ),
        pytest.param(
            (4, 5, 3),
            (3,),
            (0, 5),
        ),
        pytest.param(
            (5,),
            (2, 4, 5, 3),
            (0, 5),
        ),
        pytest.param(
            (2, 4, 5, 3),
            (3,),
            (-1, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (2, 1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
    ],
)
def test_matmul(lhs_shape, rhs_shape, bounds, helpers):
    """
    Test matmul.
    """

    configuration = helpers.configuration()

    minimum, maximum = bounds

    lhs_cst = list(np.random.randint(minimum, maximum, size=lhs_shape))
    rhs_cst = list(np.random.randint(minimum, maximum, size=rhs_shape))

    @fhe.compiler({"x": "encrypted"})
    def lhs_operator(x):
        return x @ rhs_cst

    @fhe.compiler({"x": "encrypted"})
    def rhs_operator(x):
        return lhs_cst @ x

    @fhe.compiler({"x": "encrypted"})
    def lhs_function(x):
        return np.matmul(x, rhs_cst)

    @fhe.compiler({"x": "encrypted"})
    def rhs_function(x):
        return np.matmul(lhs_cst, x)

    lhs_inputset = [np.random.randint(minimum, maximum, size=lhs_shape) for i in range(100)]
    rhs_inputset = [np.random.randint(minimum, maximum, size=rhs_shape) for i in range(100)]

    lhs_operator_circuit = lhs_operator.compile(lhs_inputset, configuration)
    rhs_operator_circuit = rhs_operator.compile(rhs_inputset, configuration)
    lhs_function_circuit = lhs_function.compile(lhs_inputset, configuration)
    rhs_function_circuit = rhs_function.compile(rhs_inputset, configuration)

    lhs_sample = np.random.randint(minimum, maximum, size=lhs_shape)
    rhs_sample = np.random.randint(minimum, maximum, size=rhs_shape)

    helpers.check_execution(lhs_operator_circuit, lhs_operator, lhs_sample)
    helpers.check_execution(rhs_operator_circuit, rhs_operator, rhs_sample)
    helpers.check_execution(lhs_function_circuit, lhs_function, lhs_sample)
    helpers.check_execution(rhs_function_circuit, rhs_function, rhs_sample)


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,bounds",
    [
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 3),
        ),
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 127),
        ),
        pytest.param(
            (1, 2),
            (2, 1),
            (0, 3),
        ),
        pytest.param(
            (3, 3),
            (3, 3),
            (0, 3),
        ),
        pytest.param(
            (2, 1),
            (1, 2),
            (0, 7),
        ),
        pytest.param(
            (2,),
            (2,),
            (0, 7),
        ),
        pytest.param(
            (5, 5),
            (5,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (5, 5),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (5, 5),
            (-63, 63),
        ),
        pytest.param(
            (2,),
            (2, 7),
            (-63, 0),
        ),
        pytest.param(
            (5,),
            (5, 3),
            (0, 3),
        ),
        pytest.param(
            (5, 3),
            (3,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (4, 5, 3),
            (-5, 5),
        ),
        pytest.param(
            (4, 5, 3),
            (3,),
            (0, 5),
        ),
        pytest.param(
            (5,),
            (2, 4, 5, 3),
            (0, 5),
        ),
        pytest.param(
            (2, 4, 5, 3),
            (3,),
            (-1, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (2, 1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
    ],
)
def test_matmul_enc_enc_and_clear(lhs_shape, rhs_shape, bounds, helpers):
    """
    Test matmul.
    """

    configuration = helpers.configuration()

    minimum, maximum = bounds

    # Matmul of clear values and encrypted matrices
    @fhe.compiler({"x": "encrypted", "y": "clear"})
    def lhs_operator_clear(x, y):
        return x @ y

    # Matmul of two encrypted matrices
    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def enc_function_xy(x, y):
        return np.matmul(x, y)

    # Put all the dual operand functions in a list

    # FIXME: add lhs_operator_clear to this list to
    # re-enable the testing with clear values
    dual_operand_functions = [enc_function_xy]

    # Compile each dual operand function and test it on random data
    for func in dual_operand_functions:
        dual_op_inputset = [
            (
                np.random.randint(minimum, maximum, size=lhs_shape),
                np.random.randint(minimum, maximum, size=rhs_shape),
            )
            for i in range(100)
        ]
        dual_op_circuit = func.compile(dual_op_inputset, configuration)

        lhs_sample, rhs_sample = np.random.randint(
            minimum, maximum, size=lhs_shape
        ), np.random.randint(minimum, maximum, size=rhs_shape)

        helpers.check_execution(dual_op_circuit, func, [lhs_sample, rhs_sample], retries=3)


@pytest.mark.parametrize("bitwidth", [4, 10])
@pytest.mark.parametrize("signed", [True, False])
def test_matmul_zero(bitwidth, signed, helpers):
    """
    Test matmul.
    """

    lhs_shape = (2, 1)
    rhs_shape = (1, 2)
    range_lhs = (-(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1) if signed else (0, 2**bitwidth)

    configuration = helpers.configuration()

    # Matmul of two encrypted matrices
    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def enc_function_xy(x, y):
        return x * y

    dual_op_inputset = [
        (
            np.random.randint(range_lhs[0], range_lhs[1], size=lhs_shape),
            np.zeros(rhs_shape, dtype=np.int64),
        )
        for i in range(100)
    ]
    dual_op_circuit = enc_function_xy.compile(dual_op_inputset, configuration)

    lhs_sample, rhs_sample = np.random.randint(
        range_lhs[0], range_lhs[1], size=lhs_shape
    ), np.zeros(rhs_shape, dtype=np.int64)

    helpers.check_execution(dual_op_circuit, enc_function_xy, [lhs_sample, rhs_sample], retries=3)
