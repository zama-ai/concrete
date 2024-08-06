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
def test_constant_matmul(lhs_shape, rhs_shape, bounds, helpers):
    """
    Test matmul where one of the operators is a constant.
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

    lhs_sample = lhs_inputset[-1]
    rhs_sample = rhs_inputset[-1]

    helpers.check_execution(lhs_operator_circuit, lhs_operator, lhs_sample)
    helpers.check_execution(rhs_operator_circuit, rhs_operator, rhs_sample)
    helpers.check_execution(lhs_function_circuit, lhs_function, lhs_sample)
    helpers.check_execution(rhs_function_circuit, rhs_function, rhs_sample)


test_matmul_shape_and_bounds = [
    (
        (3, 2),
        (2, 3),
        (0, 3),
    ),
    (
        (3, 2),
        (2, 3),
        (0, 127),
    ),
    (
        (1, 2),
        (2, 1),
        (0, 3),
    ),
    (
        (3, 3),
        (3, 3),
        (0, 3),
    ),
    (
        (2, 1),
        (1, 2),
        (0, 7),
    ),
    (
        (2,),
        (2,),
        (0, 7),
    ),
    (
        (5, 5),
        (5,),
        (0, 3),
    ),
    (
        (5,),
        (5, 5),
        (0, 3),
    ),
    (
        (5,),
        (5, 5),
        (-63, 63),
    ),
    (
        (2,),
        (2, 7),
        (-63, 0),
    ),
    (
        (5,),
        (5, 3),
        (0, 3),
    ),
    (
        (5, 3),
        (3,),
        (0, 3),
    ),
    (
        (5,),
        (4, 5, 3),
        (-5, 5),
    ),
    (
        (4, 5, 3),
        (3,),
        (0, 5),
    ),
    (
        (5,),
        (2, 4, 5, 3),
        (0, 5),
    ),
    (
        (2, 4, 5, 3),
        (3,),
        (-1, 5),
    ),
    (
        (5, 4, 3),
        (3, 2),
        (0, 5),
    ),
    (
        (4, 3),
        (5, 3, 2),
        (0, 5),
    ),
    (
        (2, 5, 4, 3),
        (3, 2),
        (0, 5),
    ),
    (
        (5, 4, 3),
        (1, 3, 2),
        (0, 5),
    ),
    (
        (1, 4, 3),
        (5, 3, 2),
        (0, 5),
    ),
    (
        (5, 4, 3),
        (2, 1, 3, 2),
        (0, 5),
    ),
    (
        (2, 1, 4, 3),
        (5, 3, 2),
        (0, 5),
    ),
]


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,bounds,clear_rhs",
    [
        (lhs_shape, rhs_shape, bounds, clear)
        for lhs_shape, rhs_shape, bounds in test_matmul_shape_and_bounds
        for clear in [False, True]
    ],
)
def test_matmul(lhs_shape, rhs_shape, bounds, clear_rhs, helpers):
    """
    Test matmul.
    """

    configuration = helpers.configuration()

    minimum, maximum = bounds

    @fhe.compiler({"x": "encrypted", "y": "clear"})
    def clear(x, y):
        return x @ y

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def encrypted(x, y):
        return np.matmul(x, y)

    implementation = clear if clear_rhs else encrypted

    inputset = [
        (
            np.random.randint(minimum, maximum, size=lhs_shape),
            np.random.randint(minimum, maximum, size=rhs_shape),
        )
        for _ in range(100)
    ]
    circuit = implementation.compile(inputset, configuration)

    sample = list(inputset[-1])

    helpers.check_execution(circuit, implementation, sample, retries=3)


@pytest.mark.parametrize("bit_width", [4, 10])
@pytest.mark.parametrize("signed", [True, False])
def test_zero_matmul(bit_width, signed, helpers):
    """
    Test matmul where one of the operators is all zeros.
    """

    configuration = helpers.configuration()

    lhs_shape = (2, 1)
    rhs_shape = (1, 2)

    bounds = (-(2 ** (bit_width - 1)), 2 ** (bit_width - 1) - 1) if signed else (0, 2**bit_width)
    minimum, maximum = bounds

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def function(x, y):
        return x * y

    inputset = [
        (
            np.random.randint(minimum, maximum, size=lhs_shape),
            np.zeros(rhs_shape, dtype=np.int64),
        )
        for i in range(100)
    ]
    circuit = function.compile(inputset, configuration)

    sample = [
        np.random.randint(minimum, maximum, size=lhs_shape),
        np.zeros(rhs_shape, dtype=np.int64),
    ]

    helpers.check_execution(circuit, function, sample, retries=3)
