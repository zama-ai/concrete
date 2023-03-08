"""
Tests of execution of matmul operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,bounds",
    [
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 3),
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
            (0, 5),
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
            (0, 5),
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

    @cnp.compiler({"x": "encrypted"})
    def lhs_operator(x):
        return x @ rhs_cst

    @cnp.compiler({"x": "encrypted"})
    def rhs_operator(x):
        return lhs_cst @ x

    @cnp.compiler({"x": "encrypted"})
    def lhs_function(x):
        return np.matmul(x, rhs_cst)

    @cnp.compiler({"x": "encrypted"})
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
