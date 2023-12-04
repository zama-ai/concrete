"""
Tests of execution of multivariate extension.
"""

import inspect

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer
from concrete.fhe.values import ValueDescription


def x_if_y_else_zero(x, y):
    """
    Multivariate function for keeping the value of x
    or setting it to 0, depending on the value of y.
    """
    return np.where(y, x, 0)


def multi(x, y, z):
    """
    Multivariate function that results in multiple tables in the table lookup.
    """
    result = x * y
    result[0] -= z
    result[2] += z
    return result


cases = [
    [
        ("x_if_y_else_zero", lambda x, y: fhe.multivariate(x_if_y_else_zero)(x, y)),
        [
            ValueDescription(
                Integer(is_signed=False, bit_width=3),
                shape=(),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=1),
                shape=(),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.CASTED,
    ],
    [
        ("x_if_y_else_zero", lambda x, y: fhe.multivariate(x_if_y_else_zero)(x, y)),
        [
            ValueDescription(
                Integer(is_signed=False, bit_width=3),
                shape=(2,),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=1),
                shape=(),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.PROMOTED,
    ],
    [
        ("x_if_y_else_zero", lambda x, y: fhe.multivariate(x_if_y_else_zero)(x, y)),
        [
            ValueDescription(
                Integer(is_signed=True, bit_width=3),
                shape=(),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=1),
                shape=(2,),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.CASTED,
    ],
    [
        ("x_if_y_else_zero", lambda x, y: fhe.multivariate(x_if_y_else_zero)(x, y)),
        [
            ValueDescription(
                Integer(is_signed=True, bit_width=3),
                shape=(2,),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=1),
                shape=(2,),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.PROMOTED,
    ],
    [
        ("multi", lambda x, y, z: fhe.multivariate(multi)(x, y, z)),
        [
            ValueDescription(
                Integer(is_signed=False, bit_width=2),
                shape=(3,),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=True, bit_width=2),
                shape=(3,),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=2),
                shape=(),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.CASTED,
    ],
    [
        ("multi", lambda x, y, z: fhe.multivariate(multi)(x, y, z)),
        [
            ValueDescription(
                Integer(is_signed=True, bit_width=2),
                shape=(3, 2),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=False, bit_width=2),
                shape=(1, 2),
                is_encrypted=True,
            ),
            ValueDescription(
                Integer(is_signed=True, bit_width=2),
                shape=(2,),
                is_encrypted=True,
            ),
        ],
        fhe.MultivariateStrategy.PROMOTED,
    ],
]


@pytest.mark.parametrize("operation,values,strategy", cases)
def test_multivariate(operation, values, strategy, helpers):
    """
    Test multivariate extension.
    """

    name, function = operation

    print()
    print()
    print(
        f"{name}({', '.join(str(value) for value in values)})"
        + (f" {{{strategy}}}" if strategy is not None else "")
    )
    print()
    print()

    signature = inspect.signature(function)
    parameters = list(signature.parameters.keys())

    parameter_encryption_statuses = {parameter: "encrypted" for parameter in parameters}
    configuration = helpers.configuration()

    if strategy is not None:
        configuration = configuration.fork(multivariate_strategy_preference=strategy)

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = [
        tuple(
            np.random.randint(value.dtype.min(), value.dtype.max() + 1, size=value.shape)
            for value in values
        )
        for _ in range(100)
    ]
    circuit = compiler.compile(inputset, configuration)

    samples = [
        [np.zeros(value.shape, dtype=np.int64) for value in values],
        [np.ones(value.shape, dtype=np.int64) * value.dtype.max() for value in values],
        [
            np.random.randint(value.dtype.min(), value.dtype.max() + 1, size=value.shape)
            for value in values
        ],
    ]
    for sample in samples:
        helpers.check_execution(circuit, function, sample, retries=5)
