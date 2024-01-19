"""
Tests of execution of identity extension.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "sample,expected_output",
    [
        (0, 0),
        (1, 1),
        (-1, -1),
        (10, 10),
        (-10, -10),
    ],
)
def test_plain_identity(sample, expected_output):
    """
    Test plain evaluation of identity extension.
    """
    assert fhe.identity(sample) == expected_output


operations = [
    lambda x: fhe.identity(x),
    lambda x: fhe.identity(x) + 100,
]

cases = []
for function in operations:
    for bit_width in [1, 2, 3, 4, 5, 8, 12]:
        for is_signed in [False, True]:
            for shape in [(), (3,), (2, 3)]:
                cases += [
                    [
                        function,
                        bit_width,
                        is_signed,
                        shape,
                    ]
                ]


@pytest.mark.parametrize(
    "function,bit_width,is_signed,shape",
    cases,
)
def test_identity(function, bit_width, is_signed, shape, helpers):
    """
    Test encrypted evaluation of identity extension.
    """

    dtype = Integer(is_signed, bit_width)

    inputset = [np.random.randint(dtype.min(), dtype.max() + 1, size=shape) for _ in range(100)]
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, {"x": "encrypted"})
    circuit = compiler.compile(inputset, configuration)

    for value in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, value, retries=3)
