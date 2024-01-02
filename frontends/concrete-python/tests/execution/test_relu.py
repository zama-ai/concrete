"""
Tests of execution of round bit pattern operation.
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
        (-1, 0),
        (10, 10),
        (-10, 0),
    ],
)
def test_plain_relu(sample, expected_output):
    """
    Test plain evaluation of relu.
    """
    assert fhe.relu(sample) == expected_output


operations = [
    lambda x: fhe.relu(x),
    lambda x: fhe.relu(x) + 100,
]
cases = [
    # fhe.relu(int1), should result in fhe.zero()
    [
        operation,
        1,
        True,
        (),
        0,
        2,
    ]
    for operation in operations
] + [
    # fhe.relu should use an optimized TLU when it's assigned bit-width is bigger than the original
    [
        lambda x: fhe.relu(x) + (x + 10),
        3,
        True,
        (),
        10,
        2,
    ]
]

with_tlu = set()
for function in operations:
    for bit_width in [1, 2, 3, 4, 5, 8, 12, 16]:
        for is_signed in [False, True]:
            for shape in [(), (3,), (2, 3)]:
                for threshold in [5, 7]:
                    for chunk_size in [2, 3]:
                        if bit_width < threshold:
                            key = (bit_width, is_signed)
                            if key in with_tlu:
                                continue
                            with_tlu.add(key)

                        cases += [
                            [
                                function,
                                bit_width,
                                is_signed,
                                shape,
                                threshold,
                                chunk_size,
                            ]
                        ]


@pytest.mark.parametrize(
    "function,bit_width,is_signed,shape,threshold,chunk_size",
    cases,
)
def test_relu(function, bit_width, is_signed, shape, threshold, chunk_size, helpers):
    """
    Test encrypted evaluation of relu.
    """

    dtype = Integer(is_signed, bit_width)

    inputset = [np.random.randint(dtype.min(), dtype.max() + 1, size=shape) for _ in range(100)]
    configuration = helpers.configuration().fork(
        relu_on_bits_threshold=threshold,
        relu_on_bits_chunk_size=chunk_size,
    )

    compiler = fhe.Compiler(function, {"x": "encrypted"})
    circuit = compiler.compile(inputset, configuration)

    for value in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, value, retries=3)
