"""
Tests of execution of min and max operations.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer
from concrete.fhe.values import ValueDescription

cases = []
for operation in ["max", "min"]:
    for bit_width in range(1, 5):
        for is_signed in [False, True]:
            for shape in [(), (4,), (3, 3)]:
                for keepdims in [False, True]:
                    for strategy in [
                        fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
                        fhe.MinMaxStrategy.THREE_TLU_CASTED,
                        fhe.MinMaxStrategy.CHUNKED,
                    ]:
                        cases.append(
                            [
                                operation,
                                bit_width,
                                is_signed,
                                shape,
                                None,
                                keepdims,
                                strategy,
                            ],
                        )
                        for axis in range(len(shape)):
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    axis,
                                    keepdims,
                                    strategy,
                                ],
                            )
                        cases.append(
                            [
                                operation,
                                bit_width,
                                is_signed,
                                shape,
                                -1,
                                keepdims,
                                strategy,
                            ],
                        )
                        if len(shape) == 2:
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    (0, 1),
                                    keepdims,
                                    strategy,
                                ],
                            )
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    -2,
                                    keepdims,
                                    strategy,
                                ],
                            )
                        if len(shape) == 3:
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    (0, 1),
                                    keepdims,
                                    strategy,
                                ],
                            )
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    (0, 2),
                                    keepdims,
                                    strategy,
                                ],
                            )
                            cases.append(
                                [
                                    operation,
                                    bit_width,
                                    is_signed,
                                    shape,
                                    (1, 2),
                                    keepdims,
                                    strategy,
                                ],
                            )

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "operation,bit_width,is_signed,shape,axis,keepdims,strategy",
    random.sample(cases, 100),
)
def test_min_max(
    operation,
    bit_width,
    is_signed,
    shape,
    axis,
    keepdims,
    strategy,
    helpers,
):
    """
    Test np.min/np.max on encrypted values.
    """

    dtype = Integer(is_signed=is_signed, bit_width=bit_width)
    description = ValueDescription(dtype, shape=shape, is_encrypted=True)

    print()
    print()
    print(
        f"np.{operation}({description}, axis={axis}, keepdims={keepdims})"
        + (f" {{{strategy}}}" if strategy is not None else "")
    )
    print()
    print()

    assert operation in {"min", "max"}

    def function(x):
        if operation == "min":
            return np.min(x, axis=axis, keepdims=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)

    parameter_encryption_statuses = {"x": "encrypted"}
    configuration = helpers.configuration()

    if strategy is not None:
        configuration = configuration.fork(min_max_strategy_preference=[strategy])

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = [np.random.randint(dtype.min(), dtype.max() + 1, size=shape) for _ in range(100)]

    circuit = compiler.compile(inputset, configuration)

    samples = [
        np.zeros(shape, dtype=np.int64),
        np.ones(shape, dtype=np.int64) * dtype.min(),
        np.ones(shape, dtype=np.int64) * dtype.max(),
        np.random.randint(dtype.min(), dtype.max() + 1, size=shape),
        np.random.randint(dtype.min(), dtype.max() + 1, size=shape),
        np.random.randint(dtype.min(), dtype.max() + 1, size=shape),
    ]
    for sample in samples:
        helpers.check_execution(circuit, function, sample, retries=5)
