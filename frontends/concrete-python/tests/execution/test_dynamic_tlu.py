"""
Tests of execution of dynamic tlu operation.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer

from ..conftest import USE_MULTI_PRECISION

cases = []
for input_bit_width in range(1, 3):
    for input_is_signed in [False, True]:
        for output_bit_width in range(1, 3):
            for output_is_signed in [False, True]:
                input_shape = random.choice([(), (2,), (3, 2)])
                cases.append(
                    pytest.param(
                        input_bit_width,
                        input_is_signed,
                        input_shape,
                        output_bit_width,
                        output_is_signed,
                        id=(
                            f"{'' if input_is_signed else 'u'}int{input_bit_width}"
                            f" -> "
                            f"{'' if output_is_signed else 'u'}int{output_bit_width}"
                            f" {{ input_shape={input_shape} }}"
                        ),
                    )
                )

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "input_bit_width,input_is_signed,input_shape,output_bit_width,output_is_signed",
    cases,
)
def test_dynamic_tlu(
    input_bit_width,
    input_is_signed,
    input_shape,
    output_bit_width,
    output_is_signed,
    helpers,
):
    """
    Test dynamic tlu.
    """

    input_dtype = Integer(is_signed=input_is_signed, bit_width=input_bit_width)
    output_dtype = Integer(is_signed=output_is_signed, bit_width=output_bit_width)

    def function(x, y):
        return y[x]

    compiler = fhe.Compiler(function, {"x": "encrypted", "y": "clear"})
    inputset = [
        (
            np.random.randint(
                input_dtype.min(),
                input_dtype.max() + 1,
                size=input_shape,
            ),
            np.random.randint(
                output_dtype.min(),
                output_dtype.max() + 1,
                size=(
                    2
                    ** (
                        input_bit_width
                        if USE_MULTI_PRECISION
                        else max(input_bit_width, output_bit_width)
                    ),
                ),
            ),
        )
        for _ in range(100)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    samples = [
        [
            np.random.randint(
                input_dtype.min(),
                input_dtype.max() + 1,
                size=input_shape,
            ),
            np.random.randint(
                output_dtype.min(),
                output_dtype.max() + 1,
                size=(
                    2
                    ** (
                        input_bit_width
                        if USE_MULTI_PRECISION
                        else max(input_bit_width, output_bit_width)
                    ),
                ),
            ),
        ]
        for _ in range(5)
    ]
    for sample in samples:
        helpers.check_execution(circuit, function, sample, retries=3)
