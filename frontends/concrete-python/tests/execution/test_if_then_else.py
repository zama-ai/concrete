"""
Tests of execution of `if_then_else` extension.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer
from concrete.fhe.values import EncryptedScalar, EncryptedTensor

# pylint: disable=redefined-outer-name

functions = [
    lambda condition, when_true, when_false: np.where(condition, when_true, when_false),
    lambda condition, when_true, when_false: np.where(condition, when_true, when_false) + 100,
]
condition_descriptions = [
    EncryptedTensor(Integer(is_signed=False, bit_width=1), shape=shape)
    for shape in [(), (2,), (3, 2)]
]
when_true_descriptions = [
    EncryptedTensor(Integer(is_signed, bit_width), shape=shape)
    for is_signed in [False, True]
    for bit_width in [3, 4, 5]
    for shape in [(), (2,), (3, 2)]
]
when_false_descriptions = [
    EncryptedTensor(Integer(is_signed, bit_width), shape=shape)
    for is_signed in [False, True]
    for bit_width in [3, 4, 5]
    for shape in [(), (2,), (3, 2)]
]
chunk_sizes = [
    2,
    3,
]

cases = []
for function in functions:
    for condition_description in condition_descriptions:
        for when_true_description in when_true_descriptions:
            for when_false_description in when_false_descriptions:
                for chunk_size in chunk_sizes:
                    cases.append(
                        (
                            function,
                            condition_description,
                            when_true_description,
                            when_false_description,
                            chunk_size,
                        )
                    )

cases = random.sample(cases, 100)
cases.append(
    (
        # special case of increased bit-width for condition
        lambda condition, when_true, when_false: (
            np.where(condition, when_true, when_false) + (condition + 100)
        ),
        EncryptedScalar(Integer(is_signed=False, bit_width=1)),
        EncryptedScalar(Integer(is_signed=False, bit_width=4)),
        EncryptedScalar(Integer(is_signed=False, bit_width=4)),
        2,
    )
)


@pytest.mark.parametrize(
    "function,condition_description,when_true_description,when_false_description,chunk_size",
    cases,
)
def test_if_then_else(
    function,
    condition_description,
    when_true_description,
    when_false_description,
    chunk_size,
    helpers,
):
    """
    Test encrypted evaluation of `if_then_else` extension.
    """

    print()
    print()
    print(
        f"[{when_true_description}] "
        f"if [{condition_description}] "
        f"else [{when_false_description}] "
        f"{{{chunk_size=}}}"
    )
    print()
    print()

    inputset = [
        (
            np.random.randint(
                condition_description.dtype.min(),
                condition_description.dtype.max() + 1,
                size=condition_description.shape,
            ),
            np.random.randint(
                when_true_description.dtype.min(),
                when_true_description.dtype.max() + 1,
                size=when_true_description.shape,
            ),
            np.random.randint(
                when_false_description.dtype.min(),
                when_false_description.dtype.max() + 1,
                size=when_false_description.shape,
            ),
        )
        for _ in range(100)
    ]
    configuration = helpers.configuration().fork(if_then_else_chunk_size=chunk_size)

    compiler = fhe.Compiler(
        function,
        {
            "condition": "encrypted",
            "when_true": "encrypted",
            "when_false": "encrypted",
        },
    )
    circuit = compiler.compile(inputset, configuration)

    for sample in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, list(sample), retries=3)
