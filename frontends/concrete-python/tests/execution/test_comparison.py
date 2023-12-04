"""
Tests of execution of comparison operations.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer
from concrete.fhe.values import ValueDescription

cases = [
    # uint1 (.) int1 triggers a special case in subtraction trick
    [
        # operation
        ("==", lambda x, y: x == y),
        # bit widths
        1,
        1,
        # signednesses
        False,
        True,
        # shapes
        (),
        (),
        # strategy
        fhe.ComparisonStrategy.THREE_TLU_CASTED,
    ]
]
for lhs_bit_width in range(1, 6):
    for rhs_bit_width in range(1, 6):
        for lhs_is_signed in [False, True]:
            for rhs_is_signed in [False, True]:
                cases += [
                    [
                        # operation
                        operation,
                        # bit widths
                        lhs_bit_width,
                        rhs_bit_width,
                        # signednesses
                        lhs_is_signed,
                        rhs_is_signed,
                        # shapes
                        (),
                        (),
                        # strategy
                        None,
                    ]
                    for operation in [("==", lambda x, y: x == y), ("<", lambda x, y: x < y)]
                ]

for _ in range(35):
    cases.append(
        [
            # operation
            random.choice(
                [
                    ("==", lambda x, y: x == y),
                    ("!=", lambda x, y: x != y),
                    ("<", lambda x, y: x < y),
                    ("<=", lambda x, y: x <= y),
                    (">", lambda x, y: x > y),
                    (">=", lambda x, y: x >= y),
                ]
            ),
            # bit widths
            random.choice([1, 2, 3, 4, 5, 6, 8, 14, 15, 16]),
            random.choice([1, 2, 3, 4, 5, 6, 8, 14, 15, 16]),
            # signednesses
            random.choice([False, True]),
            random.choice([False, True]),
            # shapes
            random.choice([(), (2,), (3, 2)]),
            random.choice([(), (2,), (3, 2)]),
            # strategy
            random.choice(
                [
                    fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
                    fhe.ComparisonStrategy.THREE_TLU_CASTED,
                    fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
                    fhe.ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED,
                    fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED,
                    fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED,
                    fhe.ComparisonStrategy.CHUNKED,
                ]
            ),
        ]
    )

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "operation,"
    "lhs_bit_width,rhs_bit_width,"
    "lhs_is_signed,rhs_is_signed,"
    "lhs_shape,rhs_shape,"
    "strategy",
    cases,
)
def test_comparison(
    operation,
    lhs_bit_width,
    rhs_bit_width,
    lhs_is_signed,
    rhs_is_signed,
    lhs_shape,
    rhs_shape,
    strategy,
    helpers,
):
    """
    Test comparison operations between encrypted integers.
    """

    name, function = operation

    lhs_dtype = Integer(is_signed=lhs_is_signed, bit_width=lhs_bit_width)
    rhs_dtype = Integer(is_signed=rhs_is_signed, bit_width=rhs_bit_width)

    lhs_description = ValueDescription(lhs_dtype, shape=lhs_shape, is_encrypted=True)
    rhs_description = ValueDescription(rhs_dtype, shape=rhs_shape, is_encrypted=True)

    print()
    print()
    print(
        f"[{lhs_description}] ({name}) [{rhs_description}]"
        + (f" {{{strategy}}}" if strategy is not None else "")
    )
    print()
    print()

    parameter_encryption_statuses = {"x": "encrypted", "y": "encrypted"}
    configuration = helpers.configuration().fork(use_insecure_key_cache=False)

    if strategy is not None:
        configuration = configuration.fork(comparison_strategy_preference=[strategy])

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = [
        (
            np.random.randint(lhs_dtype.min(), lhs_dtype.max() + 1, size=lhs_shape),
            np.random.randint(rhs_dtype.min(), rhs_dtype.max() + 1, size=rhs_shape),
        )
        for _ in range(100)
    ]
    circuit = compiler.compile(inputset, configuration)

    samples = [
        [
            np.zeros(lhs_shape, dtype=np.int64),
            np.zeros(rhs_shape, dtype=np.int64),
        ],
        [
            np.ones(lhs_shape, dtype=np.int64) * lhs_dtype.min(),
            np.ones(rhs_shape, dtype=np.int64) * rhs_dtype.min(),
        ],
        [
            np.ones(lhs_shape, dtype=np.int64) * lhs_dtype.max(),
            np.ones(rhs_shape, dtype=np.int64) * rhs_dtype.min(),
        ],
        [
            np.ones(lhs_shape, dtype=np.int64) * lhs_dtype.max(),
            np.ones(rhs_shape, dtype=np.int64) * rhs_dtype.max(),
        ],
        [
            np.random.randint(lhs_dtype.min(), lhs_dtype.max() + 1, size=lhs_shape),
            np.random.randint(rhs_dtype.min(), rhs_dtype.max() + 1, size=rhs_shape),
        ],
    ]
    for sample in samples:
        helpers.check_execution(
            circuit,
            function,
            sample,
            retries=5,
            only_simulation=(max(lhs_bit_width, rhs_bit_width) > 7),
        )
