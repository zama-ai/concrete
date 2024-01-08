"""
Tests of execution of minimum and maximum operations.
"""

import random

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.dtypes import Integer
from concrete.fhe.values import ValueDescription

KNOWN_MULTI_BUG = "_mark_xfail_if_multi"

cases = [
    [
        # operation
        (
            "minimum_optimized_x",
            lambda x, y: np.minimum(fhe.hint(x, bit_width=5), y),  # type: ignore
        ),
        # bit widths
        4,
        4,
        # signednesses
        False,
        True,
        # shapes
        (),
        (),
        # strategy
        fhe.MinMaxStrategy.CHUNKED,
    ],
    [
        # operation
        (
            "minimum_optimized_x",
            lambda x, y: np.minimum(fhe.hint(x, bit_width=5), y),  # type: ignore
        ),
        # bit widths
        4,
        4,
        # signednesses
        True,
        False,
        # shapes
        (2,),
        (),
        # strategy
        fhe.MinMaxStrategy.CHUNKED,
    ],
    [
        # operation
        (
            "maximum_optimized_y",
            lambda x, y: np.maximum(x, fhe.hint(y, bit_width=4)),  # type: ignore
        ),
        # bit widths
        4,
        3,
        # signednesses
        True,
        False,
        # shapes
        (),
        (2, 3),
        # strategy
        fhe.MinMaxStrategy.CHUNKED,
    ],
    [
        # operation
        (
            "maximum_optimized_y",
            lambda x, y: np.maximum(x, fhe.hint(y, bit_width=4)),  # type: ignore
        ),
        # bit widths
        4,
        3,
        # signednesses
        False,
        True,
        # shapes
        (),
        (),
        # strategy
        fhe.MinMaxStrategy.CHUNKED,
    ],
]
cases += [
    [
        # operation
        operation,
        # bit widths
        1,
        1,
        # signednesses
        lhs_is_signed,
        rhs_is_signed,
        # shapes
        (),
        (),
        # strategy
        fhe.MinMaxStrategy.CHUNKED,
    ]
    for lhs_is_signed in [False, True]
    for rhs_is_signed in [False, True]
    for operation in [
        (
            "maximum" + KNOWN_MULTI_BUG,
            lambda x, y: np.maximum(x, y),
        ),
    ]
]
for lhs_bit_width in range(1, 5):
    for rhs_bit_width in range(1, 5):
        strategies = []
        if lhs_bit_width <= 3 and rhs_bit_width <= 3:
            strategies += [
                fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
                fhe.MinMaxStrategy.THREE_TLU_CASTED,
            ]
        else:
            strategies += [
                fhe.MinMaxStrategy.CHUNKED,
            ]

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
                        random.choice([(), (2,), (3, 2)]),
                        random.choice([(), (2,), (3, 2)]),
                        # strategy
                        strategy,
                    ]
                    for operation in [
                        ("minimum" + KNOWN_MULTI_BUG, lambda x, y: np.minimum(x, y)),
                        ("maximum" + KNOWN_MULTI_BUG, lambda x, y: np.maximum(x, y)),
                    ]
                    for strategy in strategies
                ]

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "operation,"
    "lhs_bit_width,rhs_bit_width,"
    "lhs_is_signed,rhs_is_signed,"
    "lhs_shape,rhs_shape,"
    "strategy",
    cases,
)
def test_minimum_maximum(
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
        f"{name}({lhs_description}, {rhs_description})"
        + (f" {{{strategy}}}" if strategy is not None else "")
    )
    print()
    print()

    parameter_encryption_statuses = {"x": "encrypted", "y": "encrypted"}
    configuration = helpers.configuration()
    can_be_known_failure = (
        name.endswith("xfail_if_multi")
        and configuration.parameter_selection_strategy == fhe.ParameterSelectionStrategy.MULTI
    )

    if strategy is not None:
        configuration = configuration.fork(min_max_strategy_preference=[strategy])

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = [
        (
            np.random.randint(lhs_dtype.min(), lhs_dtype.max() + 1, size=lhs_shape),
            np.random.randint(rhs_dtype.min(), rhs_dtype.max() + 1, size=rhs_shape),
        )
        for _ in range(100)
    ]
    try:
        circuit = compiler.compile(inputset, configuration)
    except AssertionError:
        pytest.xfail("Known cp bug")

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
        try:
            helpers.check_execution(circuit, function, sample, retries=5)
        except RuntimeError as exc:
            if (
                str(exc) == "RuntimeError: Can't compile: Parametrization of TFHE operations failed"
                and can_be_known_failure
            ):
                pytest.xfail("Known multi output bug, bad compilation")
            raise
        except AssertionError:
            if can_be_known_failure:
                pytest.xfail("Known multi output bug, bad result")
            raise
