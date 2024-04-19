"""
Tests of utilities related to representation of computation.
"""

import numpy as np
import pytest

from concrete.fhe.representation.utils import format_constant


@pytest.mark.parametrize(
    "constant,maximum_length,keep_newlines,expected_result",
    [
        pytest.param(
            1,
            45,
            True,
            "1",
        ),
        pytest.param(
            np.uint32,
            45,
            True,
            "uintc",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            45,
            True,
            "[[1 2]\n [3 4]]",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            45,
            False,
            "[[1 2] [3 4]]",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2], [3, 4], [5, 6]]),
            45,
            True,
            "[[1 2]\n [3 4]\n [5 6]\n...\n[1 2]\n [3 4]\n [5 6]]",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2], [3, 4], [5, 6]]),
            45,
            False,
            "[[1 2] [3 4] [5 6] [ ... ] [1 2] [3 4] [5 6]]",
        ),
    ],
)
def test_format_constant(constant, maximum_length, keep_newlines, expected_result):
    """
    Test `format_constant` function.
    """

    assert format_constant(constant, maximum_length, keep_newlines) == expected_result
