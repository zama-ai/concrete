"""
Tests of utilities related to data types.
"""

import pytest

from concrete.fhe.dtypes import Float, SignedInteger, UnsignedInteger
from concrete.fhe.dtypes.utils import combine_dtypes


@pytest.mark.parametrize(
    "dtypes,expected_result",
    [
        pytest.param(
            [Float(64), Float(64)],
            Float(64),
        ),
        pytest.param(
            [Float(32), Float(64)],
            Float(64),
        ),
        pytest.param(
            [Float(16), Float(64)],
            Float(64),
        ),
        pytest.param(
            [Float(32), Float(16)],
            Float(32),
        ),
        pytest.param(
            [Float(16), Float(16)],
            Float(16),
        ),
        pytest.param(
            [SignedInteger(5), Float(64)],
            Float(64),
        ),
        pytest.param(
            [Float(32), SignedInteger(5)],
            Float(32),
        ),
        pytest.param(
            [SignedInteger(5), Float(16)],
            Float(16),
        ),
        pytest.param(
            [SignedInteger(5), SignedInteger(6)],
            SignedInteger(6),
        ),
        pytest.param(
            [UnsignedInteger(5), UnsignedInteger(6)],
            UnsignedInteger(6),
        ),
        pytest.param(
            [SignedInteger(5), UnsignedInteger(6)],
            SignedInteger(7),
        ),
        pytest.param(
            [SignedInteger(5), UnsignedInteger(4)],
            SignedInteger(5),
        ),
        pytest.param(
            [UnsignedInteger(6), SignedInteger(5)],
            SignedInteger(7),
        ),
        pytest.param(
            [UnsignedInteger(4), SignedInteger(5)],
            SignedInteger(5),
        ),
    ],
)
def test_combine_dtypes(dtypes, expected_result):
    """
    Test `combine_dtypes` function.
    """

    assert combine_dtypes(dtypes) == expected_result
