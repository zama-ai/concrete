"""
Tests of `Float` data type.
"""

import pytest

from concrete.fhe.dtypes import Float


@pytest.mark.parametrize(
    "bit_width,expected_error,expected_message",
    [
        pytest.param(
            128,
            ValueError,
            "Float(128) is not supported (bit width must be one of 16, 32 or 64)",
        ),
        pytest.param(
            "abc",
            ValueError,
            "Float('abc') is not supported (bit width must be one of 16, 32 or 64)",
        ),
    ],
)
def test_float_bad_init(bit_width, expected_error, expected_message):
    """
    Test `__init__` method of `Float` data type with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Float(bit_width)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "lhs,rhs,expected_result",
    [
        pytest.param(
            Float(32),
            Float(32),
            True,
        ),
        pytest.param(
            Float(32),
            Float(64),
            False,
        ),
        pytest.param(
            Float(32),
            "Float(32)",
            False,
        ),
        pytest.param(
            "Float(32)",
            Float(32),
            False,
        ),
    ],
)
def test_float_eq(lhs, rhs, expected_result):
    """
    Test `__eq__` method of `Float` data type.
    """

    assert (lhs == rhs) == expected_result
    assert (rhs == lhs) == expected_result


@pytest.mark.parametrize(
    "data_type,expected_result",
    [
        pytest.param(
            Float(16),
            "float16",
        ),
        pytest.param(
            Float(32),
            "float32",
        ),
        pytest.param(
            Float(64),
            "float64",
        ),
    ],
)
def test_float_str(data_type, expected_result):
    """
    Test `__str__` method of `Float` data type.
    """

    assert str(data_type) == expected_result
