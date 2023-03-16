"""
Tests of `Integer` data type.
"""

import numpy as np
import pytest

from concrete.fhe.dtypes import Integer, SignedInteger, UnsignedInteger


@pytest.mark.parametrize(
    "value,force_signed,expected_result",
    [
        pytest.param(
            -4,
            False,
            SignedInteger(3),
        ),
        pytest.param(
            -3,
            False,
            SignedInteger(3),
        ),
        pytest.param(
            -2,
            False,
            SignedInteger(2),
        ),
        pytest.param(
            -1,
            False,
            SignedInteger(1),
        ),
        pytest.param(
            0,
            False,
            UnsignedInteger(1),
        ),
        pytest.param(
            1,
            False,
            UnsignedInteger(1),
        ),
        pytest.param(
            2,
            False,
            UnsignedInteger(2),
        ),
        pytest.param(
            3,
            False,
            UnsignedInteger(2),
        ),
        pytest.param(
            4,
            False,
            UnsignedInteger(3),
        ),
        pytest.param(
            -4,
            True,
            SignedInteger(3),
        ),
        pytest.param(
            -3,
            True,
            SignedInteger(3),
        ),
        pytest.param(
            -2,
            True,
            SignedInteger(2),
        ),
        pytest.param(
            -1,
            True,
            SignedInteger(1),
        ),
        pytest.param(
            0,
            True,
            SignedInteger(1),
        ),
        pytest.param(
            1,
            True,
            SignedInteger(2),
        ),
        pytest.param(
            2,
            True,
            SignedInteger(3),
        ),
        pytest.param(
            3,
            True,
            SignedInteger(3),
        ),
        pytest.param(
            4,
            True,
            SignedInteger(4),
        ),
        pytest.param(
            np.array([0, 1]),
            False,
            UnsignedInteger(1),
        ),
        pytest.param(
            np.array([0, 1]),
            True,
            SignedInteger(2),
        ),
        pytest.param(
            [-1, 1],
            False,
            SignedInteger(2),
        ),
        pytest.param(
            [-1, 1],
            True,
            SignedInteger(2),
        ),
    ],
)
def test_integer_that_can_represent(value, force_signed, expected_result):
    """
    Test `that_can_represent` function of `Integer` data type.
    """

    assert Integer.that_can_represent(value, force_signed) == expected_result


@pytest.mark.parametrize(
    "value,force_signed,expected_error,expected_message",
    [
        pytest.param(
            "abc",
            False,
            ValueError,
            "Integer cannot represent 'abc'",
        ),
        pytest.param(
            "abc",
            True,
            ValueError,
            "Integer cannot represent 'abc'",
        ),
        pytest.param(
            4.2,
            False,
            ValueError,
            "Integer cannot represent 4.2",
        ),
        pytest.param(
            4.2,
            True,
            ValueError,
            "Integer cannot represent 4.2",
        ),
        pytest.param(
            np.array([2.2, 1.1]),
            False,
            ValueError,
            "Integer cannot represent array([2.2, 1.1])",
        ),
        pytest.param(
            np.array([2.2, 1.1]),
            True,
            ValueError,
            "Integer cannot represent array([2.2, 1.1])",
        ),
        pytest.param(
            [1, (), 3],
            True,
            ValueError,
            "Integer cannot represent [1, (), 3]",
        ),
    ],
)
def test_integer_bad_that_can_represent(value, force_signed, expected_error, expected_message):
    """
    Test `that_can_represent` function of `Integer` data type with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Integer.that_can_represent(value, force_signed)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "constructor,bit_width,expected_error,expected_message",
    [
        pytest.param(
            SignedInteger,
            0,
            ValueError,
            "SignedInteger(0) is not supported (bit width must be a positive integer)",
        ),
        pytest.param(
            UnsignedInteger,
            0,
            ValueError,
            "UnsignedInteger(0) is not supported (bit width must be a positive integer)",
        ),
        pytest.param(
            SignedInteger,
            -1,
            ValueError,
            "SignedInteger(-1) is not supported (bit width must be a positive integer)",
        ),
        pytest.param(
            UnsignedInteger,
            -1,
            ValueError,
            "UnsignedInteger(-1) is not supported (bit width must be a positive integer)",
        ),
        pytest.param(
            SignedInteger,
            "abc",
            ValueError,
            "SignedInteger('abc') is not supported (bit width must be a positive integer)",
        ),
        pytest.param(
            UnsignedInteger,
            "abc",
            ValueError,
            "UnsignedInteger('abc') is not supported (bit width must be a positive integer)",
        ),
    ],
)
def test_integer_bad_init(constructor, bit_width, expected_error, expected_message):
    """
    Test `__init__` method of `Integer` data type with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        constructor(bit_width)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "lhs,rhs,expected_result",
    [
        pytest.param(
            SignedInteger(5),
            SignedInteger(5),
            True,
        ),
        pytest.param(
            UnsignedInteger(5),
            UnsignedInteger(5),
            True,
        ),
        pytest.param(
            SignedInteger(5),
            SignedInteger(6),
            False,
        ),
        pytest.param(
            SignedInteger(6),
            SignedInteger(5),
            False,
        ),
        pytest.param(
            UnsignedInteger(5),
            UnsignedInteger(6),
            False,
        ),
        pytest.param(
            UnsignedInteger(6),
            UnsignedInteger(5),
            False,
        ),
        pytest.param(
            SignedInteger(5),
            UnsignedInteger(5),
            False,
        ),
        pytest.param(
            UnsignedInteger(5),
            SignedInteger(5),
            False,
        ),
        pytest.param(
            SignedInteger(5),
            "SignedInteger(5)",
            False,
        ),
        pytest.param(
            "SignedInteger(5)",
            SignedInteger(5),
            False,
        ),
    ],
)
def test_integer_eq(lhs, rhs, expected_result):
    """
    Test `__eq__` method of `Integer` data type.
    """

    assert (lhs == rhs) == expected_result
    assert (rhs == lhs) == expected_result


@pytest.mark.parametrize(
    "data_type,expected_result",
    [
        pytest.param(
            UnsignedInteger(4),
            "uint4",
        ),
        pytest.param(
            UnsignedInteger(7),
            "uint7",
        ),
        pytest.param(
            SignedInteger(4),
            "int4",
        ),
        pytest.param(
            SignedInteger(7),
            "int7",
        ),
    ],
)
def test_integer_str(data_type, expected_result):
    """
    Test `__str__` method of `Integer` data type.
    """

    assert str(data_type) == expected_result


@pytest.mark.parametrize(
    "data_type,expected_result",
    [
        pytest.param(
            UnsignedInteger(1),
            0,
        ),
        pytest.param(
            UnsignedInteger(3),
            0,
        ),
        pytest.param(
            UnsignedInteger(5),
            0,
        ),
        pytest.param(
            SignedInteger(1),
            -1,
        ),
        pytest.param(
            SignedInteger(3),
            -4,
        ),
        pytest.param(
            SignedInteger(5),
            -16,
        ),
    ],
)
def test_integer_min(data_type, expected_result):
    """
    Test `min` method of `Integer` data type.
    """

    assert data_type.min() == expected_result


@pytest.mark.parametrize(
    "data_type,expected_result",
    [
        pytest.param(
            UnsignedInteger(1),
            1,
        ),
        pytest.param(
            UnsignedInteger(3),
            7,
        ),
        pytest.param(
            UnsignedInteger(5),
            31,
        ),
        pytest.param(
            SignedInteger(1),
            0,
        ),
        pytest.param(
            SignedInteger(3),
            3,
        ),
        pytest.param(
            SignedInteger(5),
            15,
        ),
    ],
)
def test_integer_max(data_type, expected_result):
    """
    Test `max` method of `Integer` data type.
    """

    assert data_type.max() == expected_result


@pytest.mark.parametrize(
    "data_type,value,expected_result",
    [
        pytest.param(
            UnsignedInteger(1),
            -4,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            -3,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            -2,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            -1,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            0,
            True,
        ),
        pytest.param(
            UnsignedInteger(1),
            1,
            True,
        ),
        pytest.param(
            UnsignedInteger(1),
            2,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            3,
            False,
        ),
        pytest.param(
            UnsignedInteger(1),
            4,
            False,
        ),
        pytest.param(
            UnsignedInteger(2),
            -4,
            False,
        ),
        pytest.param(
            UnsignedInteger(2),
            -3,
            False,
        ),
        pytest.param(
            UnsignedInteger(2),
            -2,
            False,
        ),
        pytest.param(
            UnsignedInteger(2),
            -1,
            False,
        ),
        pytest.param(
            UnsignedInteger(2),
            0,
            True,
        ),
        pytest.param(
            UnsignedInteger(2),
            1,
            True,
        ),
        pytest.param(
            UnsignedInteger(2),
            2,
            True,
        ),
        pytest.param(
            UnsignedInteger(2),
            3,
            True,
        ),
        pytest.param(
            UnsignedInteger(2),
            4,
            False,
        ),
        pytest.param(
            UnsignedInteger(3),
            -4,
            False,
        ),
        pytest.param(
            UnsignedInteger(3),
            -3,
            False,
        ),
        pytest.param(
            UnsignedInteger(3),
            -2,
            False,
        ),
        pytest.param(
            UnsignedInteger(3),
            -1,
            False,
        ),
        pytest.param(
            UnsignedInteger(3),
            0,
            True,
        ),
        pytest.param(
            UnsignedInteger(3),
            1,
            True,
        ),
        pytest.param(
            UnsignedInteger(3),
            2,
            True,
        ),
        pytest.param(
            UnsignedInteger(3),
            3,
            True,
        ),
        pytest.param(
            UnsignedInteger(3),
            4,
            True,
        ),
        pytest.param(
            SignedInteger(1),
            -4,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            -3,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            -2,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            -1,
            True,
        ),
        pytest.param(
            SignedInteger(1),
            0,
            True,
        ),
        pytest.param(
            SignedInteger(1),
            1,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            2,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            3,
            False,
        ),
        pytest.param(
            SignedInteger(1),
            4,
            False,
        ),
        pytest.param(
            SignedInteger(2),
            -4,
            False,
        ),
        pytest.param(
            SignedInteger(2),
            -3,
            False,
        ),
        pytest.param(
            SignedInteger(2),
            -2,
            True,
        ),
        pytest.param(
            SignedInteger(2),
            -1,
            True,
        ),
        pytest.param(
            SignedInteger(2),
            0,
            True,
        ),
        pytest.param(
            SignedInteger(2),
            1,
            True,
        ),
        pytest.param(
            SignedInteger(2),
            2,
            False,
        ),
        pytest.param(
            SignedInteger(2),
            3,
            False,
        ),
        pytest.param(
            SignedInteger(2),
            4,
            False,
        ),
        pytest.param(
            SignedInteger(3),
            -4,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            -3,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            -2,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            -1,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            0,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            1,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            2,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            3,
            True,
        ),
        pytest.param(
            SignedInteger(3),
            4,
            False,
        ),
    ],
)
def test_integer_can_represent(data_type, value, expected_result):
    """
    Test `can_represent` method of `Integer` data type.
    """

    assert data_type.can_represent(value) == expected_result
