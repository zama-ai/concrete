"""
Tests of `Value` class.
"""

import numpy as np
import pytest

from concrete.fhe.dtypes import Float, SignedInteger, UnsignedInteger
from concrete.fhe.values import (
    ClearScalar,
    ClearTensor,
    EncryptedScalar,
    EncryptedTensor,
    ValueDescription,
)


@pytest.mark.parametrize(
    "value,is_encrypted,expected_result",
    [
        pytest.param(
            True,
            True,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=True),
        ),
        pytest.param(
            True,
            False,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=False),
        ),
        pytest.param(
            False,
            True,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=True),
        ),
        pytest.param(
            False,
            False,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=False),
        ),
        pytest.param(
            0,
            False,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=False),
        ),
        pytest.param(
            np.int32(0),
            True,
            ValueDescription(dtype=UnsignedInteger(1), shape=(), is_encrypted=True),
        ),
        pytest.param(
            0.0,
            False,
            ValueDescription(dtype=Float(64), shape=(), is_encrypted=False),
        ),
        pytest.param(
            np.float64(0.0),
            True,
            ValueDescription(dtype=Float(64), shape=(), is_encrypted=True),
        ),
        pytest.param(
            np.float32(0.0),
            False,
            ValueDescription(dtype=Float(32), shape=(), is_encrypted=False),
        ),
        pytest.param(
            np.float16(0.0),
            True,
            ValueDescription(dtype=Float(16), shape=(), is_encrypted=True),
        ),
        pytest.param(
            [True, False, True],
            False,
            ValueDescription(dtype=UnsignedInteger(1), shape=(3,), is_encrypted=False),
        ),
        pytest.param(
            [True, False, True],
            True,
            ValueDescription(dtype=UnsignedInteger(1), shape=(3,), is_encrypted=True),
        ),
        pytest.param(
            [0, 3, 1, 2],
            False,
            ValueDescription(dtype=UnsignedInteger(2), shape=(4,), is_encrypted=False),
        ),
        pytest.param(
            np.array([0, 3, 1, 2], dtype=np.int32),
            True,
            ValueDescription(dtype=UnsignedInteger(2), shape=(4,), is_encrypted=True),
        ),
        pytest.param(
            np.array([0.2, 3.4, 1.5, 2.0], dtype=np.float64),
            False,
            ValueDescription(dtype=Float(64), shape=(4,), is_encrypted=False),
        ),
        pytest.param(
            np.array([0.2, 3.4, 1.5, 2.0], dtype=np.float32),
            True,
            ValueDescription(dtype=Float(32), shape=(4,), is_encrypted=True),
        ),
        pytest.param(
            np.array([0.2, 3.4, 1.5, 2.0], dtype=np.float16),
            False,
            ValueDescription(dtype=Float(16), shape=(4,), is_encrypted=False),
        ),
    ],
)
def test_value_of(value, is_encrypted, expected_result):
    """
    Test `of` function of `Value` class.
    """

    assert ValueDescription.of(value, is_encrypted) == expected_result


@pytest.mark.parametrize(
    "value,is_encrypted,expected_error,expected_message",
    [
        pytest.param(
            "abc",
            False,
            ValueError,
            "Concrete cannot represent 'abc'",
        ),
        pytest.param(
            [1, (), 3],
            False,
            ValueError,
            "Concrete cannot represent [1, (), 3]",
        ),
    ],
)
def test_value_bad_of(value, is_encrypted, expected_error, expected_message):
    """
    Test `of` function of `Value` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        ValueDescription.of(value, is_encrypted)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "lhs,rhs,expected_result",
    [
        pytest.param(
            ClearScalar(SignedInteger(5)),
            ValueDescription(dtype=SignedInteger(5), shape=(), is_encrypted=False),
            True,
        ),
        pytest.param(
            ClearTensor(UnsignedInteger(5), shape=(3, 2)),
            ValueDescription(dtype=UnsignedInteger(5), shape=(3, 2), is_encrypted=False),
            True,
        ),
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            ValueDescription(dtype=SignedInteger(5), shape=(), is_encrypted=True),
            True,
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(5), shape=(3, 2)),
            ValueDescription(dtype=UnsignedInteger(5), shape=(3, 2), is_encrypted=True),
            True,
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(5), shape=(3, 2)),
            ClearScalar(SignedInteger(3)),
            False,
        ),
        pytest.param(
            ClearScalar(SignedInteger(3)),
            "ClearScalar(SignedInteger(3))",
            False,
        ),
        pytest.param(
            "ClearScalar(SignedInteger(3))",
            ClearScalar(SignedInteger(3)),
            False,
        ),
    ],
)
def test_value_eq(lhs, rhs, expected_result):
    """
    Test `__eq__` method of `Value` class.
    """

    assert (lhs == rhs) == expected_result
    assert (rhs == lhs) == expected_result


@pytest.mark.parametrize(
    "data_type,expected_result",
    [
        pytest.param(
            ClearScalar(SignedInteger(5)),
            "ClearScalar<int5>",
        ),
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            "EncryptedScalar<int5>",
        ),
        pytest.param(
            ClearTensor(SignedInteger(5), shape=(3, 2)),
            "ClearTensor<int5, shape=(3, 2)>",
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3, 2)),
            "EncryptedTensor<int5, shape=(3, 2)>",
        ),
    ],
)
def test_value_str(data_type, expected_result):
    """
    Test `__str__` method of `Value` class.
    """

    assert str(data_type) == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            ClearScalar(SignedInteger(5)),
            True,
        ),
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            False,
        ),
    ],
)
def test_value_is_clear(value, expected_result):
    """
    Test `is_clear` property of `Value` class.
    """

    assert value.is_clear == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            True,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3, 2)),
            False,
        ),
    ],
)
def test_value_is_scalar(value, expected_result):
    """
    Test `is_scalar` property of `Value` class.
    """

    assert value.is_scalar == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            0,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3,)),
            1,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3, 2)),
            2,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(5, 3, 2)),
            3,
        ),
    ],
)
def test_value_ndim(value, expected_result):
    """
    Test `ndim` property of `Value` class.
    """

    assert value.ndim == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            EncryptedScalar(SignedInteger(5)),
            1,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3,)),
            3,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(3, 2)),
            6,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(5, 3, 2)),
            30,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(1,)),
            1,
        ),
        pytest.param(
            EncryptedTensor(SignedInteger(5), shape=(1, 1)),
            1,
        ),
    ],
)
def test_value_size(value, expected_result):
    """
    Test `size` property of `Value` class.
    """

    assert value.size == expected_result
