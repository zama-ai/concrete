"""Test file for HDK's data types helpers"""

import pytest

from hdk.common.data_types.dtypes_helpers import (
    value_is_encrypted_integer,
    value_is_encrypted_unsigned_integer,
)
from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            ClearValue(Integer(8, is_signed=False)),
            False,
            id="ClearValue 8 bits unsigned Integer",
        ),
        pytest.param(
            EncryptedValue(Integer(8, is_signed=True)),
            True,
            id="EncryptedValue 8 bits signed Integer",
        ),
    ],
)
def test_value_is_encrypted_integer(value: Integer, expected_result: bool):
    """Test value_is_encrypted_integer helper"""
    assert value_is_encrypted_integer(value) == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            ClearValue(Integer(8, is_signed=False)),
            False,
            id="ClearValue 8 bits unsigned Integer",
        ),
        pytest.param(
            EncryptedValue(Integer(8, is_signed=True)),
            False,
            id="EncryptedValue 8 bits signed Integer",
        ),
        pytest.param(
            EncryptedValue(Integer(8, is_signed=False)),
            True,
            id="EncryptedValue 8 bits unsigned Integer",
        ),
    ],
)
def test_value_is_encrypted_unsigned_integer(value: Integer, expected_result: bool):
    """Test value_is_encrypted_unsigned_integer helper"""
    assert value_is_encrypted_unsigned_integer(value) == expected_result
