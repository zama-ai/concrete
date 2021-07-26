"""Test file for HDK's data types helpers"""

import pytest

from hdk.common.data_types.base import BaseDataType
from hdk.common.data_types.dtypes_helpers import (
    find_type_to_hold_both_lossy,
    mix_values_determine_holding_dtype,
    value_is_encrypted_integer,
    value_is_encrypted_unsigned_integer,
)
from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import BaseValue, ClearValue, EncryptedValue


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
def test_value_is_encrypted_integer(value: BaseValue, expected_result: bool):
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
def test_value_is_encrypted_unsigned_integer(value: BaseValue, expected_result: bool):
    """Test value_is_encrypted_unsigned_integer helper"""
    assert value_is_encrypted_unsigned_integer(value) == expected_result


class UnsupportedDataType(BaseDataType):
    """Test helper class to represent an UnsupportedDataType"""


@pytest.mark.parametrize(
    "dtype1,dtype2,expected_mixed_dtype",
    [
        pytest.param(Integer(6, True), Integer(6, True), Integer(6, True), id="int6, int6, int6"),
        pytest.param(
            Integer(6, False), Integer(6, False), Integer(6, False), id="uint6, uint6, uint6"
        ),
        pytest.param(Integer(6, True), Integer(6, False), Integer(7, True), id="int6, uint6, int7"),
        pytest.param(Integer(6, False), Integer(6, True), Integer(7, True), id="uint6, int6, int7"),
        pytest.param(Integer(6, True), Integer(5, False), Integer(6, True), id="int6, uint5, int6"),
        pytest.param(Integer(5, False), Integer(6, True), Integer(6, True), id="uint5, int6, int6"),
        pytest.param(
            UnsupportedDataType(),
            UnsupportedDataType(),
            None,
            id="unsupported, unsupported, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            Integer(6, True),
            UnsupportedDataType(),
            None,
            id="int6, unsupported, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            UnsupportedDataType(),
            Integer(6, True),
            None,
            id="unsupported, int6, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_mix_data_types(
    dtype1: BaseDataType,
    dtype2: BaseDataType,
    expected_mixed_dtype: BaseDataType,
):
    """Test find_type_to_hold_both_lossy helper"""
    assert expected_mixed_dtype == find_type_to_hold_both_lossy(dtype1, dtype2)


@pytest.mark.parametrize(
    "value1,value2,expected_mixed_value",
    [
        pytest.param(
            EncryptedValue(Integer(7, False)),
            EncryptedValue(Integer(7, False)),
            EncryptedValue(Integer(7, False)),
            id="euint7, euint7, euint7",
        ),
        pytest.param(
            EncryptedValue(Integer(7, False)),
            ClearValue(Integer(7, False)),
            EncryptedValue(Integer(7, False)),
            id="euint7, cuint7, euint7",
        ),
        pytest.param(
            ClearValue(Integer(7, False)),
            EncryptedValue(Integer(7, False)),
            EncryptedValue(Integer(7, False)),
            id="cuint7, euint7, euint7",
        ),
        pytest.param(
            ClearValue(Integer(7, False)),
            ClearValue(Integer(7, False)),
            ClearValue(Integer(7, False)),
            id="cuint7, cuint7, cuint7",
        ),
    ],
)
def test_mix_values(value1: BaseValue, value2: BaseValue, expected_mixed_value: BaseValue):
    """Test mix_values helper"""

    assert expected_mixed_value == mix_values_determine_holding_dtype(value1, value2)
