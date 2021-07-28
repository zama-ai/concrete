"""Test file for values classes"""

import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import BaseValue, ClearValue, EncryptedValue


@pytest.mark.parametrize(
    "value,expected_repr_str",
    [
        pytest.param(
            ClearValue(Integer(8, is_signed=False)),
            "ClearValue<Integer<unsigned, 8 bits>>",
            id="ClearValue 8 bits unsigned Integer",
        ),
        pytest.param(
            EncryptedValue(Integer(8, is_signed=True)),
            "EncryptedValue<Integer<signed, 8 bits>>",
            id="EncryptedValue 8 bits signed Integer",
        ),
    ],
)
def test_values_repr(value: BaseValue, expected_repr_str: str):
    """Test value repr"""
    assert value.__repr__() == expected_repr_str
