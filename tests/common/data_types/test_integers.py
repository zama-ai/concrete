"""Test file for HDK's common/data_types/integers.py"""

import random

import pytest

from hdk.common.data_types.integers import Integer, SignedInteger, UnsignedInteger


@pytest.mark.parametrize(
    "integer,expected_min,expected_max",
    [
        pytest.param(Integer(8, is_signed=False), 0, 255, id="8 bits unsigned Integer"),
        pytest.param(UnsignedInteger(8), 0, 255, id="8 bits UnsignedInteger"),
        pytest.param(Integer(8, is_signed=True), -128, 127, id="8 bits signed Integer"),
        pytest.param(SignedInteger(8), -128, 127, id="8 bits SignedInteger"),
        pytest.param(Integer(32, is_signed=False), 0, 4_294_967_295, id="32 bits unsigned Integer"),
        pytest.param(UnsignedInteger(32), 0, 4_294_967_295, id="32 bits UnsignedInteger"),
        pytest.param(
            Integer(32, is_signed=True),
            -2_147_483_648,
            2_147_483_647,
            id="32 bits signed Integer",
        ),
        pytest.param(
            SignedInteger(32),
            -2_147_483_648,
            2_147_483_647,
            id="32 bits SignedInteger",
        ),
    ],
)
def test_basic_integers(integer: Integer, expected_min: int, expected_max: int):
    """Test integer class basic functions"""
    assert integer.min_value() == expected_min
    assert integer.max_value() == expected_max

    assert integer.can_represent_value(random.randint(expected_min, expected_max))
    assert not integer.can_represent_value(expected_min - 1)
    assert not integer.can_represent_value(expected_max + 1)


@pytest.mark.parametrize(
    "integer,expected_repr_str",
    [
        pytest.param(
            Integer(8, is_signed=False),
            "Integer<unsigned, 8 bits>",
            id="8 bits unsigned Integer",
        ),
        pytest.param(
            Integer(8, is_signed=True),
            "Integer<signed, 8 bits>",
            id="8 bits signed Integer",
        ),
        pytest.param(
            Integer(32, is_signed=False),
            "Integer<unsigned, 32 bits>",
            id="32 bits unsigned Integer",
        ),
        pytest.param(
            Integer(32, is_signed=True),
            "Integer<signed, 32 bits>",
            id="32 bits signed Integer",
        ),
    ],
)
def test_integers_repr(integer: Integer, expected_repr_str: str):
    """Test integer repr"""
    assert integer.__repr__() == expected_repr_str
