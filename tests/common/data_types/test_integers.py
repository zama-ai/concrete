"""Test file for integers data types"""

import random

import pytest

from concrete.common.data_types.integers import (
    Integer,
    SignedInteger,
    UnsignedInteger,
    make_integer_to_hold,
)


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


@pytest.mark.parametrize(
    "values,force_signed,expected_result",
    [
        ([0], False, Integer(1, is_signed=False)),
        ([0], True, Integer(2, is_signed=True)),
        ([1], False, Integer(1, is_signed=False)),
        ([1], True, Integer(2, is_signed=True)),
        ([-1], False, Integer(2, is_signed=True)),
        ([-2], False, Integer(2, is_signed=True)),
        ([0, 1], False, Integer(1, is_signed=False)),
        ([0, 1], True, Integer(2, is_signed=True)),
        ([7], False, Integer(3, is_signed=False)),
        ([7], True, Integer(4, is_signed=True)),
        ([8], False, Integer(4, is_signed=False)),
        ([8], True, Integer(5, is_signed=True)),
        ([-7], False, Integer(4, is_signed=True)),
        ([-8], False, Integer(4, is_signed=True)),
        ([-7, -8], False, Integer(4, is_signed=True)),
        ([-9], False, Integer(5, is_signed=True)),
        ([-9], True, Integer(5, is_signed=True)),
        ([0, 127], False, Integer(7, is_signed=False)),
        ([0, 127], True, Integer(8, is_signed=True)),
        ([0, 128], False, Integer(8, is_signed=False)),
        ([0, 128], True, Integer(9, is_signed=True)),
        ([-1, 127], False, Integer(8, is_signed=True)),
        ([-256, 127], False, Integer(9, is_signed=True)),
        ([-128, 127], False, Integer(8, is_signed=True)),
        ([-128, 128], False, Integer(9, is_signed=True)),
        ([-13, 4], False, Integer(5, is_signed=True)),
        ([42, 1019], False, Integer(10, is_signed=False)),
    ],
)
def test_make_integer_to_hold(values, force_signed, expected_result):
    """Test make_integer_to_hold"""
    assert expected_result == make_integer_to_hold(values, force_signed)
