"""Test file for float data types"""


import pytest

from concrete.common.data_types.floats import Float, Float32, Float64


@pytest.mark.parametrize(
    "float_,expected_repr_str",
    [
        pytest.param(
            Float32(),
            "Float<32 bits>",
            id="Float32",
        ),
        pytest.param(
            Float(32),
            "Float<32 bits>",
            id="32 bits Float",
        ),
        pytest.param(
            Float64(),
            "Float<64 bits>",
            id="Float64",
        ),
        pytest.param(
            Float(64),
            "Float<64 bits>",
            id="64 bits Float",
        ),
    ],
)
def test_floats_repr(float_: Float, expected_repr_str: str):
    """Test float repr"""
    assert float_.__repr__() == expected_repr_str


@pytest.mark.parametrize(
    "float_1,float_2,expected_equal",
    [
        pytest.param(Float32(), Float(32), True),
        pytest.param(Float(64), Float32(), False),
        pytest.param(Float64(), Float(64), True),
    ],
)
def test_floats_eq(float_1: Float, float_2: Float, expected_equal: bool):
    """Test float eq"""
    assert expected_equal == (float_1 == float_2)
    assert expected_equal == (float_2 == float_1)
