"""
Tests of tfhers data type.
"""

import pytest

from concrete.fhe.tfhers.dtypes import TFHERSIntegerType, TFHERSParams


def default_params() -> TFHERSParams:
    """Default tfhers params used for testing."""
    return TFHERSParams(
        761,
        1,
        2048,
        6.36835566258815e-06,
        3.1529322391500584e-16,
        23,
        1,
        3,
        5,
        4,
        4,
        5,
        -40.05,
        None,
        True,
    )


@pytest.mark.parametrize(
    "msg_modulus, msg_width, carry_modulus, carry_width",
    [
        pytest.param(5, 2, 4, 2),
        pytest.param(4, 2, 5, 2),
        pytest.param(8, 4, 4, 2),
        pytest.param(4, 2, 64, 10),
    ],
)
def test_bit_widths_inconsistency(msg_modulus, msg_width, carry_modulus, carry_width):
    """Test bit widths incosistency"""
    params = default_params()
    params.message_modulus = msg_modulus
    params.carry_modulus = carry_modulus
    with pytest.raises(ValueError, match="inconsistency"):
        TFHERSIntegerType(False, 16, carry_width, msg_width, params)
