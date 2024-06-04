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
    "msg_modulus, msg_width, carry_modulus, carry_width, error_msg",
    [
        pytest.param(
            5,
            2,
            4,
            2,
            r"inconsistency between msg_modulus\(5\), and msg_width\(2\). "
            r"msg_modulus should be 2\*\*msg_width",
        ),
        pytest.param(
            4,
            2,
            5,
            2,
            r"inconsistency between carry_modulus\(5\), and carry_width\(2\). "
            r"carry_modulus should be 2\*\*carry_width",
        ),
        pytest.param(
            8,
            4,
            4,
            2,
            r"inconsistency between msg_modulus\(8\), and msg_width\(4\). "
            r"msg_modulus should be 2\*\*msg_width",
        ),
        pytest.param(
            4,
            2,
            64,
            10,
            r"inconsistency between carry_modulus\(64\), and carry_width\(10\). "
            r"carry_modulus should be 2\*\*carry_width",
        ),
    ],
)
def test_bit_widths_inconsistency(msg_modulus, msg_width, carry_modulus, carry_width, error_msg):
    """Test bit widths incosistency"""
    params = default_params()
    params.message_modulus = msg_modulus
    params.carry_modulus = carry_modulus
    with pytest.raises(ValueError, match=error_msg):
        TFHERSIntegerType(False, 16, carry_width, msg_width, params)
