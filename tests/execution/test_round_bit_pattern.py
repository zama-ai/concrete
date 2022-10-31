"""
Tests of execution of round bit pattern operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "sample,lsbs_to_remove,expected_output",
    [
        (0b_0000_0000, 3, 0b_0000_0000),
        (0b_0000_0001, 3, 0b_0000_0000),
        (0b_0000_0010, 3, 0b_0000_0000),
        (0b_0000_0011, 3, 0b_0000_0000),
        (0b_0000_0100, 3, 0b_0000_1000),
        (0b_0000_0101, 3, 0b_0000_1000),
        (0b_0000_0110, 3, 0b_0000_1000),
        (0b_0000_0111, 3, 0b_0000_1000),
        (0b_0000_1000, 3, 0b_0000_1000),
        (0b_0000_1001, 3, 0b_0000_1000),
        (0b_0000_1010, 3, 0b_0000_1000),
        (0b_0000_1011, 3, 0b_0000_1000),
        (0b_0000_1100, 3, 0b_0001_0000),
        (0b_0000_1101, 3, 0b_0001_0000),
        (0b_0000_1110, 3, 0b_0001_0000),
        (0b_0000_1111, 3, 0b_0001_0000),
    ],
)
def test_plain_round_bit_pattern(sample, lsbs_to_remove, expected_output):
    """
    Test round bit pattern in evaluation context.
    """
    assert cnp.round_bit_pattern(sample, lsbs_to_remove) == expected_output


@pytest.mark.parametrize(
    "sample,lsbs_to_remove,expected_error,expected_message",
    [
        (
            np.array([3.2, 4.1]),
            3,
            TypeError,
            "Expected input elements to be integers but they are dtype[float64]",
        ),
        (
            "foo",
            3,
            TypeError,
            "Expected input to be an int or a numpy array but it's str",
        ),
    ],
)
def test_bad_plain_round_bit_pattern(sample, lsbs_to_remove, expected_error, expected_message):
    """
    Test round bit pattern in evaluation context with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        cnp.round_bit_pattern(sample, lsbs_to_remove)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "input_bits,lsbs_to_remove",
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
    ],
)
def test_round_bit_pattern(input_bits, lsbs_to_remove, helpers):
    """
    Test round bit pattern in evaluation context.
    """

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return np.abs(50 * np.sin(cnp.round_bit_pattern(x, lsbs_to_remove))).astype(np.int64)

    circuit = function.compile([(2**input_bits) - 1], helpers.configuration(), virtual=True)
    helpers.check_execution(circuit, function, np.random.randint(0, 2**input_bits))
