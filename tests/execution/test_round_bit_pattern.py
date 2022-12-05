"""
Tests of execution of round bit pattern operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp
from concrete.numpy.representation.utils import format_constant


@pytest.mark.parametrize(
    "sample,lsbs_to_remove,expected_output",
    [
        (0b_0000_0011, 0, 0b_0000_0011),
        (0b_0000_0100, 0, 0b_0000_0100),
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
    assert cnp.round_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove) == expected_output


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
def test_bad_plain_round_bit_pattern(
    sample,
    lsbs_to_remove,
    expected_error,
    expected_message,
):
    """
    Test round bit pattern in evaluation context with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        cnp.round_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove)

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
        x_rounded = cnp.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return np.abs(50 * np.sin(x_rounded)).astype(np.int64)

    circuit = function.compile([(2**input_bits) - 1], helpers.configuration(), virtual=True)
    helpers.check_execution(circuit, function, np.random.randint(0, 2**input_bits))


def test_auto_rounding(helpers):
    """
    Test round bit pattern with auto rounding.
    """

    # with auto adjust rounders configuration
    # ---------------------------------------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 5 bits, which means we need to remove 6 of the least significant bits

    rounder1 = cnp.AutoRounder(target_msbs=5)

    @cnp.compiler({"x": "encrypted"})
    def function1(x):
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=rounder1)
        return np.sqrt(z).astype(np.int64)

    inputset1 = range(1000)
    function1.trace(inputset1, helpers.configuration(), auto_adjust_rounders=True)

    assert rounder1.lsbs_to_remove == 6

    # manual
    # ------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 3 bits, which means we need to remove 8 of the least significant bits

    rounder2 = cnp.AutoRounder(target_msbs=3)

    @cnp.compiler({"x": "encrypted"})
    def function2(x):
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=rounder2)
        return np.sqrt(z).astype(np.int64)

    inputset2 = range(1000)
    cnp.AutoRounder.adjust(function2, inputset2)

    assert rounder2.lsbs_to_remove == 8

    # complicated case
    # ----------------

    # have 2 ** 8 entries during evaluation, it won't matter after compilation
    entries3 = list(range(2**8))
    # we have 8-bit inputs for this table, and we only want to use first 5-bits
    for i in range(0, 2**8, 2**3):
        # so we set every 8th entry to a 4-bit value
        entries3[i] = np.random.randint(0, (2**4) - (2**2))
    # when this tlu is applied to an 8-bit value with 5-bit msb rounding, result will be 4-bits
    table3 = cnp.LookupTable(entries3)
    # and this is the rounder for table1, which should have lsbs_to_remove of 3
    rounder3 = cnp.AutoRounder(target_msbs=5)

    # have 2 ** 8 entries during evaluation, it won't matter after compilation
    entries4 = list(range(2**8))
    # we have 4-bit inputs for this table, and we only want to use first 2-bits
    for i in range(0, 2**4, 2**2):
        # so we set every 4th entry to an 8-bit value
        entries4[i] = np.random.randint(2**7, 2**8)
    # when this tlu is applied to a 4-bit value with 2-bit msb rounding, result will be 8-bits
    table4 = cnp.LookupTable(entries4)
    # and this is the rounder for table2, which should have lsbs_to_remove of 2
    rounder4 = cnp.AutoRounder(target_msbs=2)

    @cnp.compiler({"x": "encrypted"})
    def function3(x):
        a = cnp.round_bit_pattern(x, lsbs_to_remove=rounder3)
        b = table3[a]
        c = cnp.round_bit_pattern(b, lsbs_to_remove=rounder4)
        d = table4[c]
        return d

    inputset3 = range((2**8) - (2**3))
    circuit3 = function3.compile(
        inputset3,
        helpers.configuration(),
        auto_adjust_rounders=True,
        virtual=True,
    )

    assert rounder3.lsbs_to_remove == 3
    assert rounder4.lsbs_to_remove == 2

    table3_formatted_string = format_constant(table3.table, 25)
    table4_formatted_string = format_constant(table4.table, 25)

    helpers.check_str(
        f"""

%0 = x                                               # EncryptedScalar<uint8>
%1 = round_bit_pattern(%0, lsbs_to_remove=3)         # EncryptedScalar<uint8>
%2 = tlu(%1, table={table3_formatted_string})        # EncryptedScalar<uint4>
%3 = round_bit_pattern(%2, lsbs_to_remove=2)         # EncryptedScalar<uint4>
%4 = tlu(%3, table={table4_formatted_string})        # EncryptedScalar<uint8>
return %4

        """,
        str(circuit3.graph.format(show_bounds=False)),
    )


def test_auto_rounding_without_adjustment():
    """
    Test round bit pattern with auto rounding but without adjustment.
    """

    rounder = cnp.AutoRounder(target_msbs=5)

    def function(x):
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=rounder)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        function(100)

    assert str(excinfo.value) == (
        "AutoRounders cannot be used before adjustment, "
        "please call AutoRounder.adjust with the function that will be compiled "
        "and provide the exact inputset that will be used for compilation"
    )


def test_auto_rounding_with_empty_inputset():
    """
    Test round bit pattern with auto rounding but with empty inputset.
    """

    rounder = cnp.AutoRounder(target_msbs=5)

    def function(x):
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=rounder)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(ValueError) as excinfo:
        cnp.AutoRounder.adjust(function, [])

    assert str(excinfo.value) == "AutoRounders cannot be adjusted with an empty inputset"


def test_auto_rounding_recursive_adjustment():
    """
    Test round bit pattern with auto rounding but with recursive adjustment.
    """

    rounder = cnp.AutoRounder(target_msbs=5)

    def function(x):
        cnp.AutoRounder.adjust(function, range(10))
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=rounder)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        cnp.AutoRounder.adjust(function, range(10))

    assert str(excinfo.value) == "AutoRounders cannot be adjusted recursively"


def test_auto_rounding_construct_in_function():
    """
    Test round bit pattern with auto rounding but rounder is constructed within the function.
    """

    def function(x):
        y = x + 1000
        z = cnp.round_bit_pattern(y, lsbs_to_remove=cnp.AutoRounder(target_msbs=5))
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        cnp.AutoRounder.adjust(function, range(10))

    assert str(excinfo.value) == (
        "AutoRounders cannot be constructed during adjustment, "
        "please construct AutoRounders outside the function and reference it"
    )
