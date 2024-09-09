"""
Tests of execution of truncate bit pattern operation.
"""

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.representation.utils import format_constant


@pytest.mark.parametrize(
    "sample,lsbs_to_remove,expected_output",
    [
        (0b_0000_0011, 0, 0b_0000_0011),
        (0b_0000_0100, 0, 0b_0000_0100),
        (0b_0000_0000, 3, 0b_0000_0000),
        (0b_0000_0001, 3, 0b_0000_0000),
        (0b_0000_0010, 3, 0b_0000_0000),
        (0b_0000_0011, 3, 0b_0000_0000),
        (0b_0000_0100, 3, 0b_0000_0000),
        (0b_0000_0101, 3, 0b_0000_0000),
        (0b_0000_0110, 3, 0b_0000_0000),
        (0b_0000_0111, 3, 0b_0000_0000),
        (0b_0000_1000, 3, 0b_0000_1000),
        (0b_0000_1001, 3, 0b_0000_1000),
        (0b_0000_1010, 3, 0b_0000_1000),
        (0b_0000_1011, 3, 0b_0000_1000),
        (0b_0000_1100, 3, 0b_0000_1000),
        (0b_0000_1101, 3, 0b_0000_1000),
        (0b_0000_1110, 3, 0b_0000_1000),
        (0b_0000_1111, 3, 0b_0000_1000),
    ],
)
def test_plain_truncate_bit_pattern(sample, lsbs_to_remove, expected_output):
    """
    Test truncate bit pattern in evaluation context.
    """
    assert fhe.truncate_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove) == expected_output


@pytest.mark.parametrize(
    "sample,lsbs_to_remove,expected_error,expected_message",
    [
        (
            np.array([3.2, 4.1]),
            3,
            TypeError,
            f"Expected input elements to be integers but they are {type(np.array([3.2, 4.1]).dtype).__name__}",  # noqa: E501
        ),
        (
            "foo",
            3,
            TypeError,
            "Expected input to be an int or a numpy array but it's str",
        ),
    ],
)
def test_bad_plain_truncate_bit_pattern(
    sample,
    lsbs_to_remove,
    expected_error,
    expected_message,
):
    """
    Test truncate bit pattern in evaluation context with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        fhe.truncate_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove)

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
@pytest.mark.parametrize(
    "mapper",
    [
        pytest.param(
            lambda x: x,
            id="x",
        ),
        pytest.param(
            lambda x: x + 10,
            id="x + 10",
        ),
        pytest.param(
            lambda x: x**2,
            id="x ** 2",
        ),
        pytest.param(
            lambda x: fhe.univariate(lambda x: x if x >= 0 else 0)(x),
            id="relu",
        ),
    ],
)
def test_truncate_bit_pattern(input_bits, lsbs_to_remove, mapper, helpers):
    """
    Test truncate bit pattern.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        x_truncated = fhe.truncate_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return mapper(x_truncated)

    upper_bound = 2**input_bits
    inputset = [0, upper_bound - 1]

    circuit = function.compile(inputset, helpers.configuration())
    helpers.check_execution(circuit, function, np.random.randint(0, upper_bound), retries=3)

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


@pytest.mark.parametrize(
    "input_bits,lsbs_to_remove",
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_truncate_bit_pattern_unsigned_range(input_bits, lsbs_to_remove, helpers):
    """
    Test truncate bit pattern in unsigned range.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.truncate_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)

    inputset = range(0, 2**input_bits)
    circuit = function.compile(inputset, helpers.configuration())

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


@pytest.mark.parametrize(
    "input_bits,lsbs_to_remove",
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_truncate_bit_pattern_signed_range(input_bits, lsbs_to_remove, helpers):
    """
    Test truncate bit pattern in signed range.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.truncate_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)

    inputset = range(-(2 ** (input_bits - 1)), 2 ** (input_bits - 1))
    circuit = function.compile(inputset, helpers.configuration())

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


@pytest.mark.parametrize(
    "input_bits,lsbs_to_remove",
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_truncate_bit_pattern_unsigned_range_assigned(input_bits, lsbs_to_remove, helpers):
    """
    Test truncate bit pattern in unsigned range with a big bit-width assigned.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        truncated = fhe.truncate_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return (truncated**2) + (63 - x)

    inputset = range(0, 2**input_bits)
    circuit = function.compile(inputset, helpers.configuration())

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


@pytest.mark.parametrize(
    "input_bits,lsbs_to_remove",
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_truncate_bit_pattern_signed_range_assigned(input_bits, lsbs_to_remove, helpers):
    """
    Test truncate bit pattern in signed range with a big bit-width assigned.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        truncated = fhe.truncate_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return (truncated**2) + (63 - x)

    inputset = range(-(2 ** (input_bits - 1)), 2 ** (input_bits - 1))
    circuit = function.compile(inputset, helpers.configuration())

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


def test_truncate_bit_pattern_identity(helpers, pytestconfig):
    """
    Test truncate bit pattern used multiple times outside TLUs.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        truncated = fhe.truncate_bit_pattern(x, lsbs_to_remove=2)
        return truncated + truncated

    inputset = range(-20, 20)
    circuit = function.compile(inputset, helpers.configuration())

    expected_mlir = (
        """

module {
  func.func @function(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
    %0 = "FHE.lsb"(%arg0) : (!FHE.esint<7>) -> !FHE.esint<7>
    %1 = "FHE.sub_eint"(%arg0, %0) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    %2 = "FHE.reinterpret_precision"(%1) : (!FHE.esint<7>) -> !FHE.esint<6>
    %3 = "FHE.lsb"(%2) : (!FHE.esint<6>) -> !FHE.esint<6>
    %4 = "FHE.sub_eint"(%2, %3) : (!FHE.esint<6>, !FHE.esint<6>) -> !FHE.esint<6>
    %5 = "FHE.reinterpret_precision"(%4) : (!FHE.esint<6>) -> !FHE.esint<7>
    %6 = "FHE.add_eint"(%5, %5) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    return %6 : !FHE.esint<7>
  }
}

        """  # noqa: E501
        if pytestconfig.getoption("precision") == "multi"
        else """

module {
  func.func @function(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
    %0 = "FHE.lsb"(%arg0) : (!FHE.esint<7>) -> !FHE.esint<7>
    %1 = "FHE.sub_eint"(%arg0, %0) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    %2 = "FHE.reinterpret_precision"(%1) : (!FHE.esint<7>) -> !FHE.esint<6>
    %3 = "FHE.lsb"(%2) : (!FHE.esint<6>) -> !FHE.esint<6>
    %4 = "FHE.sub_eint"(%2, %3) : (!FHE.esint<6>, !FHE.esint<6>) -> !FHE.esint<6>
    %5 = "FHE.reinterpret_precision"(%4) : (!FHE.esint<6>) -> !FHE.esint<7>
    %6 = "FHE.add_eint"(%5, %5) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    return %6 : !FHE.esint<7>
  }
}

        """  # noqa: E501
    )

    helpers.check_str(expected_mlir, circuit.mlir)


def test_auto_truncating(helpers):
    """
    Test truncate bit pattern with auto truncating.
    """

    # with auto adjust truncators configuration
    # ---------------------------------------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 5 bits, which means we need to remove 6 of the least significant bits

    truncator1 = fhe.AutoTruncator(target_msbs=5)

    @fhe.compiler({"x": "encrypted"})
    def function1(x):
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator1)
        return np.sqrt(z).astype(np.int64)

    inputset1 = range(1000)
    function1.trace(inputset1, helpers.configuration(), auto_adjust_truncators=True)

    assert truncator1.lsbs_to_remove == 6

    # manual
    # ------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 3 bits, which means we need to remove 8 of the least significant bits

    truncator2 = fhe.AutoTruncator(target_msbs=3)

    @fhe.compiler({"x": "encrypted"})
    def function2(x):
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator2)
        return np.sqrt(z).astype(np.int64)

    inputset2 = range(1000)
    fhe.AutoTruncator.adjust(function2, inputset2)

    assert truncator2.lsbs_to_remove == 8

    # complicated case
    # ----------------

    # have 2 ** 8 entries during evaluation, it won't matter after compilation
    entries3 = list(range(2**8))
    # we have 8-bit inputs for this table, and we only want to use first 5-bits
    for i in range(0, 2**8, 2**3):
        # so we set every 8th entry to a 4-bit value
        entries3[i] = np.random.randint(0, (2**4) - (2**2))
    # when this tlu is applied to an 8-bit value with 5-bit msb truncating, result will be 4-bits
    table3 = fhe.LookupTable(entries3)
    # and this is the truncator for table1, which should have lsbs_to_remove of 3
    truncator3 = fhe.AutoTruncator(target_msbs=5)

    # have 2 ** 8 entries during evaluation, it won't matter after compilation
    entries4 = list(range(2**8))
    # we have 4-bit inputs for this table, and we only want to use first 2-bits
    for i in range(0, 2**4, 2**2):
        # so we set every 4th entry to an 8-bit value
        entries4[i] = np.random.randint(2**7, 2**8)
    # when this tlu is applied to a 4-bit value with 2-bit msb truncating, result will be 8-bits
    table4 = fhe.LookupTable(entries4)
    # and this is the truncator for table2, which should have lsbs_to_remove of 2
    truncator4 = fhe.AutoTruncator(target_msbs=2)

    @fhe.compiler({"x": "encrypted"})
    def function3(x):
        a = fhe.truncate_bit_pattern(x, lsbs_to_remove=truncator3)
        b = table3[a]
        c = fhe.truncate_bit_pattern(b, lsbs_to_remove=truncator4)
        d = table4[c]
        return d

    inputset3 = range((2**8) - (2**3))
    circuit3 = function3.compile(
        inputset3,
        helpers.configuration(),
        auto_adjust_truncators=True,
    )

    assert truncator3.lsbs_to_remove == 3
    assert truncator4.lsbs_to_remove == 2

    table3_formatted_string = format_constant(table3.table, 25)
    table4_formatted_string = format_constant(table4.table, 25)

    helpers.check_str(
        f"""

%0 = x                                                 # EncryptedScalar<uint8>
%1 = truncate_bit_pattern(%0, lsbs_to_remove=3)        # EncryptedScalar<uint8>
%2 = tlu(%1, table={table3_formatted_string})          # EncryptedScalar<uint4>
%3 = truncate_bit_pattern(%2, lsbs_to_remove=2)        # EncryptedScalar<uint4>
%4 = tlu(%3, table={table4_formatted_string})          # EncryptedScalar<uint8>
return %4

        """,
        str(circuit3.graph.format(show_bounds=False)),
    )


def test_auto_truncating_without_adjustment():
    """
    Test truncate bit pattern with auto truncating but without adjustment.
    """

    truncator = fhe.AutoTruncator(target_msbs=5)

    def function(x):
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        function(100)

    assert str(excinfo.value) == (
        "AutoTruncators cannot be used before adjustment, "
        "please call AutoTruncator.adjust with the function that will be compiled "
        "and provide the exact inputset that will be used for compilation"
    )


def test_auto_truncating_with_empty_inputset():
    """
    Test truncate bit pattern with auto truncating but with empty inputset.
    """

    truncator = fhe.AutoTruncator(target_msbs=5)

    def function(x):
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(ValueError) as excinfo:
        fhe.AutoTruncator.adjust(function, [])

    assert str(excinfo.value) == "AutoTruncators cannot be adjusted with an empty inputset"


def test_auto_truncating_recursive_adjustment():
    """
    Test truncate bit pattern with auto truncating but with recursive adjustment.
    """

    truncator = fhe.AutoTruncator(target_msbs=5)

    def function(x):
        fhe.AutoTruncator.adjust(function, range(10))
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        fhe.AutoTruncator.adjust(function, range(10))

    assert str(excinfo.value) == "AutoTruncators cannot be adjusted recursively"


def test_auto_truncating_construct_in_function():
    """
    Test truncate bit pattern with auto truncating but truncator is constructed within the function.
    """

    def function(x):
        y = x + 1000
        z = fhe.truncate_bit_pattern(y, lsbs_to_remove=fhe.AutoTruncator(target_msbs=5))
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        fhe.AutoTruncator.adjust(function, range(10))

    assert str(excinfo.value) == (
        "AutoTruncators cannot be constructed during adjustment, "
        "please construct AutoTruncators outside the function and reference it"
    )
