"""
Tests of execution of round bit pattern operation.
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
    assert fhe.round_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove) == expected_output


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
        fhe.round_bit_pattern(sample, lsbs_to_remove=lsbs_to_remove)

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
@pytest.mark.parametrize(
    "overflow",
    [
        True,
        False,
    ],
)
def test_round_bit_pattern(input_bits, lsbs_to_remove, mapper, overflow, helpers):
    """
    Test round bit pattern.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        x_rounded = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return mapper(x_rounded)

    upper_bound = (2**input_bits) - (2 ** (lsbs_to_remove - 1))
    inputset = [0, upper_bound - 1]

    if overflow:
        upper_bound = 2**input_bits
        inputset.append(upper_bound - 1)

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
def test_round_bit_pattern_unsigned_range(input_bits, lsbs_to_remove, helpers):
    """
    Test round bit pattern in unsigned range.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)

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
def test_round_bit_pattern_signed_range(input_bits, lsbs_to_remove, helpers):
    """
    Test round bit pattern in signed range.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)

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
def test_round_bit_pattern_unsigned_range_assigned(input_bits, lsbs_to_remove, helpers):
    """
    Test round bit pattern in unsigned range with a big bit-width assigned.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return (rounded**2) + (50_000 - x)

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
def test_round_bit_pattern_signed_range_assigned(input_bits, lsbs_to_remove, helpers):
    """
    Test round bit pattern in signed range with a big bit-width assigned.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        return (rounded**2) + (50_000 - x)

    inputset = range(-(2 ** (input_bits - 1)), 2 ** (input_bits - 1))
    circuit = function.compile(inputset, helpers.configuration())

    for value in inputset:
        helpers.check_execution(circuit, function, value, retries=3)


def test_round_bit_pattern_no_overflow_protection(helpers, pytestconfig):
    """
    Test round bit pattern without overflow protection.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=2)
        return rounded**2

    inputset = range(-32, 32)
    circuit = function.compile(inputset, helpers.configuration())

    expected_mlir = (
        """

module {
  func.func @main(%arg0: !FHE.esint<7>) -> !FHE.eint<11> {
    %0 = "FHE.round"(%arg0) : (!FHE.esint<7>) -> !FHE.esint<5>
    %c2_i3 = arith.constant 2 : i3
    %cst = arith.constant dense<[0, 16, 64, 144, 256, 400, 576, 784, 1024, 1296, 1600, 1936, 2304, 2704, 3136, 3600, 4096, 3600, 3136, 2704, 2304, 1936, 1600, 1296, 1024, 784, 576, 400, 256, 144, 64, 16]> : tensor<32xi64>
    %1 = "FHE.apply_lookup_table"(%0, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<11>
    return %1 : !FHE.eint<11>
  }
}

        """  # noqa: E501
        if pytestconfig.getoption("precision") == "multi"
        else """

module {
  func.func @main(%arg0: !FHE.esint<11>) -> !FHE.eint<11> {
    %c16_i12 = arith.constant 16 : i12
    %0 = "FHE.mul_eint_int"(%arg0, %c16_i12) : (!FHE.esint<11>, i12) -> !FHE.esint<11>
    %1 = "FHE.round"(%0) : (!FHE.esint<11>) -> !FHE.esint<5>
    %c2_i12 = arith.constant 2 : i12
    %cst = arith.constant dense<[0, 16, 64, 144, 256, 400, 576, 784, 1024, 1296, 1600, 1936, 2304, 2704, 3136, 3600, 4096, 3600, 3136, 2704, 2304, 1936, 1600, 1296, 1024, 784, 576, 400, 256, 144, 64, 16]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<11>
    return %2 : !FHE.eint<11>
  }
}

        """  # noqa: E501
    )

    helpers.check_str(expected_mlir, circuit.mlir)


def test_round_bit_pattern_identity(helpers, pytestconfig):
    """
    Test round bit pattern used multiple times outside TLUs.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=2, overflow_protection=False)
        return rounded + rounded

    inputset = range(-20, 20)
    circuit = function.compile(inputset, helpers.configuration())

    expected_mlir = (
        """

module {
  func.func @main(%arg0: !FHE.esint<6>) -> !FHE.esint<7> {
    %0 = "FHE.round"(%arg0) : (!FHE.esint<6>) -> !FHE.esint<4>
    %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, -32, -28, -24, -20, -16, -12, -8, -4]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%0, %cst) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.esint<7>
    %2 = "FHE.add_eint"(%1, %1) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    return %2 : !FHE.esint<7>
  }
}

        """  # noqa: E501
        if pytestconfig.getoption("precision") == "multi"
        else """

module {
  func.func @main(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
    %c2_i8 = arith.constant 2 : i8
    %0 = "FHE.mul_eint_int"(%arg0, %c2_i8) : (!FHE.esint<7>, i8) -> !FHE.esint<7>
    %1 = "FHE.round"(%0) : (!FHE.esint<7>) -> !FHE.esint<4>
    %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, -32, -28, -24, -20, -16, -12, -8, -4]> : tensor<16xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.esint<7>
    %3 = "FHE.add_eint"(%2, %2) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    return %3 : !FHE.esint<7>
  }
}

        """  # noqa: E501
    )

    helpers.check_str(expected_mlir, circuit.mlir)


def test_auto_rounding(helpers):
    """
    Test round bit pattern with auto rounding.
    """

    # with auto adjust rounders configuration
    # ---------------------------------------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 5 bits, which means we need to remove 6 of the least significant bits

    rounder1 = fhe.AutoRounder(target_msbs=5)

    @fhe.compiler({"x": "encrypted"})
    def function1(x):
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder1)
        return np.sqrt(z).astype(np.int64)

    inputset1 = range(1000)
    function1.trace(inputset1, helpers.configuration(), auto_adjust_rounders=True)

    assert rounder1.lsbs_to_remove == 6

    # manual
    # ------

    # y has the max value of 1999, so it's 11 bits
    # our target msb is 3 bits, which means we need to remove 8 of the least significant bits

    rounder2 = fhe.AutoRounder(target_msbs=3)

    @fhe.compiler({"x": "encrypted"})
    def function2(x):
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder2)
        return np.sqrt(z).astype(np.int64)

    inputset2 = range(1000)
    fhe.AutoRounder.adjust(function2, inputset2)

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
    table3 = fhe.LookupTable(entries3)
    # and this is the rounder for table1, which should have lsbs_to_remove of 3
    rounder3 = fhe.AutoRounder(target_msbs=5)

    # have 2 ** 8 entries during evaluation, it won't matter after compilation
    entries4 = list(range(2**8))
    # we have 4-bit inputs for this table, and we only want to use first 2-bits
    for i in range(0, 2**4, 2**2):
        # so we set every 4th entry to an 8-bit value
        entries4[i] = np.random.randint(2**7, 2**8)
    # when this tlu is applied to a 4-bit value with 2-bit msb rounding, result will be 8-bits
    table4 = fhe.LookupTable(entries4)
    # and this is the rounder for table2, which should have lsbs_to_remove of 2
    rounder4 = fhe.AutoRounder(target_msbs=2)

    @fhe.compiler({"x": "encrypted"})
    def function3(x):
        a = fhe.round_bit_pattern(x, lsbs_to_remove=rounder3)
        b = table3[a]
        c = fhe.round_bit_pattern(b, lsbs_to_remove=rounder4)
        d = table4[c]
        return d

    inputset3 = range((2**8) - (2**3))
    circuit3 = function3.compile(
        inputset3,
        helpers.configuration(),
        auto_adjust_rounders=True,
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

    rounder = fhe.AutoRounder(target_msbs=5)

    def function(x):
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder)
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

    rounder = fhe.AutoRounder(target_msbs=5)

    def function(x):
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(ValueError) as excinfo:
        fhe.AutoRounder.adjust(function, [])

    assert str(excinfo.value) == "AutoRounders cannot be adjusted with an empty inputset"


def test_auto_rounding_recursive_adjustment():
    """
    Test round bit pattern with auto rounding but with recursive adjustment.
    """

    rounder = fhe.AutoRounder(target_msbs=5)

    def function(x):
        fhe.AutoRounder.adjust(function, range(10))
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder)
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        fhe.AutoRounder.adjust(function, range(10))

    assert str(excinfo.value) == "AutoRounders cannot be adjusted recursively"


def test_auto_rounding_construct_in_function():
    """
    Test round bit pattern with auto rounding but rounder is constructed within the function.
    """

    def function(x):
        y = x + 1000
        z = fhe.round_bit_pattern(y, lsbs_to_remove=fhe.AutoRounder(target_msbs=5))
        return np.sqrt(z).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        fhe.AutoRounder.adjust(function, range(10))

    assert str(excinfo.value) == (
        "AutoRounders cannot be constructed during adjustment, "
        "please construct AutoRounders outside the function and reference it"
    )


def test_overflowing_round_bit_pattern_with_lsbs_to_remove_of_one(helpers):
    """
    Test round bit pattern where overflow is detected when only one bit is to be removed.
    """

    # pylint: disable=invalid-name

    def subgraph(v0):
        v1 = v0.astype(np.float32)
        v2 = 0
        v3 = np.add(v1, v2)
        v4 = np.array([[3, -4, -9, -12, 5]])
        v5 = np.subtract(v3, v4)
        v6 = 0.8477082011119198
        v7 = np.multiply(v6, v5)
        v8 = [-0.16787443, -0.4266992, 0.33739513, 0.15412766, -0.39808342]
        v9 = np.add(v7, v8)
        v10 = 0.16666666666666666
        v11 = np.multiply(v10, v9)
        v12 = 0.5
        v13 = np.add(v11, v12)
        v14 = 1
        v15 = np.minimum(v14, v13)
        v16 = 0
        v17 = np.maximum(v16, v15)
        v18 = np.multiply(v9, v17)
        v19 = 7.667494564113974
        v20 = np.divide(v18, v19)
        v21 = -8
        v22 = np.add(v20, v21)
        v23 = np.rint(v22)
        v24 = -8
        v25 = 7
        v26 = np.clip(v23, v24, v25)
        v27 = v26.astype(np.int64)
        return v27

    def function(v0):
        v1 = np.array(
            [
                [2, -5, 1, 6, -4],
                [2, 7, 6, -3, 5],
                [0, -3, 4, -2, -5],
                [-5, 5, -4, 7, 3],
                [-2, 0, 2, 4, -4],
            ]
        )
        v2 = np.matmul(v0, v1)
        v3 = fhe.round_bit_pattern(v2, lsbs_to_remove=1)
        v4 = subgraph(v3)
        v5 = np.array(
            [
                [2, -3, 5, 2, -5],
                [-6, -1, -2, 7, 4],
                [-6, 6, -6, 3, 3],
                [-7, -1, -5, 1, 1],
                [-3, -1, 5, -1, -5],
            ]
        )
        v6 = np.matmul(v4, v5)
        return v6

    # pylint: enable=invalid-name

    sample = np.array([[-7, 5, -8, -8, -8]])

    compiler = fhe.Compiler(function, {"v0": "encrypted"})
    circuit = compiler.compile(inputset=[sample], configuration=helpers.configuration())

    helpers.check_execution(circuit, function, sample, retries=5)
