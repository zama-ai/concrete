"""
Tests of execution of round bit pattern operation.
"""

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.compilation.configuration import Exactness
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


def test_round_bit_pattern_no_overflow_protection(helpers):
    """
    Test round bit pattern without overflow protection.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=2)
        return rounded**2

    inputset = range(-32, 32)
    configuration = helpers.configuration()
    circuit = function.compile(inputset, configuration)

    expected_mlir = (
        """

module {
  func.func @function(%arg0: !FHE.esint<7>) -> !FHE.eint<11> {
    %0 = "FHE.round"(%arg0) : (!FHE.esint<7>) -> !FHE.esint<5>
    %c2_i3 = arith.constant 2 : i3
    %cst = arith.constant dense<[0, 16, 64, 144, 256, 400, 576, 784, 1024, 1296, 1600, 1936, 2304, 2704, 3136, 3600, 4096, 3600, 3136, 2704, 2304, 1936, 1600, 1296, 1024, 784, 576, 400, 256, 144, 64, 16]> : tensor<32xi64>
    %1 = "FHE.apply_lookup_table"(%0, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<11>
    return %1 : !FHE.eint<11>
  }
}

        """  # noqa: E501
        if not configuration.single_precision
        else """

module {
  func.func @function(%arg0: !FHE.esint<11>) -> !FHE.eint<11> {
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


def test_round_bit_pattern_identity(helpers):
    """
    Test round bit pattern used multiple times outside TLUs.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        rounded = fhe.round_bit_pattern(x, lsbs_to_remove=2, overflow_protection=False)
        return rounded + rounded

    inputset = range(-20, 20)
    configuration = helpers.configuration()
    circuit = function.compile(inputset, configuration)

    expected_mlir = (
        """

module {
  func.func @function(%arg0: !FHE.esint<6>) -> !FHE.esint<7> {
    %0 = "FHE.round"(%arg0) : (!FHE.esint<6>) -> !FHE.esint<4>
    %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, -32, -28, -24, -20, -16, -12, -8, -4]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%0, %cst) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.esint<7>
    %2 = "FHE.add_eint"(%1, %1) : (!FHE.esint<7>, !FHE.esint<7>) -> !FHE.esint<7>
    return %2 : !FHE.esint<7>
  }
}

        """  # noqa: E501
        if not configuration.single_precision
        else """

module {
  func.func @function(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
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

%0 = x                                                                                        # EncryptedScalar<uint8>
%1 = round_bit_pattern(%0, lsbs_to_remove=3, overflow_protection=True, exactness=None)        # EncryptedScalar<uint8>
%2 = tlu(%1, table={table3_formatted_string})                                                 # EncryptedScalar<uint4>
%3 = round_bit_pattern(%2, lsbs_to_remove=2, overflow_protection=True, exactness=None)        # EncryptedScalar<uint4>
%4 = tlu(%3, table={table4_formatted_string})                                                 # EncryptedScalar<uint8>
return %4

        """,  # noqa: E501
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


def test_round_bit_pattern_overflow_to_sign_bit(helpers):
    """
    Test round bit pattern where it's applies to a signed value and p-1 bits are removed.
    """

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.round_bit_pattern(x - 4, lsbs_to_remove=3) // (2**3)

    inputset = range(10)
    configuration = helpers.configuration()

    circuit = function.compile(inputset, configuration)
    circuit.keys.generate()

    for x in inputset:
        helpers.check_execution(circuit, function, x, retries=3)


def test_round_bit_pattern_approximate_enabling(helpers):
    """
    Test round bit pattern various activation paths.
    """

    @fhe.compiler({"x": "encrypted"})
    def function_default(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove=8)

    @fhe.compiler({"x": "encrypted"})
    def function_exact(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove=8, exactness=Exactness.EXACT)

    @fhe.compiler({"x": "encrypted"})
    def function_approx(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove=8, exactness=Exactness.APPROXIMATE)

    inputset = [-(2**10), 2**10 - 1]
    configuration = helpers.configuration()

    circuit_default_default = function_default.compile(inputset, configuration)
    circuit_default_exact = function_default.compile(
        inputset, configuration.fork(rounding_exactness=Exactness.EXACT)
    )
    circuit_default_approx = function_default.compile(
        inputset, configuration.fork(rounding_exactness=Exactness.APPROXIMATE)
    )
    circuit_exact = function_exact.compile(
        inputset, configuration.fork(rounding_exactness=Exactness.APPROXIMATE)
    )
    circuit_approx = function_approx.compile(
        inputset, configuration.fork(rounding_exactness=Exactness.EXACT)
    )

    assert circuit_approx.complexity < circuit_exact.complexity
    assert circuit_exact.complexity == circuit_default_default.complexity
    assert circuit_exact.complexity == circuit_default_exact.complexity
    assert circuit_approx.complexity == circuit_default_approx.complexity


@pytest.mark.parametrize(
    "accumulator_precision,reduced_precision,signed,conf",
    [
        (8, 4, True, fhe.ApproximateRoundingConfig(False, 4)),
        (7, 4, False, fhe.ApproximateRoundingConfig(False, 4)),
        (9, 3, True, fhe.ApproximateRoundingConfig(True, False)),
        (8, 3, False, fhe.ApproximateRoundingConfig(True, False)),
        (7, 3, False, fhe.ApproximateRoundingConfig(True, 3)),
        (7, 2, True, fhe.ApproximateRoundingConfig(False, 2)),
        (7, 2, False, fhe.ApproximateRoundingConfig(False, False, False, False)),
        (8, 1, True, fhe.ApproximateRoundingConfig(False, 1)),
        (8, 1, False, fhe.ApproximateRoundingConfig(True, False)),
        (6, 5, False, fhe.ApproximateRoundingConfig(True, 6)),
        (6, 5, False, fhe.ApproximateRoundingConfig(True, 5)),
    ],
)
def test_round_bit_pattern_approximate_off_by_one_errors(
    accumulator_precision, reduced_precision, signed, conf, helpers
):
    """
    Test round bit pattern off by 1 errors.
    """
    lsbs_to_remove = accumulator_precision - reduced_precision

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        x = fhe.univariate(lambda x: x)(x)
        x = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        x = x // 2**lsbs_to_remove
        return x

    if signed:
        inputset = [-(2 ** (accumulator_precision - 1)), 2 ** (accumulator_precision - 1) - 1]
    else:
        inputset = [0, 2**accumulator_precision - 1]

    configuration = helpers.configuration()
    circuit_exact = function.compile(inputset, configuration)
    circuit_approx = function.compile(
        inputset,
        configuration.fork(
            approximate_rounding_config=conf, rounding_exactness=Exactness.APPROXIMATE
        ),
    )
    # check it's better even with bad conf
    assert circuit_approx.complexity < circuit_exact.complexity

    # avoiding overflows
    if signed:
        testset = [
            -(2 ** (accumulator_precision - 1)),
            2 ** (accumulator_precision - 1) - 2**lsbs_to_remove,
        ]
    else:
        testset = [0, 2**accumulator_precision - 2**lsbs_to_remove]

    nb_error = 0
    for x in testset:
        approx = circuit_approx.encrypt_run_decrypt(x)
        approx_simu = circuit_approx.simulate(x)
        exact = circuit_exact.simulate(x)
        delta_simu = abs(approx_simu - exact)
        delta = abs(approx - exact)
        assert delta_simu <= 1
        assert delta <= 1
        nb_error += delta > 0

    nb_transitions = 2 ** (accumulator_precision - reduced_precision)
    assert nb_error <= 3 * nb_transitions  # of the same order as transitions but small sample size


@pytest.mark.parametrize(
    "signed,physical",
    [(signed, physical) for signed in (True, False) for physical in (True, False)],
)
def test_round_bit_pattern_approximate_clippping(signed, physical, helpers):
    """
    Test round bit pattern clipping.
    """
    accumulator_precision = 6
    reduced_precision = 3
    lsbs_to_remove = accumulator_precision - reduced_precision

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        x = fhe.univariate(lambda x: x)(x)
        x = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
        x = x // 2**lsbs_to_remove
        return x

    if signed:
        input_domain = range(-(2 ** (accumulator_precision - 1)), 2 ** (accumulator_precision - 1))
    else:
        input_domain = range(0, 2 ** (accumulator_precision))

    configuration = helpers.configuration()
    approx_conf = fhe.ApproximateRoundingConfig(
        logical_clipping=not physical,
        approximate_clipping_start_precision=physical and reduced_precision,
        reduce_precision_after_approximate_clipping=False,
    )
    no_clipping_conf = fhe.ApproximateRoundingConfig(
        logical_clipping=False, approximate_clipping_start_precision=False
    )
    assert approx_conf.logical_clipping or approx_conf.approximate_clipping_start_precision
    circuit_clipping = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=approx_conf, rounding_exactness=Exactness.APPROXIMATE
        ),
    )
    circuit_no_clipping = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=no_clipping_conf, rounding_exactness=Exactness.APPROXIMATE
        ),
    )

    if signed:
        clipped_output_domain = range(-(2 ** (reduced_precision - 1)), 2 ** (reduced_precision - 1))
    else:
        clipped_output_domain = range(0, 2**reduced_precision)

    # With clipping
    for x in input_domain:
        assert (
            circuit_clipping.encrypt_run_decrypt(x) in clipped_output_domain
        ), circuit_clipping.mlir  # no overflow
        assert circuit_clipping.simulate(x) in clipped_output_domain

    # Without clipping
    # overflow
    assert circuit_no_clipping.simulate(input_domain[-1]) not in clipped_output_domain


@pytest.mark.parametrize(
    "signed,accumulator_precision",
    [
        (signed, accumulator_precision)
        for signed in (True, False)
        for accumulator_precision in (13, 24)
    ],
)
def test_round_bit_pattern_approximate_acc_to_6_costs(signed, accumulator_precision, helpers):
    """
    Test round bit pattern speedup when approximatipn is activated.
    """
    reduced_precision = 6
    lsbs_to_remove = accumulator_precision - reduced_precision

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        x = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove, overflow_protection=True)
        x = x // 2**lsbs_to_remove
        return x

    # with overflow
    if signed:
        input_domain = [-(2 ** (accumulator_precision - 1)), 2 ** (accumulator_precision - 1) - 1]
    else:
        input_domain = [0, 2 ** (accumulator_precision) - 1]

    configuration = helpers.configuration().fork(
        single_precision=False,
        parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI,
        composable=True,
    )
    circuit_exact = function.compile(input_domain, configuration)
    approx_conf_fastest = fhe.ApproximateRoundingConfig(approximate_clipping_start_precision=6)
    approx_conf_safest = fhe.ApproximateRoundingConfig(approximate_clipping_start_precision=100)
    circuit_approx_fastest = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=approx_conf_fastest,
            rounding_exactness=Exactness.APPROXIMATE,
        ),
    )
    circuit_approx_safest = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=approx_conf_safest, rounding_exactness=Exactness.APPROXIMATE
        ),
    )
    assert circuit_approx_safest.complexity < circuit_exact.complexity
    assert circuit_approx_fastest.complexity < circuit_approx_safest.complexity

    @fhe.compiler({"x": "encrypted"})
    def function(x):  # pylint: disable=function-redefined
        x = fhe.round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove, overflow_protection=False)
        x = x // 2**lsbs_to_remove
        return x

    # without overflow
    if signed:
        input_domain = [-(2 ** (accumulator_precision - 1)), 2 ** (accumulator_precision - 2) - 2]
    else:
        input_domain = [0, 2 ** (accumulator_precision - 1) - 2]

    circuit_exact_no_ovf = function.compile(input_domain, configuration)
    circuit_approx_fastest_no_ovf = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=approx_conf_fastest,
            rounding_exactness=Exactness.APPROXIMATE,
        ),
    )
    circuit_approx_safest_no_ovf = function.compile(
        input_domain,
        configuration.fork(
            approximate_rounding_config=approx_conf_safest, rounding_exactness=Exactness.APPROXIMATE
        ),
    )
    assert circuit_approx_fastest_no_ovf.complexity == circuit_approx_safest_no_ovf.complexity
    assert circuit_approx_safest_no_ovf.complexity < circuit_exact_no_ovf.complexity
    assert circuit_exact_no_ovf.complexity < circuit_exact.complexity
