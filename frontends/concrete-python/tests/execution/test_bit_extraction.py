"""
Tests of execution of bit extraction.
"""

import random

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "sample,operation,expected_output",
    [
        (0b_0110_0100, lambda x: fhe.bits(x)[0], 0b_0),
        (0b_0110_0100, lambda x: fhe.bits(x)[1], 0b_0),
        (0b_0110_0100, lambda x: fhe.bits(x)[2], 0b_1),
        (0b_0110_0100, lambda x: fhe.bits(x)[3], 0b_0),
        (0b_0110_0100, lambda x: fhe.bits(x)[4], 0b_0),
        (0b_0110_0100, lambda x: fhe.bits(x)[5], 0b_1),
        (0b_0110_0100, lambda x: fhe.bits(x)[6], 0b_1),
        (0b_0110_0100, lambda x: fhe.bits(x)[7], 0b_0),
        (0b_0110_0100, lambda x: fhe.bits(x)[30], 0b_0),
        # --------------------------------------------
        (0b_1001_1011, lambda x: fhe.bits(x)[0], 0b_1),
        (0b_1001_1011, lambda x: fhe.bits(x)[1], 0b_1),
        (0b_1001_1011, lambda x: fhe.bits(x)[2], 0b_0),
        (0b_1001_1011, lambda x: fhe.bits(x)[3], 0b_1),
        (0b_1001_1011, lambda x: fhe.bits(x)[4], 0b_1),
        (0b_1001_1011, lambda x: fhe.bits(x)[5], 0b_0),
        (0b_1001_1011, lambda x: fhe.bits(x)[6], 0b_0),
        (0b_1001_1011, lambda x: fhe.bits(x)[7], 0b_1),
        (0b_1001_1011, lambda x: fhe.bits(x)[30], 0b_0),
        # --------------------------------------------
        (0b_0110_0100, lambda x: fhe.bits(x)[1:3], 0b_10),
        (0b_0110_0100, lambda x: fhe.bits(x)[:3], 0b_100),
        (0b_0110_0100, lambda x: fhe.bits(x)[1:], 0b_0110_010),
        (0b_0110_0100, lambda x: fhe.bits(x)[1:6:2], 0b_100),
        (0b_0110_0100, lambda x: fhe.bits(x)[3:1:-1], 0b_10),
        (0b_0110_0100, lambda x: fhe.bits(x)[2::-1], 0b_001),
        (0b_0110_0100, lambda x: fhe.bits(x)[30:32], 0b_00),
        # --------------------------------------------
        (0b_1001_1011, lambda x: fhe.bits(x)[1:3], 0b_01),
        (0b_1001_1011, lambda x: fhe.bits(x)[:3], 0b_011),
        (0b_1001_1011, lambda x: fhe.bits(x)[1:], 0b_1001_101),
        (0b_1001_1011, lambda x: fhe.bits(x)[1:6:2], 0b_011),
        (0b_1001_1011, lambda x: fhe.bits(x)[3:1:-1], 0b_01),
        (0b_1001_1011, lambda x: fhe.bits(x)[2::-1], 0b_110),
        (0b_1001_1011, lambda x: fhe.bits(x)[30:32], 0b_00),
    ],
)
def test_plain_bit_extraction(sample, operation, expected_output):
    """
    Test plain bit extraction.
    """

    assert operation(sample) == expected_output


@pytest.mark.parametrize(
    "sample,operation,expected_error,expected_message",
    [
        (
            100,
            lambda x: fhe.bits(x)[1.1:3],  # type: ignore
            ValueError,
            "Extracting bits using a non integer start (e.g., 1.1) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[-2:3],
            ValueError,
            "Extracting bits using a negative start (e.g., -2) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[1:3.1],  # type: ignore
            ValueError,
            "Extracting bits using a non integer stop (e.g., 3.1) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[1:-2],
            ValueError,
            "Extracting bits using a negative stop (e.g., -2) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[1:3:1.1],  # type: ignore
            ValueError,
            "Extracting bits using a non integer step (e.g., 1.1) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[1:3:0],
            ValueError,
            "Extracting bits using zero step isn't supported",
        ),
        (
            0b_1001_1011,
            lambda x: fhe.bits(x)[::-1],
            ValueError,
            "Extracting bits in reverse (step < 0) isn't supported without providing the start bit",
        ),
        (
            0b_1001_1011,
            lambda x: fhe.bits(x)[-1],
            ValueError,
            "Extracting bits from the back (index == -1 < 0) isn't supported",
        ),
        (
            100,
            lambda x: fhe.bits(x)[2.1],  # type: ignore
            ValueError,
            "Bits of 100 cannot be extracted using 2.1 since it's not an integer or a slice",
        ),
        (
            3.2,
            lambda x: fhe.bits(x)[0],
            ValueError,
            "Bits of 3.2 cannot be extracted since it's not an integer",
        ),
        (
            -100,
            lambda x: fhe.bits(x)[1:],
            ValueError,
            (
                "Extracting bits without an upper bound (stop is None) "
                "isn't supported on signed values (e.g., -100)"
            ),
        ),
    ],
)
def test_bad_plain_bit_extraction(
    sample,
    operation,
    expected_error,
    expected_message,
):
    """
    Test plain bit extraction with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        operation(sample)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "input_bit_width,input_is_signed,operation",
    [
        # unsigned
        pytest.param(3, False, lambda x: fhe.bits(x)[0:3], id="unsigned-3b[0:3]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[0], id="unsigned-5b[0]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1], id="unsigned-5b[1]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[2], id="unsigned-5b[2]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[3], id="unsigned-5b[3]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[4], id="unsigned-5b[4]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[30], id="unsigned-5b[30]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1:3], id="unsigned-5b[1:3]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[:3], id="unsigned-5b[:3]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1:], id="unsigned-5b[1:]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1:6:2], id="unsigned-5b[1:6:2]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[3:1:-1], id="unsigned-5b[3:1:-1]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[2::-1], id="unsigned-5b[2::-1]"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1:30:10], id="unsigned-5b[1:30:10]"),
        # signed
        pytest.param(3, True, lambda x: fhe.bits(x)[0:3], id="signed-3b[0:3]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[0], id="signed-5b[0]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[1], id="signed-5b[1]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[2], id="signed-5b[2]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[3], id="signed-5b[3]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[4], id="signed-5b[4]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[30], id="signed-5b[30]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[1:3], id="signed-5b[1:3]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[:3], id="signed-5b[:3]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[1:6:2], id="signed-5b[1:6:2]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[3:1:-1], id="signed-5b[3:1:-1]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[2::-1], id="signed-5b[2::-1]"),
        pytest.param(5, True, lambda x: fhe.bits(x)[1:30:10], id="signed-5b[1:30:10]"),
        # unsigned (result bit-width increased)
        pytest.param(3, False, lambda x: fhe.bits(x)[0:3] + 100, id="unsigned-3b[0:3] + 100"),
        pytest.param(5, False, lambda x: fhe.bits(x)[0] + 100, id="unsigned-5b[0] + 100"),
        pytest.param(5, False, lambda x: fhe.bits(x)[1:3] + 100, id="unsigned-5b[1:3] + 100"),
        # signed (result bit-width increased)
        pytest.param(3, True, lambda x: fhe.bits(x)[0:3], id="signed-3b[0:3] + 100"),
        pytest.param(5, True, lambda x: fhe.bits(x)[0] + 100, id="signed-5b[0] + 100"),
        pytest.param(5, True, lambda x: fhe.bits(x)[1:3] + 100, id="signed-5b[1:3] + 100"),
    ],
)
def test_bit_extraction(input_bit_width, input_is_signed, operation, helpers):
    """
    Test bit extraction.
    """

    lower_bound = 0 if not input_is_signed else -(2 ** (input_bit_width - 1))
    upper_bound = 2 ** (input_bit_width if not input_is_signed else (input_bit_width - 1))

    sizes = [(), (2,), (3, 2)]
    for size in sizes:
        inputset = [
            np.random.randint(lower_bound, upper_bound, size=size)
            for _ in range(2**input_bit_width)
        ]

        compiler = fhe.Compiler(operation, {"x": "encrypted"})
        circuit = compiler.compile(inputset, helpers.configuration())
        values = inputset if len(inputset) <= 8 else random.sample(inputset, 8)
        for value in values:
            helpers.check_execution(circuit, operation, value, retries=3)


def mlir_count_ops(mlir, operation):
    """
    Count op in mlir.
    """
    return sum(operation in line for line in mlir.splitlines())


def test_highest_bit_extraction_mlir(helpers):
    """
    Test bit extraction of the highest bit. Saves one lsb.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return fhe.bits(x)[precision - 1]

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == precision - 1
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_bits_extraction_to_same_bitwidth_mlir(helpers):
    """
    Test bit extraction to same.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return tuple(fhe.bits(x)[i] for i in range(precision))

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == precision - 1
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_bits_extraction_to_bigger_bitwidth_mlir(helpers):
    """
    Test bit extraction to bigger bitwidth.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return tuple(fhe.bits(x)[i] + (2**precision + 1) for i in range(precision))

    circuit = operation.compile(inputset, helpers.configuration())
    print(circuit.mlir)
    assert mlir_count_ops(circuit.mlir, "lsb") == precision
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_seq_bits_extraction_to_same_bitwidth_mlir(helpers):
    """
    Test sequential bit extraction to smaller bitwidth.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return tuple(fhe.bits(x)[i] + (2**precision - 2) for i in range(precision))

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == precision
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_seq_bits_extraction_to_smaller_bitwidth_mlir(helpers):
    """
    Test sequential bit extraction to smaller bitwidth.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return tuple(fhe.bits(x)[i] for i in range(precision))

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == precision - 1
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_seq_bits_extraction_to_bigger_bitwidth_mlir(helpers):
    """
    Test sequential bit extraction to bigger bitwidth.
    """

    precision = 8
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return tuple(fhe.bits(x)[i] + 2 ** (precision + 1) for i in range(precision))

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == precision
    assert mlir_count_ops(circuit.mlir, "lookup") == 0


def test_bit_extract_to_1_tlu(helpers):
    """
    Test bit extract as 1 tlu for small precision.
    """
    precision = 3
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):
        return fhe.bits(x)[0:2]

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == 0
    assert mlir_count_ops(circuit.mlir, "lookup") == 1

    precision = 4
    inputset = list(range(2**precision))

    @fhe.compiler({"x": "encrypted"})
    def operation(x):  # pylint: disable=function-redefined
        return fhe.bits(x)[0:2]

    circuit = operation.compile(inputset, helpers.configuration())
    assert mlir_count_ops(circuit.mlir, "lsb") == 2
    assert mlir_count_ops(circuit.mlir, "lookup") == 0
