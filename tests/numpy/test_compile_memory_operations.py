"""Test module for memory operations."""

import numpy
import pytest

from concrete.common.data_types import UnsignedInteger
from concrete.common.values import EncryptedTensor
from concrete.numpy import compile_numpy_function


@pytest.mark.parametrize(
    "function,parameters,inputset,test_input,expected_output",
    [
        pytest.param(
            lambda x: x.flatten(),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3, 2)) for _ in range(10)],
            [[0, 1], [1, 2], [2, 3]],
            [0, 1, 1, 2, 2, 3],
        ),
        pytest.param(
            lambda x: x.flatten(),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 4, 5, 6)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 4, 5, 6)) for _ in range(10)],
            (numpy.arange(720) % 10).reshape((2, 3, 4, 5, 6)),
            (numpy.arange(720) % 10),
        ),
        pytest.param(
            lambda x: x.reshape((1, 3)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3,)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3,)) for _ in range(10)],
            [5, 9, 1],
            [[5, 9, 1]],
        ),
        pytest.param(
            lambda x: x.reshape((3, 1)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3,)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3,)) for _ in range(10)],
            [5, 9, 1],
            [[5], [9], [1]],
        ),
        pytest.param(
            lambda x: x.reshape((3, 2)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3, 2)) for _ in range(10)],
            [[0, 1], [1, 2], [2, 3]],
            [[0, 1], [1, 2], [2, 3]],
        ),
        pytest.param(
            lambda x: x.reshape((3, 2)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3)) for _ in range(10)],
            [[0, 1, 1], [2, 2, 3]],
            [[0, 1], [1, 2], [2, 3]],
        ),
        pytest.param(
            lambda x: x.reshape(-1),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3, 2)) for _ in range(10)],
            [[0, 1], [1, 2], [2, 3]],
            [0, 1, 1, 2, 2, 3],
        ),
        pytest.param(
            lambda x: x.reshape((2, 2, 3)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(4, 3)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(4, 3)) for _ in range(10)],
            (numpy.arange(12) % 10).reshape((4, 3)),
            (numpy.arange(12) % 10).reshape((2, 2, 3)),
        ),
        pytest.param(
            lambda x: x.reshape((4, 3)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 2, 3)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 2, 3)) for _ in range(10)],
            (numpy.arange(12) % 10).reshape((2, 2, 3)),
            (numpy.arange(12) % 10).reshape((4, 3)),
        ),
        pytest.param(
            lambda x: x.reshape((3, 2, 2)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3, 4)) for _ in range(10)],
            (numpy.arange(12) % 10).reshape((3, 4)),
            (numpy.arange(12) % 10).reshape((3, 2, 2)),
        ),
        pytest.param(
            lambda x: x.reshape((3, 4)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(3, 2, 2)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(3, 2, 2)) for _ in range(10)],
            (numpy.arange(12) % 10).reshape((3, 2, 2)),
            (numpy.arange(12) % 10).reshape((3, 4)),
        ),
        pytest.param(
            lambda x: x.reshape((5, 3, 2)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(6, 5)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(6, 5)) for _ in range(10)],
            (numpy.arange(30) % 10).reshape((6, 5)),
            (numpy.arange(30) % 10).reshape((5, 3, 2)),
        ),
        pytest.param(
            lambda x: x.reshape((5, 6)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 5)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 5)) for _ in range(10)],
            (numpy.arange(30) % 10).reshape((2, 3, 5)),
            (numpy.arange(30) % 10).reshape((5, 6)),
        ),
        pytest.param(
            lambda x: x.reshape((6, 4, 30)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 4, 5, 6)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 4, 5, 6)) for _ in range(10)],
            (numpy.arange(720) % 10).reshape((2, 3, 4, 5, 6)),
            (numpy.arange(720) % 10).reshape((6, 4, 30)),
        ),
        pytest.param(
            lambda x: x.reshape((2, 60, 6)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 4, 5, 6)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 4, 5, 6)) for _ in range(10)],
            (numpy.arange(720) % 10).reshape((2, 3, 4, 5, 6)),
            (numpy.arange(720) % 10).reshape((2, 60, 6)),
        ),
        pytest.param(
            lambda x: x.reshape((6, 6, -1)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 2, 3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 2, 3, 4)) for _ in range(10)],
            (numpy.arange(144) % 10).reshape((2, 3, 2, 3, 4)),
            (numpy.arange(144) % 10).reshape((6, 6, -1)),
        ),
        pytest.param(
            lambda x: x.reshape((6, -1, 12)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 2, 3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 2, 3, 4)) for _ in range(10)],
            (numpy.arange(144) % 10).reshape((2, 3, 2, 3, 4)),
            (numpy.arange(144) % 10).reshape((6, -1, 12)),
        ),
        pytest.param(
            lambda x: x.reshape((-1, 18, 4)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 2, 3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 2, 3, 4)) for _ in range(10)],
            (numpy.arange(144) % 10).reshape((2, 3, 2, 3, 4)),
            (numpy.arange(144) % 10).reshape((-1, 18, 4)),
        ),
    ],
)
def test_memory_operation_run_correctness(
    function,
    parameters,
    inputset,
    test_input,
    expected_output,
    default_compilation_configuration,
    check_array_equality,
):
    """
    Test correctness of results when running a compiled function with memory operators.

    e.g.,
    - flatten
    - reshape
    """
    circuit = compile_numpy_function(
        function,
        parameters,
        inputset,
        default_compilation_configuration,
    )

    actual = circuit.run(numpy.array(test_input, dtype=numpy.uint8))
    expected = numpy.array(expected_output, dtype=numpy.uint8)

    check_array_equality(actual, expected)


@pytest.mark.parametrize(
    "function,parameters,inputset,error,match",
    [
        pytest.param(
            lambda x: x.reshape((-1, -1, 2)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 4)) for _ in range(10)],
            ValueError,
            "shapes are not compatible (old shape (2, 3, 4), new shape (-1, -1, 2))",
        ),
        pytest.param(
            lambda x: x.reshape((3, -1, 3)),
            {
                "x": EncryptedTensor(UnsignedInteger(4), shape=(2, 3, 4)),
            },
            [numpy.random.randint(0, 2 ** 4, size=(2, 3, 4)) for _ in range(10)],
            ValueError,
            "shapes are not compatible (old shape (2, 3, 4), new shape (3, -1, 3))",
        ),
    ],
)
def test_memory_operation_failed_compilation(
    function,
    parameters,
    inputset,
    error,
    match,
    default_compilation_configuration,
):
    """
    Test compilation failures of compiled function with memory operations.

    e.g.,
    - reshape
    """

    with pytest.raises(error) as excinfo:
        compile_numpy_function(
            function,
            parameters,
            inputset,
            default_compilation_configuration,
        )

    assert (
        str(excinfo.value) == match
    ), f"""

Actual Output
=============
{excinfo.value}

Expected Output
===============
{match}

        """
