"""Test file for intermediate node to MLIR converter."""

import random

import numpy
import pytest

from concrete.common.data_types import UnsignedInteger
from concrete.common.values import EncryptedScalar, EncryptedTensor
from concrete.numpy import compile_numpy_function


@pytest.mark.parametrize(
    "function_to_compile,parameters,inputset,expected_error_type,expected_error_message",
    [
        pytest.param(
            lambda x, y: x * y,
            {
                "x": EncryptedScalar(UnsignedInteger(3)),
                "y": EncryptedScalar(UnsignedInteger(3)),
            },
            [(random.randint(0, 7), random.randint(0, 7)) for _ in range(10)] + [(7, 7)],
            NotImplementedError,
            "Multiplication "
            "between "
            "EncryptedScalar<uint6> "
            "and "
            "EncryptedScalar<uint6> "
            "cannot be converted to MLIR yet",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": EncryptedScalar(UnsignedInteger(3)),
                "y": EncryptedScalar(UnsignedInteger(3)),
            },
            [(random.randint(5, 7), random.randint(0, 5)) for _ in range(10)],
            NotImplementedError,
            "Subtraction "
            "of "
            "EncryptedScalar<uint3> "
            "from "
            "EncryptedScalar<uint3> "
            "cannot be converted to MLIR yet",
        ),
        pytest.param(
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2,)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(2,)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(2,)),
                    numpy.random.randint(0, 2 ** 3, size=(2,)),
                )
                for _ in range(10)
            ]
            + [(numpy.array([7, 7]), numpy.array([7, 7]))],
            NotImplementedError,
            "Dot product "
            "between "
            "EncryptedTensor<uint7, shape=(2,)> "
            "and "
            "EncryptedTensor<uint7, shape=(2,)> "
            "cannot be converted to MLIR yet",
        ),
        pytest.param(
            lambda x: numpy.ones(shape=(2, 3), dtype=numpy.uint32) @ x,
            {"x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)]
            + [numpy.array([[7, 7], [7, 7], [7, 7]])],
            NotImplementedError,
            "Matrix multiplication "
            "between "
            "ClearTensor<uint6, shape=(2, 3)> "
            "and "
            "EncryptedTensor<uint5, shape=(3, 2)> "
            "cannot be converted to MLIR yet "
            "(notice the encrypted value is in the right hand side which is not supported)",
        ),
    ],
)
def test_fail_node_conversion(
    function_to_compile,
    parameters,
    inputset,
    expected_error_type,
    expected_error_message,
    default_compilation_configuration,
):
    """Test function for failed intermediate node conversion."""

    with pytest.raises(expected_error_type) as excinfo:
        compile_numpy_function(
            function_to_compile, parameters, inputset, default_compilation_configuration
        )

    assert str(excinfo.value) == expected_error_message
