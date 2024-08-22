"""
Tests of `Tracer` class.
"""

import numpy as np
import pytest

from concrete.fhe.dtypes import UnsignedInteger
from concrete.fhe.tracing import Tracer
from concrete.fhe.tracing.typing import uint4
from concrete.fhe.values import EncryptedTensor


def bad_assignment(x):
    """
    Function with an invalid assignment.
    """
    idx = [[1, 2], [1, 2, 3], [1]]
    x[idx] = 0
    return x


@pytest.mark.parametrize(
    "function,parameters,expected_error,expected_message",
    [
        pytest.param(
            lambda x: np.ravel(x),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            RuntimeError,
            "Function 'np.ravel' is not supported",
        ),
        pytest.param(
            lambda x: np.sum(x, initial=42),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            RuntimeError,
            "Function 'np.sum' is not supported with kwarg 'initial'",
        ),
        pytest.param(
            lambda x: np.absolute(x, where=False),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(1, 2, 3))},
            RuntimeError,
            "Function 'np.absolute' is not supported with kwarg 'where'",
        ),
        pytest.param(
            lambda x: np.multiply.outer(x, [1, 2, 3]),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(4,))},
            RuntimeError,
            "Only __call__ hook is supported for numpy ufuncs",
        ),
        pytest.param(
            lambda x: x.astype(uint4),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(4,))},
            ValueError,
            "`astype` method must be called with a numpy type "
            "for compilation (e.g., value.astype(np.int64))",
        ),
        pytest.param(
            lambda x: x + 1 if x else x + x,
            {"x": EncryptedTensor(UnsignedInteger(7), shape=())},
            RuntimeError,
            "Branching within circuits is not possible",
        ),
        pytest.param(
            lambda x: x[["abc", 3, 2.2, (1,)]],
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            ValueError,
            "Tracer<output=EncryptedTensor<uint7, shape=(3, 2)>> "
            "cannot be indexed with ['abc', 3, 2.2, (1,)]",
        ),
        pytest.param(
            bad_assignment,
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            ValueError,
            "Tracer<output=EncryptedTensor<uint7, shape=(3, 2)>>[[[1, 2], [1, 2, 3], [1]]] "
            "cannot be assigned 0",
        ),
    ],
)
def test_tracer_bad_trace(function, parameters, expected_error, expected_message):
    """
    Test `trace` function of `Tracer` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Tracer.trace(function, parameters)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "function,parameters,expected_message",
    [
        pytest.param(
            lambda x: x.astype(np.int8),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.int16),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.int32),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.uint8),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.uint16),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.uint32),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
        pytest.param(
            lambda x: x.astype(np.uint64),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(3, 2))},
            (
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            ),
        ),
    ],
)
def test_tracer_warning_trace(function, parameters, expected_message, capsys):
    """
    Test `trace` function of `Tracer` class with parameters that result in a warning.
    """

    Tracer.trace(function, parameters)

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_message
