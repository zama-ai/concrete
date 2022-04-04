"""
Tests of `Tracer` class.
"""

import numpy as np
import pytest

from concrete.numpy.dtypes import UnsignedInteger
from concrete.numpy.tracing import Tracer
from concrete.numpy.values import EncryptedTensor


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
            lambda x: np.multiply.outer(x, [1, 2, 3]),
            {"x": EncryptedTensor(UnsignedInteger(7), shape=(4,))},
            RuntimeError,
            "Only __call__ hook is supported for numpy ufuncs",
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
