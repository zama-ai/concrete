"""
Tests errors returned by the compiler.
"""

import inspect

import numpy as np
import pytest

from concrete import fhe

# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-argument, unused-variable, no-member, unused-argument, function-redefined, expression-not-assigned
# same disables for ruff:
# ruff: noqa: N805, E501, F841, ARG002, F811, B015, RUF001


def test_non_composable(helpers):
    """
    Test optimizer error for lack of refresh.
    """

    @fhe.compiler({"x": "encrypted"})
    def circuit(x):
        return x * 2

    line = inspect.currentframe().f_lineno - 2
    inputset = range(100)
    config = helpers.configuration().fork(composable=True, parameter_selection_strategy="MULTI")

    with pytest.raises(RuntimeError) as excinfo:
        circuit = circuit.compile(inputset, config)

    assert (
        str(excinfo.value)
        == f"Program can not be composed (see https://docs.zama.ai/concrete/compilation/common_errors#id-9.-non-composable-circuit): \
At location test_optimizer_errors.py:{line}:0:\nThe noise of the node 0 is contaminated by noise coming straight from the input \
(partition: 0, coeff: 4.00)."
    )


def test_unfeasible(helpers):
    """
    Test optimizer error for unfeasible circuit.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def a(x):
            return fhe.refresh(x * 10)

        @fhe.function({"x": "encrypted"})
        def b(x):
            return fhe.refresh(x * 1000)

    line = inspect.currentframe().f_lineno - 2
    inputset = [np.random.randint(1, 1000, size=()) for _ in range(100)]

    with pytest.raises(RuntimeError) as excinfo:
        module = Module.compile({"a": inputset, "b": inputset}, p_error=0.000001)

    assert (
        str(excinfo.value)
        == f"Unfeasible noise constraint encountered (see https://docs.zama.ai/concrete/compilation/common_errors#id-8.-unfeasible-noise-constraint): \
At location test_optimizer_errors.py:{line}:0:\n21990232555520000000σ²Br[0] + 1σ²K[0] + 1σ²M[0] < (2²)**-4.5 (0bits partition:0 count:1, dom=73)."
    )
