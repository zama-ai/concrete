"""Test file for numpy tracing"""

import inspect

import numpy
import pytest

from concrete.common.data_types.integers import Integer
from concrete.common.representation import intermediate as ir
from concrete.common.values import ClearScalar, EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(
            {"x": EncryptedScalar(Integer(32, is_signed=True))},
        ),
    ],
)
@pytest.mark.parametrize(
    "function_to_trace",
    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint is not happy
    # with it
    [lambda x: numpy.invert(x), lambda x: numpy.bitwise_not(x)],
)
def test_trace_numpy_fails_for_invert(inputs, function_to_trace):
    """Check we catch calls to numpy.invert and tell user to change their code"""

    with pytest.raises(RuntimeError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert (
        "NPTracer does not manage the following func: invert. Please replace by calls to "
        "bitwise_xor with appropriate mask" in str(excinfo.value)
    )


def test_trace_numpy_ufuncs_not_supported():
    """Testing a failure case of trace_numpy_function"""
    inputs = {"x": EncryptedScalar(Integer(128, is_signed=True))}

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x: numpy.add.reduce(x)  # noqa: E731

    with pytest.raises(NotImplementedError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert "Only __call__ method is supported currently" in str(excinfo.value)


def test_trace_numpy_ufuncs_no_kwargs_no_extra_args():
    """Test a case where kwargs are not allowed and too many inputs are passed"""
    inputs = {
        "x": EncryptedScalar(Integer(32, is_signed=True)),
        "y": EncryptedScalar(Integer(32, is_signed=True)),
        "z": EncryptedScalar(Integer(32, is_signed=True)),
    }

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x, y, z: numpy.add(x, y, z)  # noqa: E731

    with pytest.raises(AssertionError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    # numpy only passes ufunc.nin tracers so the extra arguments are passed as kwargs
    assert "**kwargs are currently not supported for numpy ufuncs, ufunc: add" in str(excinfo.value)

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x, y, z: numpy.add(x, y, out=z)  # noqa: E731

    with pytest.raises(AssertionError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert "**kwargs are currently not supported for numpy ufuncs, ufunc: add" in str(excinfo.value)


def test_nptracer_get_tracing_func_for_np_functions_not_implemented():
    """Check NPTracer in case of not-implemented function"""
    with pytest.raises(NotImplementedError) as excinfo:
        tracing.NPTracer.get_tracing_func_for_np_function(numpy.conjugate)

    assert "NPTracer does not yet manage the following func: conjugate" in str(excinfo.value)


@pytest.mark.parametrize(
    "operation,exception_type,match",
    [
        pytest.param(
            lambda x: x + "fail",
            TypeError,
            "unsupported operand type(s) for +: 'NPTracer' and 'str'",
        ),
        pytest.param(
            lambda x: "fail" + x,
            TypeError,
            'can only concatenate str (not "NPTracer") to str',
        ),
        pytest.param(
            lambda x: x - "fail",
            TypeError,
            "unsupported operand type(s) for -: 'NPTracer' and 'str'",
        ),
        pytest.param(
            lambda x: "fail" - x,
            TypeError,
            "unsupported operand type(s) for -: 'str' and 'NPTracer'",
        ),
        pytest.param(
            lambda x: x * "fail",
            TypeError,
            "can't multiply sequence by non-int of type 'NPTracer'",
        ),
        pytest.param(
            lambda x: "fail" * x,
            TypeError,
            "can't multiply sequence by non-int of type 'NPTracer'",
        ),
        pytest.param(
            lambda x: x / "fail",
            TypeError,
            "unsupported operand type(s) for /: 'NPTracer' and 'str'",
        ),
        pytest.param(
            lambda x: "fail" / x,
            TypeError,
            "unsupported operand type(s) for /: 'str' and 'NPTracer'",
        ),
        pytest.param(
            lambda x: x // "fail",
            TypeError,
            "unsupported operand type(s) for //: 'NPTracer' and 'str'",
        ),
        pytest.param(
            lambda x: "fail" // x,
            TypeError,
            "unsupported operand type(s) for //: 'str' and 'NPTracer'",
        ),
        pytest.param(
            lambda x, y: x / y, NotImplementedError, "Can't manage binary operator truediv"
        ),
        pytest.param(
            lambda x, y: x // y, NotImplementedError, "Can't manage binary operator floordiv"
        ),
    ],
)
def test_nptracer_unsupported_operands(operation, exception_type, match):
    """Test cases where NPTracer cannot be used with other operands."""
    tracers = [
        tracing.NPTracer([], ir.Input(ClearScalar(Integer(32, True)), param_name, idx), 0)
        for idx, param_name in enumerate(inspect.signature(operation).parameters.keys())
    ]

    with pytest.raises(exception_type) as exc_info:
        _ = operation(*tracers)

    assert match in str(exc_info)


@pytest.mark.parametrize(
    "lambda_f,params",
    [
        (
            lambda x: numpy.reshape(x, (5, 3)),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(7, 5)),
            },
        ),
    ],
)
def test_errors_with_generic_function(lambda_f, params):
    "Test some errors with generic function"
    with pytest.raises(ValueError) as excinfo:
        tracing.trace_numpy_function(lambda_f, params)

    assert "shapes are not compatible (old shape (7, 5), new shape (5, 3))" in str(excinfo.value)
