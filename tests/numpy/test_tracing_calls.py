"""Test file for numpy tracing"""

from copy import deepcopy

import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.representation import intermediate as ir
from concrete.common.values import EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing

OPERATIONS_TO_TEST = [ir.Add, ir.Sub, ir.Mul]

# Functions from tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC, whose output
# is a float64, whatever the input type
LIST_OF_UFUNC_WHOSE_OUTPUT_IS_FLOAT64 = set(
    [
        numpy.arccos,
        numpy.arccosh,
        numpy.arcsin,
        numpy.arcsinh,
        numpy.arctan,
        numpy.arctanh,
        numpy.cbrt,
        numpy.ceil,
        numpy.cos,
        numpy.cosh,
        numpy.deg2rad,
        numpy.degrees,
        numpy.exp,
        numpy.exp2,
        numpy.expm1,
        numpy.fabs,
        numpy.floor,
        numpy.log,
        numpy.log10,
        numpy.log1p,
        numpy.log2,
        numpy.rad2deg,
        numpy.radians,
        numpy.rint,
        numpy.sin,
        numpy.sinh,
        numpy.spacing,
        numpy.sqrt,
        numpy.tan,
        numpy.tanh,
        numpy.trunc,
    ]
)

# Functions from tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC, whose output
# is a boolean, whatever the input type
LIST_OF_UFUNC_WHOSE_OUTPUT_IS_BOOL = set(
    [
        numpy.isfinite,
        numpy.isinf,
        numpy.isnan,
        numpy.signbit,
        numpy.logical_not,
    ]
)


@pytest.mark.parametrize(
    "inputs,expected_output_node",
    [
        pytest.param(
            {"x": EncryptedScalar(Integer(7, is_signed=False))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(32, is_signed=True))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(64, is_signed=True))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(128, is_signed=True))},
            ir.GenericFunction,
            marks=pytest.mark.xfail(strict=True, raises=NotImplementedError),
        ),
        pytest.param(
            {"x": EncryptedScalar(Float(64))},
            ir.GenericFunction,
        ),
    ],
)
@pytest.mark.parametrize(
    "function_to_trace_def",
    [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 1],
)
def test_trace_numpy_supported_unary_ufuncs(inputs, expected_output_node, function_to_trace_def):
    """Function to trace supported numpy ufuncs"""

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    # pylint: disable=cell-var-from-loop
    function_to_trace = lambda x: function_to_trace_def(x)  # noqa: E731
    # pylint: enable=cell-var-from-loop

    op_graph = tracing.trace_numpy_function(function_to_trace, inputs)

    assert len(op_graph.output_nodes) == 1
    assert isinstance(op_graph.output_nodes[0], expected_output_node)
    assert len(op_graph.output_nodes[0].outputs) == 1

    if function_to_trace_def in LIST_OF_UFUNC_WHOSE_OUTPUT_IS_FLOAT64:
        assert op_graph.output_nodes[0].outputs[0] == EncryptedScalar(Float(64))
    elif function_to_trace_def in LIST_OF_UFUNC_WHOSE_OUTPUT_IS_BOOL:

        # Boolean function
        assert op_graph.output_nodes[0].outputs[0] == EncryptedScalar(Integer(8, is_signed=False))
    else:

        # Function keeping more or less input type
        input_node_type = inputs["x"]

        expected_output_node_type = deepcopy(input_node_type)

        expected_output_node_type.dtype.bit_width = max(
            expected_output_node_type.dtype.bit_width, 32
        )

        assert op_graph.output_nodes[0].outputs[0] == expected_output_node_type


@pytest.mark.parametrize("np_function", tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC)
def test_nptracer_get_tracing_func_for_np_functions(np_function):
    """Test NPTracer get_tracing_func_for_np_function"""

    expected_tracing_func = tracing.NPTracer.UFUNC_ROUTING[np_function]

    assert tracing.NPTracer.get_tracing_func_for_np_function(np_function) == expected_tracing_func


def subtest_tracing_calls(
    function_to_trace,
    input_value_input_and_expected_output_tuples,
    check_array_equality,
):
    """Test memory function managed by GenericFunction node of the form numpy.something"""
    for input_value, input_, expected_output in input_value_input_and_expected_output_tuples:

        op_graph = tracing.trace_numpy_function(function_to_trace, {"x": input_value})
        output_node = op_graph.output_nodes[0]

        node_results = op_graph.evaluate({0: input_})
        evaluated_output = node_results[output_node]
        assert isinstance(evaluated_output, type(expected_output)), type(evaluated_output)
        check_array_equality(evaluated_output, expected_output)


@pytest.mark.parametrize(
    "function_to_trace,input_value_input_and_expected_output_tuples",
    [
        (
            lambda x: numpy.transpose(x),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([[0, 2], [1, 3]]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4, 8).reshape(2, 2),
                    numpy.array([[4, 6], [5, 7]]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(42),
                ),
            ],
        ),
        (
            lambda x: numpy.transpose(x) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(3, 5).transpose(),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(84),
                ),
            ],
        ),
        (
            lambda x: numpy.ravel(x),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.array([42], dtype=numpy.int64),
                ),
            ],
        ),
        (
            lambda x: numpy.reshape(x, (5, 3)) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(5, 3),
                ),
            ],
        ),
    ],
)
def test_tracing_numpy_calls(
    function_to_trace,
    input_value_input_and_expected_output_tuples,
    check_array_equality,
):
    """Test memory function managed by GenericFunction node of the form numpy.something"""
    subtest_tracing_calls(
        function_to_trace, input_value_input_and_expected_output_tuples, check_array_equality
    )


@pytest.mark.parametrize(
    "function_to_trace,input_value_input_and_expected_output_tuples",
    [
        (
            lambda x: x.transpose() + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(3, 5).transpose(),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(84),
                ),
            ],
        ),
        (
            lambda x: x.ravel(),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.array([42], dtype=numpy.int64),
                ),
            ],
        ),
        (
            lambda x: x.reshape((5, 3)) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(5, 3),
                ),
            ],
        ),
        pytest.param(
            lambda x: x.reshape((5, 3)),
            [
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    None,
                )
            ],
            marks=pytest.mark.xfail(strict=True, raises=AssertionError),
        ),
    ],
)
def test_tracing_ndarray_calls(
    function_to_trace,
    input_value_input_and_expected_output_tuples,
    check_array_equality,
):
    """Test memory function managed by GenericFunction node of the form ndarray.something"""
    subtest_tracing_calls(
        function_to_trace, input_value_input_and_expected_output_tuples, check_array_equality
    )
