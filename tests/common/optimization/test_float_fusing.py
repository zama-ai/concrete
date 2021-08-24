"""Test file for float subgraph fusing"""

from inspect import signature

import numpy
import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.optimization.topological import fuse_float_operations
from hdk.common.values import EncryptedValue
from hdk.hnumpy.tracing import trace_numpy_function


def no_fuse(x):
    """No fuse"""
    return x + 2


def no_fuse_unhandled(x, y):
    """No fuse unhandled"""
    x_1 = x + 0.7
    y_1 = y + 1.3
    intermediate = x_1 + y_1
    return intermediate.astype(numpy.int32)


def simple_fuse_not_output(x):
    """Simple fuse not output"""
    intermediate = x.astype(numpy.float64)
    intermediate = intermediate.astype(numpy.int32)
    return intermediate + 2


def simple_fuse_output(x):
    """Simple fuse output"""
    return x.astype(numpy.float64).astype(numpy.int32)


def complex_fuse_indirect_input(x, y):
    """Complex fuse"""
    intermediate = x + y
    intermediate = intermediate + 2
    intermediate = intermediate.astype(numpy.float32)
    intermediate = intermediate.astype(numpy.int32)
    x_p_1 = intermediate + 1.5
    x_p_2 = intermediate + 2.7
    x_p_3 = numpy.rint(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def complex_fuse_direct_input(x, y):
    """Complex fuse"""
    x_p_1 = x + 1.5
    x_p_2 = x + 2.7
    x_p_3 = numpy.rint(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


@pytest.mark.parametrize(
    "function_to_trace,fused",
    [
        pytest.param(no_fuse, False, id="no_fuse"),
        pytest.param(no_fuse_unhandled, False, id="no_fuse_unhandled"),
        pytest.param(simple_fuse_not_output, True, id="no_fuse"),
        pytest.param(simple_fuse_output, True, id="no_fuse"),
        pytest.param(complex_fuse_indirect_input, True, id="complex_fuse_indirect_input"),
        pytest.param(complex_fuse_direct_input, True, id="complex_fuse_direct_input"),
    ],
)
@pytest.mark.parametrize("input_", [0, 2, 42, 44])
def test_fuse_float_operations(function_to_trace, fused, input_):
    """Test function for fuse_float_operations"""

    params_names = signature(function_to_trace).parameters.keys()

    op_graph = trace_numpy_function(
        function_to_trace,
        {param_name: EncryptedValue(Integer(32, True)) for param_name in params_names},
    )
    orig_num_nodes = len(op_graph.graph)
    fuse_float_operations(op_graph)
    fused_num_nodes = len(op_graph.graph)

    if fused:
        assert fused_num_nodes < orig_num_nodes
    else:
        assert fused_num_nodes == orig_num_nodes

    input_ = numpy.int32(input_)

    num_params = len(params_names)
    inputs = (input_,) * num_params
    assert function_to_trace(*inputs) == op_graph(*inputs)
