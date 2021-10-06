"""Test file for float subgraph fusing"""

import random
from inspect import signature

import numpy
import pytest

from concrete.common.data_types.integers import Integer
from concrete.common.optimization.topological import fuse_float_operations
from concrete.common.values import EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing
from concrete.numpy.tracing import trace_numpy_function


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


def complex_fuse_indirect_input(function, x, y):
    """Complex fuse"""
    intermediate = x + y
    intermediate = intermediate + 2
    intermediate = intermediate.astype(numpy.float32)
    intermediate = intermediate.astype(numpy.int32)
    x_p_1 = intermediate + 1.5
    x_p_2 = intermediate + 2.7
    x_p_3 = function(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def complex_fuse_direct_input(function, x, y):
    """Complex fuse"""
    x_p_1 = x + 0.1
    x_p_2 = x + 0.2
    x_p_3 = function(x_p_1 + x_p_2)
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
        pytest.param(
            lambda x, y: complex_fuse_indirect_input(numpy.rint, x, y),
            True,
            id="complex_fuse_indirect_input_with_rint",
        ),
        pytest.param(
            lambda x, y: complex_fuse_direct_input(numpy.rint, x, y),
            True,
            id="complex_fuse_direct_input_with_rint",
        ),
    ],
)
@pytest.mark.parametrize("input_", [0, 2, 42, 44])
def test_fuse_float_operations(function_to_trace, fused, input_):
    """Test function for fuse_float_operations"""

    params_names = signature(function_to_trace).parameters.keys()

    op_graph = trace_numpy_function(
        function_to_trace,
        {param_name: EncryptedScalar(Integer(32, True)) for param_name in params_names},
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


# TODO: #199 To be removed when doing tensor management
def test_tensor_no_fuse():
    """Test case to verify float fusing is only applied on functions on scalars."""

    ndim = random.randint(1, 3)
    tensor_shape = tuple(random.randint(1, 10) for _ in range(ndim + 1))

    def tensor_no_fuse(x):
        intermediate = x.astype(numpy.float64)
        intermediate = intermediate.astype(numpy.int32)
        return intermediate + numpy.ones(tensor_shape)

    function_to_trace = tensor_no_fuse
    params_names = signature(function_to_trace).parameters.keys()

    op_graph = trace_numpy_function(
        function_to_trace,
        {
            param_name: EncryptedTensor(Integer(32, True), shape=tensor_shape)
            for param_name in params_names
        },
    )
    orig_num_nodes = len(op_graph.graph)
    fuse_float_operations(op_graph)
    fused_num_nodes = len(op_graph.graph)

    assert orig_num_nodes == fused_num_nodes


def subtest_fuse_float_unary_operations_correctness(fun):
    """Test a unary function with fuse_float_operations."""

    # Some manipulation to avoid issues with domain of definitions of functions
    if fun == numpy.arccosh:
        input_list = [1, 2, 42, 44]
        super_fun_list = [complex_fuse_direct_input]
    elif fun in [numpy.arctanh, numpy.arccos, numpy.arcsin, numpy.arctan]:
        input_list = [0, 0.1, 0.2]
        super_fun_list = [complex_fuse_direct_input]
    else:
        input_list = [0, 2, 42, 44]
        super_fun_list = [complex_fuse_direct_input, complex_fuse_indirect_input]

    for super_fun in super_fun_list:

        for input_ in input_list:

            def get_function_to_trace():
                return lambda x, y: super_fun(fun, x, y)

            function_to_trace = get_function_to_trace()

            params_names = signature(function_to_trace).parameters.keys()

            op_graph = trace_numpy_function(
                function_to_trace,
                {param_name: EncryptedScalar(Integer(32, True)) for param_name in params_names},
            )
            orig_num_nodes = len(op_graph.graph)
            fuse_float_operations(op_graph)
            fused_num_nodes = len(op_graph.graph)

            assert fused_num_nodes < orig_num_nodes

            input_ = numpy.int32(input_)

            num_params = len(params_names)
            inputs = (input_,) * num_params

            assert function_to_trace(*inputs) == op_graph(*inputs)


LIST_OF_UFUNC_WHICH_HAVE_INTEGER_ONLY_SOURCES = {
    numpy.bitwise_and,
    numpy.bitwise_or,
    numpy.bitwise_xor,
    numpy.gcd,
    numpy.lcm,
    numpy.ldexp,
    numpy.left_shift,
    numpy.logical_and,
    numpy.logical_not,
    numpy.logical_or,
    numpy.logical_xor,
    numpy.remainder,
    numpy.right_shift,
}


def subtest_fuse_float_binary_operations_correctness(fun):
    """Test a binary functions with fuse_float_operations, with a constant as a source."""

    for i in range(4):

        # Know if the function is defined for integer inputs
        if fun in LIST_OF_UFUNC_WHICH_HAVE_INTEGER_ONLY_SOURCES:
            if i not in [0, 2]:
                continue

        # The .astype(numpy.float64) that we have in cases 0 and 2 is here to force
        # a float output even for functions which return an integer (eg, XOR), such
        # that our frontend always try to fuse them

        # For bivariate functions: fix one of the inputs
        if i == 0:
            # With an integer in first position
            def get_function_to_trace():
                return lambda x, y: fun(3, x + y).astype(numpy.float64).astype(numpy.int32)

        elif i == 1:
            # With a float in first position
            def get_function_to_trace():
                return lambda x, y: fun(2.3, x + y).astype(numpy.int32)

        elif i == 2:
            # With an integer in second position
            def get_function_to_trace():
                return lambda x, y: fun(x + y, 4).astype(numpy.float64).astype(numpy.int32)

        else:
            # With a float in second position
            def get_function_to_trace():
                return lambda x, y: fun(x + y, 5.7).astype(numpy.int32)

        input_list = [0, 2, 42, 44]

        # Domain of definition
        if fun in [numpy.true_divide, numpy.remainder, numpy.floor_divide, numpy.fmod]:
            input_list = [2, 42, 44]

        for input_ in input_list:

            function_to_trace = get_function_to_trace()

            params_names = signature(function_to_trace).parameters.keys()

            op_graph = trace_numpy_function(
                function_to_trace,
                {param_name: EncryptedScalar(Integer(32, True)) for param_name in params_names},
            )
            orig_num_nodes = len(op_graph.graph)
            fuse_float_operations(op_graph)
            fused_num_nodes = len(op_graph.graph)

            assert fused_num_nodes < orig_num_nodes

            input_ = numpy.int32(input_)

            num_params = len(params_names)
            inputs = (input_,) * num_params

            assert function_to_trace(*inputs) == op_graph(*inputs)


def subtest_fuse_float_binary_operations_dont_support_two_variables(fun):
    """Test a binary function with fuse_float_operations, with no constant as
    a source."""

    def get_function_to_trace():
        return lambda x, y: fun(x, y).astype(numpy.int32)

    function_to_trace = get_function_to_trace()

    params_names = signature(function_to_trace).parameters.keys()

    with pytest.raises(NotImplementedError, match=r"Can't manage binary operator"):
        trace_numpy_function(
            function_to_trace,
            {param_name: EncryptedScalar(Integer(32, True)) for param_name in params_names},
        )


@pytest.mark.parametrize("fun", tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC)
def test_ufunc_operations(fun):
    """Test functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""

    if fun.nin == 1:
        subtest_fuse_float_unary_operations_correctness(fun)
    elif fun.nin == 2:
        subtest_fuse_float_binary_operations_correctness(fun)
        subtest_fuse_float_binary_operations_dont_support_two_variables(fun)
    else:
        raise NotImplementedError("Only unary and binary functions are tested for now")
